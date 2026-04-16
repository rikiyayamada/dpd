import sys
import os
from pathlib import Path
import logging
import concurrent.futures
import multiprocessing as mp

logger = logging.getLogger(__name__)

if sys.platform == 'linux':
    if 'DISPLAY' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'
        logger.info('MUJOCO_GL=egl')
elif sys.platform == 'darwin':
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    logger.info('PYTORCH_ENABLE_MPS_FALLBACK=1')

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.utils.train_utils import RolloutPolicy as RobomimicRolloutPolicy
import h5py
from tqdm import tqdm

from dpd.diffusion_policy import Trainer, DiffusionPolicy
from utils.rotation_transformer import RotationTransformer

def mkdir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
OmegaConf.register_new_resolver("mkdir", mkdir, replace=True)

class RolloutPolicy(RobomimicRolloutPolicy):
    def __init__(self, policy: DiffusionPolicy, obs_keys, abs_action=False):
        self.policy = policy
        self.obs_keys = obs_keys
        self.abs_action = abs_action
        if self.abs_action:
            self.rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

    def start_episode(self):
        self.policy.reset()

    def __call__(self, ob, goal=None):
        state = np.concatenate([ob[k] for k in self.obs_keys], axis=-1)
        action =  self.policy(state)
        if self.abs_action:
            pos = action[..., :3]
            rot_6d = action[..., 3:9]
            gripper = action[..., 9:]
            rot_axis_angle = self.rotation_transformer.inverse(rot_6d)
            action = np.concatenate([pos, rot_axis_angle, gripper], axis=-1)
        return action

def rollout_worker(args):
    worker_id, env_meta, cfg, policy_state_dict, state_dim, action_dim, epoch, video_dir = args
    
    f_null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(f_null, 1)
    os.dup2(f_null, 2)
    
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": list(cfg.obs_keys)}})

    np.random.seed(cfg.seed + worker_id)
    torch.manual_seed(cfg.seed + worker_id)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta['env_name'],
        render=False,
        render_offscreen=cfg.rollout.save_video,
    )

    policy = DiffusionPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=cfg.trainer.embed_dim,
        mlp_hidden_dim=cfg.trainer.mlp_hidden_dim,
        cond_dim=cfg.trainer.cond_dim,
        cond_hidden_dims=cfg.trainer.cond_hidden_dims,
        num_blocks=cfg.trainer.num_blocks,
        history_len=cfg.trainer.history_len,
        history_encoder_hidden_dims=cfg.trainer.history_encoder_hidden_dims,
        dropout=cfg.trainer.dropout,
        noise_scheduler=instantiate(cfg.trainer.noise_scheduler),
        inference_steps=cfg.trainer.inference_steps,
    ).to('cpu')
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    rollout_policy = RolloutPolicy(policy, list(cfg.obs_keys), cfg.abs_action)

    if cfg.rollout.save_video:
        video_dir = video_dir / f'epoch_{epoch}'
        video_dir.mkdir(parents=True, exist_ok=True)

    rollout_stats, _ = TrainUtils.rollout_with_stats(
        policy=rollout_policy,
        envs={env.name: env},
        horizon=cfg.rollout.horizon,
        num_episodes=1,
        render=False,
        video_path=str(video_dir / f'{worker_id}.mp4'),
        video_skip=cfg.rollout.video_speed,
        terminate_on_success=True,
    )
    
    return rollout_stats[env.name]

@hydra.main(config_path='.', config_name='config', version_base=None)
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": list(cfg.obs_keys)}})

    env_meta = FileUtils.get_env_metadata_from_dataset(cfg.dataset_path)
    controller_configs = env_meta['env_kwargs']['controller_configs']
    for part_name, part_cfg in controller_configs['body_parts'].items():
        if cfg.abs_action:
            part_cfg['input_type'] = 'absolute'
        else:
            part_cfg['input_type'] = 'delta'
    
    state_trajs = []
    action_trajs = []
    with h5py.File(cfg.dataset_path, 'r') as f:
        names = list(f['data'].keys())[:cfg.num_demos]
        if cfg.abs_action:
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
        for name in names:
            demo = f['data'][name]
            obs = np.concatenate([demo['obs'][key][:] for key in cfg.obs_keys], axis=-1)
            last_obs = np.concatenate([demo['next_obs'][key][-1:] for key in cfg.obs_keys], axis=-1)
            state_trajs.append(np.concatenate((obs, last_obs), axis=0))
            if cfg.abs_action:
                actions = demo['actions_abs'][:]
                pos = actions[..., :3]
                rot_axis_angle = actions[..., 3:6]
                gripper = actions[..., 6:]
                rot_6d = rotation_transformer.forward(rot_axis_angle)
                actions = np.concatenate([pos, rot_6d, gripper], axis=-1)
            else:
                actions = demo['actions'][:]
            action_trajs.append(actions)
    state_dim = state_trajs[0].shape[-1]
    action_dim = action_trajs[0].shape[-1]
    
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    tb_writer = SummaryWriter(log_dir=str(output_dir / 'tb'))
    video_dir = None
    if cfg.rollout.save_video:
        video_dir = output_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)

    trainer: Trainer = instantiate(
        cfg.trainer,
        state_trajs=state_trajs,
        action_trajs=action_trajs,
        tb_writer=tb_writer,
        output_dir=output_dir,
        _convert_="all",
    )

    # env = EnvUtils.create_env_from_metadata(
    #     env_meta=env_meta,
    #     env_name=env_meta['env_name'],
    #     render=False,
    #     render_offscreen=cfg.rollout.save_video,
    # )

    # def eval(epoch, policy):
    #     rollout_stats, _ = TrainUtils.rollout_with_stats(
    #         policy=RolloutPolicy(policy, cfg.obs_keys, cfg.abs_action),
    #         envs={env.name: env},
    #         horizon=cfg.rollout.horizon,
    #         num_episodes=cfg.rollout.num_episodes,
    #         render=False,
    #         video_dir=str(video_dir),
    #         epoch=epoch,
    #         video_skip=cfg.rollout.video_speed,
    #         terminate_on_success=True,
    #     )
    #     rollout_stats = rollout_stats[env.name]
    #     success_rate = rollout_stats.get('Success_Rate', 0.0)
    #     # for key, value in rollout_stats.items():
    #     #     logger.info(f'{key}: {value:.4f}' if isinstance(value, float) else f'{key}: {value}')
    #     #     if key.startswith('Time_'):
    #     #         tb_writer.add_scalar(f'Timing_Stats/Rollout_{key[5:]}', value, epoch)
    #     #     else:
    #     #         tb_writer.add_scalar(f'Rollout/{key}', value, epoch)
    #     logger.info(f'Success Rate: {success_rate:.4f}')
    #     tb_writer.add_scalar(f'Success Rate', success_rate, epoch)
    #     return success_rate   

    def eval(epoch, policy):
        policy_state_dict = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        num_workers = cfg.rollout.num_workers
        worker_args_list = [
            (i, env_meta, cfg, policy_state_dict, state_dim, action_dim, epoch, video_dir)
            for i in range(cfg.rollout.num_episodes)
        ]
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            all_stats = list(tqdm(
                executor.map(rollout_worker, worker_args_list), 
                total=len(worker_args_list), 
            ))
        successes = [s.get('Success_Rate', 0.0) for s in all_stats]
        success_rate = sum(successes) / len(successes) if successes else 0.0
        logger.info(f'Success Rate: {success_rate:.4f}')
        tb_writer.add_scalar(f'Success Rate', success_rate, epoch)
        return success_rate

    if cfg.diffusion_policy_path is None:
        best_score = 0
        best_state_dict = None
        try:
            for epoch, policy in trainer.train():
                logging.info(f'------------ Rollout at Epoch {epoch} ------------')
                score = eval(epoch, policy)
                if best_score < score:
                    best_score = score
                    best_state_dict = policy.state_dict()
        finally:
            torch.save({
                'state_dict': trainer.ema_model.model.state_dict(),
            }, output_dir / 'final_diffusion_policy.pt')
            if best_state_dict is not None:
                torch.save({
                    'state_dict': best_state_dict,
                }, output_dir / 'best_diffusion_policy.pt')
    else:
        policy = DiffusionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=cfg.trainer.embed_dim,
            mlp_hidden_dim=cfg.trainer.mlp_hidden_dim,
            cond_dim=cfg.trainer.cond_dim,
            cond_hidden_dims=cfg.trainer.cond_hidden_dims,
            num_blocks=cfg.trainer.num_blocks,
            history_len=cfg.trainer.history_len,
            history_encoder_hidden_dims=cfg.trainer.history_encoder_hidden_dims,
            dropout=cfg.trainer.dropout,
            noise_scheduler=instantiate(cfg.trainer.noise_scheduler),
            inference_steps=cfg.trainer.inference_steps,
        ).to(cfg.trainer.device)
        policy.load_state_dict(torch.load(cfg.diffusion_policy_path, map_location=cfg.trainer.device)['state_dict'])
        eval(0, policy)

if __name__ == '__main__':
    main()