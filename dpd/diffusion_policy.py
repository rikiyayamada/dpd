from dataclasses import dataclass
from typing import Callable, Any
from collections import deque
from pathlib import Path

import torch
from torch import nn
from diffusers import SchedulerMixin
from diffusers.optimization import get_scheduler
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from .nets import DiffusionModel, MLP
from .utils import Normalizer, EMAModel

class HistoryEncoder(nn.Module):
    def __init__(
        self,
        state_dim,
        history_len,
        hidden_dims,
        output_dim,
        dropout,
    ):
        super().__init__()
        self.history_len = history_len
        self.mlp = MLP(
            input_dim=state_dim * history_len,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )
    
    def forward(
        self,
        history,
    ):
        batch_size = history.shape[0]
        curr_state = history[:, -1, :]
        if self.history_len > 1:
            diff = (history[:, 1:, :] - history[:, :-1, :]).reshape(batch_size, -1)
            x = torch.cat((diff, curr_state), dim=-1)
        else:
            x = curr_state
        out = self.mlp(x)
        return out

class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        embed_dim,
        mlp_hidden_dim,
        cond_dim,
        cond_hidden_dims,
        num_blocks,
        history_len,
        history_encoder_hidden_dims,
        dropout,
        noise_scheduler: SchedulerMixin,
        inference_steps,
        state_normalizer: Normalizer = None,
        action_normalizer: Normalizer = None,
    ):
        super().__init__()

        self.history_encoder = HistoryEncoder(
            state_dim=state_dim,
            history_len=history_len,
            hidden_dims=history_encoder_hidden_dims,
            output_dim=cond_dim,
            dropout=dropout,
        )
        
        self.diffusion_model = DiffusionModel(
            data_dim=action_dim,
            embed_dim=embed_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            cond_dim=cond_dim,
            cond_hidden_dims=cond_hidden_dims,
            num_blocks=num_blocks,
            dropout=dropout,
            noise_scheduler=noise_scheduler,
        )
        
        self.action_dim = action_dim
        self.history_len = history_len
        self.inference_steps = inference_steps
        self.state_normalizer = state_normalizer or Normalizer(data=None, mode='identity', data_dim=state_dim)
        self.action_normalizer = action_normalizer or Normalizer(data=None, mode='identity', data_dim=action_dim)
    
    def compute_loss(
        self,
        history,
        action,
    ):
        loss = self.diffusion_model.compute_loss(
            x=action,
            cond=self.history_encoder(history),
        )
        return loss
    
    @property
    def device(self):
        return next(self.parameters()).device

    def reset(self):
        self.history = deque([], maxlen=self.history_len)
        self.noise = torch.randn(
            1, self.action_dim,
            dtype=torch.float32,
            device=self.device,
        )
    
    @torch.no_grad()
    def forward(
        self,
        state: np.ndarray,
    ):
        self.eval()

        normed_state = self.state_normalizer(torch.from_numpy(state).float().to(self.device))
        if self.history:
            self.history.append(normed_state)
        else:
            self.history.extend(normed_state.unsqueeze(0).repeat(self.history_len, 1))

        history = torch.stack(list(self.history)).unsqueeze(0)

        normed_action = self.diffusion_model.inference(
            inference_steps=self.inference_steps,
            cond=self.history_encoder(history),
            noise=self.noise,
        ).squeeze(0)

        action = self.action_normalizer.unnormalize(normed_action).cpu().numpy()
        return action
    
class Dataset(TorchDataset):
    def __init__(
        self,
        state_trajs: list[np.ndarray],
        action_trajs: list[np.ndarray],
        history_len,
    ):
        super().__init__()
        
        self.action_trajs = action_trajs
        self.history_len = history_len
        
        self.padded_state_trajs = []
        for state_traj in state_trajs:
            padded_traj = np.concatenate((
                np.repeat(state_traj[0:1], history_len - 1, axis=0),
                state_traj,
            ), axis=0)
            self.padded_state_trajs.append(padded_traj) 
        
        self.idxs = []
        for traj_idx, action_traj in enumerate(action_trajs):
            for t in range(len(action_traj)):
                self.idxs.append((traj_idx, t))
                
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, idx):
        traj_idx, t = self.idxs[idx]

        history = self.padded_state_trajs[traj_idx][t : t + self.history_len]

        action = self.action_trajs[traj_idx][t]
        
        return history, action

@dataclass
class Trainer:
    state_trajs: list[np.ndarray]
    action_trajs: list[np.ndarray]
    embed_dim: int
    mlp_hidden_dim: int
    cond_dim: int
    cond_hidden_dims: list[int]
    num_blocks: int
    history_len: int
    history_encoder_hidden_dims: list[int]
    dropout: float
    noise_scheduler: SchedulerMixin
    inference_steps: int
    optimizer_factory: Callable[..., Optimizer]
    ema_model_factory: Callable[..., EMAModel]
    val_ratio: float
    batch_size: int
    lr_scheduler: str
    lr_warmup_steps: int
    epochs: int
    yield_every: Any
    device: Any
    tb_writer: SummaryWriter
    output_dir: Any

    def train(self):
        self.output_dir = Path(self.output_dir)

        state_dim = self.state_trajs[0].shape[-1]
        action_dim = self.action_trajs[0].shape[-1]

        state_normalizer = Normalizer(np.concatenate(self.state_trajs))
        action_normalizer = Normalizer(np.concatenate(self.action_trajs))
        
        normed_state_trajs = [state_normalizer(torch.from_numpy(traj)) for traj in self.state_trajs]
        normed_action_trajs = [action_normalizer(torch.from_numpy(traj)) for traj in self.action_trajs]

        self.diffusion_policy = DiffusionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=self.embed_dim,
            mlp_hidden_dim=self.mlp_hidden_dim,
            cond_dim=self.cond_dim,
            cond_hidden_dims=self.cond_hidden_dims,
            num_blocks=self.num_blocks,
            history_len=self.history_len,
            history_encoder_hidden_dims=self.history_encoder_hidden_dims,
            dropout=self.dropout,
            noise_scheduler=self.noise_scheduler,
            inference_steps=self.inference_steps,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
        ).to(self.device)

        optimizer = self.optimizer_factory(params=self.diffusion_policy.parameters())

        self.ema_model = self.ema_model_factory(model=self.diffusion_policy)

        random_idxs = np.random.permutation(len(normed_action_trajs))
        val_len = int(len(normed_action_trajs) * self.val_ratio)
        train_idxs = random_idxs[val_len:]
        val_idxs = random_idxs[:val_len]
        
        train_dataset = Dataset(
            state_trajs=[normed_state_trajs[i] for i in train_idxs],
            action_trajs=[normed_action_trajs[i] for i in train_idxs],
            history_len=self.history_len,
        )
        val_dataset = Dataset(
            state_trajs=[normed_state_trajs[i] for i in val_idxs],
            action_trajs=[normed_action_trajs[i] for i in val_idxs],
            history_len=self.history_len,
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
        )
        
        lr_scheduler = get_scheduler(
            name=self.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=len(train_dataloader) * self.epochs,
        )
        
        for epoch in tqdm(range(1, self.epochs + 1), desc='Epochs', dynamic_ncols=True):
            self.diffusion_policy.train()
            train_losses = []
            for batch in train_dataloader:
                batch = [data.to(self.device) for data in batch]
                train_loss = self.diffusion_policy.compute_loss(*batch)
                train_losses.append(train_loss.item())
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                self.ema_model.update(self.diffusion_policy)
            avg_train_loss = np.mean(train_losses)
            
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = [data.to(self.device) for data in batch]
                    val_loss = self.ema_model.model.compute_loss(*batch)
                    total_val_loss += val_loss.item() * len(batch[0])
            avg_val_loss = total_val_loss / len(val_dataloader.dataset)
                
            self.tb_writer.add_scalars(
                main_tag=f'DiffusionPolicy/Loss',
                tag_scalar_dict={'Train': avg_train_loss, 'Val': avg_val_loss},
                global_step=epoch,
            )
            self.tb_writer.add_scalar(tag='lr', scalar_value=lr_scheduler.get_last_lr()[0], global_step=epoch)
            
            if (epoch in self.yield_every) if isinstance(self.yield_every, list) else (epoch % self.yield_every == 0):
                yield epoch, self.ema_model.model