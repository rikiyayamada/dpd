import math

import torch
from torch import nn
from diffusers import SchedulerMixin

class DiTBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        mlp_hidden_dim,
        cond_dim,
        cond_hidden_dims,
        dropout,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, embed_dim),
        )

        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            MLP(
                input_dim=cond_dim,
                hidden_dims=cond_hidden_dims,
                output_dim=embed_dim * 3,
                dropout=dropout,
            )
        )

        nn.init.zeros_(self.adaln_modulation[1].net[-1].weight)
        nn.init.zeros_(self.adaln_modulation[1].net[-1].bias)
    
    def forward(
        self,
        x,
        cond,
    ):
        shift, scale, gate = self.adaln_modulation(cond).chunk(3, dim=-1)

        modulated_x = self.norm(x) * (1.0 + scale) + shift
        h = self.mlp(modulated_x)
        out = x + h * gate
        
        return out

class DiT(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        mlp_hidden_dim,
        cond_dim,
        cond_hidden_dims,
        num_blocks,
        output_dim,
        dropout,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        self.blocks = nn.ModuleList([
            DiTBlock(
                embed_dim=embed_dim,
                mlp_hidden_dim=mlp_hidden_dim,
                cond_dim=cond_dim,
                cond_hidden_dims=cond_hidden_dims,
                dropout=dropout,
            ) for _ in range(num_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)

        self.final_adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, embed_dim * 2),
        )
        
        self.output_layer = nn.Linear(embed_dim, output_dim)

        nn.init.zeros_(self.final_adaln_modulation[1].weight)
        nn.init.zeros_(self.final_adaln_modulation[1].bias)

        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(
        self,
        x,
        cond,
    ):
        embed = self.input_proj(x)
        
        for block in self.blocks:
            embed = block(embed, cond)
        
        shift, scale = self.final_adaln_modulation(cond).chunk(2, dim=-1)
        modulated_embed = self.final_norm(embed) * (1.0 + scale) + shift
        out = self.output_layer(modulated_embed)
        
        return out
        
class SinPosEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        assert embed_dim % 2 == 0
        half_dim = embed_dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -scale)
        self.register_buffer("freq", freq, persistent=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
    
    def forward(self, x):
        phase = x[:, None] * self.freq[None, :]
        embed = torch.cat((phase.sin(), phase.cos()), dim=-1)
        
        out = self.mlp(embed)
        return out

class DiffusionModel(nn.Module):
    def __init__(
        self,
        data_dim,
        embed_dim,
        mlp_hidden_dim,
        cond_dim,
        cond_hidden_dims,
        num_blocks,
        dropout,
        noise_scheduler: SchedulerMixin,
    ):
        super().__init__()
        
        self.data_dim = data_dim
        self.noise_predictor = DiT(
            input_dim=data_dim,
            embed_dim=embed_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            cond_dim=cond_dim * 2,
            cond_hidden_dims=cond_hidden_dims,
            num_blocks=num_blocks,
            output_dim=data_dim,
            dropout=dropout
        )
        
        self.diffusion_step_encoder = SinPosEmbedding(cond_dim)
        
        self.noise_scheduler = noise_scheduler
    
    def predict_noise(
        self,
        noisy_x,
        cond,
        diffusion_step,
    ):
        concat_cond = torch.cat((
            cond,
            self.diffusion_step_encoder(diffusion_step),
        ), dim=-1)
        
        pred_noise = self.noise_predictor(noisy_x, concat_cond)
        return pred_noise
    
    def compute_loss(
        self,
        x,
        cond,
    ):
        device = x.device
        batch_size = x.shape[0]

        noise = torch.randn_like(x)
        
        diffusion_steps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(batch_size,),
            dtype=torch.long,
            device=device,
        )
        
        noisy_x = self.noise_scheduler.add_noise(
            original_samples=x,
            noise=noise,
            timesteps=diffusion_steps,
        )
        
        pred_noise = self.predict_noise(
            noisy_x=noisy_x,
            cond=cond,
            diffusion_step=diffusion_steps,
        )

        assert self.noise_scheduler.config.prediction_type == 'epsilon'
        
        loss = nn.functional.mse_loss(pred_noise, noise)
        return loss

    def inference(
        self,
        inference_steps,
        cond,
        noise=None,
    ):
        device = cond.device
        batch_size = cond.shape[0]

        self.noise_scheduler.set_timesteps(
            num_inference_steps=inference_steps,
            device=device,
        )
        
        noisy_x = noise if noise is not None else torch.randn(
            batch_size, self.data_dim,
            dtype=torch.float32,
            device=device,
        )
        
        for timestep in self.noise_scheduler.timesteps:
            pred_noise = self.predict_noise(
                noisy_x=noisy_x,
                cond=cond,
                diffusion_step=timestep.expand((batch_size,)),
            )
            
            noisy_x = self.noise_scheduler.step(
                model_output=pred_noise,
                timestep=timestep.item(),
                sample=noisy_x
            ).prev_sample
        
        x = noisy_x
        return x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        activation_factory=nn.SiLU,
        dropout=0.0,
    ):
        super().__init__()

        layers = []
        in_features = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(activation_factory())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)