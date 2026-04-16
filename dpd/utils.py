import copy

import torch
from torch import nn
import numpy as np

class Normalizer(nn.Module):
    def __init__(
        self,
        data: np.ndarray,
        mode: str = 'minmax',
        output_min=-1.0,
        output_max=1.0,
        eps=1e-4,
        data_dim=None,
    ):
        super().__init__()

        self.mode = mode
        if self.mode == 'minmax':
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)

            data_range = data_max - data_min
            output_range = output_max - output_min
            
            ignore_dim = data_range < eps
            data_range[ignore_dim] = output_range

            scale = output_range / data_range
            offset = output_min - data_min * scale
            offset[ignore_dim] = (output_max + output_min) / 2.0 - data_min[ignore_dim]
        elif self.mode == 'std':
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)

            ignore_dim = std < eps
            std[ignore_dim] = 1.0 

            scale = 1.0 / std
            offset = -mean / std
        elif self.mode == 'identity':
            scale = np.full((data_dim,), 1.0, dtype=np.float32)
            offset = np.full((data_dim,), 0.0, dtype=np.float32)
        else:
            raise ValueError(f'Unsupported mode: {mode}')
        
        self.register_buffer('scale', torch.from_numpy(scale).float())
        self.register_buffer('offset', torch.from_numpy(offset).float())
        
    def forward(self, x: torch.Tensor):
        return x.float() * self.scale + self.offset

    def unnormalize(self, x: torch.Tensor):
        return (x.float() - self.offset) / self.scale

class EMA:
    def __init__(
        self,
        decay,
    ):
        self.decay = decay
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = new_value * (1.0 - self.decay) + self.value * self.decay

class EMAModel:
    def __init__(
        self,
        model: nn.Module,
        decay=None,
        update_after_steps=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
    ):
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.decay = decay
        self.update_after_steps = update_after_steps
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.update_count = 0
    
    def get_decay(self, optimization_step):
        if self.decay is None:
            step = max(0, optimization_step - self.update_after_steps - 1)
            value = 1 - (1 + step / self.inv_gamma) ** -self.power
            if step <= 0:
                return 0.0
            decay =  max(self.min_value, min(value, self.max_value))
        else:
            decay = self.decay
        return decay
    
    @torch.no_grad()
    def update(
        self,
        new_model: nn.Module,
    ):
        decay = self.get_decay(self.update_count)

        for new_param, ema_param in zip(new_model.parameters(), self.model.parameters()):
            ema_param.lerp_(new_param, 1.0 - decay)
        for new_buffer, ema_buffer in zip(new_model.buffers(), self.model.buffers()):
            ema_buffer.copy_(new_buffer)
        
        self.update_count += 1