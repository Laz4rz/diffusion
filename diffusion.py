import time
import torch
from torch import nn
import matplotlib.pyplot as plt


class LinearNoiseScheduler(nn.Module):
    """
    precomputes alphas, betas and alphas cumulative product
    reparametrization trick terms are also precomputed
    registers precomputed values as buffers
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_timesteps = num_timesteps
        beta = torch.linspace(beta_start, beta_end, num_timesteps)
        alpha = 1 - beta
        alpha_cumprod = torch.exp(torch.log(alpha).cumsum())

        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer("alpha_cumprod_sqrt", torch.sqrt(alpha))
        self.register_buffer("one_alpha_cumprod_sqrt", torch.sqrt(1-alpha))

    def q_sample(self, x0, t, noise=None):
        """
        x0: [Batch, C, H, W]
        t: [Batch]
        noise: Optional [Batch, C, H, W] (If None, generate it)
        """
        

def extract(input_tensor: torch.Tensor, t_indices, x_shape):
    """
    takes alpha/beta/alpha (scalar) cumprod from t_indices and expands shape to batch shape (B, C, H, W)
    """
    batch_size = x_shape[0]
    assert batch_size == len(t_indices), "batch size and indices length does not match"

    out = torch.gather(
        input_tensor, -1, t_indices.to(input_tensor.device))
    
    reshape_shape = (batch_size, ) + (1,) * (len(x_shape) - 1)

    return out.reshape(reshape_shape)
