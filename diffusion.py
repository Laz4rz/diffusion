import math
import os
import time
from typing import Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# --- Optional type/decorator deps (safe fallbacks if not installed) ---
try:
    from jaxtyping import Float, Int64, Int  # type: ignore
except Exception:  # pragma: no cover
    Float = Int64 = Int = Any  # type: ignore

try:
    from beartype import beartype  # type: ignore
except Exception:  # pragma: no cover
    def beartype(f):  # type: ignore
        return f


# --------------------------- Scheduler --------------------------- #
class LinearNoiseScheduler(nn.Module):
    """
    Precomputes α_t, β_t, \bar{α}_t and useful derived terms.
    Uses a *posterior variance* for sampling (critical for stability).
    """

    def __init__(self, num_timesteps: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.num_timesteps = num_timesteps

        beta = torch.linspace(beta_start, beta_end, num_timesteps)
        alpha = 1.0 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        alpha_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alpha_cumprod.dtype), alpha_cumprod[:-1]], dim=0
        )

        # base buffers
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)

        # forward / reparameterization constants
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod",
            torch.sqrt((1.0 - alpha_cumprod).clamp_min(1e-20)),
        )

        # reverse step constants
        self.register_buffer("sqrt_alpha_inverse", alpha.pow(-0.5))
        self.register_buffer(
            "beta_over_sqrt_one_minus_alpha_cumprod",
            beta / torch.sqrt((1.0 - alpha_cumprod).clamp_min(1e-20)),
        )

        # posterior variance (\tilde{β}_t) per Ho et al. 2020
        posterior_variance = beta * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.register_buffer("sqrt_posterior_variance", posterior_variance.clamp_min(1e-20).sqrt())

    @beartype
    def q_sample(
        self,
        x0: Float[Tensor, "b c h w"],
        t: Int64[Tensor, "b"],
        noise: Float[Tensor, "b c h w"] | None = None,
    ) -> Float[Tensor, "b c h w"]:
        """Forward diffusion: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε"""
        if noise is None:
            noise = torch.randn_like(x0)
        sac = extract(self.sqrt_alpha_cumprod, t, x0.shape)
        somac = extract(self.sqrt_one_minus_alpha_cumprod, t, x0.shape)
        return sac * x0 + somac * noise


@beartype
def extract(
    input_tensor: Float[Tensor, "timesteps"],
    t_indices: Int[Tensor, "batch"],
    x_shape: torch.Size | tuple,
) -> Float[Tensor, "batch ..."]:
    """Gathers 1D schedule values by t and reshapes to broadcast over x."""
    batch_size = t_indices.shape[0]
    out = input_tensor.gather(-1, t_indices.to(input_tensor.device))
    reshape_shape = (batch_size,) + (1,) * (len(x_shape) - 1)
    return out.reshape(reshape_shape)


# --------------------------- Model --------------------------- #

def timestep_embedding(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    """Sinusoidal timestep embedding, shape: (B, dim)."""
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int | None = None):
        super().__init__()
        ng = max(1, min(8, out_ch))  # small GroupNorm works well on tiny data
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(ng, out_ch)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(ng, out_ch)
        self.act2 = nn.SiLU()

        self.time_proj = None
        if time_dim is not None:
            self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))

        # simple residual for overfitting ease
        self.residual = (in_ch == out_ch)

    def forward(self, x: Tensor, t_emb: Tensor | None = None) -> Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act1(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)

        if self.residual:
            h = h + x
        return h


class TinyUNet(nn.Module):
    def __init__(self, input_dim: int, base: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 256), nn.SiLU(), nn.Linear(256, time_dim)
        )
        self.inc = ConvBlock(input_dim, base, time_dim)
        self.down1 = ConvBlock(base, base * 2, time_dim)
        self.mid = ConvBlock(base * 2, base * 2, time_dim)
        self.up1 = ConvBlock(base * 2, base, time_dim)
        self.outc = nn.Conv2d(base, input_dim, kernel_size=3, padding=1)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        h = self.inc(x, t_emb)
        h = self.down1(h, t_emb)
        h = self.mid(h, t_emb)
        h = self.up1(h, t_emb)
        return self.outc(h)


# --------------------------- Training helpers --------------------------- #
@beartype
def train_step(
    model: nn.Module,
    scheduler: LinearNoiseScheduler,
    optimizer: torch.optim.Optimizer,
    x0_batch: Float[Tensor, "b c h w"],
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    b = x0_batch.shape[0]
    t = torch.randint(0, scheduler.num_timesteps, (b,), device=x0_batch.device)
    noise = torch.randn_like(x0_batch)
    xt = scheduler.q_sample(x0_batch, t, noise)

    noise_pred = model(xt, t)
    loss = F.mse_loss(noise_pred, noise)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
@beartype
def sample(
    model: nn.Module,
    scheduler: LinearNoiseScheduler,
    image_shape: tuple[int, ...],
    device: torch.device | str,
) -> Float[Tensor, "b c h w"]:
    model.eval()
    x = torch.randn(image_shape, device=device)
    for i in reversed(range(scheduler.num_timesteps)):
        t = torch.full((x.shape[0],), i, device=device, dtype=torch.long)

        noise_pred = model(x, t)
        sqrt_alpha_inv = extract(scheduler.sqrt_alpha_inverse, t, x.shape)
        coeff = extract(scheduler.beta_over_sqrt_one_minus_alpha_cumprod, t, x.shape)
        mean = sqrt_alpha_inv * (x - coeff * noise_pred)

        if i > 0:
            noise = torch.randn_like(x)
            sqrt_post = extract(scheduler.sqrt_posterior_variance, t, x.shape)
            x = mean + sqrt_post * noise
        else:
            x = mean
    return x


# --------------------------- Main: single-sample overfit --------------------------- #
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid, save_image

    # 0) Repro & device
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    print(f"Using device: {device}")

    # 1) Load *one* MNIST example and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2.0) - 1.0),
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)

    # Pick a specific digit to make visual inspection easy (e.g., digit 1)
    KEEP_DIGIT = 1
    if KEEP_DIGIT is not None:
        idxs = (train_ds.targets == KEEP_DIGIT).nonzero(as_tuple=True)[0]
        idx = int(idxs[0].item())
    else:
        idx = 0

    x0_single, _ = train_ds[idx]  # (1, 28, 28) in [-1,1]
    x0_single = x0_single.unsqueeze(0)  # (1, 1, 28, 28)
    print(f"Selected sample index {idx} with shape {tuple(x0_single.shape)}")

    # 2) Repeat the same sample to form a small batch (helps optimization)
    BATCH = 32
    x0_batch = x0_single.repeat(BATCH, 1, 1, 1).to(device)

    # 3) Model & scheduler intentionally sized for *overfitting*
    scheduler = LinearNoiseScheduler(num_timesteps=200).to(device)
    model = TinyUNet(input_dim=1, base=64, time_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 4) Train for a fixed number of steps (or early stop if it crushes the loss)
    os.makedirs("results/ddpm", exist_ok=True)

    steps = 5000
    ema = 0.0
    alpha = 0.98  # for a smoothed loss display
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        loss_val = train_step(model, scheduler, optimizer, x0_batch)
        ema = alpha * ema + (1 - alpha) * loss_val if step > 1 else loss_val

        if step % 100 == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            print(f"step {step:5d} | loss {loss_val:.6f} | ema {ema:.6f} | {elapsed:.1f}s")

        # quick qualitative check: sample occasionally
        if step % 1000 == 0:
            with torch.no_grad():
                x_samples = sample(model, scheduler, (16, 1, 28, 28), device)
                x_vis = ((x_samples + 1.0) / 2.0).clamp(0, 1)
                grid = make_grid(x_vis, nrow=8)
                save_path = f"results/ddpm/overfit_samples_step_{step}.png"
                save_image(grid, save_path)
                print(f"Saved {save_path}")

        # simple early stop if it's totally overfit
        if ema < 1e-4 and step > 1000:
            print("Early stopping: loss sufficiently low.")
            break

    # 5) Final sampling & save
    with torch.no_grad():
        x_samples = sample(model, scheduler, (16, 1, 28, 28), device)
        x_vis = ((x_samples + 1.0) / 2.0).clamp(0, 1)
        grid = make_grid(x_vis, nrow=8)
        save_path = "results/ddpm/overfit_final.png"
        save_image(grid, save_path)
        print(f"Saved final samples to {save_path}")

    # 6) Quick quantitative sanity check on the *training* example
    with torch.no_grad():
        model.eval()
        b = 256
        t = torch.randint(0, scheduler.num_timesteps, (b,), device=device)
        x0_eval = x0_single.to(device).repeat(b, 1, 1, 1)
        noise = torch.randn_like(x0_eval)
        xt = scheduler.q_sample(x0_eval, t, noise)
        eps = model(xt, t)
        mse = F.mse_loss(eps, noise).item()
        print(f"Noise-pred MSE on training sample (random t): {mse:.6f}")
