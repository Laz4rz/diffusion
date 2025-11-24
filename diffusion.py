import torch
from torch import nn, Tensor
import torch.nn.functional as F
from jaxtyping import Float, Int64, Int
from beartype import beartype


class LinearNoiseScheduler(nn.Module):
    """
    precomputes alphas, betas and alphas cumulative product
    reparametrization trick terms are also precomputed
    registers precomputed values as buffers
    """
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        beta = torch.linspace(beta_start, beta_end, num_timesteps)
        alpha = 1. - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)

        # base constants
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        
        # x_t, reperametrization constants
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1. - alpha_cumprod))

        # x_{t-1} constants
        self.register_buffer('sqrt_beta', beta.pow(0.5))
        self.register_buffer("sqrt_alpha_inverse", torch.pow(alpha, -0.5))
        self.register_buffer("beta_over_sqrt_one_minus_alpha_cumprod", 
                             self.beta / self.sqrt_one_minus_alpha_cumprod)


    @beartype
    def q_sample(
        self, 
        x0: Float[Tensor, "b c h w"], 
        t: Int64[Tensor, "b"], 
        noise: Float[Tensor, "b c h w"] | None = None
    ) -> Float[Tensor, "b c h w"]:
        """
        for training we want to pass known noise so we have a ground truth to use with loss
        """
        sac = extract(self.sqrt_alpha_cumprod, t, x0.shape)
        somac = extract(self.sqrt_one_minus_alpha_cumprod, t, x0.shape)

        if noise is None:
            noise = torch.randn_like(x0)
            # each pixel is getting its own noise value
            # hence we later need to predict same number of noise values
            # at model output

        xt = x0 * sac + noise * somac

        return xt


@beartype
def extract(
    input_tensor: Float[Tensor, "timesteps"], 
    t_indices: Int[Tensor, "batch"], 
    x_shape: torch.Size | tuple
) -> Float[Tensor, "batch ..."]:
    """
    pretty constans from buffer specific 
    Extracts values from a 1D tensor based on indices and reshapes
    to (batch, 1, 1, ...).
    """
    batch_size = t_indices.shape[0]
    
    # Gather values
    out = input_tensor.gather(-1, t_indices.to(input_tensor.device))
    
    # Create shape: (batch, 1, 1, 1)
    reshape_shape = (batch_size,) + (1,) * (len(x_shape) - 1)
    
    return out.reshape(reshape_shape)


class SimpleUnet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.downblock = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        ) # (B, 64, H, W)

        self.time_embedder = nn.Sequential(
            nn.Embedding(1000, 64),
            nn.Linear(64, 64),
            nn.ReLU(),
        ) # (B, 64, )

        self.upblock = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, input_dim, kernel_size=3, padding=1),
        ) # (B, input_dim, H, W)
        # upblock needs to return unconstrained values as noise follows N(0, 1)

    def forward(self, x, t):
        # x: (B, input_dim, 28, 28)
        x_down = self.downblock(x) # (B, 64, H, W)
        time_emb = self.time_embedder(t)[:, :, None, None] # (B, 64, 1, 1)
        x_down = x_down + time_emb

        return self.upblock(x_down) # (B, input_dim, 28, 28)
    

def train_batch(model, scheduler: LinearNoiseScheduler, optimizer, x0: torch.Tensor):
    """
    model: SimpleUnet
    scheduler: LinearNoiseScheduler
    optimizer: torch.optim.Optimizer
    x0: Clean image batch [B, C, H, W]
    """
    model.train()
    optimizer.zero_grad()

    batch_size = x0.shape[0]
    
    # 1. Sample random timesteps t for every image in the batch
    t = torch.randint(low=0, high=scheduler.num_timesteps, size=(batch_size,), device=x0.device)
    
    # 2. Create the noise (epsilon)
    noise = torch.randn_like(x0)
    
    # 3. Create the noisy image (x_t)
    xt = scheduler.q_sample(x0, t, noise)
    
    # 4. Predict the noise
    noise_pred = model(xt, t)
    
    # 5. Calculate Loss and Backprop
    loss = F.mse_loss(noise_pred, noise)
    
    loss.backward()
    optimizer.step()
    
    return loss

@torch.no_grad()
def sample(model, scheduler, image_shape, device):
    """
    model: Trained SimpleUnet
    scheduler: LinearNoiseScheduler
    image_shape: Tuple (B, C, H, W) e.g. (16, 1, 28, 28)
    """
    # 1. Start with pure noise
    x = torch.randn(image_shape, device=device)
    
    # 2. Iterate backwards from T-1 down to 0
    #    range(num_timesteps - 1, -1, -1)
    for i in reversed(range(scheduler.num_timesteps)):
        t = torch.full((x.shape[0], ), i, device=device)
        noise_pred = model(x, t)

        # x_{t-1}
        sqrt_beta = extract(scheduler.sqrt_beta, t, x.shape)
        alpha_inverse = extract(scheduler.sqrt_alpha_inverse, t, x.shape)
        bosomac = extract(scheduler.beta_over_sqrt_one_minus_alpha_cumprod, t, x.shape)
        x_prev = alpha_inverse * (x - bosomac * noise_pred)

        if i > 0:
            x_prev += torch.randn_like(x_prev) * sqrt_beta
        
        x = x_prev

    return x_prev.detach()
