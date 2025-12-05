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

        # x_{t-1} constants for the reverse step
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

        xt = x0 * sac + noise * somac

        return xt


@beartype
def extract(
    input_tensor: Float[Tensor, "timesteps"], 
    t_indices: Int[Tensor, "batch"], 
    x_shape: torch.Size | tuple
) -> Float[Tensor, "batch ..."]:
    """
    Extracts values from a 1D tensor based on indices and reshapes
    to (batch, 1, 1, ...).
    """
    batch_size = t_indices.shape[0]
    out = input_tensor.gather(-1, t_indices.to(input_tensor.device))
    reshape_shape = (batch_size,) + (1,) * (len(x_shape) - 1)
    
    return out.reshape(reshape_shape)

class SelfAttention(nn.Module):
    def __init__(self, channels): # Removed 'size' argument
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        b, c, h, w = x.shape
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x_flat = x.view(b, c, -1).swapaxes(1, 2)
        
        x_ln = self.ln(x_flat)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x_flat
        attention_value = self.ff_self(attention_value) + attention_value
        
        # Reshape back: (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        return attention_value.swapaxes(2, 1).view(b, c, h, w)

class SimpleUnet(nn.Module):
    def __init__(self, input_dim, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        # --- ENCODER (Down) ---
        self.down1 = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3, padding=1),
            nn.GELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.GELU()
        )

        # --- BOTTLENECK ATTENTION ---
        if self.use_attention:
            self.attn = SelfAttention(channels=64) 

        # --- TIME EMBEDDING ---
        self.time_embedder = nn.Sequential(
            nn.Embedding(1000, 64),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        # --- DECODER (Up) ---
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(96, 16, 4, stride=2, padding=1),
            nn.GELU()
        )
        self.up3 = nn.Conv2d(48, input_dim, 3, padding=1)

    def forward(self, x, t):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        time_emb = self.time_embedder(t)[:, :, None, None] 
        x3 = x3 + time_emb 

        if self.use_attention:
            x3 = self.attn(x3)

        x_up = self.up1(x3)
        if x_up.shape[-2:] != x2.shape[-2:]:
            x_up = F.interpolate(x_up, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x_up = torch.cat([x_up, x2], dim=1) 

        x_up = self.up2(x_up)
        if x_up.shape[-2:] != x1.shape[-2:]:
            x_up = F.interpolate(x_up, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x_up = torch.cat([x_up, x1], dim=1) 

        return self.up3(x_up)


@beartype
def train_batch(
    model, 
    scheduler, 
    optimizer, 
    x0, 
    use_weighted_loss=True
): 
    """
    Performs one full training step (Forward -> Loss -> Backward -> Clip -> Step).
    Returns a dictionary of metrics (detached tensors) for logging.
    """
    model.train()
    optimizer.zero_grad()

    batch_size = x0.shape[0]
    
    t = torch.randint(low=0, high=scheduler.num_timesteps, size=(batch_size,), device=x0.device)
    
    noise = torch.randn_like(x0)
    
    xt = scheduler.q_sample(x0, t, noise)
    
    noise_pred = model(xt, t)
    
    if use_weighted_loss:
        loss_unreduced = F.mse_loss(noise_pred, noise, reduction='none')
        
        weights = torch.ones_like(x0)
        weights[x0 > -0.5] = 10.0
        
        loss = (loss_unreduced * weights).mean()
    else:
        loss = F.mse_loss(noise_pred, noise)
    
    loss.backward()
    
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    return {
        "loss": loss.detach(),
        "grad_norm": grad_norm.detach(),
        "pred_mean": noise_pred.detach().mean(),
        "pred_std": noise_pred.detach().std()
    }


@torch.no_grad()
@beartype
def sample(
    model: nn.Module, 
    scheduler: LinearNoiseScheduler, 
    image_shape: tuple[int, ...], 
    device: torch.device | str
) -> Float[Tensor, "b c h w"]:
    """
    Generates images from pure noise by iterating backwards from T to 0.
    """
    # 1. Start with pure noise
    x = torch.randn(image_shape, device=device)
    
    # 2. Iterate backwards from T-1 down to 0
    for i in reversed(range(scheduler.num_timesteps)):
        t = torch.full((x.shape[0], ), i, device=device, dtype=torch.long)
        
        # Predict noise
        noise_pred = model(x, t)

        # Get the pre-calculated coefficients for this timestep
        sqrt_alpha_inv = extract(scheduler.sqrt_alpha_inverse, t, x.shape)
        beta_over_sigma = extract(scheduler.beta_over_sqrt_one_minus_alpha_cumprod, t, x.shape)
        sqrt_beta = extract(scheduler.sqrt_beta, t, x.shape) # This is sigma_t

        # Calculate the Mean (mu_theta)
        # Formula: 1/sqrt(alpha) * (x - beta/sqrt(1-alpha_bar) * eps_theta)
        mean = sqrt_alpha_inv * (x - beta_over_sigma * noise_pred)

        # Update x to x_{t-1}
        # If i > 0, add noise. If i == 0, just return the mean.
        if i > 0:
            noise = torch.randn_like(x)
            x = mean + sqrt_beta * noise
        else:
            x = mean

    return x

if __name__ == "__main__":
    import os
    import time
    import shutil
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    
    # --------------------------------------------------------------------------
    # 1. Data Loader (Optimized for Device)
    # --------------------------------------------------------------------------
    
    def synthetic_shapes_loader(batch_size, num_samples, shape_type="mix", img_size=28, device='cpu'):
        """
        Generates synthetic data and moves it to the specific device immediately.
        """
        
        def get_single_shape(name):
            # Create on CPU first to use efficient drawing logic
            img = torch.full((1, img_size, img_size), -1.0)
            center = img_size // 2
            
            if name == "cross":
                img[:, center, :] = 1.0
                img[:, :, center] = 1.0
                
            elif name == "square":
                r = img_size // 4
                img[:, center-r:center+r, center-r:center+r] = 1.0
                
            elif name == "circle":
                y, x = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), indexing='ij')
                radius = img_size // 4
                mask = ((x - center)**2 + (y - center)**2) <= radius**2
                img[:, mask] = 1.0
            
            return img

        # 1. Generate Data (on CPU)
        if shape_type == "mix":
            shapes = ["cross", "square", "circle"]
            chunk_size = num_samples // len(shapes)
            data_chunks = []
            for s in shapes:
                base_img = get_single_shape(s)
                data_chunks.append(base_img.repeat(chunk_size, 1, 1, 1))
            
            remainder = num_samples - (chunk_size * len(shapes))
            if remainder > 0:
                data_chunks.append(get_single_shape(shapes[0]).repeat(remainder, 1, 1, 1))
            data = torch.cat(data_chunks, dim=0)
        else:
            base_img = get_single_shape(shape_type)
            data = base_img.repeat(num_samples, 1, 1, 1)
        
        # 2. Move to Device IMMEDIATELY (The Optimization)
        print(f"Moving {len(data)} samples to {device}...")
        data = data.to(device)
        targets = torch.zeros(len(data), dtype=torch.long).to(device)
        
        # 3. Create Dataset/Loader
        # Note: num_workers must be 0 when using CUDA tensors in a Dataset
        dataset = TensorDataset(data, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return loader

    # --------------------------------------------------------------------------
    # 2. Visualization Helpers
    # --------------------------------------------------------------------------

    def visualize_forward_process(scheduler, sample_img, device, save_path):
        with torch.no_grad():
            # sample_img is already on device, but .to(device) is safe to call again
            sample_img = sample_img.to(device)
            noise_seed = torch.randn_like(sample_img)
            timeline = torch.linspace(0, scheduler.num_timesteps - 1, steps=10).long()
            frames = []
            
            for step in timeline:
                t_step = torch.full((1,), step.item(), device=device, dtype=torch.long)
                frames.append(scheduler.q_sample(sample_img, t_step, noise_seed).cpu())
            
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            for ax, frame, step in zip(axes.flatten(), frames, timeline.tolist()):
                img = (frame[0] + 1) / 2
                ax.imshow(img.squeeze().clamp(0, 1), cmap="gray")
                ax.set_title(f"t={step}")
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)

    def sample_and_save_snapshot(model, scheduler, sample_shape, device, epoch, save_path):
        n_vis_samples = 4
        snapshot_steps = torch.linspace(scheduler.num_timesteps - 1, 0, steps=6, dtype=torch.long).tolist()
        
        x = torch.randn((n_vis_samples, *sample_shape), device=device)
        history = [] 

        for i in reversed(range(scheduler.num_timesteps)):
            t = torch.full((n_vis_samples,), i, device=device, dtype=torch.long)
            noise_pred = model(x, t)

            sqrt_alpha_inv = extract(scheduler.sqrt_alpha_inverse, t, x.shape)
            beta_over_sigma = extract(scheduler.beta_over_sqrt_one_minus_alpha_cumprod, t, x.shape)
            sqrt_beta = extract(scheduler.sqrt_beta, t, x.shape)
            mean = sqrt_alpha_inv * (x - beta_over_sigma * noise_pred)

            if i > 0:
                noise = torch.randn_like(x)
                x = mean + sqrt_beta * noise
            else:
                x = mean
            
            if i in snapshot_steps:
                history.append(x.detach().cpu())

        fig, axs = plt.subplots(n_vis_samples, len(snapshot_steps), figsize=(12, 6))
        for row in range(n_vis_samples):
            for col, step_idx in enumerate(snapshot_steps):
                img_tensor = history[col][row] 
                img_disp = (img_tensor + 1) / 2
                img_disp = img_disp.clamp(0, 1).squeeze(0)
                axs[row, col].imshow(img_disp, cmap='gray')
                if row == 0: axs[row, col].set_title(f"t={step_idx}")
                axs[row, col].axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    def get_grad_norm(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    # --------------------------------------------------------------------------
    # 3. Main Execution
    # --------------------------------------------------------------------------
    
    # Configuration
    NUM_EPOCHS = 50000
    BATCH_SIZE = 64
    LR = 3e-4
    LOG_INTERVAL = 25
    EVAL_INTERVAL = 25
    
    # Setup
    if os.path.exists("results/ddpm"):
        shutil.rmtree("results/ddpm")
    os.makedirs("results/ddpm", exist_ok=True)
    
    torch.manual_seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    synchronize = torch.mps.synchronize if torch.backends.mps.is_available() else (torch.cuda.synchronize if torch.cuda.is_available() else lambda: None)
    print(f"Using device: {device}")

    # Dataset Selection - NOW PASSING DEVICE
    train_loader = synthetic_shapes_loader(
        batch_size=BATCH_SIZE, 
        num_samples=256, 
        shape_type="mix",
        device=device,
        img_size=16
    )
    
    test_loader = synthetic_shapes_loader(
        batch_size=32, 
        num_samples=32, 
        shape_type="mix",
        device=device,
        img_size=16
    )
    
    print(f"Train Dataset Size: {len(train_loader.dataset)} | Test Dataset Size: {len(test_loader.dataset)}")
    x_example, _ = next(iter(train_loader))
    print(f"Shape of one batch (next(iter...)): {x_example.shape}")

    
    # Model Setup
    scheduler = LinearNoiseScheduler(num_timesteps=250).to(device)
    model = SimpleUnet(input_dim=1, use_attention=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Visualize Forward Process
    # Note: next(iter(loader)) is already on GPU now
    sample_batch = next(iter(train_loader))[0] 
    visualize_forward_process(scheduler, sample_batch[:1], device, "results/ddpm/forward_overview.png")
    sample_shape = sample_batch.shape[1:]

    # Main Training Loop
    training_start_time = time.perf_counter()
    seen_samples = 0
    
    # Initialize Accumulators on Device
    acc_metrics = {
        "loss": torch.tensor(0.0, device=device),
        "grad_norm": torch.tensor(0.0, device=device),
        "pred_mean": torch.tensor(0.0, device=device),
        "pred_std": torch.tensor(0.0, device=device)
    }
    acc_steps = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        
        for x, _ in train_loader:
            metrics = train_batch(model, scheduler, optimizer, x, use_weighted_loss=True)
            
            for k, v in metrics.items():
                acc_metrics[k] += v
            
            acc_steps += 1

        seen_samples += len(train_loader.dataset)
        synchronize()
        
        # Logging
        if (epoch + 1) % LOG_INTERVAL == 0:
            avg = {k: v.item() / acc_steps for k, v in acc_metrics.items()}
            
            elapsed = time.perf_counter() - training_start_time
            
            print(f"Epoch {epoch+1} | Samples: {seen_samples} | Time: {elapsed:.0f}s")
            print(f"  > Loss: {avg['loss']:.5f}")
            print(f"  > Grads: {avg['grad_norm']:.4f}") 
            print(f"  > Preds: μ={avg['pred_mean']:.3f}, σ={avg['pred_std']:.3f}")
            
            for k in acc_metrics: 
                acc_metrics[k].zero_()
            acc_steps = 0
             
        # Evaluation
        if (epoch + 1) % EVAL_INTERVAL == 0 or epoch == NUM_EPOCHS - 1 or epoch == 0:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for x, _ in test_loader:
                    t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
                    noise = torch.randn_like(x)
                    xt = scheduler.q_sample(x, t, noise)
                    noise_pred = model(xt, t)
                    loss = F.mse_loss(noise_pred, noise)
                    test_loss += loss.item()

            print(f" >> Epoch {epoch+1} Validation Loss: {test_loss/len(test_loader):.5f}")
            save_file = f"results/ddpm/ddpm_epoch_{epoch+1}.png"
            sample_and_save_snapshot(model, scheduler, sample_shape, device, epoch, save_file)