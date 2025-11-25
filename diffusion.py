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


class SimpleUnet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Down: 28 -> 14 -> 7 (or 9 -> 5 -> 3)
        self.downblock = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.GELU(),
        )

        self.time_embedder = nn.Sequential(
            nn.Embedding(1000, 64),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        # Up: 7 -> 14 -> 28 (or 3 -> 6 -> 12)
        self.upblock = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, input_dim, 3, padding=1)
        )

    def forward(self, x, t):
        # 1. Downsample
        x_down = self.downblock(x) 
        
        # 2. Add Time Embedding
        time_emb = self.time_embedder(t)[:, :, None, None] 
        x_down = x_down + time_emb 

        # 3. Upsample
        output = self.upblock(x_down)
        
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
        return output


@beartype
def train_batch(
    model: nn.Module, 
    scheduler: LinearNoiseScheduler, 
    optimizer: torch.optim.Optimizer, 
    x0: Float[Tensor, "b c h w"],
    use_weighted_loss: bool = True
) -> Float[Tensor, ""]: 
    """
    Performs one step of training. 
    If use_weighted_loss is True, pixels with values > -0.5 (signal) 
    are weighted 10x more than background pixels.
    """
    model.train()
    optimizer.zero_grad()

    batch_size = x0.shape[0]
    
    # 1. Sample random timesteps t
    t = torch.randint(low=0, high=scheduler.num_timesteps, size=(batch_size,), device=x0.device)
    
    # 2. Create the noise (epsilon)
    noise = torch.randn_like(x0)
    
    # 3. Create the noisy image (x_t)
    xt = scheduler.q_sample(x0, t, noise)
    
    # 4. Predict the noise
    noise_pred = model(xt, t)
    
    # 5. Calculate Loss
    if use_weighted_loss:
        # Calculate per-pixel squared error
        loss_unreduced = F.mse_loss(noise_pred, noise, reduction='none')
        
        # Define Weights: Background (-1) gets 1.0, Signal (>-0.5) gets 10.0
        weights = torch.ones_like(x0)
        weights[x0 > -0.5] = 10.0
        
        # Apply weights and reduce
        loss = (loss_unreduced * weights).mean()
    else:
        # Standard MSE
        loss = F.mse_loss(noise_pred, noise)
    
    loss.backward()
    optimizer.step()
    
    return loss


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
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import datasets, transforms
    import shutil
    import matplotlib.pyplot as plt

    # Create results folder
    if os.path.exists("results/ddpm"):
        shutil.rmtree("results/ddpm")
    os.makedirs("results/ddpm", exist_ok=True)

    # 1. Setup Data with [-1, 1] Normalization
    def mnist_loader(batch_size=64, keep_digit=0, max_examples=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1) 
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # --- 1. FILTER CLASS (Train & Test) ---
        if keep_digit is not None:
            # Filter Train
            train_idx = train_dataset.targets == keep_digit
            train_dataset.data = train_dataset.data[train_idx]
            train_dataset.targets = train_dataset.targets[train_idx]
            
            # Filter Test
            test_idx = test_dataset.targets == keep_digit
            test_dataset.data = test_dataset.data[test_idx]
            test_dataset.targets = test_dataset.targets[test_idx]
        
        # --- 2. LIMIT EXAMPLES (Train only) ---
        # We usually limit training data to test few-shot capabilities, 
        # but we keep the full test set to see if it generalizes.
        if max_examples is not None:
            limit = min(max_examples, len(train_dataset.data))
            train_dataset.data = train_dataset.data[:limit]
            train_dataset.targets = train_dataset.targets[:limit]
            print(f"Limiting training set to {limit} examples.")

        # --- 3. REPEAT TO FILL BATCH (Train only) ---
        # If we have fewer training samples than batch_size, repeat them
        current_len = len(train_dataset.data)
        if current_len < batch_size and current_len > 0:
            import math
            repeats = math.ceil(batch_size / current_len)
            print(f"Train set size ({current_len}) < Batch Size ({batch_size}). Repeating data {repeats} times.")
            
            train_dataset.data = train_dataset.data.repeat(repeats, 1, 1)
            train_dataset.targets = train_dataset.targets.repeat(repeats)

        print(f"Final Stats | Digit: {keep_digit} | Train Size: {len(train_dataset)} | Test Size: {len(test_dataset)}")
        
        # Drop_last=True for train to avoid unstable partial batches
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # Shuffle=False and drop_last=False for test
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
        return train_loader, test_loader
    
    def synthetic_cross_loader(batch_size=64, num_samples=1000):
        """
        Returns a dataloader that yields a synthetic 9x9 image 
        with a cross pattern, repeated 'num_samples' times.
        Range is [-1, 1] for diffusion compatibility.
        """
        # 1. Create the canvas (1 channel, 9x9) with background -1
        # Shape: [1, 28, 28]
        size = 25
        img = torch.full((1, size, size), -1.0)
        
        # 2. Draw the cross with foreground 1
        # For a 28x28 grid, the center index is 14
        img[:, size // 2, :] = 1.0  # Horizontal line (Middle Row)
        img[:, :, size // 2] = 1.0  # Vertical line (Middle Column)
        
        # 3. Repeat this single example to create a full dataset
        # Shape becomes: [num_samples, 1, 28, 28]
        data = img.repeat(num_samples, 1, 1, 1)
        
        # 4. Create dummy labels (just 0s)
        targets = torch.zeros(num_samples, dtype=torch.long)
        
        # 5. Create Dataset and Loader
        dataset = TensorDataset(data, targets)
        
        # shuffle=True is technically irrelevant for identical data, 
        # but good practice to keep the API consistent.
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return loader


    # 2. Setup Device & Seed
    torch.manual_seed(0)
    train_loader, test_loader = mnist_loader(batch_size=512, keep_digit=1, max_examples=1)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    synchronize = torch.mps.synchronize if torch.backends.mps.is_available() else (torch.cuda.synchronize if torch.cuda.is_available() else lambda: None)
    train_loader = synthetic_cross_loader(batch_size=512, num_samples=1024)
    test_loader = synthetic_cross_loader(batch_size=1, num_samples=1)

    print(f"Using device: {device}")

    # 3. Instantiate Model & Scheduler
    # Scheduler buffers move to device automatically when .to(device) is called
    scheduler = LinearNoiseScheduler(num_timesteps=250).to(device)
    model = SimpleUnet(input_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Visualize Forward Process Overview
    with torch.no_grad():
        samp = next(iter(train_loader))[0][:1].to(device)
        noise_seed = torch.randn_like(samp)
        timeline = torch.linspace(0, scheduler.num_timesteps - 1, steps=10).long()
        frames = []
        for step in timeline:
            t_step = torch.full((1,), step.item(), device=device, dtype=torch.long)
            frames.append(scheduler.q_sample(samp, t_step, noise_seed).cpu())
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for ax, frame, step in zip(axes.flatten(), frames, timeline.tolist()):
            img = (frame[0] + 1) / 2
            ax.imshow(img.squeeze().clamp(0, 1), cmap="gray")
            ax.set_title(f"t={step}")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig("results/ddpm/forward_overview.png")
        plt.close(fig)

    num_epochs = 50000
    
    sample_shape = next(iter(train_loader))[0].shape

    training_start_time = time.perf_counter()
    # 4. Training Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for x, _ in train_loader:
            x = x.to(device)
            
            # We use the train_batch helper we defined earlier
            # It handles sampling t, adding noise, predicting, and backprop
            loss_val = train_batch(model, scheduler, optimizer, x)
            
            # train_batch returns a detached tensor or float, so we just add it
            train_loss += loss_val

        synchronize()
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}, Loss: {train_loss/len(train_loader):.5f}, Elapsed time: {time.perf_counter() - training_start_time:.2f}s")
        
        if (epoch + 1) % 2000 == 0 or epoch == num_epochs - 1 or epoch == 0:
            # 5. Evaluation & Sampling
            model.eval()
            test_loss = 0
            
            # Validation Loss (How well do we predict noise on unseen data?)
            with torch.no_grad():
                for x, _ in test_loader:
                    x = x.to(device)
                    # Replicate train logic but without backprop
                    t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
                    noise = torch.randn_like(x)
                    xt = scheduler.q_sample(x, t, noise)
                    noise_pred = model(xt, t)
                    loss = F.mse_loss(noise_pred, noise)
                    test_loss += loss.item()

            # Sampling with Snapshots
            # We will generate 4 samples and visualize 6 steps for each
            n_vis_samples = 4
            # Define specific timesteps we want to see (e.g., 999, 800, 600, 400, 200, 0)
            snapshot_steps = torch.linspace(
                scheduler.num_timesteps - 1, 0, steps=6, dtype=torch.long
            ).tolist()
            
            # 1. Start with pure noise
            x = torch.randn((n_vis_samples, sample_shape[1], sample_shape[2], sample_shape[3]), device=device)
            history = [] # To store the snapshots

            # 2. Manual Sampling Loop to capture intermediates
            for i in reversed(range(scheduler.num_timesteps)):
                t = torch.full((n_vis_samples,), i, device=device, dtype=torch.long)
                
                # Predict noise
                noise_pred = model(x, t)

                # Get coefficients (Reusing your extract logic)
                sqrt_alpha_inv = extract(scheduler.sqrt_alpha_inverse, t, x.shape)
                beta_over_sigma = extract(scheduler.beta_over_sqrt_one_minus_alpha_cumprod, t, x.shape)
                sqrt_beta = extract(scheduler.sqrt_beta, t, x.shape)

                # Calculate the Mean
                mean = sqrt_alpha_inv * (x - beta_over_sigma * noise_pred)

                # Update x
                if i > 0:
                    noise = torch.randn_like(x)
                    x = mean + sqrt_beta * noise
                else:
                    x = mean
                
                # Save snapshot if this step is in our list
                if i in snapshot_steps:
                    history.append(x.detach().cpu())

            # 3. Visualization
            # Rows = Different Samples, Cols = Time Steps
            fig, axs = plt.subplots(n_vis_samples, len(snapshot_steps), figsize=(12, 6))
            
            # history is currently [t=999, t=800, ..., t=0]
            
            for row in range(n_vis_samples):
                for col, step_idx in enumerate(snapshot_steps):
                    # Get the specific image tensor
                    img_tensor = history[col][row] 
                    
                    # Inverse Transform: [-1, 1] -> [0, 1]
                    img_disp = (img_tensor + 1) / 2
                    img_disp = img_disp.clamp(0, 1).squeeze(0)

                    axs[row, col].imshow(img_disp, cmap='gray')
                    
                    # Only set labels on the top row
                    if row == 0:
                        axs[row, col].set_title(f"t={step_idx}")
                    axs[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"results/ddpm/ddpm_epoch_{epoch+1}.png")
            plt.close(fig)

            print(f"Test Loss: {test_loss/len(test_loader):.5f}")
