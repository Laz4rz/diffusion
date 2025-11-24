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

    def forward(self, x, t):
        x_down = self.downblock(x)
        time_emb = self.time_embedder(t)[:, :, None, None] 
        x_down = x_down + time_emb
        return self.upblock(x_down)


@beartype
def train_batch(
    model: nn.Module, 
    scheduler: LinearNoiseScheduler, 
    optimizer: torch.optim.Optimizer, 
    x0: Float[Tensor, "b c h w"]
) -> Float[Tensor, ""]: # Returns a scalar tensor (loss)
    """
    Performs one step of training: Corrupts image, predicts noise, calculates gradient.
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
    
    # 5. Calculate Loss and Backprop
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
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt

    # Create results folder
    os.makedirs("results/ddpm", exist_ok=True)

    # 1. Setup Data with [-1, 1] Normalization
    def mnist_loader(batch_size=64, keep_digit=0):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Transform from [0, 1] to [-1, 1]
            transforms.Lambda(lambda t: (t * 2) - 1) 
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # --- FILTERING LOGIC ---
        if keep_digit is not None:
            # 1. Create a mask for the desired class
            train_idx = train_dataset.targets == keep_digit
            test_idx = test_dataset.targets == keep_digit
            
            # 2. Slice the internal data tensors to keep only that class
            train_dataset.data = train_dataset.data[train_idx]
            train_dataset.targets = train_dataset.targets[train_idx]
            
            test_dataset.data = test_dataset.data[test_idx]
            test_dataset.targets = test_dataset.targets[test_idx]
            
            print(f"Filtered dataset to digit '{keep_digit}'. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
        # -----------------------
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader

    # 2. Setup Device & Seed
    torch.manual_seed(0)
    train_loader, test_loader = mnist_loader(keep_digit=1)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    synchronize = torch.mps.synchronize if torch.backends.mps.is_available() else (torch.cuda.synchronize if torch.cuda.is_available() else lambda: None)

    print(f"Using device: {device}")

    # 3. Instantiate Model & Scheduler
    # Scheduler buffers move to device automatically when .to(device) is called
    scheduler = LinearNoiseScheduler().to(device)
    model = SimpleUnet(input_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    num_epochs = 10

    # 4. Training Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        start_time = time.perf_counter()
        
        for x, _ in train_loader:
            x = x.to(device)
            
            # We use the train_batch helper we defined earlier
            # It handles sampling t, adding noise, predicting, and backprop
            loss_val = train_batch(model, scheduler, optimizer, x)
            
            # train_batch returns a detached tensor or float, so we just add it
            train_loss += loss_val

        synchronize()
        print(f"Epoch {epoch+1}, Loss: {train_loss/len(train_loader):.5f}, Time: {time.perf_counter() - start_time:.2f}s")
        
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
        x = torch.randn((n_vis_samples, 1, 28, 28), device=device)
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
