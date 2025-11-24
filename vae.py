import time
import torch
import matplotlib.pyplot as plt


class VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
        )

        self.mu_layer = torch.nn.Linear(256, latent_dim)
        self.logvar_layer = torch.nn.Linear(256, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, input_dim),
            torch.nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def decode(self, z):
        return self.decoder(z)

    def reparametrize(self, mu, logvar):
        # We need "std" (sigma) for: z = mu + sigma * epsilon
        # var = exp(log_var)
        # std = sqrt(var) = var^(0.5)
        # std = (exp(log_var))^(0.5) = exp(0.5 * log_var)
        return mu + logvar * torch.randn_like(torch.exp(0.5*logvar))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

def elbo_loss(x, x_hat, mu, logvar):
    reconstruction = (x - x_hat).pow(2).sum()
    regularization = torch.sum(mu.pow(2)+logvar.exp() - logvar-1, axis=1).sum()
    N = x.shape[0]

    return 1/(2*N) * (reconstruction + regularization)

def mnist_loader():
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    torch.manual_seed(0)

    train_loader, test_loader = mnist_loader()
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    synchronize = torch.mps.synchronize if torch.backends.mps.is_available() else (torch.cuda.synchronize if torch.cuda.is_available() else lambda: None)
    test_generation_indices = torch.randint(0, len(test_loader.dataset), (4,))

    print(f"Using device: {device}")

    vae = VAE(input_dim=28*28, latent_dim=20).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0
        start_time = time.perf_counter()
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = vae(x)
            loss = elbo_loss(x, x_hat, mu, logvar)
            loss.backward()
            train_loss += loss.detach()
            optimizer.step()
        synchronize()
        print(f"Epoch {epoch+1}, Loss: {train_loss/len(train_loader)}, Time: {time.perf_counter() - start_time:.2f}s")
        vae.eval()
        test_loss = 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                x_hat, mu, logvar = vae(x)
                loss = elbo_loss(x, x_hat, mu, logvar)
                test_loss += loss.item()
            # Create a single figure with subplots for all test indices
            fig, axs = plt.subplots(2, len(test_generation_indices), figsize=(len(test_generation_indices)*3, 6))
            
            for i, idx in enumerate(test_generation_indices):
                original = test_loader.dataset[idx][0].to(device)
                mu, logvar = vae.encode(original.unsqueeze(0))
                z = vae.reparametrize(mu, logvar)
                generated = vae.decode(z).squeeze(0).cpu()

                axs[0, i].imshow(original.cpu().view(28, 28), cmap='gray')
                axs[0, i].set_title(f"Original {i+1}")
                axs[0, i].axis('off')
                axs[1, i].imshow(generated.view(28, 28), cmap='gray')
                axs[1, i].set_title(f"Reconstructed {i+1}")
                axs[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"results/vae/vae_epoch_{epoch+1}.png")
            plt.close(fig)
        print(f"Test Loss: {test_loss/len(test_loader)}")


