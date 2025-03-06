# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load and preprocess CIFAR data
transform = transforms.Compose([transforms.ToTensor()])
cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Take n samples and normalize
# dimension of cifar is 
n_samples = 500  # Increased samples since CIFAR has more classes
selected_idx = torch.randperm(len(cifar.data))[:n_samples]
X = torch.tensor(cifar.data[selected_idx]).float() / 255.0
X = X.reshape(n_samples, -1)  # Flatten to (n_samples, 3072)
labels = torch.tensor(cifar.targets)[selected_idx]

# Project to 2D using PCA for visualization
pca = PCA()
X_pca_full = pca.fit_transform(X)
X_pca = torch.tensor(X_pca_full[:, :2])  # Keep first 2 components for plotting

# Plot eigenvalues
plt.figure(figsize=(10, 5))
plt.plot(pca.explained_variance_ratio_[:50], 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Eigenvalue Spectrum (first 50 components)')
plt.grid(True)
plt.savefig('results/cifar/eigenvalues.png')
plt.close()

# Animation parameters with more points and log space
T = 200  # Increased number of frames
# Create timesteps in log space from 1e-3 to 2
timesteps = torch.exp(torch.linspace(np.log(1e-3), np.log(8), T))

def compute_P_t_e(x, data, t):
    """Compute P_t_e according to equation (2) in paper"""
    n = len(data)
    Delta_t = 1 - torch.exp(-2*t)
    diff = x.unsqueeze(0) - data * torch.exp(-t)
    P_t_e = torch.mean(torch.exp(-torch.sum(diff**2, dim=1)/(2*Delta_t))) / ((2*np.pi*Delta_t)**(x.shape[0]/2))
    return P_t_e

def compute_gaussian_entropy(t):
    """Compute entropy of a standard Gaussian"""
    d = X.shape[1]  # Full dimension of CIFAR (3072)
    return d/2 * (1 + np.log(2*np.pi))

def compute_entropy_and_excess(data, t, n_samples=10000):
    """Compute entropy of P_t_e and excess entropy by Monte Carlo sampling"""
    Delta_t = 1 - torch.exp(-2*t)
    # Sample points from noisy distribution
    samples = torch.randn(n_samples, data.shape[1]) * torch.sqrt(Delta_t) + \
             data[torch.randint(0, len(data), (n_samples,))] * torch.exp(-t)
    
    # Compute entropy
    entropy = torch.tensor(0.0)
    for x in samples:
        p = compute_P_t_e(x, data, t)
        if p > 0:
            entropy -= (1/n_samples) * torch.log(p)
            
    # Compute excess entropy f(t) = s(t) - s_G(t)
    gaussian_entropy = compute_gaussian_entropy(t)
    excess_entropy = entropy.item() - gaussian_entropy
    
    return entropy.item(), excess_entropy

# Calculate entropy and excess entropy over time
entropies = []
excess_entropies = []
for t in tqdm(timesteps):
    entropy, excess = compute_entropy_and_excess(X, t)  # Using full dimensional data
    entropies.append(entropy)
    excess_entropies.append(excess)

# Plot entropy and excess entropy with log scale x-axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.semilogx(timesteps, entropies, label='Total entropy s(t)')
ax1.semilogx(timesteps, [compute_gaussian_entropy(t) for t in timesteps], '--', label='Gaussian entropy')
ax1.set_xlabel('Time t (log scale)')
ax1.set_ylabel('Entropy')
ax1.set_title('Total and Gaussian entropy over time')
ax1.grid(True)
ax1.legend()

ax2.semilogx(timesteps, excess_entropies)
ax2.set_xlabel('Time t (log scale)')
ax2.set_ylabel('Excess entropy f(t)')
ax2.set_title('Excess entropy over time')
ax2.grid(True)

plt.tight_layout()
plt.savefig('results/cifar/entropy_analysis.png')
plt.close()

# Setup the diffusion animation figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Initialize scatter plot with colors for different classes
scatter = ax1.scatter([], [], c=[], alpha=0.5, cmap='tab10')
ax1.set_xlim(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1)
ax1.set_ylim(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1)

# Initialize excess entropy plot with log scale x-axis
line, = ax2.semilogx([], [])
ax2.set_xlim(1e-3, 8)
ax2.set_ylim(min(excess_entropies), max(excess_entropies))
ax2.set_xlabel('Time t (log scale)')
ax2.set_ylabel('Excess entropy f(t)')
ax2.grid(True)

def update(frame):
    t = timesteps[frame]
    # Add noise based on timestep
    noise_scale = torch.sqrt(1 - torch.exp(-2*t))
    noisy_data = X_pca * torch.exp(-t) + noise_scale * torch.randn_like(X_pca)
    
    scatter.set_offsets(noisy_data.numpy())
    scatter.set_array(labels.numpy())
    ax1.set_title(f't = {t:.3e}')
    
    # Update excess entropy plot
    line.set_data(timesteps[:frame+1], excess_entropies[:frame+1])
    
    return scatter, line

# Create animation
anim = FuncAnimation(fig, update, frames=T, interval=50, blit=True)

# Save animation
anim.save('results/cifar/diffusion_and_excess_entropy.gif', writer='pillow')
plt.close()
