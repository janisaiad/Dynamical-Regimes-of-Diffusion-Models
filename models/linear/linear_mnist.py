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

# Load and preprocess MNIST data (only digits 0 and 1)
transform = transforms.Compose([transforms.ToTensor()])
mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Filter for classes 0 and 1
idx = (mnist.targets == 0) | (mnist.targets == 1)
mnist.data = mnist.data[idx]
mnist.targets = mnist.targets[idx]

# Take n samples and normalize
n_samples = 1000
selected_idx = torch.randperm(len(mnist.data))[:n_samples]
X = mnist.data[selected_idx].float() / 255.0
X = X.reshape(n_samples, -1)

# Project to 2D using PCA
pca = PCA(n_components=2)
X_pca = torch.tensor(pca.fit_transform(X), device=device)

# Animation parameters
T = 100  # Number of frames
timesteps = torch.linspace(0, 2, T)

# Setup the figure
fig, ax = plt.subplots(figsize=(8, 8))

# Initialize scatter plot
scatter = ax.scatter([], [], alpha=0.5)
ax.set_xlim(X_pca.cpu()[:, 0].min() - 1, X_pca.cpu()[:, 0].max() + 1)
ax.set_ylim(X_pca.cpu()[:, 1].min() - 1, X_pca.cpu()[:, 1].max() + 1)

def update(frame):
    t = timesteps[frame]
    # Add noise based on timestep
    noise_scale = torch.sqrt(1 - torch.exp(-2*t))
    noisy_data = X_pca * torch.exp(-t) + noise_scale * torch.randn_like(X_pca)
    
    scatter.set_offsets(noisy_data.cpu().numpy())
    ax.set_title(f't = {t:.2f}')
    return scatter,

# Create animation
anim = FuncAnimation(fig, update, frames=T, interval=50, blit=True)

# Save animation
anim.save('results/MNIST/diffusion_2d.gif', writer='pillow')
plt.close()
