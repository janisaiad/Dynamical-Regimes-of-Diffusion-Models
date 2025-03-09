import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for initial 2D Gaussian mixture
means = [np.array([-5, -5]), np.array([5, 5])]  # Centers of the two modes
covs = [np.array([[0.2, 0], [0, 0.2]]), np.array([[0.2, 0], [0, 0.2]])]  # Covariance matrices
weights = [0.5, 0.5]  # Equal weights for both modes

# Generate grid for visualization
x = np.linspace(-8, 8, 100)
y = np.linspace(-8, 8, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Calculate Gaussian mixture
Z = np.zeros_like(X)
for mean, cov, weight in zip(means, covs, weights):
    rv = multivariate_normal(mean, cov)
    Z += weight * rv.pdf(pos)

# Plot and save initial mixture
plt.figure(figsize=(8, 8))
plt.contourf(X, Y, Z, levels=20, cmap='plasma')
plt.colorbar()
plt.title('Initial 2D Gaussian Mixture')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('beamer_reports/mixture.png')
plt.close()

# Time parameters for Brownian motion
T = 8.0  # Total time
n_steps = 100  # Number of time steps
dt = T/n_steps  # Time step
t = np.linspace(0, T, n_steps)

# Generate Brownian trajectories (2 for each mode)
trajectories = []
for mean in means:
    for _ in range(2):  # 2 trajectories per mode
        # Initialize trajectory
        traj = np.zeros((n_steps, 2))
        traj[0] = np.random.multivariate_normal(mean, covs[0])
        
        # Generate Brownian motion
        for i in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt)/5, size=2)  # Scaled down Brownian increments
            traj[i] = traj[i-1] + 3*dW # 3*dW for more visible motion
            
        trajectories.append(traj)

# Save trajectories plot
plt.figure(figsize=(8, 8))
for i, traj in enumerate(trajectories):
    plt.plot(traj[:, 0], traj[:, 1], label=f'Trajectory {i+1}')
plt.title('Brownian Motion Trajectories')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('beamer_reports/brownian.png')
plt.close()

# Generate and save final Gaussian distribution
final_trajectories = np.array([traj[-1] for traj in trajectories])
final_mean = np.mean(final_trajectories, axis=0)
final_cov = np.cov(final_trajectories.T)

# Calculate final Gaussian
Z_final = multivariate_normal(final_mean, final_cov).pdf(pos)

# Plot and save final Gaussian
plt.figure(figsize=(8, 8))
plt.contourf(X, Y, Z_final, levels=20, cmap='plasma')
plt.colorbar()
plt.title('Final 2D Gaussian Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('beamer_reports/gaussian.png')
plt.close()

# Save trajectories data to markdown file
with open('beamer_reports/brownian.md', 'w') as f:
    f.write('# Brownian Motion Trajectories\n\n')
    f.write('Initial positions:\n\n')
    for i, traj in enumerate(trajectories):
        f.write(f'Trajectory {i+1}: ({traj[0,0]:.3f}, {traj[0,1]:.3f})\n')
    f.write('\nFinal positions:\n\n')
    for i, traj in enumerate(trajectories):
        f.write(f'Trajectory {i+1}: ({traj[-1,0]:.3f}, {traj[-1,1]:.3f})\n')
    
    # Save full trajectories
    f.write('\nFull trajectories:\n\n')
    for i, traj in enumerate(trajectories):
        f.write(f'\nTrajectory {i+1}:\n')
        for t in range(len(traj)):
            f.write(f't={t*dt:.2f}: ({traj[t,0]:.3f}, {traj[t,1]:.3f})\n')
