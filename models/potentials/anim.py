import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from pathlib import Path
import datetime
from tqdm import tqdm

class DiffusionHeatmapAnimation:
    def __init__(self, n_points=1000, d=2, n_samples=5000, alpha=0.5, t_range=(0.01, 5), n_frames=200):
        """
        Initialise l'animation de la distribution de probabilité pour le processus de diffusion.
        
        Args:
            n_points (int): Nombre de points pour la grille 2D
            d (int): Dimension (fixée à 2 pour la visualisation)
            n_samples (int): Nombre d'échantillons pour l'estimation
            alpha (float): log(n)/d, ratio pour l'exponentialité
            t_range (tuple): Intervalle temporel
            n_frames (int): Nombre d'images pour l'animation
        """
        self.n_points = n_points
        self.d = d
        self.n_samples = n_samples
        self.alpha = alpha
        self.t_range = t_range
        self.n_frames = n_frames
        
        # Génère les données initiales (distribution gaussienne 2D)
        self.data = cp.random.randn(n_samples, d)
        
        # Crée la grille 2D pour la heatmap
        x = cp.linspace(-4, 4, n_points)
        y = cp.linspace(-4, 4, n_points)
        self.X, self.Y = cp.meshgrid(x, y)
        
        # Setup de l'animation
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Crée l'échelle de temps logarithmique
        self.times = cp.logspace(cp.log10(t_range[0]), cp.log10(t_range[1]), n_frames)
        
    def compute_theoretical_density(self, t):
        """Calcule la densité de probabilité théorique au temps t."""
        delta_t = 1 - cp.exp(-2*t)
        Z = cp.zeros_like(self.X)
        positions = cp.stack([self.X, self.Y], axis=-1)
        
        # Calcul de la distribution théorique
        for a in self.data:
            mean = a * cp.exp(-t)
            Z += cp.exp(-0.5 * cp.sum((positions - mean.reshape(1,1,2))**2 / delta_t, axis=-1))
            
        return cp.asnumpy(Z / (2*np.pi*delta_t) / self.n_samples)
    
    def compute_empirical_density(self, t):
        """Calcule la densité de probabilité empirique au temps t."""
        # Applique la diffusion aux données
        data_t = self.data * cp.exp(-t) + cp.sqrt(1 - cp.exp(-2*t)) * cp.random.randn(self.n_samples, self.d)
        
        # Calcule la densité sur la grille par convolution avec noyau gaussien
        Z = cp.zeros_like(self.X)
        positions = cp.vstack([self.X.ravel(), self.Y.ravel()])
        
        for x, y in data_t:
            delta = positions - cp.array([[x], [y]])
            kernel = cp.exp(-0.5 * cp.sum(delta**2, axis=0))
            Z += kernel.reshape(self.n_points, self.n_points)
            
        return cp.asnumpy(Z / self.n_samples)
        
    def init_animation(self):
        """Initialise l'animation."""
        for ax in [self.ax1, self.ax2]:
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            
        self.im1 = self.ax1.imshow(np.zeros((self.n_points, self.n_points)), 
                                  extent=[-4, 4, -4, 4],
                                  origin='lower',
                                  cmap='viridis')
        self.im2 = self.ax2.imshow(np.zeros((self.n_points, self.n_points)), 
                                  extent=[-4, 4, -4, 4],
                                  origin='lower',
                                  cmap='viridis')
                                  
        self.ax1.set_title('Distribution théorique')
        self.ax2.set_title('Distribution empirique')
        plt.colorbar(self.im1, ax=self.ax1)
        plt.colorbar(self.im2, ax=self.ax2)
        return [self.im1, self.im2]
    
    def update(self, frame):
        """Met à jour l'animation pour chaque frame."""
        t = float(self.times[frame])
        Z_theo = self.compute_theoretical_density(t)
        Z_emp = self.compute_empirical_density(t)
        
        self.im1.set_array(Z_theo)
        self.im2.set_array(Z_emp)
        
        max_val = max(np.max(Z_theo), np.max(Z_emp))
        self.im1.set_clim(0, max_val)
        self.im2.set_clim(0, max_val)
        
        self.ax1.set_title(f'Distribution théorique (t = {t:.3f})')
        self.ax2.set_title(f'Distribution empirique (t = {t:.3f})')
        
        return [self.im1, self.im2]
    
    def create_animation(self, save_path=None):
        """Crée et sauvegarde l'animation."""
        anim = FuncAnimation(self.fig, self.update, frames=tqdm(range(self.n_frames)),
                           init_func=self.init_animation, blit=True,
                           interval=50)
        
        if save_path:
            writer = animation.PillowWriter(fps=20)
            save_path = str(save_path).replace('.mp4', '.gif')
            anim.save(save_path, writer=writer)
            plt.close()
        else:
            plt.show()
        
        return anim

def main():
    # Crée deux animations avec différents ratios alpha
    configs = [
        {"n_samples": 100, "alpha": 0.1},  # Peu d'échantillons
        {"n_samples": 5000, "alpha": 2.0}  # Beaucoup d'échantillons
    ]
    
    # Sauvegarde les animations
    save_dir = Path(__file__).parent.parent.parent / "results" / "animations"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, config in tqdm(enumerate(configs), total=len(configs), desc="Generating animations"):
        print(f"Génération de l'animation {i+1}/2 avec alpha={config['alpha']}")
        animator = DiffusionHeatmapAnimation(
            n_points=100,
            d=2,
            n_samples=config['n_samples'],
            alpha=config['alpha'],
            t_range=(0.01, 5),
            n_frames=200
        )
        
        save_path = save_dir / f"diffusion_heatmap_alpha{config['alpha']}_{timestamp}.gif"
        animator.create_animation(save_path=str(save_path))
        print(f"Animation sauvegardée dans: {save_path}")

if __name__ == "__main__":
    main()
