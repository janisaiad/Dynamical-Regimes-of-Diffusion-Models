import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm

class DiffusionVolumeAnalysis:
    """Analyse volumétrique des modèles de diffusion."""
    
    def __init__(self, n_samples, dim, sigma=1.0, device=0):
        """
        Args:
            n_samples (int): Nombre de points dans le jeu d'entraînement
            dim (int): Dimension de l'espace
            sigma (float): Écart-type du bruit gaussien
            device (int): GPU device ID
        """
        cp.cuda.Device(device).use() # Use GPU device
        self.n = n_samples
        self.d = dim
        self.sigma = sigma
        self.alpha = np.log(n_samples) / dim
        
    def generate_training_data(self):
        """Génère des données d'entraînement synthétiques."""
        # Génère deux clusters gaussiens
        self.data = cp.random.normal(0, 1, (self.n, self.d))
        # Normalise pour avoir |x|^2 ~ d
        self.data = self.data * cp.sqrt(self.d / cp.sum(self.data**2, axis=1, keepdims=True))
        return self.data
    
    def compute_delta_t(self, t):
        """Calcule Δ_t = 1 - exp(-2t)."""
        return 1 - cp.exp(-2 * t)
    
    def compute_ball_volume(self, t):
        """Calcule le volume d'une boule en dimension d avec rayon √(dΔ_t)."""
        delta_t = self.compute_delta_t(t)
        # Log du volume de la boule unitaire en dimension d
        log_unit_ball = (self.d/2) * cp.log(2 * cp.pi * cp.e / self.d)
        # Ajout du facteur d'échelle
        return log_unit_ball + (self.d/2) * cp.log(delta_t)
    
    def compute_empirical_volume(self, t):
        """Calcule le volume de M^e."""
        S_G = (self.d/2) * (1 + cp.log(2 * cp.pi * self.compute_delta_t(t)))
        return cp.log(self.n) + S_G
    
    def compute_population_volume(self, t):
        """Calcule le volume de M."""
        # Approximation simple de l'entropie de la population
        s_t = (self.d/2) * (1 + cp.log(2 * cp.pi * (self.compute_delta_t(t) + self.sigma**2)))
        return self.d * s_t
    
    def find_collapse_time(self, t_range):
        """Trouve le temps de collapse t_C."""
        times = cp.linspace(t_range[0], t_range[1], 1000)
        v_emp = cp.array([self.compute_empirical_volume(t).get() for t in times])
        v_pop = cp.array([self.compute_population_volume(t).get() for t in times])
        
        # Trouve où les volumes sont égaux
        diff = cp.abs(v_emp - v_pop)
        t_c_idx = cp.argmin(diff)
        return times[t_c_idx]
    
    def visualize_volumes(self, t_range):
        """Visualise l'évolution des volumes."""
        times = cp.linspace(t_range[0], t_range[1], 100)
        v_emp = cp.array([self.compute_empirical_volume(t).get() for t in times]).get()
        v_pop = cp.array([self.compute_population_volume(t).get() for t in times]).get()
        times = times.get()
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, v_emp, label='Volume empirique (M^e)')
        plt.plot(times, v_pop, label='Volume population (M)')
        plt.axvline(x=self.find_collapse_time(t_range).get(), color='r', 
                   linestyle='--', label='t_C')
        plt.xlabel('Temps t')
        plt.ylabel('log(Volume)')
        plt.title(f'Évolution des volumes (d={self.d}, n={self.n})')
        plt.legend()
        plt.grid(True)
        plt.show()

def run_experiment():
    """Exécute une expérience de démonstration."""
    # Paramètres
    dims = [100, 500, 1000]
    n_samples = 10000
    
    for dim in dims:
        print(f"\nAnalyse pour dimension {dim}")
        analyzer = DiffusionVolumeAnalysis(n_samples, dim)
        analyzer.generate_training_data()
        
        # Trouve et affiche t_C
        t_c = analyzer.find_collapse_time((0, 5))
        print(f"Temps de collapse t_C ≈ {t_c.get():.3f}")
        
        # Visualise les volumes
        analyzer.visualize_volumes((0, 5))
        
        # Calcule l'entropie excédentaire
        t = cp.linspace(0, 5, 100)
        excess_entropy = (analyzer.compute_empirical_volume(t) - 
                         analyzer.compute_population_volume(t)) / analyzer.d
        print(f"Entropie excédentaire maximale: {cp.max(excess_entropy).get():.3f}")

if __name__ == "__main__":
    run_experiment()
