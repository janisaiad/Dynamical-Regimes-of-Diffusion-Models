import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional
import logging
import json
from pathlib import Path
import datetime
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.animation as animation

# Configuration globale
CONFIG = {
    "n_samples": 10000,    # Nombre d'échantillons
    "dim": 100,            # Dimension de l'espace
    "sigma": 1.0,          # Écart-type du bruit gaussien
    "device": 0,           # ID du GPU à utiliser
    "t_range": (0, 10),     # Intervalle temporel pour l'analyse
    "n_points": 10000,      # Nombre de points pour la discrétisation
}

@dataclass
class DiffusionConfig:
    """Configuration pour l'analyse volumétrique de diffusion."""
    n_samples: int
    dim: int
    sigma: float = 1.0
    device: int = 0
    t_range: Tuple[float, float] = (0, 5)
    n_points: int = 1000
    save_dir: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            n_samples=config_dict["n_samples"],
            dim=config_dict["dim"],
            sigma=config_dict["sigma"],
            device=config_dict["device"],
            t_range=config_dict["t_range"],
            n_points=config_dict["n_points"]
        )

    def to_dict(self):
        return {
            "n_samples": self.n_samples,
            "dim": self.dim,
            "sigma": self.sigma,
            "device": self.device,
            "t_range": self.t_range,
            "n_points": self.n_points,
            "save_dir": self.save_dir
        }

class DiffusionVolumeAnalysis:
    """Analyse volumétrique des modèles de diffusion avec support CUDA."""
    
    def __init__(self, config: DiffusionConfig):
        """
        Initialise l'analyseur avec la configuration donnée.
        
        Args:
            config (DiffusionConfig): Configuration de l'analyse
        """
        self.config = config
        self.logger = self._setup_logger()
        
        try:
            cp.cuda.Device(config.device).use()
            self.logger.info(f"Utilisation du GPU {config.device}")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation du GPU: {e}")
            raise
            
        self.n = config.n_samples
        self.d = config.dim
        self.sigma = config.sigma
        self.alpha = np.log(config.n_samples) / config.dim
        
    def _setup_logger(self) -> logging.Logger:
        """Configure le logger pour l'analyse."""
        logger = logging.getLogger("DiffusionAnalysis")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Handler pour la console
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # Création du dossier logs s'il n'existe pas
            log_dir = Path(__file__).parent.parent.parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Handler pour le fichier dans le dossier logs
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"diffusion_analysis_{timestamp}.log"
            log_path = log_dir / log_file
            
            file_handler = logging.FileHandler(log_path)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Les logs sont sauvegardés dans: {log_path}")
            
        return logger
        
    def generate_training_data(self) -> cp.ndarray:
        """
        Génère des données d'entraînement synthétiques normalisées.
        
        Returns:
            cp.ndarray: Données générées de forme (n_samples, dim)
        """
        self.logger.info("Génération des données d'entraînement...")
        
        # Génère des clusters gaussiens
        self.data = cp.random.normal(0, 1, (self.n, self.d))
        
        # Normalisation pour avoir |x|^2 ~ d
        norms = cp.sqrt(cp.sum(self.data**2, axis=1, keepdims=True))
        self.data = self.data * cp.sqrt(self.d) / norms
        
        self.logger.info(f"Données générées: forme {self.data.shape}")
        return self.data
    
    def compute_delta_t(self, t: float) -> cp.ndarray:
        """
        Calcule Δ_t = 1 - exp(-2t).
        
        Args:
            t (float): Temps
            
        Returns:
            cp.ndarray: Valeur de Δ_t
        """
        return 1 - cp.exp(-2 * t)
    
    def compute_volumes(self, t: float) -> Tuple[float, float]:
        """
        Calcule les volumes empirique et de population.
        
        Args:
            t (float): Temps
            
        Returns:
            Tuple[float, float]: (volume empirique, volume population)
        """
        delta_t = self.compute_delta_t(t)
        
        # Volume empirique (M^e)
        S_G = (self.d/2) * (1 + cp.log(2 * cp.pi * delta_t))
        v_emp = cp.log(self.n) + S_G
        
        # Volume population (M)
        s_t = (self.d/2) * (1 + cp.log(2 * cp.pi * (delta_t + self.sigma**2)))
        v_pop = self.d * s_t
        
        return float(v_emp), float(v_pop)
    
    def find_collapse_time(self) -> float:
        """
        Trouve le temps de collapse t_C.
        
        Returns:
            float: Temps de collapse estimé
        """
        self.logger.info("Recherche du temps de collapse...")
        
        times = cp.linspace(self.config.t_range[0], 
                          self.config.t_range[1], 
                          self.config.n_points)
        
        min_diff = float('inf')
        t_c = None
        
        for t in tqdm(times):
            v_emp, v_pop = self.compute_volumes(float(t))
            diff = abs(v_emp - v_pop)
            
            if diff < min_diff:
                min_diff = diff
                t_c = float(t)
        
        self.logger.info(f"Temps de collapse trouvé: t_C ≈ {t_c:.3f}")
        return t_c
    
    def analyze_and_save(self):
        """Effectue l'analyse complète et sauvegarde les résultats."""
        # Génère les données
        self.generate_training_data()
        
        # Trouve t_C
        t_c = self.find_collapse_time()
        
        # Calcule l'entropie excédentaire
        t = cp.linspace(self.config.t_range[0], 
                       self.config.t_range[1], 
                       100)
        
        excess_entropies = []
        for ti in t:
            v_emp, v_pop = self.compute_volumes(float(ti))
            excess_entropy = (v_emp - v_pop) / self.d
            excess_entropies.append(float(excess_entropy))
        
        max_excess_entropy = max(excess_entropies)
        
        # Sauvegarde des résultats
        results = {
            "config": self.config.to_dict(),
            "t_c": t_c,
            "max_excess_entropy": max_excess_entropy,
            "excess_entropies": excess_entropies,
            "times": t.tolist()
        }
        
        if self.config.save_dir:
            save_path = Path(self.config.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            with open(save_path / "results.json", "w") as f:
                json.dump(results, f, indent=4)
            
            self.logger.info(f"Résultats sauvegardés dans {save_path}")
        
        return results

class DiffusionAnimation2D:
    def __init__(self, n_points=5, d=2, t_range=(0.01, 5), n_frames=200):
        self.n_points = n_points
        self.d = d
        self.t_range = t_range
        self.n_frames = n_frames
        
        # Génère les points initiaux
        self.points = np.random.randn(n_points, d)
        # Normalise les points
        self.points = self.points / np.linalg.norm(self.points, axis=1)[:, np.newaxis]
        
        # Setup de l'animation
        self.fig, (self.ax, self.ax_time) = plt.subplots(2, 1, figsize=(10, 12), 
                                                         gridspec_kw={'height_ratios': [4, 1]})
        self.circles = []
        
        # Crée l'échelle de temps logarithmique
        self.times = np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), n_frames)
        
    def init_animation(self):
        """Initialise l'animation."""
        self.ax.set_xlim(-2.5, 2.5)
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        
        # Crée les cercles initiaux
        self.circles = [Circle((0, 0), 0.1, alpha=0.3) for _ in range(self.n_points)]
        for circle in self.circles:
            self.ax.add_artist(circle)
        
        # Configure l'axe temporel
        self.ax_time.set_xlim(self.t_range)
        self.ax_time.set_xscale('log')
        self.ax_time.set_ylim(-0.1, 1.1)
        self.ax_time.set_xlabel('Temps (échelle log)')
        self.ax_time.grid(True)
        
        # Ligne verticale pour indiquer le temps actuel
        self.time_line = self.ax_time.axvline(x=self.t_range[0], color='r')
        
        return self.circles + [self.time_line]
    
    def update(self, frame):
        """Met à jour l'animation pour chaque frame."""
        t = self.times[frame]
        
        # Calcule Delta_t
        Delta_t = 1 - np.exp(-2*t)
        
        # Met à jour chaque cercle
        for i, (circle, point) in enumerate(zip(self.circles, self.points)):
            # Position du centre
            center = point * np.exp(-t)
            # Rayon
            radius = np.sqrt(Delta_t)
            
            # Met à jour le cercle
            circle.center = center
            circle.radius = radius
            
            # Couleur basée sur l'indice du point
            circle.set_facecolor(plt.cm.viridis(i/self.n_points))
            
        # Met à jour le titre et l'indicateur de temps
        self.ax.set_title(f'Temps t = {t:.3f}\nΔt = {Delta_t:.3f}')
        self.time_line.set_xdata([t, t])
        
        # Ajoute un point sur l'axe temporel
        self.ax_time.scatter(t, 0.5, color='red', alpha=0.5, s=1)
        
        return self.circles + [self.time_line]
    
    def create_animation(self, save_path=None):
        """Crée et sauvegarde l'animation."""
        anim = FuncAnimation(self.fig, self.update, frames=self.n_frames,
                           init_func=self.init_animation, blit=True,
                           interval=100)
        
        if save_path:
            writer = animation.PillowWriter(fps=20)
            save_path = str(save_path).replace('.mp4', '.gif')
            anim.save(save_path, writer=writer)
            plt.close()
        else:
            plt.show()
        
        return anim

def main():
    # Configuration avec échelle logarithmique
    config = {
        "n_points": 10,           # Nombre de points
        "d": 2,                   # Dimension (2D)
        "t_range": (0.01, 5),     # Intervalle de temps (commence à 0.01 pour log)
        "n_frames": 200           # Nombre de frames pour l'animation
    }
    
    # Crée l'animation
    animator = DiffusionAnimation2D(**config)
    
    # Sauvegarde l'animation
    save_dir = Path(__file__).parent.parent.parent / "results" / "animations"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"diffusion_animation_log_{timestamp}.gif"
    
    animator.create_animation(save_path=str(save_path))
    print(f"Animation sauvegardée dans: {save_path}")

if __name__ == "__main__":
    main()
