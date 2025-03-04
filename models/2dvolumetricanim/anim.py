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
    "dim": 100,            # Dimension de l'espace
    "sigma": 1.0,          # Écart-type du bruit gaussien
    "device": 0,           # ID du GPU à utiliser
    "t_range": (0, 10),     # Intervalle temporel pour l'analyse
    "n_points": 10000,      # Nombre de points pour la discrétisation
}

@dataclass
class DiffusionConfig:
    """Configuration pour l'analyse volumétrique de diffusion."""
    dim: int
    sigma: float = 1.0
    device: int = 0
    t_range: Tuple[float, float] = (0, 5)
    n_points: int = 1000
    save_dir: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            dim=config_dict["dim"],
            sigma=config_dict["sigma"],
            device=config_dict["device"],
            t_range=config_dict["t_range"],
            n_points=config_dict["n_points"]
        )

    def to_dict(self):
        return {
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
            
        self.d = config.dim
        self.sigma = config.sigma
        
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
        
        # Volume empirique (M^e) - calculé exactement pour la gaussienne
        S_G = (self.d/2) * (1 + cp.log(2 * cp.pi * delta_t))
        v_emp = S_G
        
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
        self.trajectories = [[] for _ in range(n_points)]  # Pour stocker les trajectoires
        
        # Crée l'échelle de temps logarithmique
        self.times = np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), n_frames)
        
        # Calcul du temps de collapse (quand les cercles se touchent)
        self.t_collapse = self.compute_collapse_time()
        
    def compute_collapse_time(self):
        """Calcule le temps approximatif de collapse."""
        # Trouve la plus petite distance entre deux points
        min_dist = float('inf')
        for i in range(self.n_points):
            for j in range(i + 1, self.n_points):
                dist = np.linalg.norm(self.points[i] - self.points[j])
                min_dist = min(min_dist, dist)
        
        # Le collapse se produit quand 2*radius = min_dist*exp(-t)
        # 2*sqrt(1-exp(-2t)) = min_dist*exp(-t)
        # Résolution numérique approximative
        t = 0.01
        while t < 5:
            if 2*np.sqrt(1-np.exp(-2*t)) >= min_dist*np.exp(-t):
                return t
            t += 0.01
        return t
        
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
        
        # Ligne verticale pour le temps de collapse
        self.collapse_line = self.ax_time.axvline(x=self.t_collapse, color='g', 
                                                 linestyle='--', alpha=0.5)
        self.ax_time.text(self.t_collapse, 1.05, f't_c = {self.t_collapse:.3f}', 
                         color='g', ha='center')
        
        return self.circles + [self.time_line, self.collapse_line]
    
    def update(self, frame):
        """Met à jour l'animation pour chaque frame."""
        t = self.times[frame]
        
        # Calcule Delta_t
        Delta_t = 1 - np.exp(-2*t)
        
        # Met à jour chaque cercle et trace les trajectoires
        for i, (circle, point) in enumerate(zip(self.circles, self.points)):
            # Position du centre
            center = point * np.exp(-t)
            # Rayon
            radius = np.sqrt(Delta_t)
            
            # Met à jour le cercle
            circle.center = center
            circle.radius = radius
            
            # Couleur basée sur l'indice du point
            color = plt.cm.viridis(i/self.n_points)
            circle.set_facecolor(color)
            
            # Ajoute le point à la trajectoire et trace
            self.trajectories[i].append(center)
            if len(self.trajectories[i]) > 1:
                traj = np.array(self.trajectories[i])
                self.ax.plot(traj[-2:, 0], traj[-2:, 1], color=color, alpha=0.3)
            
        # Met à jour le titre et l'indicateur de temps
        title = f'Temps t = {t:.3f}\nΔt = {Delta_t:.3f}'
        if abs(t - self.t_collapse) < (self.times[1] - self.times[0]):
            title += '\nCollapse!'
        self.ax.set_title(title)
        
        self.time_line.set_xdata([t, t])
        
        # Ajoute un point sur l'axe temporel
        self.ax_time.scatter(t, 0.5, color='red', alpha=0.5, s=1)
        
        return self.circles + [self.time_line, self.collapse_line]
    
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
        "n_points": 8,            # Nombre de points réduit pour mieux voir
        "d": 2,                   # Dimension (2D)
        "t_range": (0.01, 5),     # Intervalle de temps
        "n_frames": 200           # Nombre de frames pour l'animation
    }
    
    # Crée l'animation
    animator = DiffusionAnimation2D(**config)
    
    # Sauvegarde l'animation
    save_dir = Path(__file__).parent.parent.parent / "results" / "animations"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"diffusion_animation_trajectories_{timestamp}.gif"
    
    animator.create_animation(save_path=str(save_path))
    print(f"Animation sauvegardée dans: {save_path}")

if __name__ == "__main__":
    main()
