import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.animation as animation
import matplotlib
matplotlib.use('Agg')

def compute_potential_gpu(q: cp.ndarray, t: float, d: int, mu_tilde: float = 1.0) -> cp.ndarray:
    """
    Calcule le potentiel V(q,t) sur GPU selon l'équation (8).
    
    Args:
        q (cp.ndarray): Points où évaluer le potentiel
        t (float): Temps 
        d (int): Dimension
        mu_tilde (float): Paramètre μ̃ du modèle
        
    Returns:
        cp.ndarray: Valeurs du potentiel V(q,t)
    """
    return 0.5 * q**2 - 2 * mu_tilde**2 * cp.log(cp.cosh(q * cp.exp(-t) * cp.sqrt(d)))

def create_potential_animation(d: int = 100, 
                             q_range: tuple = (-3, 3),
                             n_points: int = 1000,
                             n_frames: int = 200,
                             fps: int = 30) -> None:
    """
    Crée une animation GIF du potentiel évoluant dans le temps en échelle logarithmique.
    
    Args:
        d (int): Dimension
        q_range (tuple): Intervalle pour q
        n_points (int): Nombre de points pour l'échantillonnage
        n_frames (int): Nombre d'images dans l'animation
        fps (int): Images par seconde pour le GIF
    """
    # Préparation des données sur GPU
    q = cp.linspace(q_range[0], q_range[1], n_points)
    t_switch = 0.5 * cp.log(d)
    
    # Temps en échelle logarithmique
    t_min, t_max = 0.01 * t_switch, 5 * t_switch
    times = cp.logspace(cp.log10(t_min), cp.log10(t_max), n_frames)
    
    # Configuration de l'animation
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [])
    ax.grid(True)
    
    # Limites fixes pour l'animation
    ax.set_xlim(q_range)
    V_min = float(cp.min(compute_potential_gpu(q, times[-1], d)))
    V_max = float(cp.max(compute_potential_gpu(q, times[0], d)))
    ax.set_ylim(V_min - 0.5, V_max + 0.5)
    
    ax.set_xlabel('q')
    ax.set_ylabel('V(q,t)')
    title = ax.set_title('')
    
    def init():
        line.set_data([], [])
        return line,
    
    def animate(frame):
        t = times[frame]
        V = compute_potential_gpu(q, t, d)
        line.set_data(cp.asnumpy(q), cp.asnumpy(V))
        title.set_text(f't/tS = {float(t/t_switch):.2f}')
        return line, title
    
    # Création de l'animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=n_frames, interval=1000//fps, 
                                 blit=True)
    
    # Sauvegarde
    save_dir = Path('results/potentials')
    save_dir.mkdir(parents=True, exist_ok=True)
    anim.save(save_dir / f'potential_evolution_d{d}.gif', 
              writer='pillow', fps=fps)
    plt.close()

    # Libération de la mémoire GPU
    cp.get_default_memory_pool().free_all_blocks()

