import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.animation as animation
import matplotlib
import tqdm
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

def calculate_optimal_q_range(t: float, d: int, mu_tilde: float = 1.0) -> tuple:
    """
    Calcule une plage optimale pour q en fonction du temps t.
    
    Args:
        t (float): Temps actuel
        d (int): Dimension
        mu_tilde (float): Paramètre du modèle
        
    Returns:
        tuple: Plage (q_min, q_max) adaptée au temps t
    """
    # Le facteur important est exp(-t) * sqrt(d)
    scaling_factor = cp.exp(-t) * cp.sqrt(d)
    
    if scaling_factor > 1:  # Temps plus petit que t_s
        # Pour les temps courts, nous avons besoin d'une plage plus large
        q_scale = 2.5 / scaling_factor
        return (-q_scale, q_scale)
    else:
        # Pour les temps longs, nous nous concentrons sur les minima qui sont autour de ±1
        return (-2.5, 2.5)

def create_potential_animation(d: int = 100, 
                             base_q_range: tuple = (-3, 3),
                             n_points: int = 1000,
                             n_frames: int = 2000,
                             fps: int = 30,
                             adaptive_scaling: bool = True) -> None:
    """
    Crée une animation GIF du potentiel évoluant dans le temps en échelle logarithmique.
    
    Args:
        d (int): Dimension
        base_q_range (tuple): Intervalle de base pour q (utilisé si adaptive_scaling=False)
        n_points (int): Nombre de points pour l'échantillonnage
        n_frames (int): Nombre d'images dans l'animation
        fps (int): Images par seconde pour le GIF
        adaptive_scaling (bool): Si True, adapte l'échelle des axes en fonction du temps
    """
    # Calcul du temps critique
    t_switch = 0.5 * cp.log(d)
    
    # Temps en échelle logarithmique autour du temps critique
    t_min, t_max = 0.01 * t_switch, 5 * t_switch
    times = cp.logspace(cp.log10(t_min), cp.log10(t_max), n_frames)
    
    # Configuration de l'animation
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [])
    ax.grid(True)
    
    # Initialisation de la plage q (sera mise à jour si adaptive_scaling=True)
    if not adaptive_scaling:
        q_range = base_q_range
        q = cp.linspace(q_range[0], q_range[1], n_points)
        
        # Limites fixes pour l'animation si non adaptatif
        ax.set_xlim(q_range)
        V_min = float(cp.min(compute_potential_gpu(q, times[-1], d)))
        V_max = float(cp.max(compute_potential_gpu(q, times[0], d)))
        ax.set_ylim(V_min - 0.5, V_max + 0.5)
    
    ax.set_xlabel('q')
    ax.set_ylabel('V(q,t)')
    title = ax.set_title('')
    
    # Points de référence pour les minima théoriques (après la séparation)
    minima_points = cp.array([-1.0, 1.0])
    minima_scatter = ax.scatter([], [], color='red', s=50, zorder=3)
    
    # Ligne en pointillé pour V=0
    zero_line = ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Légende sur le temps critique
    t_crit_text = ax.text(0.02, 0.98, f't_s ≈ {float(t_switch):.3f}', 
                         transform=ax.transAxes, fontsize=10,
                         verticalalignment='top')
    
    def init():
        line.set_data([], [])
        minima_scatter.set_offsets(np.empty((0, 2)))
        return line, minima_scatter
    
    def animate(frame):
        t = times[frame]
        
        # Mise à jour des limites et des points q si adaptatif
        if adaptive_scaling:
            q_range = calculate_optimal_q_range(t, d)
            q = cp.linspace(q_range[0], q_range[1], n_points)
            ax.set_xlim(q_range)
        else:
            q = cp.linspace(base_q_range[0], base_q_range[1], n_points)
        
        # Calcul du potentiel
        V = compute_potential_gpu(q, t, d)
        
        # Mise à jour de la courbe
        line.set_data(cp.asnumpy(q), cp.asnumpy(V))
        
        # Adaptation des limites y pour chaque frame
        if adaptive_scaling:
            V_min = float(cp.min(V))
            V_max = float(cp.max(V))
            
            # Ajouter de la marge
            y_margin = 0.1 * (V_max - V_min) if V_max > V_min else 0.5
            ax.set_ylim(V_min - y_margin, V_max + y_margin)
        
        # Mise à jour du titre avec le rapport t/t_switch
        ratio = float(t/t_switch)
        title.set_text(f't/tS = {ratio:.2f}')
        
        # Mettre à jour les points de minimum du potentiel
        # Ils apparaissent progressivement à mesure que t diminue en dessous de t_s
        if t < t_switch:
            # Pour les petits t, les minima sont proches de ±1
            minima_y = compute_potential_gpu(minima_points, t, d)
            minima_data = np.column_stack((cp.asnumpy(minima_points), cp.asnumpy(minima_y)))
            minima_scatter.set_offsets(minima_data)
            # Convertir la valeur CuPy en float Python pour éviter l'erreur TypeError
            alpha_value = float(min(1.0, (t_switch-t)/(0.5*t_switch)))
            minima_scatter.set_alpha(alpha_value)  # Apparition progressive
        else:
            minima_scatter.set_offsets(np.empty((0, 2)))
            
        return line, minima_scatter, title
    
    # Création de l'animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=n_frames, interval=1000//fps, 
                                 blit=True)
    
    # Sauvegarde
    save_dir = Path('results/potentials')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    scaling_type = "adaptive" if adaptive_scaling else "fixed"
    anim.save(save_dir / f'potential_evolution_d{d}_{scaling_type}.gif', 
              writer='pillow', fps=fps)
    plt.close()

    # Libération de la mémoire GPU
    cp.get_default_memory_pool().free_all_blocks()

if __name__ == "__main__":
    # Générer des animations pour différentes dimensions avec mise à l'échelle adaptative
    for d in tqdm.tqdm(range(10, 1000, 10)):
        create_potential_animation(d=d, adaptive_scaling=True)
        print(f"Animation créée pour d={d} avec mise à l'échelle adaptative")