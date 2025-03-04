import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.animation as animation
import matplotlib
matplotlib.use('Agg')

def compute_potential(x: np.ndarray, t: float, d: int, mu_tilde: float = 1.0, eps: float = 0.1) -> np.ndarray:
    """
    Calcule le potentiel V(x,t) pour les quatre points autour de ±m.
    
    Args:
        x (np.ndarray): Points où évaluer le potentiel
        t (float): Temps
        d (int): Dimension
        mu_tilde (float): Paramètre μ̃ du modèle
        eps (float): Taille de la perturbation ε
        
    Returns:
        np.ndarray: Valeurs du potentiel V(x,t)
    """
    m = mu_tilde * np.sqrt(d)  # Norme de m
    Gamma_t = np.exp(-2*t) + (1 - np.exp(-2*t))  # Γ_t
    
    # Calcul pour les quatre points x = ±m ± ε
    V = np.zeros_like(x)
    for i, xi in enumerate(x):
        # Point 1: m - ε
        x1 = m - eps
        A1 = (m*(1-np.exp(-t)) - eps)**2
        B1 = (m*(1+np.exp(-t)) - eps)**2
        V1 = 0.5*(m**2 + eps**2) - 2*np.log(1/(2*np.sqrt(2*np.pi*Gamma_t)) * 
                                            (np.exp(-A1/(2*Gamma_t)) + np.exp(-B1/(2*Gamma_t))))
        
        # Point 2: m + ε
        x2 = m + eps
        A2 = (m*(1-np.exp(-t)) + eps)**2
        B2 = (m*(1+np.exp(-t)) + eps)**2
        V2 = 0.5*(m**2 + eps**2) - 2*np.log(1/(2*np.sqrt(2*np.pi*Gamma_t)) * 
                                            (np.exp(-A2/(2*Gamma_t)) + np.exp(-B2/(2*Gamma_t))))
        
        # Point 3: -m - ε
        x3 = -m - eps
        A3 = (-m*(1+np.exp(-t)) - eps)**2
        B3 = (-m*(1-np.exp(-t)) - eps)**2
        V3 = 0.5*(m**2 + eps**2) - 2*np.log(1/(2*np.sqrt(2*np.pi*Gamma_t)) * 
                                            (np.exp(-A3/(2*Gamma_t)) + np.exp(-B3/(2*Gamma_t))))
        
        # Point 4: -m + ε
        x4 = -m + eps
        A4 = (-m*(1+np.exp(-t)) + eps)**2
        B4 = (-m*(1-np.exp(-t)) + eps)**2
        V4 = 0.5*(m**2 + eps**2) - 2*np.log(1/(2*np.sqrt(2*np.pi*Gamma_t)) * 
                                            (np.exp(-A4/(2*Gamma_t)) + np.exp(-B4/(2*Gamma_t))))
        
        # Interpolation entre les points pour une visualisation continue
        if xi <= -m:
            alpha = (xi - (-m-eps))/eps if eps != 0 else 0
            V[i] = (1-alpha)*V3 + alpha*V4
        elif xi <= 0:
            alpha = (xi - (-m+eps))/(m-eps)
            V[i] = (1-alpha)*V4 + alpha*V1
        elif xi <= m:
            alpha = (xi - (m-eps))/eps if eps != 0 else 0
            V[i] = (1-alpha)*V1 + alpha*V2
        else:
            V[i] = V2
            
    return V

def create_potential_animation(d: int = 100, 
                             x_range: tuple = (-3, 3),
                             n_points: int = 1000,
                             n_frames: int = 200,
                             fps: int = 30,
                             eps: float = 0.1) -> None:
    """
    Crée une animation GIF du potentiel évoluant dans le temps.
    
    Args:
        d (int): Dimension
        x_range (tuple): Intervalle pour x
        n_points (int): Nombre de points pour l'échantillonnage
        n_frames (int): Nombre d'images dans l'animation
        fps (int): Images par seconde pour le GIF
        eps (float): Taille de la perturbation ε
    """
    # Préparation des données
    x = np.linspace(x_range[0], x_range[1], n_points)
    t_switch = 0.5 * np.log(d)  # Temps critique t_s
    
    # Temps en échelle logarithmique
    t_min, t_max = 0.01 * t_switch, 5 * t_switch
    times = np.logspace(np.log10(t_min), np.log10(t_max), n_frames)
    
    # Configuration de l'animation
    fig, ax = plt.subplots(figsize=(12, 8))
    line, = ax.plot([], [], 'b-', label='Potentiel V(x,t)')
    points, = ax.plot([], [], 'ro', label='Points spéciaux')
    ax.grid(True)
    ax.legend()
    
    # Limites fixes pour l'animation
    ax.set_xlim(x_range)
    V_min = float(np.min([compute_potential(x, t, d, eps=eps) for t in [times[-1], times[0]]]))
    V_max = float(np.max([compute_potential(x, t, d, eps=eps) for t in [times[-1], times[0]]]))
    ax.set_ylim(V_min - 0.5, V_max + 0.5)
    
    ax.set_xlabel('x')
    ax.set_ylabel('V(x,t)')
    title = ax.set_title('')
    
    def init():
        line.set_data([], [])
        points.set_data([], [])
        return line, points
    
    def animate(frame):
        t = times[frame]
        V = compute_potential(x, t, d, eps=eps)
        line.set_data(x, V)
        
        # Points spéciaux
        m = np.sqrt(d)
        special_points_x = [m-eps, m+eps, -m-eps, -m+eps]
        special_points_y = compute_potential(np.array(special_points_x), t, d, eps=eps)
        points.set_data(special_points_x, special_points_y)
        
        title.set_text(f't/tS = {float(t/t_switch):.2f}')
        return line, points, title
    
    # Création de l'animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=n_frames, interval=1000//fps, 
                                 blit=True)
    
    # Sauvegarde
    save_dir = Path('results/potentials')
    save_dir.mkdir(parents=True, exist_ok=True)
    anim.save(save_dir / f'potential_evolution_4points_d{d}.gif', 
              writer='pillow', fps=fps)
    plt.close()

if __name__ == "__main__":
    # Création de l'animation pour différentes dimensions
    for d in [10, 50, 100, 500]:
        create_potential_animation(d=d, x_range=(-4, 4), eps=0.1)
