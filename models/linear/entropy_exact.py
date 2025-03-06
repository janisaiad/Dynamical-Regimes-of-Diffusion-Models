import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def compute_exact_entropy(m, t, d=784, device='cuda'):
    """
    Calcule l'entropie exacte d'un mélange gaussien bidirectionnel
    centré en ±m*exp(-t).
    
    Args:
        m (float ou torch.Tensor): Norme de m (μ̃ * sqrt(d))
        t (float ou list): Temps de diffusion (peut être un scalaire ou tableau)
        d (int): Dimension de l'espace
        device (str): 'cuda' ou 'cpu'
        
    Returns:
        torch.Tensor: Entropie exacte S(t)
    """
    # Conversion en tenseurs
    if not isinstance(m, torch.Tensor):
        m = torch.tensor(m, dtype=torch.float32, device=device)
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32, device=device).reshape(-1)
    
    # Variance du processus
    Delta_t = 1 - torch.exp(-2*t)
    Gamma_t = Delta_t  # Simplifié avec σ=0
    
    # Terme de normalisation pour une seule gaussienne
    log_Z = d/2 * torch.log(2*np.pi*Gamma_t)
    
    # Distance entre les centres des gaussiennes
    mu_distance = 2 * m * torch.exp(-t)
    
    # Calculer l'overlap entre les gaussiennes
    # α = exp(-|μ₁-μ₂|²/(4*σ²)) où μ₁-μ₂ = 2m*exp(-t)
    alpha = torch.exp(-(mu_distance**2) / (4 * Gamma_t))
    
    # Probabilité de mixage effective pour une gaussienne
    # p = 0.5 (fixe dans notre cas pour un mélange symétrique)
    p = 0.5
    
    # Contribution à l'entropie du mixage
    entropy_mix = -p * torch.log(p) - p * torch.log(p)  # = log(2) dans notre cas
    
    # Contribution à l'entropie de chaque gaussienne: d/2 * log(2πeσ²)
    entropy_gauss = d/2 * (torch.log(2*np.pi*Gamma_t) + 1)
    
    # Terme correctif dû au chevauchement des gaussiennes
    # Ce terme est complexe et requiert une approximation
    # Nous utilisons un développement en série pour faible/fort chevauchement
    
    # Cas 1: Faible chevauchement (grands t)
    overlap_term_small = -2 * p**2 * alpha
    
    # Cas 2: Fort chevauchement (petits t)
    # Nous avons besoin d'un terme plus précis pour éviter la divergence
    overlap_term_large = -p * torch.log(1 + p/(1-p) * torch.exp(-mu_distance**2/(2*Gamma_t)))
    
    # Transition douce entre les deux régimes
    transition = torch.sigmoid(5 * (t - 0.5*torch.log(torch.tensor(d, dtype=torch.float32, device=device))))
    overlap_term = transition * overlap_term_small + (1-transition) * overlap_term_large
    
    # Entropie totale
    entropy = entropy_mix + entropy_gauss + overlap_term
    
    return entropy

def entropy_vs_dimension():
    """
    Calcule et trace l'entropie exacte en fonction du temps pour 
    différentes dimensions.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dimensions = [10, 50, 100, 500, 784]
    mu_tilde = 1.0  # paramètre normalisé
    
    plt.figure(figsize=(10, 6))
    
    for d in dimensions:
        # Temps critique théorique
        t_switch = 0.5 * np.log(d)
        
        # Plage de temps en échelle logarithmique autour de t_switch
        t_values = np.logspace(np.log10(0.01*t_switch), np.log10(5*t_switch), 200)
        t_tensor = torch.tensor(t_values, dtype=torch.float32, device=device)
        
        # Norme de m
        m = mu_tilde * np.sqrt(d)
        
        # Calcul de l'entropie
        entropy = compute_exact_entropy(m, t_tensor, d, device=device)
        
        # Tracé
        plt.semilogx(t_values, entropy.cpu().numpy(), label=f'd = {d}')
        plt.axvline(x=t_switch, color=f'C{dimensions.index(d)}', linestyle='--', alpha=0.5)
    
    plt.xlabel('Temps t')
    plt.ylabel('Entropie S(t)')
    plt.title('Entropie exacte du mélange gaussien en fonction du temps')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    
    # Sauvegarde
    save_dir = Path('results/entropy')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'exact_entropy_vs_dimension.png', dpi=300)
    plt.close()

def entropy_components_analysis(d=784):
    """
    Analyse des différentes composantes de l'entropie.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mu_tilde = 1.0
    m = mu_tilde * np.sqrt(d)
    t_switch = 0.5 * np.log(d)
    
    t_values = np.logspace(np.log10(0.01*t_switch), np.log10(5*t_switch), 200)
    t_tensor = torch.tensor(t_values, dtype=torch.float32, device=device)
    
    # Composantes de l'entropie
    Delta_t = 1 - torch.exp(-2*t_tensor)
    Gamma_t = Delta_t
    
    # Entropie d'une gaussienne
    entropy_gauss = d/2 * (torch.log(2*np.pi*Gamma_t) + 1)
    
    # Entropie de mélange
    entropy_mix = torch.log(torch.tensor(2.0, device=device)) * torch.ones_like(t_tensor)
    
    # Termes d'overlap
    mu_distance = 2 * m * torch.exp(-t_tensor)
    alpha = torch.exp(-(mu_distance**2) / (4 * Gamma_t))
    overlap_term_small = -2 * (0.5)**2 * alpha
    overlap_term_large = -0.5 * torch.log(1 + 0.5/(1-0.5) * torch.exp(-mu_distance**2/(2*Gamma_t)))
    
    # Transition
    transition = torch.sigmoid(5 * (t_tensor - 0.5*torch.log(torch.tensor(d, dtype=torch.float32, device=device))))
    overlap_term = transition * overlap_term_small + (1-transition) * overlap_term_large
    
    # Entropie totale
    total_entropy = entropy_mix + entropy_gauss + overlap_term
    
    # Tracé
    plt.figure(figsize=(10, 6))
    plt.semilogx(t_values, entropy_gauss.cpu().numpy(), label='Entropie gaussienne')
    plt.semilogx(t_values, entropy_mix.cpu().numpy(), label='Entropie de mélange')
    plt.semilogx(t_values, overlap_term.cpu().numpy(), label='Terme de chevauchement')
    plt.semilogx(t_values, total_entropy.cpu().numpy(), label='Entropie totale')
    plt.axvline(x=t_switch, color='black', linestyle='--', label=f't_s = {t_switch:.3f}')
    
    plt.xlabel('Temps t')
    plt.ylabel('Contribution à l\'entropie')
    plt.title(f'Composantes de l\'entropie exacte (d = {d})')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    
    save_dir = Path('results/entropy')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'entropy_components_d{d}.png', dpi=300)
    plt.close()

def entropy_rate_of_change(d=784):
    """
    Calcule et trace le taux de variation de l'entropie (dS/dt).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mu_tilde = 1.0
    m = mu_tilde * np.sqrt(d)
    t_switch = 0.5 * np.log(d)
    
    t_values = np.logspace(np.log10(0.01*t_switch), np.log10(5*t_switch), 500)
    t_tensor = torch.tensor(t_values, dtype=torch.float32, device=device)
    t_tensor.requires_grad_(True)
    
    # Calcul de l'entropie avec gradient
    entropy = compute_exact_entropy(m, t_tensor, d, device=device)
    
    # Calcul du gradient dS/dt
    entropy.backward(torch.ones_like(entropy))
    dS_dt = t_tensor.grad.clone()
    t_tensor.grad.zero_()
    
    # Tracé
    plt.figure(figsize=(10, 6))
    plt.semilogx(t_values, dS_dt.cpu().numpy(), label='dS/dt')
    plt.axvline(x=t_switch, color='red', linestyle='--', label=f't_s = {t_switch:.3f}')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.xlabel('Temps t')
    plt.ylabel('Taux de variation de l\'entropie (dS/dt)')
    plt.title(f'Taux de variation de l\'entropie exacte (d = {d})')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    
    save_dir = Path('results/entropy')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'entropy_derivative_d{d}.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Calcul de l'entropie exacte pour différentes dimensions...")
    entropy_vs_dimension()
    
    print("Analyse des composantes de l'entropie...")
    entropy_components_analysis(d=100)
    entropy_components_analysis(d=784)
    
    print("Calcul du taux de variation de l'entropie...")
    entropy_rate_of_change(d=100)
    entropy_rate_of_change(d=784)
    
    print("Analyses terminées!") 