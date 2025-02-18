import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif, juste pour sauvegarder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def visualize_manifold_embedding(n_samples=1000, dim_intrinsic=2, dim_ambient=3):
    """
    Visualise un manifold de dimension intrinsèque dim_intrinsic plongé dans un espace de dimension dim_ambient.
    
    Args:
        n_samples (int): Nombre de points à générer
        dim_intrinsic (int): Dimension intrinsèque du manifold (D)
        dim_ambient (int): Dimension de l'espace ambiant (N)
    """
    # Génère la matrice de projection
    F = np.random.randn(dim_ambient, dim_intrinsic) / np.sqrt(dim_intrinsic)
    
    # Génère les points latents sur le manifold intrinsèque
    z = np.random.randn(n_samples, dim_intrinsic)
    
    # Projette dans l'espace ambiant avec non-linéarité
    X = np.tanh(z @ F.T)  # X est maintenant de forme (n_samples, dim_ambient)
    
    # Plot 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], 
                        c=z[:, 0],  # Colorie selon la première coordonnée latente
                        cmap='viridis',
                        alpha=0.6)
    
    plt.colorbar(scatter, label='Première coordonnée latente')
    ax.set_title(f"Manifold {dim_intrinsic}D plongé dans R^{dim_ambient}")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("x₃")
    
    # Ajoute une grille et ajuste la vue
    ax.grid(True)
    ax.view_init(elev=20, azim=45)  # Ajuste l'angle de vue
    
    # Sauvegarde
    save_dir = Path(__file__).parent.parent.parent / "results" / "manifold_viz"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"manifold_D{dim_intrinsic}_N{dim_ambient}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot sauvegardé dans: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Visualise un manifold 2D plongé dans R³
    visualize_manifold_embedding(n_samples=2000, dim_intrinsic=2, dim_ambient=3)
