import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import os

def plot_anisotropy_ellipse(fitted_params, true_params=None, scenario_name="", out_file='anisotropy_ellipse.png'):
    """Draw correlation ellipse comparing recovered vs true anisotropy."""
    angle = fitted_params.get("angle", 0.0)
    ls_major = float(fitted_params.get("range", 1.0))
    ratio = float(fitted_params.get("anisotropy_ratio", 1.0))
    ls_minor = ls_major / ratio

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    # Recovered ellipse
    ell = Ellipse((0, 0), width=2 * ls_major, height=2 * ls_minor, angle=angle,
                  ec="red", fc="salmon", alpha=0.3, lw=2,
                  label=f"Recovered (angle={angle:.1f}, ratio={ratio:.2f})")
    ax.add_patch(ell)

    if true_params is not None:
        ta = float(true_params.get("angle", 0.0))
        tr = float(true_params.get("range", 1.0))
        t_ratio = float(true_params.get("anisotropy_ratio", 1.0))
        tmin = tr / t_ratio
        ell2 = Ellipse((0, 0), width=2 * tr, height=2 * tmin, angle=ta,
                       ec="blue", fc="none", lw=2, ls="--",
                       label=f"True (angle={ta:.1f}, ratio={t_ratio:.2f})")
        ax.add_patch(ell2)

    lim = max(ls_major, float(true_params.get("range", 1.0)) if true_params else 1.0) * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, ls="--", alpha=0.3)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_title(f"Anisotropy Ellipse  -  {scenario_name}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Anisotropy ellipse saved to {out_file}")

def plot_2d_results(merged_df: pd.DataFrame, sampled_df: pd.DataFrame, domain_cfg: dict, out_file: str = 'spatial_results.png'):
    nx = domain_cfg.get('nx', 50)
    ny = domain_cfg.get('ny', 50)
    nz = domain_cfg.get('nz', 1)
    
    if nz > 1:
        # If 3D, we just take the middle slice along Z for visualization
        z_vals = sorted(merged_df['z'].unique())
        mid_z = z_vals[len(z_vals) // 2]
        plot_df = merged_df[merged_df['z'] == mid_z].copy()
        sample_plot_df = sampled_df[sampled_df['z'] == mid_z]
        title_suffix = f" (Middle Z-Slice: z={mid_z})"
    else:
        plot_df = merged_df.copy()
        sample_plot_df = sampled_df
        title_suffix = ""

    # Pivot data into 2D arrays for imshow
    # Note: we use pivot to ensure correct spatial arrangement
    try:
        true_grid = plot_df.pivot(index='y', columns='x', values='value').values
        pred_grid = plot_df.pivot(index='y', columns='x', values='pred').values
        var_grid = plot_df.pivot(index='y', columns='x', values='pred_var').values
        error_grid = np.abs(true_grid - pred_grid)
        
        # Get coordinates for plotting extents
        x_min, x_max = plot_df['x'].min(), plot_df['x'].max()
        y_min, y_max = plot_df['y'].min(), plot_df['y'].max()
        extent = [x_min, x_max, y_min, y_max]
    except Exception as e:
        print(f"Could not pivot data for plotting: {e}")
        return

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    # 1. True Domain
    im0 = axes[0].imshow(true_grid, origin='lower', extent=extent, cmap='viridis')
    axes[0].scatter(sample_plot_df['x'], sample_plot_df['y'], c='white', edgecolors='black', s=20, label='Samples')
    axes[0].set_title(f"True Synthetic Domain{title_suffix}")
    axes[0].legend(loc='upper right')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Predicted Domain
    im1 = axes[1].imshow(pred_grid, origin='lower', extent=extent, cmap='viridis')
    axes[1].set_title(f"Kriging Prediction{title_suffix}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Kriging Variance
    im2 = axes[2].imshow(var_grid, origin='lower', extent=extent, cmap='magma')
    axes[2].set_title(f"Kriging Variance{title_suffix}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Absolute Error
    im3 = axes[3].imshow(error_grid, origin='lower', extent=extent, cmap='Reds')
    axes[3].set_title(f"Absolute Error (|True - Pred|){title_suffix}")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualizations saved to {out_file}")

def plot_variogram_analysis(emp_var_df, fit_params, true_line, fitted_line, dir_var_df, out_file='variogram_results.png', time_val=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Isotropic Variogram
    ax1 = axes[0]
    ax1.scatter(emp_var_df['dist'], emp_var_df['gamma'], s=50, c='crimson', edgecolor='k', zorder=5, label='Empirical')
    
    if true_line is not None:
        ax1.plot(true_line['dist'], true_line['gamma'], 'b-', lw=2, label='True Model')
        
    if fitted_line is not None:
        ax1.plot(fitted_line['dist'], fitted_line['gamma'], 'g--', lw=2, label='Fitted Model')
        
    ax1.set_xlabel('Lag Distance', fontsize=12)
    ax1.set_ylabel('Semivariance', fontsize=12)
    ax1.set_title('Omnidirectional Variogram', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, ls='--', alpha=0.4)
    
    # 2. Directional Variograms
    ax2 = axes[1]
    
    # Plot omnidirectional (empirical) variogram as a baseline reference
    if emp_var_df is not None:
        ax2.scatter(emp_var_df['dist'], emp_var_df['gamma'], s=80, c='black', alpha=0.75, marker='^', zorder=5, label='Omnidirectional')

    if dir_var_df is not None and 'dir.hor' in dir_var_df.columns:
        directions = dir_var_df['dir.hor'].unique()
        cmap = plt.cm.tab20
        for i, d in enumerate(directions):
            sub_df = dir_var_df[dir_var_df['dir.hor'] == d]
            ax2.plot(sub_df['dist'], sub_df['gamma'], 'o-', color=cmap(i % 20), ms=5, zorder=3, label=f"{d}°")
            
        ax2.set_xlabel('Lag Distance', fontsize=12)
        ax2.set_ylabel('Semivariance', fontsize=12)
        ax2.set_title('Directional Variograms', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Variogram visualizations saved to {out_file}")

