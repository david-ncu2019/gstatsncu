import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import os

def plot_anisotropy_ellipse(fitted_params, true_params=None, scenario_name="", out_file='anisotropy_ellipse.png', time_val=None):
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

def plot_2d_results(merged_df: pd.DataFrame, sampled_df: pd.DataFrame, domain_cfg: dict, out_file: str = 'spatial_results.png', time_val=None):
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



def plot_3d_comparison_dashboard(merged_df: pd.DataFrame, sampled_df: pd.DataFrame, out_file: str = 'comparison_3d.png', time_val=None):
    """
    Plots a 3-panel 3D comparison: Ground Truth, Prediction, and Absolute Error.
    """
    fig = plt.figure(figsize=(24, 8))
    
    # Get unique coordinates
    x_vals = np.sort(merged_df['x'].unique())
    y_vals = np.sort(merged_df['y'].unique())
    z_vals = np.sort(merged_df['z'].unique())
    nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)
    i_mid, j_mid, k_mid = nx // 2, ny // 2, nz // 2
    
    # Pre-calculate matrices
    V_true = np.zeros((ny, nx, nz))
    V_pred = np.zeros((ny, nx, nz))
    for i, z in enumerate(z_vals):
        slice_df = merged_df[merged_df['z'] == z]
        V_true[:, :, i] = slice_df.pivot(index='y', columns='x', values='value').reindex(index=y_vals, columns=x_vals).values
        V_pred[:, :, i] = slice_df.pivot(index='y', columns='x', values='pred').reindex(index=y_vals, columns=x_vals).values
    
    V_diff = np.abs(V_true - V_pred)
    
    # Unified color range for True and Pred
    vmin = min(merged_df['value'].min(), merged_df['pred'].min())
    vmax = max(merged_df['value'].max(), merged_df['pred'].max())
    
    def plot_block(ax, V_data, cmap, v_min, v_max, title):
        def plot_face(X_grid, Y_grid, Z_grid, C):
            norm = plt.Normalize(vmin=v_min, vmax=v_max)
            colors = plt.cm.get_cmap(cmap)(norm(C))
            ax.plot_surface(X_grid, Y_grid, Z_grid, facecolors=colors, shade=False, edgecolor='black', linewidth=0.05)

        # 1. Top
        X, Y = np.meshgrid(x_vals, y_vals); Z = np.full_like(X, z_vals[-1]); C = V_data[:, :, -1].copy()
        mask = np.zeros((ny, nx), dtype=bool); mask[:j_mid, i_mid:] = True; Z[mask] = np.nan
        plot_face(X, Y, Z, C)
        # 2. Bottom
        X, Y = np.meshgrid(x_vals, y_vals); Z = np.full_like(X, z_vals[0]); C = V_data[:, :, 0]
        plot_face(X, Y, Z, C)
        # 3. Back (Y=max)
        X, Z_g = np.meshgrid(x_vals, z_vals); Y_p = np.full_like(X, y_vals[-1]); C = V_data[-1, :, :].T
        plot_face(X, Y_p, Z_g, C)
        # 4. Front (Y=min)
        X, Z_g = np.meshgrid(x_vals, z_vals); Y_p = np.full_like(X, y_vals[0]); C = V_data[0, :, :].T
        mask = np.zeros((nz, nx), dtype=bool); mask[k_mid:, i_mid:] = True; Y_p[mask] = np.nan
        plot_face(X, Y_p, Z_g, C)
        # 5. Right (X=max)
        Y_p, Z_g = np.meshgrid(y_vals, z_vals); X_p = np.full_like(Y_p, x_vals[-1]); C = V_data[:, -1, :].T
        mask = np.zeros((nz, ny), dtype=bool); mask[k_mid:, :j_mid] = True; X_p[mask] = np.nan
        plot_face(X_p, Y_p, Z_g, C)
        # 6. Left (X=min)
        Y_p, Z_g = np.meshgrid(y_vals, z_vals); X_p = np.full_like(Y_p, x_vals[0]); C = V_data[:, 0, :].T
        plot_face(X_p, Y_p, Z_g, C)
        # Inner faces
        Y_p, Z_g = np.meshgrid(y_vals[:j_mid], z_vals[k_mid:]); X_p = np.full_like(Y_p, x_vals[i_mid]); C = V_data[:j_mid, i_mid, k_mid:].T
        plot_face(X_p, Y_p, Z_g, C)
        X_p, Z_g = np.meshgrid(x_vals[i_mid:], z_vals[k_mid:]); Y_p = np.full_like(X_p, y_vals[j_mid]); C = V_data[j_mid, i_mid:, k_mid:].T
        plot_face(X_p, Y_p, Z_g, C)
        X_p, Y_p = np.meshgrid(x_vals[i_mid:], y_vals[:j_mid]); Z_g = np.full_like(X_p, z_vals[k_mid]); C = V_data[:j_mid, i_mid:, k_mid]
        plot_face(X_p, Y_p, Z_g, C)

        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.grid(False)
        ax.view_init(elev=30, azim=-45)

    # Panel 1: True
    ax1 = fig.add_subplot(131, projection='3d')
    plot_block(ax1, V_true, 'viridis', vmin, vmax, "Ground Truth Volume")
    
    # Panel 2: Predicted
    ax2 = fig.add_subplot(132, projection='3d')
    plot_block(ax2, V_pred, 'viridis', vmin, vmax, "Kriging Prediction")
    
    # Shared Colorbar for 1 & 2
    m1 = plt.cm.ScalarMappable(cmap='viridis')
    m1.set_array(merged_df['value'])
    m1.set_clim(vmin, vmax)
    cb1 = fig.colorbar(m1, ax=[ax1, ax2], shrink=0.5, pad=0.05, location='bottom')
    cb1.set_label('Value Units')

    # Panel 3: Difference
    ax3 = fig.add_subplot(133, projection='3d')
    diff_max = float(V_diff.max())
    plot_block(ax3, V_diff, 'Reds', 0, diff_max, "Absolute Error (|True-Pred|)")
    
    m3 = plt.cm.ScalarMappable(cmap='Reds')
    m3.set_array(V_diff)
    m3.set_clim(0, diff_max)
    cb3 = fig.colorbar(m3, ax=ax3, shrink=0.5, pad=0.05, location='bottom')
    cb3.set_label('Residual Error')

    t_str = f" [Time: {time_val}]" if time_val is not None else ""
    plt.suptitle(f"3D Volumetric Kriging Comparison{t_str}", fontsize=20, fontweight='bold', y=0.95)
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"3D Comparison Dashboard saved to {out_file}")
