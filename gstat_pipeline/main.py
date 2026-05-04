import yaml
import json
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from r_engine import generate_sgs, run_kriging, run_cv
from data_ops import apply_trend, apply_noise, sample_domain, get_spatial_heuristics

def create_grid(domain_cfg: dict) -> pd.DataFrame:
    nx = domain_cfg.get('nx', 50)
    ny = domain_cfg.get('ny', 50)
    nz = domain_cfg.get('nz', 1)
    cell_size = domain_cfg.get('cell_size', 1.0)
    
    x = np.arange(nx) * cell_size
    y = np.arange(ny) * cell_size
    
    if nz > 1:
        z = np.arange(nz) * cell_size
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return pd.DataFrame({'x': X.flatten(), 'y': Y.flatten(), 'z': Z.flatten()})
    else:
        X, Y = np.meshgrid(x, y, indexing='ij')
        return pd.DataFrame({'x': X.flatten(), 'y': Y.flatten()})

def get_metrics(df_subset):
    if len(df_subset) == 0:
        return {}
    y_true = df_subset['value'].values
    y_pred = df_subset['pred'].values
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (len(y_true)*mse / ss_tot) if ss_tot > 0 else 0.0
    return {
        'RMSE': float(np.sqrt(mse)),
        'MAE': float(mae),
        'R2': float(r2)
    }

def theoretical_gamma(h, model, nugget, sill, range_val):
    gamma = np.full_like(h, nugget + sill, dtype=float)
    if model == 'Sph':
        mask = h < range_val
        gamma[mask] = nugget + sill * (1.5 * (h[mask]/range_val) - 0.5 * (h[mask]/range_val)**3)
    elif model == 'Exp':
        gamma = nugget + sill * (1.0 - np.exp(-3.0 * h / range_val))
    elif model == 'Gau':
        gamma = nugget + sill * (1.0 - np.exp(-3.0 * (h / range_val)**2))
    return gamma

def fit_anisotropy(dir_var, base_params):
    # Hierarchical search for anisotropy
    ratios = np.arange(1.0, 5.5, 0.5)
    
    # 1. Coarse scan
    coarse_angles = np.arange(0, 180, 22.5)
    best_error = np.inf
    best_coarse_angle = 0.0
    best_ratio = 1.0
    
    dists = dir_var['dist'].values
    gammas = dir_var['gamma'].values
    dirs = dir_var['dir.hor'].values
    
    for ang in coarse_angles:
        for ratio in ratios:
            alpha_rel = np.radians(dirs - ang)
            h_major = dists * np.cos(alpha_rel)
            h_minor = dists * np.sin(alpha_rel)
            h_eq = np.sqrt(h_major**2 + (h_minor * ratio)**2)
            gamma_hat = theoretical_gamma(h_eq, base_params['model'], base_params['nugget'], base_params['sill'], base_params['range'])
            error = np.mean((gammas - gamma_hat)**2)
            if error < best_error:
                best_error = error
                best_coarse_angle = ang
                best_ratio = ratio
                
    # 2. Fine scan around best coarse angle
    fine_angles = np.arange(best_coarse_angle - 15, best_coarse_angle + 16, 5)
    # Wrap angles to [0, 180)
    fine_angles = np.mod(fine_angles, 180)
    
    best_params = {'angle': float(best_coarse_angle), 'anisotropy_ratio': float(best_ratio)}
    
    for ang in fine_angles:
        for ratio in ratios:
            alpha_rel = np.radians(dirs - ang)
            h_major = dists * np.cos(alpha_rel)
            h_minor = dists * np.sin(alpha_rel)
            h_eq = np.sqrt(h_major**2 + (h_minor * ratio)**2)
            gamma_hat = theoretical_gamma(h_eq, base_params['model'], base_params['nugget'], base_params['sill'], base_params['range'])
            error = np.mean((gammas - gamma_hat)**2)
            if error < best_error:
                best_error = error
                best_params['angle'] = float(ang)
                best_params['anisotropy_ratio'] = float(ratio)
                
    return best_params

def get_primary_structure(vgm_params):
    structures = vgm_params.get('structures', [])
    if structures:
        best_s = max(structures, key=lambda s: float(s.get('sill', 0)))
        return {
            'model': best_s.get('model', 'Sph'),
            'range': float(best_s.get('range', 1.0)),
            'sill': float(best_s.get('sill', 1.0)),
            'angle': float(best_s.get('angle', 0.0)),
            'anisotropy_ratio': float(best_s.get('anisotropy_ratio', 1.0)),
            'nugget': float(vgm_params.get('nugget', 0.0))
        }
    return {
        'model': vgm_params.get('model', 'Sph'),
        'range': float(vgm_params.get('range', 1.0)),
        'sill': float(vgm_params.get('sill', 1.0)),
        'angle': float(vgm_params.get('angle', 0.0)),
        'anisotropy_ratio': float(vgm_params.get('anisotropy_ratio', 1.0)),
        'nugget': float(vgm_params.get('nugget', 0.0))
    }

def run_pipeline(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    scenario_name = Path(config_path).stem
    out_dir = Path("output") / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    seed = config.get('random_seed', 42)
    
    # 1. Create Grid
    grid_df = create_grid(config['domain'])
    
    # 2. Run SGS
    print(f"[{scenario_name}] Running SGS...")
    true_domain = generate_sgs(grid_df, config['sgs_variogram'], nsim=1, seed=seed)
    
    # 3. Add trend & noise
    true_domain = apply_trend(true_domain, config.get('trend', {}))
    true_domain = apply_noise(true_domain, config.get('noise', {}).get('std_dev', 0.0), seed=seed)
    
    # 4. Sample
    print(f"[{scenario_name}] Sampling domain...")
    sampled_df = sample_domain(true_domain, config.get('sampling', {}), seed=seed)
    
    # 4.2 Detrending Analysis
    pre_cfg = config.get("preprocessing", {}).get("detrend", {})
    do_detrend = pre_cfg.get("enabled", False)
    auto_detect = pre_cfg.get("auto_detect", True)
    trend_order = pre_cfg.get("order", 1)
    processor = None
    
    if auto_detect or do_detrend:
        from preprocessor import analyze_trend, TrendProcessor
        print(f"[{scenario_name}] Trend Analysis...")
        trend_stats = analyze_trend(sampled_df['x'].values, sampled_df['y'].values, sampled_df['value'].values, order=trend_order)
        print(f"[{scenario_name}]   F-test p-value: {trend_stats['f_pvalue']:.4e} | R²: {trend_stats['r2']:.4f}")
        
        if auto_detect:
            do_detrend = trend_stats['recommend_detrend']
            
        if do_detrend:
            print(f"[{scenario_name}]   -> Significant trend detected. Detrending ENABLED (Order {trend_order}).")
            processor = TrendProcessor(order=trend_order)
            processor.fit(sampled_df['x'].values, sampled_df['y'].values, sampled_df['value'].values)
            # Detrend the sample data for variogram and kriging
            sampled_df['value'] = processor.detrend(sampled_df['x'].values, sampled_df['y'].values, sampled_df['value'].values)
            # Update formula to Simple/Ordinary Kriging since trend is handled
            config['kriging']['formula'] = "value ~ 1"
        else:
            print(f"[{scenario_name}]   -> No significant trend. Detrending DISABLED.")
            
    # 4.3 Normal Score Transform (NST)
    nst_cfg = config.get("preprocessing", {}).get("nst", {})
    nst_enabled = nst_cfg.get("enabled", False)
    nst_processor = None
    
    if nst_enabled:
        print(f"[{scenario_name}] Applying Normal Score Transform (GeostatsPy)...")
        from preprocessor import GeostatsPyNST
        nst_processor = GeostatsPyNST()
        sampled_df['value'] = nst_processor.fit_transform(sampled_df, vcol='value')
    
    # 4.5 Variogram Analysis
    print(f"[{scenario_name}] Performing Variogram Analysis...")
    import r_engine
    import viz
    formula = config['kriging']['formula']
    vgm_params = config['sgs_variogram']
    emp_var = r_engine.compute_variogram(sampled_df, formula)
    primary_vgm = get_primary_structure(vgm_params)
    
    # Hierarchical search seeds
    seeds = get_spatial_heuristics(sampled_df)
    
    max_dist = emp_var['dist'].max() * 1.1 if len(emp_var) > 0 else primary_vgm['range'] * 2
    
    try:
        # Use new multi-start fit with spatial heuristics
        fit_params = r_engine.fit_variogram(
            sampled_df, 
            formula, 
            primary_vgm['model'], 
            seeds, 
            primary_vgm['nugget']
        )
        
        nugget_row = fit_params[fit_params['model'] == 'Nug']
        struct_row = fit_params[fit_params['model'] != 'Nug']
        
        fitted_nugget = float(nugget_row['psill'].iloc[0]) if len(nugget_row) > 0 else 0.0
        fitted_sill = float(struct_row['psill'].iloc[0]) if len(struct_row) > 0 else primary_vgm['sill']
        fitted_range = float(struct_row['range'].iloc[0]) if len(struct_row) > 0 else primary_vgm['range']
        fitted_model = str(struct_row['model'].iloc[0]) if len(struct_row) > 0 else primary_vgm['model']
        
        dir_var = r_engine.compute_variogram(sampled_df, formula, alpha=list(range(0, 180, 15)))
        
        base_fitted_params = {'model': fitted_model, 'range': fitted_range, 'sill': fitted_sill, 'nugget': fitted_nugget}
        anis_params = fit_anisotropy(dir_var, base_fitted_params)
        
        fitted_vgm_params = {**base_fitted_params, **anis_params}
        fitted_line = r_engine.get_variogram_line(fitted_vgm_params, max_dist)
    except Exception as e:
        print(f"Failed to fit variogram: {e}")
        fitted_line = None
        fit_params = None
        fitted_vgm_params = {}
        dir_var = r_engine.compute_variogram(sampled_df, formula, alpha=list(range(0, 180, 15)))

    true_line = r_engine.get_variogram_line(vgm_params, max_dist)
    
    print(f"[{scenario_name}] Generating variogram visualizations...")
    viz.plot_variogram_analysis(emp_var, fit_params, true_line, fitted_line, dir_var, out_file=str(out_dir / "variogram_results.png"))
    if fitted_vgm_params:
        viz.plot_anisotropy_ellipse(fitted_vgm_params, primary_vgm, scenario_name=scenario_name, out_file=str(out_dir / "anisotropy_ellipse.png"))
    
    # 5. Run Kriging
    print(f"[{scenario_name}] Running Kriging...")
    kriging_cfg = config['kriging']
    try:
        pred_domain = run_kriging(sampled_df, grid_df, kriging_cfg['formula'], config['sgs_variogram'])
        
        merged_df = true_domain.copy()
        
        # 1. Backtransform NST if applied
        pred_vals = pred_domain['var1.pred'].values
        if nst_processor is not None:
            pred_vals = nst_processor.inverse_transform(pred_vals)
            
        # 2. Retrend predictions if detrending was applied
        if processor is not None:
            merged_df['pred'] = processor.retrend(grid_df['x'].values, grid_df['y'].values, pred_vals)
        else:
            merged_df['pred'] = pred_vals
            
        merged_df['pred_var'] = pred_domain['var1.var']
    except Exception as e:
        print(f"Kriging failed: {e}")
        return None
    
    # 6. Run CV
    print(f"[{scenario_name}] Running Cross Validation...")
    try:
        cv_res = run_cv(sampled_df, kriging_cfg['formula'], config['sgs_variogram'], nfold=kriging_cfg.get('cv_nfold', 5))
        
        obs_cv = cv_res['observed'].values
        pred_cv = cv_res['var1.pred'].values
        
        if nst_processor is not None:
            obs_cv = nst_processor.inverse_transform(obs_cv)
            pred_cv = nst_processor.inverse_transform(pred_cv)
            
        if processor is not None:
            obs_cv = processor.retrend(sampled_df['x'].values, sampled_df['y'].values, obs_cv)
            pred_cv = processor.retrend(sampled_df['x'].values, sampled_df['y'].values, pred_cv)
            
        cv_res_residual = obs_cv - pred_cv
        
        cv_metrics = {
            'mean_error': float(cv_res_residual.mean()),
            'rmse': float(np.sqrt((cv_res_residual**2).mean())),
            'msdr': float((cv_res['residual']**2 / cv_res['var1.var']).mean()) # keep msdr in transformed space to test kriging variance validity
        }
    except Exception as e:
        print(f"CV failed: {e}")
        cv_metrics = {"error": str(e)}
    
    # 7. Compute Metrics
    sampled_idx = sampled_df.index
    unsampled_idx = merged_df.index.difference(sampled_idx)
    
    metrics = {
        'sampled_locations': get_metrics(merged_df.loc[sampled_idx]),
        'unsampled_locations': get_metrics(merged_df.loc[unsampled_idx]),
        'overall': get_metrics(merged_df)
    }
    
    report = {
        'config': config,
        'metrics': metrics,
        'cv_metrics': cv_metrics,
        'summary_stats': {
            'true_mean': float(merged_df['value'].mean()),
            'pred_mean': float(merged_df['pred'].mean()),
            'true_var': float(merged_df['value'].var()),
            'pred_var': float(merged_df['pred'].var())
        }
    }
    
    # 8. Export JSON
    with open(out_dir / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    with open(out_dir / 'reproduction_config.json', 'w') as f:
        json.dump({'config': config, 'seed': seed}, f, indent=2)
        
    # 9. Generate Visualizations
    from viz import plot_2d_results
    print(f"[{scenario_name}] Generating spatial visualizations...")
    plot_2d_results(merged_df, sampled_df, config['domain'], out_file=str(out_dir / "spatial_results.png"))
        
    print(f"[{scenario_name}] Done! Reports and visualizations exported successfully.")
    
    return {
        'Scenario': scenario_name,
        'RMSE': metrics['overall'].get('RMSE', np.nan),
        'R2': metrics['overall'].get('R2', np.nan)
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_pipeline(sys.argv[1])
    else:
        run_pipeline("config.yaml")
