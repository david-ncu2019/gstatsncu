import numpy as np
import pandas as pd
from scipy.spatial import KDTree

def apply_trend(df: pd.DataFrame, trend_config: dict) -> pd.DataFrame:
    """
    Applies a mathematical trend to the simulated data.
    trend_config expects 'type': 'linear', 'coefficients': {'x': 0.1, 'y': 0.2, 'z': 0.05}
    """
    df_out = df.copy()
    if not trend_config or trend_config.get('type') != 'linear':
        df_out['value'] = df_out.get('sim1', df_out.iloc[:, -1])
        return df_out
        
    coeffs = trend_config.get('coefficients', {})
    trend = np.zeros(len(df_out))
    
    if 'x' in coeffs and 'x' in df_out.columns:
        trend += coeffs['x'] * df_out['x']
    if 'y' in coeffs and 'y' in df_out.columns:
        trend += coeffs['y'] * df_out['y']
    if 'z' in coeffs and 'z' in df_out.columns:
        trend += coeffs['z'] * df_out['z']
        
    df_out['trend'] = trend
    base_val = df_out.get('sim1', df_out.iloc[:, -1])
    df_out['value'] = base_val + trend
    return df_out

def apply_noise(df: pd.DataFrame, std_dev: float, seed: int = None) -> pd.DataFrame:
    """
    Adds Gaussian noise to the 'value' column.
    """
    if seed is not None:
        np.random.seed(seed)
        
    df_out = df.copy()
    noise = np.random.normal(0, std_dev, size=len(df_out))
    if 'value' not in df_out.columns:
        df_out['value'] = df_out.get('sim1', df_out.iloc[:, -1])
    df_out['noise'] = noise
    df_out['value'] += noise
    return df_out

def sample_domain(df: pd.DataFrame, sample_config: dict, seed: int = None) -> pd.DataFrame:
    """
    Samples the domain based on strategy.
    strategy: 'random_uniform' or 'clustered'
    fraction: 0.0 to 1.0
    """
    strategy = sample_config.get('strategy', 'random_uniform')
    fraction = sample_config.get('fraction', 0.1)
    n_samples = int(len(df) * fraction)
    
    if seed is not None:
        np.random.seed(seed)
        
    if strategy == 'clustered':
        # Create a few clusters
        n_clusters = sample_config.get('n_clusters', 2)
        samples_per_cluster = n_samples // n_clusters
        
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        cluster_centers_x = np.random.uniform(x_min + 0.1*x_range, x_max - 0.1*x_range, n_clusters)
        cluster_centers_y = np.random.uniform(y_min + 0.1*y_range, y_max - 0.1*y_range, n_clusters)
        
        sampled_indices = set()
        
        for i in range(n_clusters):
            # Normal distribution around center
            std_dev_x = x_range * 0.1
            std_dev_y = y_range * 0.1
            
            pts_x = np.random.normal(cluster_centers_x[i], std_dev_x, samples_per_cluster * 2) # Overgenerate
            pts_y = np.random.normal(cluster_centers_y[i], std_dev_y, samples_per_cluster * 2)
            
            # Find closest points in df
            tree = KDTree(df[['x', 'y']].values)
            distances, indices = tree.query(np.column_stack((pts_x, pts_y)))
            
            # Add unique indices until we have enough
            for idx in indices:
                if len(sampled_indices) < (i + 1) * samples_per_cluster:
                    sampled_indices.add(int(idx))
                    
        # If we didn't get enough points (due to exact duplicates or boundary issues), top up randomly
        remaining = n_samples - len(sampled_indices)
        if remaining > 0:
            available_indices = list(set(range(len(df))) - sampled_indices)
            extra_indices = np.random.choice(available_indices, remaining, replace=False)
            sampled_indices.update(int(x) for x in extra_indices)
            
        sampled_df = df.iloc[list(sampled_indices)].copy()
    else:
        # Default fallback to random_uniform
        sampled_df = df.sample(n=n_samples, random_state=seed)
        
    return sampled_df

def get_spatial_heuristics(df: pd.DataFrame) -> list:
    """
    Calculates Small, Medium, and Large spatial scales for variogram seeding.
    Small: Average distance to nearest neighbor.
    Medium: 25% of the diagonal of the bounding box.
    Large: 50% of the diagonal of the bounding box.
    """
    if len(df) < 2:
        return [1.0, 10.0, 50.0]
    
    # Identify coordinate columns
    coord_cols = [c for c in ['x', 'y', 'z'] if c in df.columns]
    if not coord_cols:
        # Fallback if no standard coordinate names
        return [1.0, 10.0, 50.0]
        
    coords = df[coord_cols].values
    
    # Small: Average distance to nearest neighbor
    tree = KDTree(coords)
    # k=2 because the closest point is the point itself (dist=0)
    distances, _ = tree.query(coords, k=2)
    avg_nn_dist = np.mean(distances[:, 1])
    
    # Bounding box diagonal
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    diagonal = np.sqrt(np.sum((max_coords - min_coords)**2))
    
    medium = 0.25 * diagonal
    large = 0.5 * diagonal
    
    # Safety check for coincident points
    small = max(avg_nn_dist, 0.0001 * diagonal) if diagonal > 0 else avg_nn_dist
    if small <= 0:
        small = 1.0 # Last resort fallback
        
    return [float(small), float(medium), float(large)]
