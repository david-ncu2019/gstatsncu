import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pandas as pd

# Define the R functions as strings
R_FUNCTIONS = """
library(gstat)
library(sp)

build_vgm <- function(vgm_list, nugget) {
    if (length(vgm_list) == 0) {
        return(vgm(nugget=nugget, model="Nug"))
    }
    
    # Base variogram
    s1 <- vgm_list[[1]]
    v <- vgm(psill=s1$sill, model=s1$model, range=s1$range, nugget=nugget, anis=c(s1$angle, 1/s1$anisotropy_ratio))
    
    # Add subsequent structures
    if (length(vgm_list) > 1) {
        for (i in 2:length(vgm_list)) {
            si <- vgm_list[[i]]
            v <- vgm(psill=si$sill, model=si$model, range=si$range, nugget=0, anis=c(si$angle, 1/si$anisotropy_ratio), add.to=v)
        }
    }
    return(v)
}

# Function to generate SGS
generate_sgs <- function(grid_df, vgm_list, vgm_nugget, nsim=1, seed=NULL) {
    if (!is.null(seed)) {
        set.seed(seed)
    }
    
    has_z <- "z" %in% colnames(grid_df)
    
    if (has_z) {
        coordinates(grid_df) <- ~x+y+z
        gridded(grid_df) <- TRUE
        loc_formula <- ~x+y+z
    } else {
        coordinates(grid_df) <- ~x+y
        gridded(grid_df) <- TRUE
        loc_formula <- ~x+y
    }
    
    v <- build_vgm(vgm_list, vgm_nugget)
    g <- gstat(formula=sim~1, locations=loc_formula, dummy=TRUE, beta=0, model=v, nmax=40)
    res <- predict(g, newdata=grid_df, nsim=nsim)
    return(as.data.frame(res))
}

# Function to run Kriging
run_kriging <- function(sampled_df, grid_df, formula_str, vgm_list, vgm_nugget) {
    has_z <- "z" %in% colnames(grid_df)
    
    if (has_z) {
        coordinates(grid_df) <- ~x+y+z
        coordinates(sampled_df) <- ~x+y+z
        gridded(grid_df) <- TRUE
    } else {
        coordinates(grid_df) <- ~x+y
        coordinates(sampled_df) <- ~x+y
        gridded(grid_df) <- TRUE
    }
    
    v <- build_vgm(vgm_list, vgm_nugget)
    f <- as.formula(formula_str)
    
    res <- krige(f, sampled_df, grid_df, model=v)
    return(as.data.frame(res))
}

# Function to run cross validation
run_cv <- function(sampled_df, formula_str, vgm_list, vgm_nugget, nfold=5) {
    has_z <- "z" %in% colnames(sampled_df)
    
    if (has_z) {
        coordinates(sampled_df) <- ~x+y+z
    } else {
        coordinates(sampled_df) <- ~x+y
    }
    
    v <- build_vgm(vgm_list, vgm_nugget)
    f <- as.formula(formula_str)
    
    cv_res <- krige.cv(f, sampled_df, model=v, nfold=nfold)
    return(as.data.frame(cv_res))
}

# Compute empirical variogram
compute_variogram <- function(sampled_df, formula_str, alpha=NULL) {
    has_z <- "z" %in% colnames(sampled_df)
    if (has_z) {
        coordinates(sampled_df) <- ~x+y+z
    } else {
        coordinates(sampled_df) <- ~x+y
    }
    f <- as.formula(formula_str)
    
    if (is.null(alpha)) {
        v <- variogram(f, sampled_df)
    } else {
        v <- variogram(f, sampled_df, alpha=as.numeric(alpha))
    }
    return(as.data.frame(v))
}

# Get theoretical variogram line
get_variogram_line <- function(vgm_list, vgm_nugget, max_dist) {
    v <- build_vgm(vgm_list, vgm_nugget)
    line <- variogramLine(v, maxdist=max_dist)
    return(as.data.frame(line))
}

# Fit variogram model with multiple range seeds
fit_variogram <- function(sampled_df, formula_str, model_type, seeds, nugget_init) {
    f <- as.formula(formula_str)
    dep_var <- all.vars(f)[1]
    total_var <- var(sampled_df[[dep_var]], na.rm=TRUE)
    
    has_z <- "z" %in% colnames(sampled_df)
    if (has_z) {
        coordinates(sampled_df) <- ~x+y+z
    } else {
        coordinates(sampled_df) <- ~x+y
    }
    
    # Compute empirical variogram
    v_emp <- variogram(f, sampled_df)
    
    best_fit <- NULL
    min_sse <- Inf
    
    # Safe nugget (max 50% of total variance)
    safe_nugget <- min(nugget_init, total_var * 0.5)
    
    for (s in seeds) {
        # Try fitting with different range seeds
        # psill is partial sill (total sill - nugget)
        v_init <- vgm(psill=total_var * 0.5, model=model_type, range=s, nugget=safe_nugget)
        v_fit <- try(fit.variogram(v_emp, v_init), silent=TRUE)
        
        if (!inherits(v_fit, "try-error")) {
            sse <- attr(v_fit, "SSErr")
            if (!is.null(sse) && sse < min_sse) {
                min_sse <- sse
                best_fit <- v_fit
            }
        }
    }
    
    if (is.null(best_fit)) {
        # Fallback to a default if all failed
        return(as.data.frame(vgm(psill=total_var * 0.5, model=model_type, range=mean(seeds), nugget=safe_nugget)))
    }
    
    return(as.data.frame(best_fit))
}
"""

# Load the R functions into the global environment
ro.r(R_FUNCTIONS)

# Python wrapper functions

def prepare_vgm_list(vgm_params: dict):
    """Converts Python dict to R list of structures"""
    nugget = float(vgm_params.get('nugget', 0.0))
    structures = vgm_params.get('structures', [])
    if not structures:
        # Fallback to single structure
        structures = [{
            'model': vgm_params.get('model', 'Sph'),
            'range': float(vgm_params.get('range', 1.0)),
            'sill': float(vgm_params.get('sill', 1.0)),
            'angle': float(vgm_params.get('angle', 0.0)),
            'anisotropy_ratio': float(vgm_params.get('anisotropy_ratio', 1.0))
        }]
    
    r_list = ro.ListVector({
        str(i): ro.ListVector({
            'model': s['model'],
            'range': float(s['range']),
            'sill': float(s['sill']),
            'angle': float(s.get('angle', 0.0)),
            'anisotropy_ratio': float(s.get('anisotropy_ratio', 1.0))
        }) for i, s in enumerate(structures)
    })
    
    return r_list, nugget

def generate_sgs(grid_df: pd.DataFrame, vgm_params: dict, nsim: int = 1, seed: int = None) -> pd.DataFrame:
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_generate_sgs = ro.globalenv['generate_sgs']
        r_seed = ro.vectors.IntVector([seed]) if seed is not None else ro.NULL
        r_vgm_list, nugget = prepare_vgm_list(vgm_params)
        res = r_generate_sgs(grid_df, r_vgm_list, nugget, nsim, r_seed)
        return ro.conversion.rpy2py(res)

def run_kriging(sampled_df: pd.DataFrame, grid_df: pd.DataFrame, formula: str, vgm_params: dict) -> pd.DataFrame:
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_run_kriging = ro.globalenv['run_kriging']
        r_vgm_list, nugget = prepare_vgm_list(vgm_params)
        res = r_run_kriging(sampled_df, grid_df, formula, r_vgm_list, nugget)
        return ro.conversion.rpy2py(res)

def run_cv(sampled_df: pd.DataFrame, formula: str, vgm_params: dict, nfold: int = 5) -> pd.DataFrame:
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_run_cv = ro.globalenv['run_cv']
        r_vgm_list, nugget = prepare_vgm_list(vgm_params)
        res = r_run_cv(sampled_df, formula, r_vgm_list, nugget, nfold)
        return ro.conversion.rpy2py(res)

def compute_variogram(sampled_df: pd.DataFrame, formula: str, alpha: list = None) -> pd.DataFrame:
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_compute = ro.globalenv['compute_variogram']
        if alpha is None:
            r_alpha = ro.NULL
        else:
            r_alpha = ro.vectors.FloatVector(alpha)
        res = r_compute(sampled_df, formula, r_alpha)
        return ro.conversion.rpy2py(res)

def get_variogram_line(vgm_params: dict, max_dist: float) -> pd.DataFrame:
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_line = ro.globalenv['get_variogram_line']
        r_vgm_list, nugget = prepare_vgm_list(vgm_params)
        res = r_line(r_vgm_list, nugget, float(max_dist))
        return ro.conversion.rpy2py(res)

def fit_variogram(sampled_df: pd.DataFrame, formula: str, model_type: str, seeds: list, nugget_init: float) -> pd.DataFrame:
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_fit = ro.globalenv['fit_variogram']
        r_seeds = ro.vectors.FloatVector(seeds)
        res = r_fit(sampled_df, formula, model_type, r_seeds, float(nugget_init))
        return ro.conversion.rpy2py(res)
