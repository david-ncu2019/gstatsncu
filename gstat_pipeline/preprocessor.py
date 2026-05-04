import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
import statsmodels.api as sm

def analyze_trend(X, Y, Z, order: int = 1):
    """
    Perform statistical tests to detect an underlying spatial trend.
    Uses adaptive LOOCV for small datasets to prevent overfitting.
    """
    coords = np.column_stack((X, Y))
    N = len(Z)

    # OLS for p-value and fit R2
    poly = PolynomialFeatures(degree=order, include_bias=False)
    X_poly = poly.fit_transform(coords)
    X_model = sm.add_constant(X_poly)
    ols_results = sm.OLS(Z, X_model).fit()
    
    f_pvalue = float(ols_results.f_pvalue)
    r2_fit = float(ols_results.rsquared)
    
    # Adaptive Cross-Validation
    if N < 1000:
        cv_pipeline = make_pipeline(PolynomialFeatures(degree=order, include_bias=False), LinearRegression())
        # Use K-fold CV with K up to 50 (effectively LOOCV for N <= 50)
        # Ensure cv_folds is at least 2
        cv_folds = max(2, min(N, 50))
        # Use cross_val_predict to get out-of-sample predictions
        Z_pred_cv = cross_val_predict(cv_pipeline, coords, Z, cv=cv_folds)
        r2_cv = float(r2_score(Z, Z_pred_cv))
        metric_used = 'cv'
        effective_r2 = r2_cv
    else:
        r2_cv = None
        metric_used = 'fit'
        effective_r2 = r2_fit

    # Decision rule: significant p-value and effective R2 > 0.05
    recommend_detrend = bool(f_pvalue < 0.05 and effective_r2 > 0.05)

    return {
        "f_pvalue": f_pvalue,
        "r2_fit": r2_fit,
        "r2_cv": r2_cv,
        "effective_r2": effective_r2,
        "metric_used": metric_used,
        "recommend_detrend": recommend_detrend,
        "tested_order": order
    }

class TrendProcessor:
    """
    Polynomial detrending pre-processor.
    Fits a surface of a given order to spatial data.
    """
    def __init__(self, order=1):
        self.order = order
        self.model = make_pipeline(PolynomialFeatures(order, include_bias=False), LinearRegression())
        
    def fit(self, X, Y, Z):
        coords = np.column_stack((X, Y))
        self.model.fit(coords, Z)
        return self
        
    def get_trend(self, X, Y):
        coords = np.column_stack((X, Y))
        return self.model.predict(coords)
        
    def detrend(self, X, Y, Z):
        trend = self.get_trend(X, Y)
        return Z - trend
        
    def retrend(self, X, Y, Z_res):
        trend = self.get_trend(X, Y)
        return Z_res + trend
        
    def get_params(self):
        lr = self.model.named_steps['linearregression']
        return {
            'order': self.order,
            'intercept': lr.intercept_,
            'coefficients': lr.coef_.tolist()
        }

class GeostatsPyNST:
    """
    Wrapper for GeostatsPy's Normal Score Transform (nscore) and back-transform (backtr).
    """
    def __init__(self):
        self._fitted = False
        self.vr = None
        self.vrg = None
        self.zmin = None
        self.zmax = None

    def fit_transform(self, df, vcol='value'):
        """
        Applies nscore and saves transformation tables.
        Returns the numpy array of normal scores.
        """
        import pandas as pd
        import geostatspy.geostats as geostats
        
        # geostats.nscore requires a DataFrame and modifies it or returns a new array
        # Signature: nscore(df, vcol, wcol=None)
        # Returns: (ns_array, vr, vrg)
        # However, looking at standard geostatspy nscore, it actually returns:
        # nscore_array, tv_vr, tv_vrg
        
        df_copy = df.copy()
        # if wcol is not used, pass None or omit if signature allows. Let's try passing weights as uniformly 1
        df_copy['wt'] = 1.0
        
        # We need to capture the output of nscore
        ns, vr, vrg = geostats.nscore(df_copy, vcol, wcol='wt')
        
        self.vr = vr
        self.vrg = vrg
        self.zmin = df_copy[vcol].min() * 0.9 # slightly lower than min for tail extrapolation
        self.zmax = df_copy[vcol].max() * 1.1 # slightly higher than max
        self._fitted = True
        
        return ns

    def inverse_transform(self, ns_array):
        """
        Applies backtr to an array of normal scores to return to original units.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit_transform before inverse_transform")
            
        import pandas as pd
        import geostatspy.geostats as geostats
        
        # Hard clip normal scores before back-transformation.
        # Kriging overshoots > 3.5 are unphysical numerical artifacts.
        # Without this, linear/power tail extrapolation maps them to massive outliers, destroying MSE/R2.
        ns_array = np.clip(ns_array, -3.5, 3.5)
        
        # backtr requires a dataframe with the normal scores
        temp_df = pd.DataFrame({'ns': ns_array})
        
        # apply backtr with default tail params (linear extrapolation)
        # backtr(df, vcol, vr, vrg, zmin, zmax, ltail, ltpar, utail, utpar)
        back_array = geostats.backtr(temp_df, 'ns', self.vr, self.vrg, 
                                     self.zmin, self.zmax, 
                                     ltail=1, ltpar=1.0, 
                                     utail=1, utpar=1.0)
        return back_array
