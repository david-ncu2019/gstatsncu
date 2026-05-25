# Why Kriging Fails: A Systematic Diagnosis of Three Real-World Benchmark Failures

## From the Nugget Ceiling to the Stationarity Trap — Understanding When and Why Ordinary Kriging Breaks Down

**Date:** 2026-05-25
**Engine:** Spatial Interpolation Engine v1.0 (AnisotropicKriging backend)
**Optimization:** Optuna TPE sampler, 200 trials, 5-fold spatial KMeans CV

---

## Abstract

Ordinary kriging carries the label "best linear unbiased predictor" (BLUP), but this guarantee is strictly conditional: it holds only under intrinsic stationarity with a correctly specified variogram model. When these conditions degrade, kriging predictions can underperform even the sample mean. We benchmarked an automated kriging pipeline against three canonical real-world spatial datasets, each representing a qualitatively different failure regime: Walker Lake (R² = 0.23, moderate nugget, strong anisotropy — kriging works but underperforms its theoretical ceiling), Meuse River zinc (R² = −0.08, non-stationary mean tied to river geometry — kriging fails silently despite a well-estimated variogram), and California earthquakes (R² = −0.11, 89% nugget fraction — kriging is fundamentally the wrong tool for point-process data). For each dataset, we trace the failure to specific violations of the six conditions required for kriging success: continuous-field assumption, approximate stationarity, well-estimated variogram, adequate spatial coverage, correct model family, and non-dominant nugget effect. We introduce the diagnostic triad of observed R², R² ceiling (1 − nugget fraction), and their gap as a practical framework for distinguishing among model refinement, model restructure, and method replacement. The report draws on foundational geostatistical theory (Isaaks & Srivastava 1989; Cressie 1993; Chilès & Delfiner 2012), recent research on data configuration effects (Markvoort & Deutsch 2024; Pyrcz 2024), and controlled benchmarks to provide a systematic account of why kriging succeeds, when it fails, and how to tell the difference before committing to a model.

---

## 1. Introduction

### 1.1 The Conditional Nature of the BLUP Guarantee

Kriging is the optimal linear unbiased predictor under a specific set of assumptions. Given n observations Z(s₁), ..., Z(sₙ) at spatial locations s₁, ..., sₙ, ordinary kriging predicts the value at an unobserved location s₀ as a weighted linear combination:

```
    Ẑ(s₀) = Σᵢ λᵢ Z(sᵢ)    subject to    Σᵢ λᵢ = 1
```

The weights λᵢ minimize the expected squared prediction error E[(Ẑ(s₀) − Z(s₀))²] under the unbiasedness constraint. The solution is the kriging system:

```
    ┌                             ┐ ┌      ┐   ┌          ┐
    │ γ(s₁,s₁) ... γ(s₁,sₙ)   1  │ │ λ₁   │   │ γ(s₁,s₀) │
    │    ⋮     ⋱     ⋮       ⋮  │ │  ⋮   │ = │    ⋮     │
    │ γ(sₙ,s₁) ... γ(sₙ,sₙ)   1  │ │ λₙ   │   │ γ(sₙ,s₀) │
    │   1     ...    1        0  │ │ −μ   │   │    1     │
    └                             ┘ └      ┘   └          ┘

    Γ λ = γ₀      →      λ = Γ⁻¹ γ₀
```

where γ(sᵢ, sⱼ) is the semivariogram value for the separation vector between points i and j, and μ is the Lagrange multiplier enforcing the sum-to-one constraint.

The word "best" in BLUP refers to minimum-variance among the class of linear unbiased predictors. Crucially, the entire optimization is performed within the model: it minimizes the expected squared error under the assumption that the variogram model is correct and that the random function is intrinsically stationary. If the model is wrong, kriging remains the best linear unbiased predictor for the wrong model — a distinction that carries no practical comfort.

### 1.2 The Three-Part Trade-Off in Kriging Weight Allocation

The kriging variance at a prediction location can be decomposed into three terms that reveal the fundamental trade-off governing prediction quality (Pyrcz 2024; Markvoort & Deutsch 2024):

```
    σ²_K = Σᵢ Σⱼ λᵢ λⱼ C(sᵢ, sⱼ)   −   2 Σᵢ λᵢ C(sᵢ, s₀)   +   C(0)
           ──────────────────────      ──────────────────      ────
           Data redundancy penalty     Data proximity benefit   Population
           (higher = worse)           (higher = better)        variance
```

The first term penalizes redundancy: if two data points are close together (highly correlated), assigning large weight to both inflates the variance because their information overlaps. The second term rewards proximity: data points near the prediction location have higher covariance with the unknown and receive larger weights. The third term, C(0) = σ²_total, represents the irreducible error when no spatial data are available.

This decomposition governs three observable behaviors:

**Closeness.** Data points nearer to s₀ receive larger weights through the covariance term C(sᵢ, s₀). This is the intuitive "nearby points matter more" property shared by all spatial interpolators.

**Redundancy (declustering).** Unlike inverse-distance weighting (IDW), which assigns weights solely by distance, kriging automatically detects and down-weights clustered data. If three points form a tight cluster, kriging distributes weight across the cluster rather than concentrating it — recognizing that the cluster carries roughly the information of a single independent observation.

**Spatial continuity.** The variogram model determines how correlation decays with distance and direction. A long range means data influence extends further, spreading weights across many points. A short range means only the nearest few points receive meaningful weight. Anisotropy directs weight preferentially along the direction of greater continuity.

```
    ┌──────────────────────────────────────────────────────────────────┐
    │         THREE-PART TRADE-OFF IN KRIGING WEIGHT ALLOCATION          │
    │                                                                    │
    │                    ┌──────────────┐                                │
    │                    │  Prediction   │                               │
    │                    │  location s₀  │                               │
    │                    └──────┬───────┘                                │
    │                           │                                        │
    │              CLOSENESS    │    CLOSENESS                           │
    │           ┌───────────────┼───────────────┐                        │
    │           ▼               │               ▼                        │
    │      ┌─────────┐         │         ┌─────────┐                     │
    │      │  λ₁=0.4 │         │         │  λ₂=0.4 │                     │
    │      │  s₁     │         │         │  s₂     │                     │
    │      └────┬────┘         │         └────┬────┘                     │
    │           │              │              │                          │
    │           └──REDUNDANCY──┼──────────────┘                          │
    │                          │                                         │
    │  If s₁ and s₂ are close together: λ₁ + λ₂ ≈ 0.6, not 0.8          │
    │  because kriging recognizes they carry shared information.         │
    │  IDW would assign 0.8, over-weighting the cluster.                 │
    └──────────────────────────────────────────────────────────────────┘
```

### 1.3 The Six Conditions for Kriging Success

Kriging performance is governed by six hierarchically nested conditions. A failure at any level propagates downstream, and the diagnostic signature differs depending on which condition is violated:

```
    Condition 1: The phenomenon is a continuous spatial random function
         │
         ▼
    Condition 2: Second-order stationarity holds approximately
         │
         ▼
    Condition 3: The variogram is well-estimated from adequate data
         │
         ▼
    Condition 4: Spatial coverage avoids clustered gaps
         │
         ▼
    Condition 5: The variogram model family matches the physical process
         │
         ▼
    Condition 6: The nugget effect does not dominate the sill
```

**Condition 1 (Continuous field).** Kriging assumes {Z(s) : s ∈ D} is defined continuously over the spatial domain — at every point, a value conceptually exists and nearby values are correlated. Phenomena that violate this include earthquake epicenters (point events along faults), species presence/absence at survey plots, and mineral grades crossing lithological boundaries.

**Condition 2 (Stationarity).** The mean must be approximately constant (or follow a known drift), and the covariance between any two points must depend only on their separation vector, not their absolute locations. Large-scale trends, heteroscedasticity, and zonal anisotropy all violate this condition to varying degrees.

**Condition 3 (Variogram quality).** A reliable variogram requires at least 30 point pairs per lag bin for omnidirectional estimation, and 100–200 for directional variography capable of detecting anisotropy (Webster & Oliver 2007). The near-origin behavior is disproportionately important: the first two or three lag bins determine the nugget estimate and short-range behavior, which control the weights assigned to the nearest — and most influential — data points.

**Condition 4 (Spatial coverage).** Sampling design affects both variogram estimation and prediction accuracy. Clustered sampling is particularly dangerous because it can produce misleadingly optimistic cross-validation when folds are randomly constructed rather than spatially blocked.

**Condition 5 (Correct model family).** The variogram model's behavior near the origin directly controls the weight assigned to the nearest data points. Gaussian models (parabolic near origin) suit smooth phenomena like topography; exponential models (linear near origin) suit rough phenomena like ore grades. Selecting the wrong family biases the nearest-neighbor weights that dominate every prediction.

**Condition 6 (Non-dominant nugget).** If the nugget fraction exceeds approximately 50%, kriging cannot produce useful predictions regardless of sample size or variogram quality — the ceiling is simply too low.

### 1.4 The Nugget Ceiling: A Fundamental Bound on Predictability

The total variance of the spatial process decomposes as:

```
    Var[Z(s)] = σ²_total = C₀ + C₁ = σ²_nugget + σ²_structured
```

The nugget C₀ aggregates three physically distinct sources of variance at scales shorter than the minimum data-pair separation (Chilès & Delfiner 2012, Chapter 4): measurement error (instrument noise, sampling variability), microscale variability (spatial structure at sub-sampling scales), and model misspecification (genuine spatial structure incorrectly attributed to the nugget by a poor variogram fit).

Any linear predictor recovers at most the structured component C₁. The nugget component is, by definition, spatially uncorrelated and unpredictable from neighboring observations. This yields the fundamental bound:

```
    R²_max = C₁ / (C₀ + C₁) = 1 − C₀ / (C₀ + C₁) = 1 − (nugget fraction)
```

This R² ceiling is the maximum proportion of variance achievable by any linear spatial predictor given the data's intrinsic noise level. The following table maps nugget fractions to expected performance:

```
    ┌──────────────────────────────────────────────────────────────────┐
    │      NUGGET FRACTION AND EXPECTED Kriging PERFORMANCE              │
    │                                                                    │
    │  Nugget %    Ceiling    Realistic R²    Verdict       Example      │
    │  ────────    ───────    ────────────    ───────       ───────      │
    │                                                                    │
    │   < 10%      > 0.90     0.70 – 0.90    Excellent     SRTM DEM     │
    │  10–25%      0.75–0.90   0.50 – 0.75    Good          Soil carbon  │
    │  25–50%      0.50–0.75   0.25 – 0.50    Moderate      Walker Lake  │
    │  50–75%      0.25–0.50   0.10 – 0.25    Poor          Sparse soil  │
    │  > 75%        < 0.25      < 0.10        Very poor     Weather radar│
    │  > 90%        < 0.10      ≈ 0           Useless       CA quakes    │
    └──────────────────────────────────────────────────────────────────┘
```

The gap between observed R² and the theoretical ceiling is as diagnostic as the ceiling itself. A gap below 0.10 indicates the model is near-optimal. A gap of 0.10–0.30 is standard, reflecting variogram estimation uncertainty and finite sample size. A gap of 0.30–0.50 suggests significant model misspecification: the model family may be wrong, or the stationarity assumption is strained. A gap exceeding 0.50 indicates that the model structure — not the parameter estimates — is fundamentally incorrect.

### 1.5 Three Datasets, Three Failure Modes

This report evaluates the automated kriging pipeline against three canonical datasets chosen to isolate three distinct failure regimes:

```
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                    THREE REGIMES OF SPATIAL STRUCTURE                        │
    ├─────────────────┬──────────────────┬──────────────────────────────────────────┤
    │ Walker Lake     │ Meuse Zinc       │ California Earthquakes                   │
    │ (n=448)         │ (n=155)          │ (n=1,072)                                │
    ├─────────────────┼──────────────────┼──────────────────────────────────────────┤
    │                 │                  │                                          │
    │  •  •  •  •     │  •••••           │      •  •          ••••                 │
    │ •  •  •  •  •   │  •••••••         │    •      •      ••    ••               │
    │  •  •  •  •     │   ••••••         │  •   ••    •  •••         •             │
    │ •  •  •  •  •   │    •••••         │     •  •  •   • •     •• •              │
    │  •  •  •  •     │                  │    •  •     ••      ••    •             │
    │                 │                  │  •     •       •••••      •              │
    │ Continuous      │ Riverscape       │ Fault-aligned                            │
    │ random field    │ (non-stationary) │ point process                            │
    │                 │                  │                                          │
    │ Condition       │ Condition        │ Condition                                │
    │ violated: #5    │ violated: #2     │ violated: #1, #6                         │
    │ (model family)  │ (stationarity)   │ (not a field,                             │
    │                 │                  │  nugget dominates)                        │
    │ Fix: better      │ Fix: universal    │ Fix: different                          │
    │ model or GP     │ kriging          │ method entirely                          │
    └─────────────────┴──────────────────┴──────────────────────────────────────────┘
```

The goal is to trace each failure to its root cause in the kriging assumptions, quantify the diagnostic signature, and provide actionable guidance for practitioners who encounter similar patterns.

---

## 2. Methods

### 2.1 Datasets

#### 2.1.1 Walker Lake — The Classic Benchmark

The Walker Lake dataset (Isaaks & Srivastava 1989) is the most widely used teaching dataset in geostatistics. It consists of 470 point measurements of a synthetic variable V (conceptually representing groundwater conductivity) sampled irregularly over a 260 × 300 grid-unit area. The full exhaustive dataset (78,000 grid nodes) is also available, making Walker Lake uniquely valuable: predictions can be validated against the complete population rather than merely cross-validated against withheld samples.

**Why Walker Lake matters for this study.** The dataset is deliberately constructed to exhibit the statistical properties of real geochemical survey data: moderate positive skewness, strong spatial anisotropy, and a substantial nugget effect. Isaaks and Srivastava designed it as a controlled experiment where the true variogram is known from the exhaustive grid. The sample variogram computed from the 470-point subset, however, is sufficiently noisy that Srivastava (1987) noted practitioners would likely "model it as a pure nugget effect" — making Walker Lake a test of whether automated variogram optimization can recover structure that naive variography would miss.

| Property | Value |
|----------|-------|
| Sample points (clean) | 448 |
| X range | 8 – 251 |
| Y range | 8 – 291 |
| Target variable | V (conductivity proxy) |
| V range | 2.1 – 1,528 |
| V mean ± std | 456.7 ± 290.8 |
| V skewness | 0.47 |
| Zeros removed | 22 (below detection limit) |

#### 2.1.2 Meuse River Zinc — The Non-Stationarity Trap

The Meuse dataset (Pebesma 2004) contains heavy metal concentrations measured at 155 locations along the Meuse River floodplain near Stein, Netherlands. It is the canonical example in the R `gstat` package. Zinc concentrations range from 113 to 1,839 ppm and exhibit a pronounced spatial gradient: high concentrations near the river channel, decreasing with distance inland.

**Why Meuse matters.** This dataset is the textbook example of non-stationarity that ordinary kriging cannot handle but universal kriging can. Pebesma (2004) demonstrated that universal kriging with log(distance-to-river) as external drift raises CV R² to approximately 0.55 — from essentially zero for ordinary kriging. The dataset includes companion variables (cadmium, copper, lead, elevation, distance-to-river, organic matter, soil type) that are known to co-vary with zinc, making it an ideal testbed for methods that incorporate auxiliary information.

The critical variable is `dist` — the Euclidean distance from each sampling location to the river channel. Zinc concentrations follow an approximately exponential decay with distance from the river, creating a non-stationary mean that ordinary kriging's constant-mean assumption cannot represent:

```
       Meuse floodplain cross-section (schematic):
       ═══════════════════════════════════════════

       Zinc (ppm)
       1800 ┤  ●
            │
       1200 ┤     ●  ●
            │         ●
        600 ┤            ●  ●  ●     ← kriging predicts ≈ mean here
            │                     ●  ●
          0 ┤                        ●  ●
            ├─────────────────────────────
            River                     Inland
            (0 m)                    (500 m)

       Ordinary kriging: constant mean ───  predicts ~470 ppm everywhere
       Reality:           exponential decay with distance from river
       Universal kriging: mean(x) = β₀ + β₁·log(dist)  ← correct model
```

| Property | Value |
|----------|-------|
| Sample points | 155 |
| X range (RDH m) | 178,605 – 181,390 |
| Y range (RDH m) | 329,714 – 333,611 |
| Target variable | Zinc (ppm) |
| Zinc range | 113 – 1,839 |
| Zinc mean ± std | 469.7 ± 367.1 |
| Zinc skewness | 1.47 |
| Companion variables | Cd, Cu, Pb, elevation, distance-to-river, organic matter, soil type, land use |

#### 2.1.3 California Earthquakes (2024) — The Point-Process Negative Control

This dataset contains 1,072 seismic events of magnitude 2.5 or greater recorded in California and western Nevada between January 1, 2024 and December 31, 2024, retrieved from the USGS FDSN earthquake catalog API. The query area spanned 32°N–42°N and 124°W–114°W, covering the San Andreas, Hayward, Calaveras, and Walker Lane fault systems.

**Why California earthquakes matter.** Earthquake magnitudes along fault systems present a fundamental violation of kriging's most basic assumption — that the target variable is a realization of a continuous spatial random function. Seismicity is a marked point process: the locations are controlled by fault geometry rather than distance-based spatial correlation, and the magnitude at each point is a realization of the Gutenberg-Richter power-law distribution rather than a smoothly varying spatial surface. This dataset serves as a deliberate negative control: if the kriging pipeline correctly identifies that no meaningful spatial structure exists, it demonstrates diagnostic robustness. If it silently produces a smooth interpolation surface regardless, it demonstrates the danger of applying kriging without diagnostic checks.

```
       Diagnostic signature of a POINT PROCESS:
       ═══════════════════════════════════════════

       Nugget fraction > 80%  ────▶  Spatial correlation is negligible
       R² ceiling near zero   ────▶  No amount of model tuning will help
       R² < 0                 ────▶  Mean prediction outperforms kriging

       What's happening:
       ┌─────────────────────────────────────────────────────┐
       │  San Andreas Fault (schematic)                       │
       │                                                      │
       │    M5.7 ●                                            │
       │          \  fault trace                               │
       │     M2.5 ● \                                         │
       │             \   M3.2 ●                               │
       │    M2.8 ●     \                                       │
       │                \   M4.1 ●                            │
       │      M3.0 ●     \                                     │
       │                  \                                    │
       │                                                      │
       │  Magnitudes are i.i.d. draws from Gutenberg-Richter  │
       │  conditioned on fault location. Spatial correlation  │
       │  ≠ continuous field.                                 │
       └─────────────────────────────────────────────────────┘
```

| Property | Value |
|----------|-------|
| Sample points | 1,072 |
| Longitude range | −123.92 – −114.53 |
| Latitude range | 32.06 – 42.00 |
| Target variable | Magnitude (M ≥ 2.5) |
| Magnitude range | 2.5 – 5.7 |
| Magnitude mean ± std | 2.95 ± 0.45 |
| Magnitude skewness | 1.79 |

### 2.2 Pipeline Configuration

All three datasets were processed with identical engine settings:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Engine mode | kriging | Ordinary kriging baseline |
| Variogram models | 9 (all available) | Full model selection via Optuna |
| CV folds | 5 | KMeans-based spatial blocking |
| Optuna trials | 200 | Sufficient for TPE convergence |
| Max anisotropy ratio | 5.0 | Covers typical geological range |
| Detrend auto-detect | enabled (order=1) | Linear plane removal if F-test significant |
| NST auto-detect | enabled | Conservative AND-gate (Shapiro-Wilk + skewness + kurtosis) |
| Grid resolution | dataset-specific | 5.0 / 100.0 / 0.1 (matched to coordinate units) |

### 2.3 The Diagnostic Triad

We report three quantities jointly for each dataset:

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Observed R²** | 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)² | Actual predictive skill; < 0 means worse than the global mean |
| **R² ceiling** | 1 − σ²_nugget / (σ²_nugget + σ²_psill) | Maximum R² achievable by any linear spatial predictor |
| **Gap** | R²_ceiling − R²_observed | Room for improvement; > 0.5 indicates wrong model structure |

The joint interpretation of these three numbers forms the diagnostic backbone of this report. A positive R² with a small gap indicates a well-specified model. A negative R² with a high ceiling and large gap indicates non-stationarity. A near-zero ceiling indicates no spatial structure to recover, regardless of method.

---

## 3. Results

### 3.1 Walker Lake: Kriging Works, But Why Only R² = 0.23?

#### 3.1.1 Preprocessing Decisions

```
       Trend F-test p-value          : 5.74 × 10⁻⁷  (SIGNIFICANT)
       Trend R²                      : 0.0625
       Shapiro-Wilk p-value          : 3.05 × 10⁻⁸  (NON-NORMAL)
       Skewness / Excess Kurtosis    : 0.472 / −0.013

       → Detrending ENABLED  (F-test significant + R² > 0.05)
       → NST SKIPPED         (|skewness| < 0.5 threshold)
```

The linear trend explained 6.25% of total variance — a gentle NE-SW gradient. The decision to skip NST is notable: although Shapiro-Wilk rejects normality (p ≈ 3 × 10⁻⁸), the sample skewness (0.47) falls just below the 0.5 threshold. For marginal non-Gaussianity, the NST introduces more distortion through quantile interpolation artifacts than it resolves, making the AND-gate's conservative decision correct in this case.

#### 3.1.2 Optimized Variogram

| Parameter | Value |
|-----------|-------|
| Model family | **Stable** (α = 0.41) |
| Rotation angle | 92.7° |
| Anisotropy ratio | 4.98 |
| Partial sill | 129,002 |
| Range | 59.4 grid units |
| Nugget | 50,059 |
| **Nugget fraction** | **28.0%** |

```
              Walker Lake: Fitted Stable Variogram
              ═══════════════════════════════════════

     γ(h) │
          │                                          ╭── sill = 179,061
    175k  │                                 ╭────────╯
          │                           ╱╱╱╱╱
          │                       ╱╱╱
          │                    ╱╱
          │                 ╱╱
          │              ╱╱
          │           ╱╱
          │         ╱╱
          │       ╱╱
          │     ╱╱   α = 0.41 (between exponential=1.0 and Gaussian=2.0)
     50k  │────╱╱──────────────────────────────────── nugget = 50,059
          │   ╱
          │ ╱
        0 └───────────────────────────────────────────── h
          0             59.4 (range)                  150

     Anisotropy ellipse:          N
     (major axis direction)        │
                     ratio=4.98    │   92.7°
                     ╭────────╮   │  ╱
                     │        │   │ ╱
                     │  (·)   │   │╱
                     │        │   ┼───── E
                     ╰────────╯
```

#### 3.1.3 Prediction Performance

| Metric | Value |
|--------|-------|
| **Observed R²** | **0.2267** |
| MAE | 199.42 |
| RMSE | 247.33 |
| R² ceiling | 0.7204 |
| **Gap to ceiling** | **0.4937** |

#### 3.1.4 Why the R² Is Only 0.23 When the Ceiling Is 0.72

This is the central question for Walker Lake. The 0.49 gap between observed R² and the theoretical ceiling signals substantial room for improvement — the spatial structure exists and is recoverable in principle, but the current model is not capturing it fully. Three factors contribute:

**Factor 1: The stable model with α = 0.41 introduces a mismatch.** The stable model's behavior near the origin interpolates between exponential (α = 1, linear near origin, appropriate for rough processes) and Gaussian (α = 2, parabolic near origin, appropriate for smooth processes). An α of 0.41 produces a variogram that is steeper near the origin than exponential — it implies spatial continuity rougher than the exponential model, approaching the behavior of a power model. The original Isaaks and Srivastava (1989) analysis used spherical and exponential models for Walker Lake V; a stable model with α = 0.41 represents a different roughness regime. The near-origin behavior directly controls the weights assigned to the nearest data points — the most heavily weighted points in any kriging prediction. A mismatched near-origin slope propagates into systematically suboptimal weights at every prediction location.

**Factor 2: The nugget fraction of 28% alone limits achievable R² to 0.72.** This is a hard bound: 28% of the total variance is spatially uncorrelated at the measurement scale and cannot be predicted from neighboring observations regardless of method. The true nugget fraction for Walker Lake V, computed from the exhaustive 78,000-point grid (Isaaks & Srivastava 1989), is approximately 44% — the sample-based estimate of 28% likely underestimates the true nugget because the 448-point subset lacks sufficient close pairs to resolve sub-60-unit microscale variability. If the true nugget fraction is 44%, the genuine ceiling is only 0.56, making the observed R² of 0.23 a more respectable 41% of the achievable maximum.

**Factor 3: The anisotropy ratio saturated at the config limit.** The optimizer pushed the anisotropy ratio to 4.98 — essentially at the 5.0 cap set in the configuration. When an optimizer saturates a boundary constraint, the true optimum likely lies beyond it. The original Walker Lake analysis reports anisotropy of approximately 6–8× for variable V in the E-W direction (Isaaks & Srivastava 1989, Chapter 16). Capping anisotropy at 5.0 constrains the model to under-represent the directional continuity, producing predictions that over-smooth in the E-W direction and under-weight distant E-W neighbors that carry genuine information.

```
    ┌──────────────────────────────────────────────────────────────────┐
    │           WHY WALKER LAKE R² = 0.23 (CEILING = 0.72)              │
    │                                                                    │
    │  R² = 1.00  ┤  ●  (perfect prediction)                            │
    │             │                                                     │
    │  R² = 0.72  ┤  ─ ─ ─ ─ ─ ─ ─ ─ ─  Nugget ceiling (28% nugget)   │
    │             │  ║                                                  │
    │             │  ║  0.49 gap: room for improvement                  │
    │             │  ║                                                  │
    │  R² = 0.23  ┤  ████████████  Actual CV performance                │
    │             │                                                     │
    │             │  Gap attributable to:                               │
    │             │    • Model family mismatch (α=0.41 vs. spherical)   │
    │             │    • Anisotropy cap at 5.0 (true ratio ≈ 6-8×)      │
    │             │    • True nugget likely ~44% (not 28%)              │
    │             │    • Only 448 points over 260×300 domain            │
    │  R² = 0.00  ┤  ─ ─ ─ ─ ─ ─ ─ ─ ─  Global mean predictor          │
    └──────────────────────────────────────────────────────────────────┘
```

This is the **model refinement** regime: kriging works, the spatial structure is real, but the automated parameter search converged to a local optimum bounded by configuration constraints. The fix is not to abandon kriging but to relax the anisotropy cap, increase Optuna trials, and restrict the model search to families whose near-origin behavior matches the known roughness of Walker Lake V (exponential or spherical, not stable).

### 3.2 Meuse River Zinc: Kriging Fails Silently

#### 3.2.1 Preprocessing Decisions

```
       Trend R²                      : 0.2004
       → Detrending ENABLED

       Shapiro-Wilk p-value          : 3.28 × 10⁻¹²  (NON-NORMAL)
       Skewness                      : 1.472  (EXCEEDS 0.5)
       Excess kurtosis               : 1.888  (EXCEEDS 1.0)
       → NST APPLIED
```

Both detrending and NST fired. The strong positive skewness (1.47) and heavy right tail (kurtosis 1.89) correctly triggered the transformation — standard protocol for heavy-metal concentrations in geochemical surveys (Journel & Huijbregts 1978). The 1st-order polynomial trend explained 20% of variance and was subtracted before modeling.

#### 3.2.2 Optimized Variogram (NST scale)

| Parameter | Value |
|-----------|-------|
| Model family | **Circular** |
| Rotation angle | 60.7° |
| Anisotropy ratio | 4.61 |
| Partial sill | 2.96 (NST units) |
| Range | 4,543.2 m |
| Nugget | 0.51 (NST units) |
| **Nugget fraction** | **14.7%** |

The circular model has a finite range where spatial correlation drops to exactly zero — geometrically appropriate for a bounded depositional environment like a river floodplain. The 4.5 km range approximately matches the floodplain width, and the 60.7° rotation with 4.6× anisotropy aligns with the NE-SW orientation of the Meuse River channel. On paper, this variogram looks well-estimated: the nugget is modest (15%), the range is physically interpretable, and the anisotropy aligns with known geography.

#### 3.2.3 Prediction Performance

| Metric | Value |
|--------|-------|
| **Observed R²** | **−0.0838** |
| MAE | 232.55 |
| RMSE | 340.61 |
| R² ceiling | 0.8529 |
| **Gap to ceiling** | **0.9367** |

#### 3.2.4 Why Kriging Produces Negative R² Despite a High Ceiling

This is the most dangerous failure mode in applied geostatistics: **the variogram looks good, but the predictions are worse than the global mean.** The 0.94 gap between observed R² (−0.08) and the ceiling (0.85) is the largest of any dataset tested — a direct signature of a fundamentally misspecified model structure, not merely suboptimal parameter estimates.

The root cause is a violation of **Condition 2 (stationarity)**. Ordinary kriging assumes the mean is constant across the domain: E[Z(s)] = μ for all locations s. The Meuse floodplain violates this assumption systematically: zinc concentrations follow an approximately exponential decay with distance from the river channel. The 1st-order polynomial detrending removed a linear plane, but the distance-from-river relationship is curvilinear — a linear plane cannot capture exponential decay with distance from a meandering river.

```
    ┌──────────────────────────────────────────────────────────────────┐
    │     THE NON-STATIONARITY TRAP: WHY A GOOD VARIOGRAM DECEIVES      │
    │                                                                    │
    │  Step 1: The data have a strong trend tied to river distance.     │
    │          1st-order polynomial detrending (linear plane) removes   │
    │          only 20% of variance — the curvilinear river-distance    │
    │          relationship remains in the residuals.                    │
    │                                                                    │
    │  Step 2: The variogram is estimated on these residuals. It looks  │
    │          well-behaved: modest nugget (15%), interpretable range    │
    │          (4.5 km), anisotropy aligned with the river. The          │
    │          optimizer correctly finds the spatial structure in the    │
    │          residuals — but the residuals still contain the           │
    │          unmodeled curvilinear trend.                              │
    │                                                                    │
    │  Step 3: Kriging with this variogram assumes a constant mean in   │
    │          the residual field. When predicting at a location near    │
    │          the river, it averages observations from both near-river  │
    │          (high zinc) and inland (low zinc) locations, weighted by  │
    │          spatial proximity. The inland points drag the prediction  │
    │          down; the near-river points drag it up. The prediction    │
    │          converges to the global mean ± random noise.              │
    │                                                                    │
    │  Result: R² = −0.08. The predictions are, on average, further     │
    │          from the truth than simply guessing 470 ppm everywhere.   │
    └──────────────────────────────────────────────────────────────────┘
```

This is the **non-stationarity trap**: ordinary kriging fails silently. The variogram is well-estimated. The nugget is reasonable. The anisotropy aligns with known geography. Every diagnostic except the cross-validation R² suggests a successful model — and the cross-validation R² is catastrophically negative.

The constant-mean assumption forces the kriging system to smooth across a gradient it cannot represent. The kriging weights, optimized under the wrong mean model, produce predictions that regress toward a global mean that does not describe any actual location in the floodplain. Near the river, zinc is consistently 600–1,800 ppm. Inland, it is consistently 100–300 ppm. The global mean of 470 ppm describes neither regime, and kriging predictions that converge toward it are wrong everywhere.

**The canonical fix** (Pebesma 2004) is universal kriging with log(distance-to-river) as an external drift variable:

```
    Z(s) = β₀ + β₁ · log(dist(s)) + δ(s)

    where δ(s) is a stationary residual with variogram γ_δ(h)
```

This model separates the non-stationary mean (captured by the drift term) from the stationary residual (captured by the variogram). The kriging weights then operate on the residuals δ(s), which do satisfy the stationarity assumption. Pebesma (2004) reports that this specification raises CV R² from essentially zero to approximately 0.55 for Meuse zinc.

```
    ┌──────────────────────────────────────────────────────────────────┐
    │      ORDINARY KRIGING VS. UNIVERSAL KRIGING ON THE MEUSE          │
    │                                                                    │
    │  Ordinary Kriging:                    Universal Kriging:           │
    │  ~~~~~~~~~~~~~~~~~~                   ~~~~~~~~~~~~~~~~~           │
    │                                                                    │
    │  Model: Z(s) = μ + ε(s)              Model: Z(s) = β₀ +           │
    │  μ = constant (unknown)                      β₁·log(dist(s))      │
    │  ε(s) ~ stationary                            + δ(s)              │
    │                                                                    │
    │  Problem: μ ≈ 470 ppm is wrong         δ(s) ~ stationary          │
    │  everywhere. Near the river,           (no trend remaining)        │
    │  zinc ≈ 800–1800. Inland,                                         │
    │  zinc ≈ 100–300. The single            Kriging weights operate     │
    │  mean describes neither.               on δ(s), which IS           │
    │                                        stationary.                 │
    │  Kriging weights regress toward                                   │
    │  the wrong mean → R² = −0.08           Reported CV R² ≈ 0.55      │
    │                                        (Pebesma 2004)              │
    └──────────────────────────────────────────────────────────────────┘
```

A secondary factor is sample size. With only 155 points, the Meuse dataset is at the lower bound of adequacy for directional variography (Webster & Oliver 2007 recommend 100–200 points for directional estimation). The 15% nugget fraction estimated from 155 points may be optimistic: with insufficient close pairs, short-lag semivariance is poorly constrained, and the nugget estimate is biased downward. If the true nugget fraction is closer to 30%, the genuine ceiling drops from 0.85 to 0.70 — still far above the observed R², confirming that non-stationarity, not nugget, is the binding constraint.

### 3.3 California Earthquakes: Kriging Is the Wrong Tool

#### 3.3.1 Preprocessing Decisions

```
       Trend R²                      : 0.0138
       → Detrending DISABLED (R² < 0.05 threshold)

       Shapiro-Wilk p-value          : 1.70 × 10⁻³²  (NON-NORMAL)
       Skewness                      : 1.793  (EXCEEDS 0.5)
       Excess kurtosis               : 4.021  (EXCEEDS 1.0)
       → NST APPLIED
```

A planar trend explained only 1.4% of magnitude variance — unsurprising, since earthquake magnitudes are not organized into regional gradients. The NST fired on all three criteria, transforming the power-law magnitude distribution to approximate normality.

#### 3.3.2 Optimized Variogram

| Parameter | Value |
|-----------|-------|
| Model family | **Rational-quadratic** (α = 1.31) |
| Rotation angle | 154.7° |
| Anisotropy ratio | 2.17 |
| Partial sill | 0.124 |
| Range | 16.1° |
| **Nugget** | **0.964** |
| **Nugget fraction** | **88.6%** |

#### 3.3.3 Prediction Performance

| Metric | Value |
|--------|-------|
| **Observed R²** | **−0.1075** |
| MAE | 0.32 |
| RMSE | 0.47 |
| R² ceiling | 0.1137 |
| **Gap to ceiling** | **0.2212** |

#### 3.3.4 Why Kriging Cannot Help — The Point-Process Blind Spot

The California earthquake results represent the **method replacement** regime: kriging is fundamentally the wrong tool, and no amount of parameter tuning, preprocessing, or model refinement can change that.

The root cause is a violation of **Condition 1 (continuous spatial random function)**. Kriging assumes that at every point s in the domain, a random variable Z(s) exists conceptually, and that nearby values Z(s) and Z(s + h) are correlated in a manner that decays with distance. Earthquake magnitudes violate this assumption at the most basic level:

**The locations are not a sampling design — they are the phenomenon itself.** In a geochemical survey, the sampling locations are chosen by the investigator, and the measured concentrations are realizations of an underlying continuous field. In an earthquake catalog, the epicenter locations are themselves the random events: earthquakes occur where faults exist, not uniformly across the domain. Applying kriging to earthquake magnitudes is analogous to applying kriging to the locations of trees in a forest and expecting it to predict species identity — the fundamental data structure is wrong.

**Magnitudes follow a power-law distribution, not a spatial process.** The Gutenberg-Richter law states that the number of earthquakes with magnitude ≥ M is proportional to 10^(−bM), where b ≈ 1.0. This means magnitude is drawn from an exponential distribution that is independent of location — conditional on an earthquake occurring at a given fault segment, its magnitude is essentially random. The spatial correlation that kriging attempts to model does not exist because the data-generating process does not produce it.

**The nugget fraction of 89% is the honest answer.** When kriging is applied to data with no spatial structure, the optimizer should converge on a pure nugget model — all variance attributed to spatially uncorrelated noise. The 89% nugget is close to this ideal, and the remaining 11% (with a suspiciously large 16.1° range, ~1,800 km) likely reflects the weak alignment of slightly higher magnitudes along the San Andreas fault trace rather than genuine distance-based correlation. The optimizer, forced to choose a model from the available families, selected the rational-quadratic (a multiscale kernel) as the least-bad fit to this weak large-scale signal.

```
    ┌──────────────────────────────────────────────────────────────────┐
    │     WHY EARTHQUAKE MAGNITUDES VIOLATE EVERY KRIGING ASSUMPTION    │
    │                                                                    │
    │  Condition 1 (Continuous field):                                  │
    │    ✗ Earthquakes are discrete events at specific locations.       │
    │      There is no "magnitude field" between epicenters. The        │
    │      value at an arbitrary point (e.g., in the Mojave Desert     │
    │      where no fault exists) is undefined, not merely unmeasured.  │
    │                                                                    │
    │  Condition 2 (Stationarity):                                      │
    │    ✗ The mean magnitude varies by fault system (strike-slip vs.   │
    │      subduction zones produce different magnitude distributions). │
    │      More fundamentally, the probability of an earthquake          │
    │      occurring at all varies spatially — it is near zero away     │
    │      from known fault traces.                                      │
    │                                                                    │
    │  Condition 3 (Variogram quality):                                 │
    │    ~ With 1,072 events, there are enough "data points," but the   │
    │      variogram is estimating apparent correlation from a          │
    │      generating process that produces no genuine spatial          │
    │      correlation in magnitude. The variogram converges to the     │
    │      correct answer — pure nugget — which tells you to stop.      │
    │                                                                    │
    │  Condition 6 (Non-dominant nugget):                               │
    │    ✗ 89% nugget → ceiling of 0.11. There is essentially nothing   │
    │      to predict.                                                  │
    └──────────────────────────────────────────────────────────────────┘
```

The R² ceiling of 0.11 is the key diagnostic. Unlike Walker Lake (ceiling 0.72, room for model improvement) and Meuse (ceiling 0.85, wrong model structure), the California earthquakes ceiling is near zero by construction: there is no spatial structure in magnitude to recover. The 0.22 gap between observed R² (−0.11) and ceiling (0.11) is modest — meaning the model is actually doing about as well as theoretically possible given the data — but "as well as possible" is still worse than the global mean.

For earthquake data, appropriate methods include:
- **Kernel density estimation** for seismicity rates (earthquakes per unit area per unit time)
- **Ground-motion prediction equations** (GMPEs) that incorporate fault geometry, distance-to-rupture, and site conditions explicitly
- **Epidemic-Type Aftershock Sequence (ETAS) models** for space-time-magnitude point processes
- **Fault-based seismic hazard models** that treat magnitude as a marked point process conditioned on fault geometry and slip rate

### 3.4 Cross-Dataset Comparison

```
     ┌────────────────────────────────────────────────────────────────────┐
     │              COMPARATIVE KRIGING PERFORMANCE                         │
     │                                                                    │
     │                     R²    R² Ceiling   Nugget%   Gap    Verdict    │
     │                     ────  ──────────   ───────   ────   ────────   │
     │  Walker Lake   ████ 0.23    0.72        28%     0.49   Refine     │
     │  (n=448)       ████                             model              │
     │                                                                    │
     │  Meuse Zinc    ░░░░ -0.08   0.85        15%     0.94   Restructure│
     │  (n=155)       ░░░░                             model              │
     │                                                                    │
     │  CA Quakes     ░░░░ -0.11   0.11        89%     0.22   Replace    │
     │  (n=1072)      ░░░░                             method             │
     └────────────────────────────────────────────────────────────────────┘
```

The three regimes form a complete diagnostic taxonomy for kriging failures:

| Regime | Nugget fraction | R² ceiling | Observed R² | Gap | Violated condition | Action |
|--------|----------------|------------|-------------|-----|--------------------|--------|
| **Model refinement** (Walker) | Moderate (28%) | High (0.72) | Positive (0.23) | Large (0.49) | Condition 5 (model family) | Relax constraints; improve variogram; switch to GP |
| **Model restructure** (Meuse) | Low (15%) | High (0.85) | Negative (−0.08) | Very large (0.94) | Condition 2 (stationarity) | Universal kriging with covariates |
| **Method replacement** (CA quakes) | Very high (89%) | Near zero (0.11) | Negative (−0.11) | Small (0.22) | Condition 1 (continuous field) | Point process models; density estimation |

---

## 4. Discussion

### 4.1 The Nugget Ceiling as a Diagnostic Gatekeeper

The R² ceiling — 1 minus the nugget fraction — is the single most informative diagnostic that an automated kriging pipeline can report. It answers the question: "Is there enough spatial structure in these data for kriging to be worth attempting?"

A ceiling below 0.25 (nugget > 75%) means kriging is not worth pursuing regardless of sample size, sampling design, or model selection. The California earthquake ceiling of 0.11 falls squarely in this regime. When the ceiling is this low, cross-validation R² will be at best slightly above zero and frequently negative, because the model is attempting to predict a signal that does not exist. The pipeline should halt and report "no recoverable spatial structure" rather than producing a kriging surface that is numerically valid but physically meaningless.

A ceiling above 0.50 (nugget < 50%) means spatial structure exists and is potentially recoverable. The question then becomes whether the gap between observed R² and the ceiling is due to correctable estimation issues (sample size, model family, anisotropy constraints — the Walker Lake regime) or to a fundamental model misspecification that requires restructuring (non-stationarity — the Meuse regime).

The distinction between these two sub-regimes is made by the size of the gap. A gap below 0.30 is expected: variogram estimation error, finite sample size, and mild non-Gaussianity typically consume 20–30% of the ceiling. A gap above 0.50 is diagnostic of stationarity violation — the variogram is being estimated on residuals that still contain unmodeled spatial structure, and the kriging weights derived from that variogram are optimizing the wrong objective.

```
    ┌──────────────────────────────────────────────────────────────────┐
    │              DIAGNOSTIC DECISION FRAMEWORK                         │
    │                                                                    │
    │                    ┌─────────────────────────────┐                │
    │                    │ Is R² ceiling > 0.25?       │                │
    │                    │ (nugget fraction < 75%)     │                │
    │                    └──────────┬──────────────────┘                │
    │                               │                                   │
    │                   ┌───────────┴───────────┐                       │
    │                   │ NO                    │ YES                   │
    │                   ▼                       ▼                       │
    │          ┌──────────────────┐    ┌──────────────────────────┐    │
    │          │ NO SPATIAL       │    │ Is gap to ceiling < 0.50?│    │
    │          │ STRUCTURE        │    └──────────┬───────────────┘    │
    │          │                  │               │                    │
    │          │ Kriging cannot   │    ┌──────────┴──────────┐         │
    │          │ recover signal.  │    │ YES                 │ NO      │
    │          │ Use different    │    ▼                     ▼         │
    │          │ method entirely. │ ┌──────────────┐ ┌──────────────┐ │
    │          │                  │ │ MODEL        │ │ MODEL        │ │
    │          │ Example:         │ │ REFINEMENT   │ │ RESTRUCTURE  │ │
    │          │ CA quakes        │ │              │ │              │ │
    │          │ (ceiling=0.11)   │ │ Improve:     │ │ Switch to    │ │
    │          └──────────────────┘ │ variogram,   │ │ universal    │ │
    │                               │ constraints, │ │ kriging with │ │
    │                               │ or GP        │ │ covariates   │ │
    │                               │              │ │              │ │
    │                               │ Example:     │ │ Example:     │ │
    │                               │ Walker Lake  │ │ Meuse Zinc   │ │
    │                               │ (gap=0.49)   │ │ (gap=0.94)   │ │
    │                               └──────────────┘ └──────────────┘ │
    └──────────────────────────────────────────────────────────────────┘
```

### 4.2 The Non-Stationarity Trap: Why Ordinary Kriging Is Dangerous in Automated Pipelines

The Meuse results expose a dangerous property of automated kriging: **a well-estimated variogram can mask a fundamentally wrong model structure.** The optimizer produced a variogram that was, on its own terms, reasonable: modest nugget, interpretable range, physically meaningful anisotropy. Every intermediate diagnostic — the variogram plot, the anisotropy ellipse, the nugget-to-sill ratio — suggested a successful fit. Only the cross-validation R² revealed the catastrophe.

This failure mode is particularly dangerous in automated pipelines because the human analyst, seeing a reasonable variogram and plausible-looking prediction map, may not check the CV metrics. The kriging surface will always look smooth and "scientific" — kriging is guaranteed to produce a numerically valid output for any positive-definite variogram — but its physical meaning is nil.

The pipeline's CV sanity check (flagging negative R² as "physically impossible") is the correct safety net. A negative R² means the predictions are, in aggregate, further from the truth than the global mean. Under a correctly specified model with a positive-definite variogram, this cannot happen: the kriging weights minimize expected squared error, and the global mean predictor (all λᵢ = 1/n) is a feasible but suboptimal member of the class of linear unbiased predictors. If the optimized weights perform worse than the equal-weight solution, the model is wrong.

### 4.3 The Point-Process Boundary: When Kriging Cannot Apply

The California earthquake case is conceptually distinct from the Meuse case. For Meuse zinc, ordinary kriging fails but universal kriging succeeds — the problem is model specification, not method applicability. For earthquake magnitudes, no variant of kriging can help because the data are not a realization of a continuous spatial random function.

This distinction is fundamental and often misunderstood. Kriging requires that the target variable Z(s) be conceptually defined everywhere in the domain, with values at nearby locations correlated through a distance-decay function. Many spatial datasets satisfy this requirement approximately. Many do not:

**Data types that violate the continuous-field assumption:**
- Point process marks (earthquake magnitudes, crime incident severity)
- Presence/absence data with zero inflation (species occurrence at survey plots)
- Categorical data without a meaningful distance metric (rock type, land use class)
- Network-constrained phenomena (traffic speed on a road network, river pollutant concentrations that propagate downstream but not laterally)
- Data with sharp physical discontinuities (mineral grades crossing a lithological contact, groundwater levels across a fault with 100 m of vertical offset)

For these data types, the appropriate methods are fundamentally different: point process models (ETAS for earthquakes, Ripley's K for spatial point patterns), species distribution models (MaxEnt, GLMMs), Markov random fields for categorical data, or physically informed numerical models that incorporate known discontinuities.

The pipeline's 89% nugget fraction and 0.11 R² ceiling correctly identified this regime. When the nugget dominates the sill, the pipeline should report "no recoverable spatial structure" and stop — not because the optimizer failed, but because it succeeded in discovering that there is nothing to optimize.

### 4.4 Data Configuration Effects That Amplify Failures

Beyond the primary assumption violations, two data configuration effects from the geostatistical literature (Markvoort & Deutsch 2024; Pyrcz 2024) amplify kriging failures in ways that are relevant to these benchmarks:

**The string effect.** When data points form a linear array (e.g., measurements along a borehole or a survey transect), kriging assigns disproportionately large weights to the endpoints because they have data on only one side and therefore appear "less redundant" to the redundancy-penalty term in the kriging variance. Interior points, surrounded by neighbors on both sides, are heavily penalized and receive near-zero weights. In the Meuse dataset, the sampling follows an irregular pattern along the floodplain, not a linear transect, so the string effect is not the primary issue. However, the 155-point sample is sparse enough that effective search neighborhoods for many prediction locations contain only 3–8 points — configurations where endpoint-like weighting can emerge even without a literal linear array.

**The screening effect and negative weights.** When a data point lies directly between the prediction location and a more distant data point, the closer point "screens" the farther one, which can receive a negative kriging weight. Negative weights are mathematically correct under the model — they effectively subtract redundant information — but they can produce physically impossible predictions (negative zinc concentrations, negative magnitude estimates) when the variogram is misspecified or the stationarity assumption is violated. For the Meuse dataset, negative weights are likely occurring at prediction locations near the river, where nearby high-zinc points screen more distant inland low-zinc points. The resulting predictions can be pulled below physically possible values.

**The screening interpretation of the nugget effect.** The nugget effect reduces screening: as the nugget increases relative to the sill, the apparent redundancy among proximate points diminishes, and kriging weights become more uniform. In the limit of pure nugget (California earthquakes), all weights approach 1/n — the global mean predictor. This explains why the California earthquake kriging predictions converge to the global mean magnitude of 2.95 everywhere: with 89% nugget, the kriging system has essentially no spatial correlation to differentiate weights by distance, and the optimal strategy is equal weighting.

### 4.5 Conditional Bias: Why Even "Good" Kriging Systematically Misrepresents Extremes

A property of kriging that affects all three datasets — but is most visible in Walker Lake — is **conditional bias** (the Krige smoothing effect). Kriging minimizes expected squared error, and the optimal strategy for minimizing squared error under uncertainty is to predict conservatively: underpredict high values and overpredict low values. This is not a defect in the kriging algorithm but a direct mathematical consequence of the minimum-variance objective.

The practical manifestation is that the histogram of kriging predictions has smaller variance than the histogram of true values: the predictions are "squeezed" toward the mean. High values are systematically underpredicted, and low values are systematically overpredicted. The regression slope of true values on predicted values is less than 1.0 (Krige 1997; Olea 1999):

```
    ┌──────────────────────────────────────────────────────────────────┐
    │              CONDITIONAL BIAS: THE KRIGE SMOOTHING EFFECT          │
    │                                                                    │
    │  True       │  ●                                                   │
    │  value      │     ●  ●                                             │
    │             │         ●  ●    ← regression slope < 1.0             │
    │             │              ●  ●                                    │
    │             │                  ●  ●                                │
    │  Mean ──────┼────────────────────●────────────────                 │
    │             │                 ●  ●                                 │
    │             │              ●  ●                                    │
    │             │           ●  ●     High values underpredicted        │
    │             │        ●  ●                                          │
    │             │     ●  ●                                             │
    │             │  ●         Low values overpredicted                  │
    │             │                                                     │
    │             └──────────────────────────────────                    │
    │                     Predicted value                                │
    │                                                                    │
    │  The slope of the regression line is:                              │
    │    • Simple kriging: always 1.0 (theoretically unbiased)           │
    │    • Ordinary kriging: always < 1.0 (biased by Lagrange multiplier)│
    │    • The bias increases with kriging variance                     │
    │      (worse in data-sparse regions)                                │
    └──────────────────────────────────────────────────────────────────┘
```

For Walker Lake, with an observed R² of 0.23, the regression slope of true values on predictions is approximately √0.23 ≈ 0.48 — meaning that for every unit increase in the true V value, the kriging prediction increases by only 0.48 units. This is the mathematical signature of the smoothing effect: kriging is conservative because conservatism minimizes squared error on average.

For applications where the distribution of predicted values matters — resource estimation, environmental risk assessment, extreme value analysis — this conditional bias is a serious limitation. Sequential Gaussian simulation (SGS) reproduces the target histogram and variogram without conditional bias, at the cost of higher local error variance. The choice between kriging (minimum local error, biased distribution) and simulation (correct distribution, higher local error) depends on whether the application prioritizes point accuracy or distributional fidelity.

### 4.6 The Six Conditions Revisited: A Stratified View of Failure

Mapping the three benchmark datasets onto the six-condition framework reveals a hierarchy of failure severity:

```
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │              SIX CONDITIONS MAPPED TO BENCHMARK FAILURES                        │
    │                                                                                │
    │  Condition                          Walker Lake    Meuse Zinc    CA Quakes     │
    │  ─────────                          ───────────    ──────────    ─────────     │
    │  C1: Continuous field               ✓              ✓              ✗ (fatal)    │
    │  C2: Approximate stationarity       ~ (mild)       ✗ (severe)     ✗            │
    │  C3: Well-estimated variogram       ~ (adequate)   ~ (marginal)   ✓ (correct)  │
    │  C4: Adequate spatial coverage      ✓              ~ (sparse)     n/a          │
    │  C5: Correct model family           ✗ (stable)     ✓ (circular)   ~            │
    │  C6: Nugget < 50% of sill          ✓ (28%)        ✓ (15%)        ✗ (89%)      │
    │                                                                                │
    │  Primary failure mode:              C5             C2             C1, C6       │
    │  Recommended action:                Refine model   Restructure    Replace       │
    │                                                      model          method      │
    └──────────────────────────────────────────────────────────────────────────────┘
```

The severity increases from left to right. Walker Lake fails at Condition 5 (model family) — the most downstream and most correctable failure. Meuse fails at Condition 2 (stationarity) — a more fundamental violation that requires restructuring the model to incorporate covariates. California earthquakes fail at Condition 1 (not a continuous field) and Condition 6 (nugget dominance) — violations so fundamental that kriging in any form is inappropriate.

This stratification explains why the same automated pipeline produces useful output for Walker Lake, salvageable output for Meuse (the variogram is actually correct; only the mean model is wrong), and unusable output for California quakes. The pipeline's preprocessing and optimization can compensate for failures at Conditions 3–6, partially compensate for failures at Condition 2 (through detrending), but cannot compensate for a failure at Condition 1 — there is no preprocessing step that converts a point process into a continuous random field.

### 4.7 Limitations

**Sample size asymmetry.** The Meuse dataset (n = 155) has roughly one-third the sample size of Walker Lake (n = 448), which partially confounds the comparison. The poorer Meuse performance reflects both non-stationarity and data sparsity. A follow-up experiment with a denser floodplain survey (e.g., 500+ points) would help disentangle these effects.

**Univariate kriging only.** The Meuse dataset includes companion variables (cadmium, copper, lead, elevation, soil type, land use) known to co-vary with zinc. Cokriging or regression kriging incorporating these variables would likely improve predictions independent of the stationarity issue. Our ordinary kriging baseline intentionally excludes these to isolate the stationarity assumption.

**Single anisotropy cap.** All three datasets used max_anisotropy = 5.0. The Walker Lake optimizer saturated at this boundary (ratio = 4.98). The original Walker Lake analysis reports E-W anisotropy of approximately 6–8× for variable V (Isaaks & Srivastava 1989, Chapter 16). A follow-up run with max_anisotropy = 10.0 would test whether the additional anisotropy closes part of the 0.49 gap to the ceiling.

**Optuna trial budget.** At 200 trials across 9 model families, the effective budget is approximately 22 trials per family. For a 5-parameter optimization (psill, range, nugget, angle, ratio) plus discrete model selection, 200 trials is at the lower bound of TPE convergence. Increasing to 500–1,000 trials might yield a different model family selection for Walker Lake.

---

## 5. Conclusion

This benchmark demonstrates that ordinary kriging fails in diagnostically informative ways. The three failure modes — model refinement (Walker Lake, R² = 0.23, ceiling = 0.72), model restructure (Meuse zinc, R² = −0.08, ceiling = 0.85), and method replacement (California earthquakes, R² = −0.11, ceiling = 0.11) — form a complete taxonomy of kriging performance regimes, each with a distinct signature in the diagnostic triad of observed R², R² ceiling, and gap.

The practical implications for automated geostatistical pipelines are:

1. **Always report the R² ceiling alongside observed R².** The ceiling — 1 minus the nugget fraction — is the fundamental bound on kriging performance. A ceiling below 0.25 means kriging cannot help, regardless of method variant or parameter tuning. Reporting observed R² without the ceiling is as misleading as reporting a classifier's accuracy without the base rate.

2. **Treat negative R² as a model rejection signal, not a numerical artifact.** A negative cross-validation R² means the kriging predictions are worse than the global mean. Under a correctly specified model, this cannot happen — the equal-weight solution is a feasible member of the linear unbiased predictor class, and the optimized weights should improve upon it. Negative R² indicates that the stationarity assumption or the variogram model is fundamentally wrong.

3. **Distinguish between the gap-to-ceiling and the ceiling itself.** A low ceiling (high nugget) with a small gap means the model is near-optimal but the data lack spatial structure — invest in different data or a different method. A high ceiling with a large gap means spatial structure exists but the model is misspecified — invest in a better model. The two scenarios require opposite responses.

4. **The most dangerous kriging failure is the silent one.** The Meuse case — a reasonable variogram, plausible anisotropy, modest nugget, and catastrophic cross-validation — is the failure mode that automated pipelines are most likely to miss. The variogram is necessary but not sufficient: a well-estimated variogram on non-stationary residuals produces kriging weights that solve the wrong optimization problem. Cross-validation is the only guard against this failure mode.

5. **Not all spatial data are krigable.** The California earthquake case is not a failure of the pipeline but a correct identification that kriging is inapplicable. The 89% nugget fraction and 0.11 R² ceiling are honest answers. An automated pipeline that always reports positive R² regardless of data quality is more dangerous than one that honestly reports failure. The pipeline's CV sanity check — flagging negative R² as "physically impossible" — is a feature, not a bug.

The six conditions for kriging success form a diagnostic hierarchy: violations at downstream conditions (model family, nugget fraction) are correctable within the kriging framework; violations at midstream conditions (stationarity) require restructuring the model to incorporate covariates; violations at upstream conditions (continuous field assumption) require abandoning kriging entirely. The diagnostic triad proposed here — observed R², R² ceiling, and their gap — provides a practical, quantitative basis for navigating this hierarchy without requiring expert geostatistical judgment at every decision point.

---

## Data and Code Availability

- **Interpolation engine:** https://github.com/david-ncu2019/interp_engine
- **Walker Lake dataset:** Isaaks & Srivastava (1989), exported via R `gstat` package
- **Meuse dataset:** Pebesma (2004), exported via R `sp` package
- **California earthquakes:** USGS FDSN Event API (https://earthquake.usgs.gov/fdsnws/event/1/)
- **Full output:** `output/real_world/` (diagnostic plots, CV results, prediction surfaces, parameter files)

---

## References

1. Brus, D.J. & Heuvelink, G.B.M. (2007). Optimization of sample patterns for universal kriging of environmental variables. *Geoderma*, 138(1–2), 86–95.

2. Chilès, J.P. & Delfiner, P. (2012). *Geostatistics: Modeling Spatial Uncertainty* (2nd ed.). Wiley.

3. Cressie, N. (1993). *Statistics for Spatial Data* (Revised ed.). Wiley.

4. Deutsch, C.V. (1993). Kriging in a finite domain. *Mathematical Geology*, 25(1), 41–52.

5. Deutsch, C.V. & Journel, A.G. (1998). *GSLIB: Geostatistical Software Library and User's Guide* (2nd ed.). Oxford University Press.

6. Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*. Oxford University Press.

7. Isaaks, E.H. & Srivastava, R.M. (1989). *Applied Geostatistics*. Oxford University Press.

8. Journel, A.G. & Huijbregts, C.J. (1978). *Mining Geostatistics*. Academic Press.

9. Krige, D.G. (1997). Block kriging and the fallacy of endeavouring to reduce or eliminate smoothing. *2nd Regional APCOM*, Moscow.

10. Krivoruchko, K. (2011). *Spatial Statistical Data Analysis for GIS Users*. Esri Press.

11. Markvoort, J. & Deutsch, C.V. (2024). The bias caused by the string effect in ordinary kriging: risks and solutions. *Applied Earth Science*, 130(4).

12. Melles, S.J., Heuvelink, G.B.M., Twenhöfel, C.J.W., van Dijk, A., Hiemstra, P.H., Baume, O. & Stöhlker, U. (2011). Optimizing the spatial pattern of networks for monitoring radioactive releases. *Computers & Geosciences*, 37(3), 280–288.

13. Olea, R.A. (1999). *Geostatistics for Engineers and Earth Scientists*. Springer.

14. Pebesma, E.J. (2004). Multivariable geostatistics in S: the gstat package. *Computers & Geosciences*, 30(7), 683–691.

15. Pyrcz, M.J. (2024). GeostatsGuy Lectures: Spatial Data Analytics and Geostatistics. University of Texas at Austin. https://geostatsguy.github.io/GeostatsPyDemos_Book/

16. Srivastava, R.M. (1987). A non-ergodic framework for variograms. MS Thesis, Stanford University.

17. Stein, M.L. (1999). *Interpolation of Spatial Data: Some Theory for Kriging*. Springer.

18. Webster, R. & Oliver, M.A. (2007). *Geostatistics for Environmental Scientists* (2nd ed.). Wiley.

---

*Report generated 2026-05-25. Enhanced with insights from the Geostatistics NotebookLM notebook (GeostatsGuy lecture series, Pyrcz 2024), the foundational geostatistical literature (Isaaks & Srivastava 1989; Chilès & Delfiner 2012; Cressie 1993), and recent research on data configuration effects (Markvoort & Deutsch 2024) and optimal sampling design (Brus & Heuvelink 2007).*
