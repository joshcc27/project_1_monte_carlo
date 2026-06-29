# Monte Carlo Pricer

## Overview

Monte Carlo-based pricing and Greek estimation toolkit for European, Asian arithmetic, and barrier options under geometric Brownian motion. Includes analytic Black-Scholes benchmarks, variance-reduction techniques, an interactive Streamlit web UI, and extensive pytest coverage.

## Repository structure

```text
app.py                          # Streamlit entry point (landing page)
pages/
    1_Pricer.py                 # Option pricing with payoff histogram
    2_Greeks.py                 # Greeks dashboard (MC vs analytic)
    3_Convergence.py            # 1/√N convergence analysis
    4_Variance_Reduction.py     # Variance reduction benchmark
src/
    bs_analytics.py             # Black-Scholes pricing and analytic Greeks
    gbm.py                      # GBM path simulation
    mc.py                       # MC pricers returning {price, stderr, ci_low, ci_high}
    payoffs.py                  # European, Asian arithmetic, and barrier payoff utilities
    variance_reduction.py       # Antithetic variates and control variate
    greeks.py                   # MC European Greeks (pathwise + finite differences)
    asian_geometric.py          # Closed-form geometric Asian pricing (CV control)
    rng.py                      # RNG wrapper (pseudo-random) and QMCGenerator (Sobol)
tests/                          # Pytest suite covering analytics, convergence, Greeks, VR
requirements.txt                # Python dependencies
pyproject.toml                  # Build config and optional dev dependencies
```

## Requirements

- Python 3.10+
- Install dependencies: `pip install -r requirements.txt`

## Running the web UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The sidebar exposes four pages:

| Page | Purpose |
| --- | --- |
| **Pricer** | Price any supported contract; choose variance reduction method; view payoff histogram |
| **Greeks** | Estimate all five Greeks via MC; compare to Black-Scholes analytic values |
| **Convergence** | Visualise 1/√N convergence rate; overlay pseudo-random vs Sobol QMC |
| **Variance Reduction** | Benchmark stderr across plain MC, antithetic, and control variate for an Asian call |

## Running tests

```bash
python -m pytest
```

## Library usage

```python
from src.rng import RNG
from src.gbm import simulate_gbm_paths
from src.mc import mc_price_european

S0, K, r, T, sigma = 100, 105, 0.02, 1.0, 0.2
rng = RNG(seed=42)
paths = simulate_gbm_paths(S0, r, sigma, T, steps=128, n_paths=50_000, rng=rng)
result = mc_price_european(paths[:, -1], K, r, T, "call")
print(result)
# {'price': ..., 'stderr': ..., 'ci_low': ..., 'ci_high': ..., 'n_paths': 50000, 'extra': {}}
```

## Variance reduction

- **Antithetic variates**: `mc_price_european_antithetic` / `mc_price_asian_arithmetic_antithetic` — pair each path with its sign-flipped counterpart to cancel first-order noise
- **Control variate**: `mc_price_asian_arithmetic_cv` — uses the closed-form geometric Asian price as a control; `control_variate(X, Y, EY)` for custom pairings

Typical stderr reduction on an Asian arithmetic call (n=100 000, σ=0.25): antithetic ≈1.4×, control variate ≈8×.

## Greeks

`mc_european_greeks` returns all five Greeks with both MC estimates and Black-Scholes analytic values:

| Greek | MC method |
| --- | --- |
| Delta | Pathwise + finite difference |
| Vega | Pathwise + finite difference |
| Gamma | Finite difference |
| Theta | Finite difference |
| Rho | Finite difference |

Common random numbers are used across bumped paths to reduce finite-difference noise.
