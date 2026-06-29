import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(page_title="MC Pricer", layout="wide")
st.title("Monte Carlo Option Pricer")
st.markdown("""
A Monte Carlo pricing and Greek estimation toolkit for European, Asian, and
barrier options under geometric Brownian motion.

Use the **sidebar** to navigate between tools:

- **Pricer** — price any supported contract with configurable variance reduction
- **Greeks** — estimate delta, vega, gamma, theta, rho against Black-Scholes analytic values
- **Convergence** — visualise the 1/√N convergence rate; compare pseudo-random vs Sobol QMC
- **Variance Reduction** — benchmark standard error across plain MC, antithetic, and control variate
""")
