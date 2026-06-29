import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rng import RNG
from src.gbm import simulate_gbm_paths
from src.mc import mc_price_asian_arithmetic
from src.variance_reduction import mc_price_asian_arithmetic_antithetic, mc_price_asian_arithmetic_cv

st.set_page_config(page_title="Variance Reduction Benchmark", layout="wide")
st.title("Variance Reduction Benchmark")

with st.sidebar:
    st.header("Parameters")
    S0    = st.number_input("Spot S0",    value=100.0, min_value=0.01, step=1.0)
    K     = st.number_input("Strike K",   value=95.0,  min_value=0.01, step=1.0)
    r     = st.number_input("Rate r",     value=0.015, min_value=-0.10, max_value=0.30, step=0.005, format="%.3f")
    T     = st.number_input("Maturity T", value=1.0,   min_value=0.01, step=0.25)
    sigma = st.number_input("Vol σ",      value=0.25,  min_value=0.01, max_value=2.0,  step=0.01, format="%.2f")
    n_paths = st.select_slider("Paths", options=[10_000, 50_000, 100_000, 200_000], value=100_000)
    steps   = st.select_slider("Steps", options=[32, 64, 128], value=64)
    seed    = st.number_input("Seed", value=42, step=1)

if st.button("Run benchmark"):
    with st.spinner("Running three methods..."):
        rng_plain = RNG(seed=int(seed))
        paths     = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, rng=rng_plain)
        plain     = mc_price_asian_arithmetic(paths, K, r, T, "call")

        rng_anti  = RNG(seed=int(seed))
        anti      = mc_price_asian_arithmetic_antithetic(S0, K, r, T, sigma, steps, n_paths, rng_anti, "call")

        rng_cv    = RNG(seed=int(seed))
        cv        = mc_price_asian_arithmetic_cv(S0, K, r, T, sigma, steps, n_paths, rng_cv, "call")

    methods = ["Plain MC", "Antithetic", "Control variate"]
    stderrs = [plain["stderr"], anti["stderr"], cv["stderr"]]
    prices  = [plain["price"],  anti["price"],  cv["price"]]
    ratios  = [plain["stderr"] / s for s in stderrs]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["steelblue", "seagreen", "coral"]
    bars   = ax.bar(methods, stderrs, color=colors, width=0.5)
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stderrs) * 0.01,
                f"×{ratio:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Standard error")
    ax.set_title("Stderr by variance reduction method (Asian arithmetic call)")
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(pd.DataFrame({
        "Method":           methods,
        "Price":            [f"{p:.4f}" for p in prices],
        "Std Error":        [f"{s:.6f}" for s in stderrs],
        "Reduction factor": [f"{ratio:.2f}×" for ratio in ratios],
    }).set_index("Method"), use_container_width=True)
