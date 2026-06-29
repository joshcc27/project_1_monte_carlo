import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from src.rng import RNG, QMCGenerator
from src.gbm import simulate_gbm_paths
from src.mc import mc_price_european
from src.bs_analytics import bs_price

st.set_page_config(page_title="Convergence Analysis", layout="wide")
st.title("Convergence Analysis")

with st.sidebar:
    st.header("Parameters")
    S0    = st.number_input("Spot S0",    value=100.0, min_value=0.01, step=1.0)
    K     = st.number_input("Strike K",   value=100.0, min_value=0.01, step=1.0)
    r     = st.number_input("Rate r",     value=0.05,  min_value=-0.10, max_value=0.30, step=0.005, format="%.3f")
    T     = st.number_input("Maturity T", value=1.0,   min_value=0.01, step=0.25)
    sigma = st.number_input("Vol σ",      value=0.20,  min_value=0.01, max_value=2.0,  step=0.01, format="%.2f")
    steps     = st.select_slider("Steps", options=[32, 64, 128], value=64)
    seed      = st.number_input("Seed", value=42, step=1)
    n_min     = st.number_input("Min paths", value=500,     step=100,  min_value=100)
    n_max     = st.number_input("Max paths", value=100_000, step=1000, min_value=1000)
    n_points  = st.slider("Points on curve", min_value=5, max_value=20, value=12)
    show_qmc  = st.checkbox("Overlay QMC (Sobol)", value=True)


@st.cache_data
def convergence_curve(S0, K, r, T, sigma, steps, seed, n_min, n_max, n_points, use_qmc):
    bs_ref = bs_price(S0, K, T, r, sigma, "call")
    counts = np.unique(np.logspace(np.log10(n_min), np.log10(n_max), n_points).astype(int))

    mc_errors, qmc_errors = [], []
    for n in counts:
        rng   = RNG(seed=seed)
        paths = simulate_gbm_paths(S0, r, sigma, T, steps, int(n), rng=rng)
        price = mc_price_european(paths[:, -1], K, r, T, "call")["price"]
        mc_errors.append(abs(price - bs_ref))

        if use_qmc:
            qmc   = QMCGenerator(dim=steps, seed=seed)
            paths = simulate_gbm_paths(S0, r, sigma, T, steps, int(n), rng=qmc)
            price = mc_price_european(paths[:, -1], K, r, T, "call")["price"]
            qmc_errors.append(abs(price - bs_ref))

    return counts, mc_errors, qmc_errors, bs_ref


if st.button("Run convergence"):
    with st.spinner("Simulating across path counts..."):
        counts, mc_errs, qmc_errs, bs_ref = convergence_curve(
            S0, K, r, T, sigma, steps, int(seed), int(n_min), int(n_max), n_points, show_qmc
        )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.loglog(counts, mc_errs, "o-", color="steelblue", label="MC (pseudo-random)")
    if show_qmc:
        ax.loglog(counts, qmc_errs, "s--", color="coral", label="QMC (Sobol)")

    ref_x = np.array([counts[0], counts[-1]])
    ref_y = mc_errs[0] * np.sqrt(counts[0] / ref_x)
    ax.loglog(ref_x, ref_y, "k:", linewidth=1, label="1/√N reference")

    ax.set_xlabel("Number of paths")
    ax.set_ylabel("|MC price − BS price|")
    ax.set_title(f"Convergence to BS price = {bs_ref:.4f}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
