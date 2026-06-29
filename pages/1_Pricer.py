import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

from src.rng import RNG
from src.gbm import simulate_gbm_paths
from src.mc import mc_price_european, mc_price_asian_arithmetic, mc_price_barrier
from src.variance_reduction import (
    mc_price_european_antithetic,
    mc_price_asian_arithmetic_antithetic,
    mc_price_asian_arithmetic_cv,
)
from src.bs_analytics import bs_price
from src.payoffs import european_payoff, asian_arithmetic_payoff, barrier_payoff

st.set_page_config(page_title="Option Pricer", layout="wide")
st.title("Option Pricer")

with st.sidebar:
    st.header("Parameters")

    st.subheader("Market")
    S0    = st.number_input("Spot S0",    value=100.0, min_value=0.01, step=1.0)
    K     = st.number_input("Strike K",   value=100.0, min_value=0.01, step=1.0)
    r     = st.number_input("Rate r",     value=0.02,  min_value=-0.10, max_value=0.30, step=0.005, format="%.3f")
    T     = st.number_input("Maturity T", value=1.0,   min_value=0.01, step=0.25)
    sigma = st.number_input("Vol σ",      value=0.20,  min_value=0.01, max_value=2.0,  step=0.01, format="%.2f")

    st.subheader("Contract")
    contract    = st.selectbox("Type", ["European", "Asian arithmetic", "Barrier"])
    option_type = st.radio("Side", ["call", "put"], horizontal=True)

    barrier = None
    barrier_type = None
    if contract == "Barrier":
        barrier      = st.number_input("Barrier level", value=85.0, step=1.0)
        barrier_type = st.selectbox("Barrier type", ["down_out", "down_in", "up_out", "up_in"])

    st.subheader("Simulation")
    n_paths = st.select_slider("Paths", options=[1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000], value=100_000)
    steps   = st.select_slider("Steps", options=[8, 16, 32, 64, 128, 252], value=64)
    seed    = st.number_input("Seed", value=42, step=1)

    st.subheader("Variance reduction")
    vr_options = ["None", "Antithetic"]
    if contract == "Asian arithmetic":
        vr_options.append("Control variate")
    vr = st.selectbox("Method", vr_options)

if st.button("Price"):
    with st.spinner("Simulating..."):
        t0  = time.perf_counter()
        rng = RNG(seed=int(seed))
        paths = None

        if contract == "European":
            if vr == "Antithetic":
                result = mc_price_european_antithetic(S0, K, r, T, sigma, steps, n_paths, rng, option_type)
            else:
                paths  = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, rng=rng)
                result = mc_price_european(paths[:, -1], K, r, T, option_type)

        elif contract == "Asian arithmetic":
            if vr == "Antithetic":
                result = mc_price_asian_arithmetic_antithetic(S0, K, r, T, sigma, steps, n_paths, rng, option_type)
            elif vr == "Control variate":
                result = mc_price_asian_arithmetic_cv(S0, K, r, T, sigma, steps, n_paths, rng, option_type)
            else:
                paths  = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, rng=rng)
                result = mc_price_asian_arithmetic(paths, K, r, T, option_type)

        elif contract == "Barrier":
            paths  = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, rng=rng)
            result = mc_price_barrier(paths, K, r, T, barrier, barrier_type, option_type)

        elapsed_ms = (time.perf_counter() - t0) * 1000

    st.subheader("Result")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Price",       f"{result['price']:.4f}")
    col2.metric("Std Error",   f"{result['stderr']:.5f}")
    col3.metric("95% CI low",  f"{result['ci_low']:.4f}")
    col4.metric("95% CI high", f"{result['ci_high']:.4f}")
    col5.metric("Runtime",     f"{elapsed_ms:.0f} ms")

    if contract == "European":
        bs = bs_price(S0, K, T, r, sigma, option_type)
        st.caption(f"Black-Scholes analytic: **{bs:.4f}**  |  Error: {abs(result['price'] - bs):.5f}")

    if paths is not None:
        disc = np.exp(-r * T)
        if contract == "European":
            payoffs = disc * european_payoff(paths[:, -1], K, option_type)
        elif contract == "Asian arithmetic":
            payoffs = disc * asian_arithmetic_payoff(paths, K, option_type)
        elif contract == "Barrier":
            payoffs = disc * barrier_payoff(paths, K, barrier, barrier_type, option_type)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(payoffs[payoffs > 0], bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.axvline(result['price'], color="crimson", linewidth=1.5, label=f"Mean = {result['price']:.3f}")
        ax.set_xlabel("Discounted payoff")
        ax.set_ylabel("Frequency")
        ax.set_title("Payoff distribution (in-the-money paths only)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Histogram not shown for antithetic/CV paths.")
