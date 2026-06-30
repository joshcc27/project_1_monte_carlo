import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from src.binomial import price_binomial
from src.bs_analytics import bs_price

st.set_page_config(page_title="Binomial Tree", layout="wide")
st.title("Binomial Tree Pricer")

with st.sidebar:
    st.header("Market Parameters")
    S0    = st.number_input("Spot S0",    value=100.0, min_value=0.01, step=1.0)
    K     = st.number_input("Strike K",   value=100.0, min_value=0.01, step=1.0)
    r     = st.number_input("Rate r",     value=0.05,  min_value=-0.10, max_value=0.30, step=0.005, format="%.3f")
    T     = st.number_input("Maturity T", value=1.0,   min_value=0.01, step=0.25)
    sigma = st.number_input("Vol σ",      value=0.20,  min_value=0.01, max_value=2.0,  step=0.01, format="%.2f")

    st.subheader("Contract")
    option_type = st.radio("Option type", ["call", "put"], horizontal=True)
    steps = st.select_slider("Steps", options=[10, 50, 100, 200, 500, 1000], value=200)

if st.button("Price"):
    with st.spinner("Pricing..."):
        binom_price = price_binomial(S0, K, r, T, sigma, steps=steps, option_type=option_type)
        bs_ref      = bs_price(S0, K, T, r, sigma, option_type)
        abs_error   = abs(binom_price - bs_ref)

    col1, col2, col3 = st.columns(3)
    col1.metric("Binomial Price", f"{binom_price:.6f}")
    col2.metric("Black-Scholes", f"{bs_ref:.6f}")
    col3.metric("Abs Error vs BS", f"{abs_error:.6f}")

    # Convergence chart across step counts.
    st.subheader("Convergence to Black-Scholes")
    step_counts = np.unique(np.logspace(1, 3, 30).astype(int))
    prices = [price_binomial(S0, K, r, T, sigma, steps=int(n), option_type=option_type)
              for n in step_counts]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(step_counts, prices, "o-", color="steelblue", markersize=4, label="Binomial")
    ax.axhline(bs_ref, color="tomato", linestyle="--", linewidth=1.5,
               label=f"Black-Scholes = {bs_ref:.4f}")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Option Price")
    ax.set_title(f"CRR Binomial convergence — {option_type} (S0={S0}, K={K}, T={T}, r={r}, σ={sigma})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
