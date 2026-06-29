import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rng import RNG
from src.greeks import mc_european_greeks

st.set_page_config(page_title="Greeks Dashboard", layout="wide")
st.title("Greeks Dashboard")

with st.sidebar:
    st.header("Parameters")
    S0    = st.number_input("Spot S0",    value=100.0, min_value=0.01, step=1.0)
    K     = st.number_input("Strike K",   value=100.0, min_value=0.01, step=1.0)
    r     = st.number_input("Rate r",     value=0.05,  min_value=-0.10, max_value=0.30, step=0.005, format="%.3f")
    T     = st.number_input("Maturity T", value=1.0,   min_value=0.01, step=0.25)
    sigma = st.number_input("Vol σ",      value=0.20,  min_value=0.01, max_value=2.0,  step=0.01, format="%.2f")
    option_type = st.radio("Side", ["call", "put"], horizontal=True)
    n_paths = st.select_slider("Paths", options=[10_000, 50_000, 100_000, 300_000], value=100_000)
    steps   = st.select_slider("Steps", options=[32, 64, 128], value=64)
    seed    = st.number_input("Seed", value=42, step=1)

if st.button("Estimate Greeks"):
    with st.spinner("Running MC..."):
        rng    = RNG(seed=int(seed))
        result = mc_european_greeks(S0, K, r, T, sigma, option_type, steps, n_paths, rng)

    greeks_order = ["delta", "vega", "gamma", "theta", "rho"]
    mc_vals, mc_errs, an_vals, labels = [], [], [], []

    for g in greeks_order:
        block = result[g]
        if "pathwise" in block:
            mc_val = block["pathwise"]
            mc_err = block["pathwise_stderr"] * 1.96
        else:
            mc_val = block["finite_difference"]
            mc_err = block["finite_difference_stderr"] * 1.96
        mc_vals.append(mc_val)
        mc_errs.append(mc_err)
        an_vals.append(block["analytic"])
        labels.append(g.capitalize())

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width/2, mc_vals, width, yerr=mc_errs, label="MC estimate",
           color="steelblue", capsize=4, error_kw={"elinewidth": 1.2})
    ax.bar(x + width/2, an_vals, width, label="Analytic (BS)",
           color="coral", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Greek value")
    ax.set_title("MC Greeks vs Black-Scholes analytic")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    st.pyplot(fig)
    plt.close(fig)

    rows = []
    for g in greeks_order:
        block  = result[g]
        mc_val = block.get("pathwise", block["finite_difference"])
        an_val = block["analytic"]
        rows.append({
            "Greek":     g.capitalize(),
            "MC":        round(mc_val, 5),
            "Analytic":  round(an_val, 5),
            "Abs error": round(abs(mc_val - an_val), 5),
            "Rel error": f"{abs(mc_val - an_val) / (abs(an_val) + 1e-12) * 100:.3f}%",
        })

    st.dataframe(pd.DataFrame(rows).set_index("Greek"), use_container_width=True)
