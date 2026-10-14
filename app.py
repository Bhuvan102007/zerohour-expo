import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from detect import make_sample_lightcurve, find_transit

st.title("Exoplanet Hunter: Transit Detector")
st.caption("Uses a documented synthetic light curve with a planted transit (period 3.5 d, depth 1%).")

noise = st.slider("Noise level", 0.0, 0.005, 0.001, 0.0005)
t, flux = make_sample_lightcurve()
flux = flux + np.random.default_rng(1).normal(0, max(noise - 0.001, 0), flux.size)

r = find_transit(t, flux)
st.success(f"Best candidate: period {r['period']:.2f} days, depth {r['depth']*100:.2f}%")
st.info("Expected answer: period 3.5 days, depth 1%. A match means the detector works.")

fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].scatter(t, flux, s=2)
ax[0].set_title("Light curve (raw)")
ax[0].set_xlabel("Time (days)")
ax[0].set_ylabel("Relative brightness")
ax[1].plot(r["periods"], r["power"])
ax[1].axvline(r["period"], color="red", linestyle="--")
ax[1].set_title("Period search (peak = candidate)")
ax[1].set_xlabel("Period (days)")
ax[1].set_ylabel("BLS power")
fig.tight_layout()
st.pyplot(fig)

with st.expander("Assumptions"):
    st.write("Synthetic data with a known transit, simple detrending (centering), Box Least Squares search over 0.5–10 day periods. Real data needs stronger detrending.")