import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

st.set_page_config(page_title="Yield Predictor (V1)", layout="centered")

logo = Image.open("SAPPA_2025_Primary_Green (1).png")

col_logo, col_title = st.columns([1, 3])
with col_logo:
    st.image(logo, width=120)
with col_title:
    st.title("Yield Predictor")
    st.caption("Estimate nut yield per tree from simple measurements.")

st.markdown("---")
st.subheader("Tree inputs")

# Load model artifact
@st.cache_resource
def load_artifact():
    art = joblib.load("yield_model.joblib")
    model = art["model"]
    feature_cols = art["features"]
    sigma = art.get("sigma", None)
    return model, feature_cols, sigma

model, feature_cols, sigma = load_artifact()

# UI inputs
shoot_count = st.number_input("Shoot count", min_value=1, value=100, step=1)
nut_count = st.number_input("Nut count (total)", min_value=0, value=80, step=1)

tree_height = st.number_input("Tree height (m)", min_value=0.1, value=7.0, step=0.1)
tree_depth = st.number_input("Tree depth / width (m)", min_value=0.1, value=3.5, step=0.1)

volume_type = st.selectbox("Volume type (optional)", ["Hedge", "Sphere", "No volume", "Cone", "Cube"], index=0)
cultivar = st.selectbox("Cultivar (optional)", ["Barton", "Choctaw", "Pawnee", "Ukulinga", "Western", "Wichita"], index=0)

# Derived
nuts_per_shoot = nut_count / shoot_count if shoot_count else np.nan
height_depth = tree_height * tree_depth
depth_over_height = tree_depth / tree_height if tree_height else np.nan

# Build input row with ALL possible fields (safe)
row = {
    "ShootCount": float(shoot_count),
    "NutCount": float(nut_count),
    "NutsPerShoot": float(nuts_per_shoot),
    "TreeHeight_m": float(tree_height),
    "TreeDepth_m": float(tree_depth),
    "HeightDepth": float(height_depth),
    "DepthOverHeight": float(depth_over_height),
    "CultivarClean": str(cultivar).strip(),
    "VolumeType": str(volume_type).strip(),
}

# Only pass the features the model expects
X = pd.DataFrame([{k: row.get(k) for k in feature_cols}])

if st.button("Predict"):
    pred = float(model.predict(X)[0])

    # Approximate 95% prediction range using training residual std dev
    if sigma is not None and np.isfinite(sigma):
        z = 1.96  # ~95% interval
        lower = max(0.0, pred - z * sigma)
        upper = pred + z * sigma
        help_text = f"Approx. 95% range: {lower:.2f} – {upper:.2f} kg/tree"
        st.metric("Predicted yield (kg/tree)", f"{pred:.2f}", help=help_text)
    else:
        st.metric("Predicted yield (kg/tree)", f"{pred:.2f}")

    st.write("Inputs used for this prediction:")
    st.dataframe(X)