import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor


# =========================
# CONFIG
# =========================
EXCEL_FILE = "Yield estimate data 2025 harvest 3-11-2025 CW.xlsx"
SHEET_NAME = "RawData"  # set to "CleanData" or your sheet name if needed
TABLE_EXPORT_CSV = None  # if you export Table1 to CSV, put path here

MODEL_OUT = "yield_model.joblib"  # saved best model

RANDOM_STATE = 42


# =========================
# LOAD DATA
# =========================
def load_data():
    if TABLE_EXPORT_CSV:
        df = pd.read_csv(TABLE_EXPORT_CSV)
        return df

    # fallback: read Excel sheet (adjust if your clean table is on a specific sheet)
    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    return df


# =========================
# CLEAN + FEATURES
# =========================
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Rename if needed (match your columns)
    rename_map = {
        "Nut yield": "NutYieldKgTree",
        "Nut yield ": "NutYieldKgTree",
        "Tree hight": "TreeHeight_m",
        "Tree width": "TreeDepth_m",
        "Shoot count": "ShootCount",
        "Nut count": "NutCount",
        "In-row spacing": "InRowSpacing_m",
        "Row width": "RowWidth_m",
        "Hegde row": "HedgeFlag",
    }
    df = df.rename(columns=rename_map)

    # strip text
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # numeric coercion
    for c in ["NutYieldKgTree", "TreeHeight_m", "TreeDepth_m", "ShootCount", "NutCount", "InRowSpacing_m", "RowWidth_m"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Cultivar clean (optional)
    if "Cultivar" in df.columns and "CultivarClean" not in df.columns:
        df["CultivarClean"] = (
            df["Cultivar"]
            .astype(str)
            .str.replace(r"\(Young\)|\(Old\)", "", regex=True)
            .str.strip()
        )

    # VolumeType from flags (if flags exist)
    def get_volume_type(row):
        if row.get("HedgeFlag") == "~":
            return "Hedge"
        if row.get("SphereFlag") == "~" or row.get("Sphere") == "~":
            return "Sphere"
        if row.get("ConeFlag") == "~" or row.get("Cone") == "~":
            return "Cone"
        if row.get("CubeFlag") == "~" or row.get("Cube") == "~":
            return "Cube"
        return "No volume"

    if "VolumeType" not in df.columns:
        df["VolumeType"] = df.apply(get_volume_type, axis=1)

    # Derived features
    df["NutsPerShoot"] = df["NutCount"] / df["ShootCount"]
    df["HeightDepth"] = df["TreeHeight_m"] * df["TreeDepth_m"]
    df["DepthOverHeight"] = df["TreeDepth_m"] / df["TreeHeight_m"]

    # Valid rows only
    df = df[
        (df["ShootCount"] > 0)
        & (df["NutCount"] >= 0)
        & (df["NutYieldKgTree"] > 0)
        & (df["TreeHeight_m"] > 0)
        & (df["TreeDepth_m"] > 0)
    ].copy()

    return df


# =========================
# BUILD PIPELINE
# =========================
def make_pipeline(feature_cols, estimator):
    # infer numeric vs categorical
    # (we treat strings as categorical)
    num_cols = [c for c in feature_cols if c not in ["CultivarClean", "VolumeType"]]
    cat_cols = [c for c in feature_cols if c in ["CultivarClean", "VolumeType"]]

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            num_cols
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cat_cols
        ))

    prep = ColumnTransformer(transformers)

    return Pipeline([
        ("prep", prep),
        ("model", estimator)
    ])


def cv_metrics(model, X, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2")
    )
    return {
        "MAE": float(-scores["test_neg_mean_absolute_error"].mean()),
        "RMSE": float(-scores["test_neg_root_mean_squared_error"].mean()),
        "R2": float(scores["test_r2"].mean())
    }


# =========================
# MAIN
# =========================
def run():
    df = prepare(load_data())

    # Candidate feature sets (increasing sophistication)
    feat_base = ["ShootCount", "NutsPerShoot", "TreeHeight_m", "TreeDepth_m"]
    feat_interactions = ["ShootCount", "NutsPerShoot", "TreeHeight_m", "TreeDepth_m", "HeightDepth", "DepthOverHeight"]
    feat_with_cat = feat_interactions + ["CultivarClean", "VolumeType"]

    candidates = [
        ("Linear base", feat_base, LinearRegression()),
        ("Ridge base", feat_base, Ridge(alpha=1.0)),
        ("Linear + interactions", feat_interactions, LinearRegression()),
        ("RF + interactions", feat_interactions, RandomForestRegressor(
            n_estimators=400, random_state=RANDOM_STATE
        )),
        ("RF + interactions + cat", feat_with_cat, RandomForestRegressor(
            n_estimators=500, random_state=RANDOM_STATE
        )),
    ]

    results = []

    # 1) Overall comparison
    for name, feats, est in candidates:
        X = df[feats]
        y = df["NutYieldKgTree"]
        pipe = make_pipeline(feats, est)
        m = cv_metrics(pipe, X, y)
        results.append({"Scope": "ALL", "Model": name, "Features": ", ".join(feats), **m})

    # 2) Split by VolumeType (Hedge vs Sphere) — often boosts accuracy
    for vt in ["Hedge", "Sphere"]:
        sub = df[df["VolumeType"] == vt].copy()
        if len(sub) < 20:
            continue

        for name, feats, est in candidates:
            # If we already filtered by a volume type, drop VolumeType to avoid silly one-hot on constant
            feats2 = [f for f in feats if f != "VolumeType"]
            X = sub[feats2]
            y = sub["NutYieldKgTree"]
            pipe = make_pipeline(feats2, est)
            m = cv_metrics(pipe, X, y)
            results.append({"Scope": vt, "Model": name, "Features": ", ".join(feats2), **m})

    out = pd.DataFrame(results).sort_values(["Scope", "R2", "MAE"], ascending=[True, False, True])
    print(out.to_string(index=False))

    # Choose best overall model (Scope=ALL, max R2 then min MAE)
    all_only = out[out["Scope"] == "ALL"].copy()
    best = all_only.sort_values(["R2", "MAE"], ascending=[False, True]).iloc[0]
    best_name = best["Model"]

    # Rebuild and fit best on all data, then save
    # Find candidate spec:
    best_spec = [c for c in candidates if c[0] == best_name][0]
    _, best_feats, best_est = best_spec
    X_best = df[best_feats]
    y_best = df["NutYieldKgTree"]

    best_pipe = make_pipeline(best_feats, best_est)
    best_pipe.fit(X_best, y_best)

    # Estimate residual std dev (used for prediction range in app)
    y_pred_best = best_pipe.predict(X_best)
    residuals = y_best - y_pred_best
    sigma = float(residuals.std(ddof=1))

    joblib.dump(
        {"model": best_pipe, "features": best_feats, "sigma": sigma},
        MODEL_OUT
    )
    print(f"\nSaved best model to: {MODEL_OUT}\nBest: {best_name} | R2={best['R2']:.3f} | MAE={best['MAE']:.3f} | Sigma={sigma:.3f}")


if __name__ == "__main__":
    run()