# winrate_regression_scikit.py
# Adds random wins/losses per deck, computes wr = wins/(wins+losses),
# then selects the BEST regressor via 5-fold CV (R²), evaluates on a
# held-out test split, saves plots.
#
# Models:
#   - Ridge (linear)
#   - Polynomial Ridge (degree=2 on numeric branch; categoricals OHE)
#   - kNN Regressor 11 Neighbours
#
# Saves:
#   - cv_r2_bar.png
#   - ridge_coef_top.png               (if best is Ridge)
#   - poly_ridge_coef_top.png          (if best is Poly Ridge)
#   - pred_vs_true_scatter.png

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# --- Paths (no args) ---
BASE_DIR = Path(__file__).resolve().parent                  # .../mtg_tools/ml
DB_PATH  = (BASE_DIR.parent / "db" / "mtgcore_demo.db").resolve()


# --- Features (same canonical lists) ---
CATEGORICAL = ["dominant_type", "main_tribe"]
NUMERIC = [
    "avg_cmc",
    "cmc_0","cmc_1","cmc_2","cmc_3","cmc_4","cmc_5","cmc_6","cmc_7_plus",
    "color_W","color_U","color_B","color_R","color_G","color_C"
]

# ---------- Data access ----------
def fetch_df(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Loads features from deck_stats. No labels needed for regression.
    """
    df = pd.read_sql_query(f"""
        SELECT deck_id, {', '.join(NUMERIC + CATEGORICAL)}
        FROM deck_stats
    """, conn)

    df[CATEGORICAL] = df[CATEGORICAL].fillna("None")
    df[NUMERIC] = df[NUMERIC].fillna(0.0)
    return df.reset_index(drop=True)

# ---------- Add synthetic winrate ----------
# First add_random_wr is terrible -> Will only lead to R2 -> 0
# def add_random_wr(df, seed: int = 42, min_games: int = 20, max_games: int = 60):
#     """
#     Adds columns: wins, losses, wr to the dataframe (random because no data).
#     Reproducible via `seed`.
#     """
#     rng = np.random.default_rng(seed)
#     n = len(df)

#     # Random win ratio to avoid overly scattered data -> Beta centered near 0.5
#     p = rng.beta(2.0, 2.0, size=n)  # symmetric, broad

#     # Sample number of games per deck
#     n_games = rng.integers(min_games, max_games + 1, size=n)

#     # Wins - Binomial(n_games, p), losses the remainder
#     wins = rng.binomial(n_games, p)
#     losses = n_games - wins
#     wr = wins / np.maximum(n_games, 1)

#     df_wr = df.copy()
#     df_wr["wins"] = wins
#     df_wr["losses"] = losses
#     df_wr["wr"] = wr.astype(float)
#     return df_wr

def add_random_wr(
    df,
    seed: int = 42,
    min_games: int = 20,
    max_games: int = 60,
    signal_strength: float = 1.2, # raise -> raise R2
    noise_std: float = 0.02,      # lower noise -> easier regression
    cat_effect_scale: float = 0.02,  # magnitude for random cat effects):
    lo: float = 0.1, # tune lowest deck win probability
    hi: float = 0.9, # tune highest deck win probability
):
    """
    Synthesizes wins/losses/wr with a stronger, learnable pattern:
      - Lower avg_cmc -> slightly higher WR (negative slope on z-scored CMC)
      - Curve shape effect from CMC bucket proportions
      - Small color effects
      - (Stable) Random categorical bumps for every seen dominant_type / main_tribe

    All randomness is reproducible via `seed`.
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    df_wr = df.copy()

    # --- Numeric effects ---
    # avg_cmc (z-scored)
    avg_cmc_z = StandardScaler().fit_transform(df_wr[["avg_cmc"]]).ravel()
    slope_cmc = -0.06  # negative slope: cheaper decks slightly better

    # CMC curve shape: convert bucket counts to proportions and weight them
    cmc_cols = ["cmc_0","cmc_1","cmc_2","cmc_3","cmc_4","cmc_5","cmc_6","cmc_7_plus"]
    cmc_counts = df_wr[cmc_cols].to_numpy(dtype=float)
    totals = np.maximum(cmc_counts.sum(axis=1, keepdims=True), 1.0)
    cmc_props = cmc_counts / totals
    # Prefer lower curve slightly, penalize very top-heavy curves
    w_cmc = np.array([0.02, 0.03, 0.02, 0.00, -0.01, -0.02, -0.03, -0.04])
    curve_effect = (cmc_props @ w_cmc)

    # Colors: tiny effects that roughly cancel on average
    color_cols = ["color_W","color_U","color_B","color_R","color_G","color_C"]
    color_mat = df_wr[color_cols].to_numpy(dtype=float)
    w_colors = np.array([+0.012, +0.006, +0.000, +0.008, +0.005, -0.004])
    color_effect = np.matmul(color_mat, w_colors)

    # --- Categorical effects (stable random per seen category) ---
    def stable_random_map(values: pd.Series, lo=-cat_effect_scale, hi=+cat_effect_scale):
        cats = sorted(set(values.fillna("None")))
        # derive a deterministic seed per category to keep effects stable across runs with same `seed`
        effects = {}
        for i, c in enumerate(cats):
            local_rng = np.random.default_rng([seed, i, 1337])  # mix in index for determinism
            effects[c] = float(local_rng.uniform(lo, hi))
        return np.array([effects.get(v if pd.notna(v) else "None", 0.0) for v in values])

    dom_effect = stable_random_map(df_wr["dominant_type"])
    tribe_effect = stable_random_map(df_wr["main_tribe"])

    # --- Combine signal + noise ---
    base = 0.50
    linear_score = (
        slope_cmc * avg_cmc_z
        + curve_effect
        + color_effect
        + dom_effect
        + tribe_effect
    )

    score = base + signal_strength * linear_score + rng.normal(0, noise_std, size=n)  
    p = np.clip(score, lo, hi)

    # --- Sample games and wins ---
    n_games = rng.integers(min_games, max_games + 1, size=n)
    wins = rng.binomial(n_games, p)
    losses = n_games - wins
    wr = wins / np.maximum(n_games, 1)

    df_wr = df.copy()
    df_wr["wins"] = wins
    df_wr["losses"] = losses
    df_wr["wr"] = wr.astype(float)
    return df_wr

# ---------- Simple feature-name helper (matches preprocessors) ----------
def get_feature_names(ct, categorical=CATEGORICAL, numeric=NUMERIC):
    """
    Works with our blocks:
      - "cat" = OneHotEncoder
      - "num" = passthrough or scaler
      - "num_poly" = Pipeline(poly -> scaler)
    """
    names = []

    # Categorical OHE names
    ohe = ct.named_transformers_.get("cat")
    if ohe is not None and hasattr(ohe, "categories_"):
        for col, cats in zip(categorical, ohe.categories_):
            names.extend([f"{col}={c}" for c in cats])
    else:
        names.extend(list(categorical))

    # Numeric branch
    if "num_poly" in ct.named_transformers_:
        poly_pipe = ct.named_transformers_["num_poly"]
        poly = getattr(poly_pipe, "named_steps", {}).get("poly", None)
        if poly is not None and hasattr(poly, "get_feature_names_out"):
            try:
                poly_names = poly.get_feature_names_out(numeric)
            except TypeError:
                poly_names = poly.get_feature_names_out()
            names.extend(list(poly_names))
        else:
            names.extend(list(numeric))
    else:
        names.extend(list(numeric))

    return names

# ---------- Plots ----------
def plot_top_coefs(pipeline, out_path: Path, top_k=20, title="Ridge coefficients (top)"):
    ridge = pipeline.named_steps["clf"]
    pre = pipeline.named_steps["pre"]
    names = get_feature_names(pre)
    coefs = np.abs(ridge.coef_.ravel())
    order = np.argsort(coefs)[::-1][:top_k]
    plt.figure()
    plt.barh(range(len(order)), coefs[order][::-1])
    plt.yticks(range(len(order)), [names[i] for i in order][::-1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_cv_r2_bar(cv_scores: dict, out_path: Path):
    names = list(cv_scores.keys())
    vals = [cv_scores[n] for n in names]
    plt.figure()
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=15)
    plt.ylabel("CV R² (mean of 5 folds)")
    plt.title("Model comparison (5-fold CV, R²)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# ---------- Models (only what we need) ----------
def build_models(seed: int = 42):
    
    # 1) Ridge Regression: OHE(cats) + Scale(nums)
    pre_ridge = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),  
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    ridge = Pipeline([
        ("pre", pre_ridge),
        ("clf", Ridge(random_state=seed)),
    ])

    # 2) Ridge Regression + Polynomial PolynomialFeatures: OHE(cats) + Scale(Poly(nums))
    pre_poly = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num_poly", Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
            ]), NUMERIC),
        ]
    )
    ridge_poly = Pipeline([
        ("pre", pre_poly),
        ("clf", Ridge(random_state=seed)),
    ])

    # 3) kNN Regressor: dense OHE(cats) + Scale(num)
    pre_knn = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ],
        sparse_threshold=0.0  # force dense array
    )
    knn_11 = Pipeline([
        ("pre", pre_knn),
        ("clf", KNeighborsRegressor(n_neighbors=11, weights="distance")),
    ])

    return {
        "Ridge": ridge,
        "PolyRidge": ridge_poly,
        "kNN": knn_11,
    }

# ---------- Plot ----------

def plot_pred_vs_true(y_true, y_pred, out_path: Path, title="Predicted vs True Winrate"):
    plt.figure()
    plt.scatter(y_true, y_pred, s=12, alpha=0.7, label="Predictions")
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color = "orange", label="y = x (perfect prediction)")
    plt.xlabel("True wr")
    plt.ylabel("Predicted wr")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# ---------- Evaluation ----------
def evaluate_models(df: pd.DataFrame, seed: int = 42):
    # Synth WR
    df_wr = add_random_wr(df, seed=seed)
    X = df_wr[CATEGORICAL + NUMERIC]
    y = df_wr["wr"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    models = build_models(seed)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    scorer = "r2"

    print(f"CV metric: R²\n")
    header = f'{"Model":<12} {"CV R2":>8}'
    print(header)
    print("-" * len(header))

    cv_scores = {}

    for name, pipe in models.items():
        score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scorer, n_jobs=-1).mean()
        cv_scores[name] = score
        print(f"{name:<12} {score:>8.3f}")

    plot_cv_r2_bar(cv_scores, BASE_DIR / "cv_r2_bar.png")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\nBest by CV R²: {best_name}\n")
    
    # Fit best on train, evaluate on test
    best_model = models[best_name].fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    header = f'{"Model":<12} {"R2":>8} {"MAE":>8} {"RMSE":>8}'
    print(header)
    print("-" * len(header))
    print(f"{best_name:<12} {r2:>8.3f} {mae:>8.3f} {rmse:>8.3f}\n")

    # Save Pred vs True Plot
    plot_pred_vs_true(y_test, y_pred, BASE_DIR / "pred_vs_true_scatter.png", title=f"{best_name}: Predicted vs True Winrate")

    # Save coeff plots if available
    if best_name == "Ridge":
        plot_top_coefs(best_model, BASE_DIR / "ridge_coef_top.png", top_k=20,
                       title="Ridge coefficients (top)")
    elif best_name == "PolyRidge":
        plot_top_coefs(best_model, BASE_DIR / "poly_ridge_coef_top.png", top_k=20,
                       title="Poly Ridge coefficients (top)")

    # 9) Log saved files
    print(f"Saved → {BASE_DIR / 'cv_r2_bar.png'}")
    print(f"Saved → {BASE_DIR / 'pred_vs_true_scatter.png'}")
    if best_name == "Ridge":
        print(f"Saved → {BASE_DIR / 'ridge_coef_top.png'}")
    elif best_name == "PolyRidge":
        print(f"Saved → {BASE_DIR / 'poly_ridge_coef_top.png'}")

def main():
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found at {DB_PATH}. Expected demo DB.")
    print(f"Using DB: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = fetch_df(conn)
    finally:
        conn.close()

    evaluate_models(df, seed=42)

if __name__ == "__main__":
    main()
