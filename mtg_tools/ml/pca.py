import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

# --- Paths (no args) ---
BASE_DIR = Path(__file__).resolve().parent
DB_PATH  = (BASE_DIR.parent / "db" / "mtgcore_demo.db").resolve()

# --- Labels & features (canonical) ---
BUCKET_NAMES = ("Aggro", "Midrange", "Control")
SOURCE_PRIORITY = ("manual", "raw")

CATEGORICAL = ["dominant_type", "main_tribe"]
NUMERIC = [
    "avg_cmc",
    "cmc_0","cmc_1","cmc_2","cmc_3","cmc_4","cmc_5","cmc_6","cmc_7_plus",
    "color_W","color_U","color_B","color_R","color_G","color_C"
]

# ---------- Data access ----------
def fetch_df(conn: sqlite3.Connection):
    stats = pd.read_sql_query(f"""
        SELECT deck_id, {', '.join(NUMERIC + CATEGORICAL)}
        FROM deck_stats
    """, conn)

    labels = pd.read_sql_query(f"""
        SELECT da.deck_id, da.archetype_name AS label, da.source
        FROM deck_archetypes da
        JOIN archetypes a ON a.name = da.archetype_name
        WHERE da.archetype_name IN ({','.join('?' for _ in BUCKET_NAMES)})
          AND a.is_obsolete = 0
    """, conn, params=list(BUCKET_NAMES))

    # choose one label per deck by source priority (deterministic tie-break)
    tie_rank = {"Control": 0, "Midrange": 1, "Aggro": 2}
    chosen = {}
    for deck_id, g in labels.groupby("deck_id"):
        for s in SOURCE_PRIORITY:
            sub = g[g["source"] == s]
            if len(sub):
                sub = sub.copy()
                sub["rank"] = sub["label"].map(tie_rank).fillna(99)
                chosen[deck_id] = sub.sort_values("rank").iloc[0]["label"]
                break

    lab = pd.DataFrame({"deck_id": list(chosen.keys()), "label": list(chosen.values())}) if chosen else pd.DataFrame(columns=["deck_id","label"])
    # Merge left for PCA - keep all decks even without labels
    df = stats.merge(lab, on="deck_id", how="left")
    # # Merge inner to compare results
    # df = lab.merge(stats, on="deck_id", how="inner")
    # df = df[df["label"].isin(BUCKET_NAMES)].reset_index(drop=True)

    # Clean
    df[CATEGORICAL] = df[CATEGORICAL].fillna("None")
    df[NUMERIC] = df[NUMERIC].fillna(0.0)
    return df

# ---------- Preprocessing ----------
def build_preprocessor(categorical=CATEGORICAL, numeric=NUMERIC) -> ColumnTransformer:
    """
    Dense OHE for categoricals, StandardScaler for numerics, drop remainder.
    """
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return ct

# ---------- Simple feature-name helper (matches preprocessors) ----------
def get_feature_names(ct, categorical=CATEGORICAL, numeric=NUMERIC):
    """
    Extract feature names for NN input.
    - "categorical" = OneHotEncoder
    - "numeric" = scaled
    """
    names = []

    # Categorical OHE names
    ohe = ct.named_transformers_.get("cat")
    if ohe is not None and hasattr(ohe, "categories_"):
        for col, cats in zip(categorical, ohe.categories_):
            names.extend([f"{col}={c}" for c in cats])
    else:
        names.extend(list(categorical))

    # Numeric names
    names.extend(list(numeric))

    return names

# ---------- Plots ----------
def plot_scree_plot(exp_var_ratios, out_path: Path):
    cumulative_exp = np.cumsum(exp_var_ratios)*100
    plt.figure()
    plt.plot(range(1, len(exp_var_ratios) + 1), cumulative_exp, marker="o", linestyle="--", linewidth=0.5)
    plt.xlabel("Principal Components")
    plt.ylabel("Percentage of explained variance [%]")
    plt.title("PCA Scree Plot on scaled data")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    
def plot_scatter_PCA_2D(df_reduced_2D, exp_var_ratios, out_path: Path):
    plt.figure()
    for name, grp in df_reduced_2D.groupby("label", dropna=False):
        label = name if pd.notna(name) else "Unknown"
        plt.scatter(grp["pc1"], grp["pc2"], s=12, alpha=0.7, label=label) 
    plt.legend(fontsize=7, markerscale=1, loc="best")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    var_txt = f"PC1= {exp_var_ratios[0]*100:.0f}%, PC2= {exp_var_ratios[1]*100:.0f}%"
    plt.title(f"PCA 2D - Percentage of explained variance: {var_txt}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_feature_analysis(pca_2D, feature_names, out_path: Path, top_k:int = 10):
    pc1_importances = pca_2D.components_[0]
    pc2_importances = pca_2D.components_[1]
    pc1_order = np.argsort(pc1_importances)[::-1][:top_k]
    pc2_order = np.argsort(pc2_importances)[::-1][:top_k]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    ax1, ax2 = axes
    ax1.barh(range(len(pc1_order)), pc1_importances[pc1_order][::-1])
    ax1.set_yticks(range(len(pc1_order)), [feature_names[i] for i in pc1_order][::-1])
    ax1.set_title("PC1 feature importance (top)")
    
    ax2.barh(range(len(pc2_order)), pc2_importances[pc2_order][::-1])
    ax2.set_yticks(range(len(pc2_order)), [feature_names[i] for i in pc2_order][::-1])
    ax2.set_title("PC2 feature importance (top)")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    
# ---------- Main ----------
def main():
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found at {DB_PATH}. Expected demo DB.")
    print(f"Using DB: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = fetch_df(conn)
    finally:
        conn.close()

    X = df[CATEGORICAL + NUMERIC]   # keep as DataFrame so ColumnTransformer can select by name
    y = df["label"].values

    # Build and fit the preprocessor (for visuals, fit on all rows is fine)
    pre = build_preprocessor()
    X_dense = pre.fit_transform(X)

    # Get feature names in the transformed order
    feat_names = get_feature_names(pre)

    # PCA - Scree Plot
    pca_full = PCA(n_components=X_dense.shape[1], random_state=42)
    pca_full.fit(X_dense)
    plot_scree_plot(pca_full.explained_variance_ratio_, BASE_DIR / "scree_plot.png")

    # PCA - 2D
    pca_2D = PCA(n_components=2, random_state=42)
    X_PCA_2D = pca_2D.fit_transform(X_dense)

    df_reduced_2D = pd.DataFrame({
        "pc1": X_PCA_2D[:, 0],
        "pc2": X_PCA_2D[:, 1],
        "label": y,
    })
    plot_scatter_PCA_2D(df_reduced_2D, pca_2D.explained_variance_ratio_, BASE_DIR / "pca_2D_scatter.png")

    # PCA 2D - Feature Analysis
    plot_feature_analysis(pca_2D, feat_names, BASE_DIR / "pca_2D_feature_analysis.png")
    
    print(f"Saved plots → {BASE_DIR / 'scree_plot.png'}, {BASE_DIR / 'pca_2D_scatter.png'}, {BASE_DIR / 'pca_2D_feature_analysis.png'}")

if __name__ == "__main__":
    main()