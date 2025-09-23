import sqlite3
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import matplotlib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
def plot_elbow_method(k_list, inertias, out_path: Path):
    plt.figure()
    plt.plot(k_list, inertias, marker="o")
    plt.xticks(k_list)
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("K-Means Elbow")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_silhouette_method(k_list, silhouettes, out_path: Path):
    plt.figure()
    plt.plot(k_list, silhouettes, marker="o")
    plt.xticks(k_list)
    plt.xlabel("k")
    plt.ylabel("Average silhouette")
    plt.title("K-Means Silhouette")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_pca_clusters_dense(X_dense, cluster_labels, out_path: Path, seed: int = 42):
    pca_2D = PCA(n_components=2, random_state=seed)
    X_pca_2D = pca_2D.fit_transform(X_dense)
    plt.figure()
    for c_lbl in np.unique(cluster_labels):
        m = cluster_labels == c_lbl  # element-wise comparison -> boolean mask array
        plt.scatter(X_pca_2D[m, 0], X_pca_2D[m, 1], s=12, alpha=0.7, label=f"cluster {c_lbl}")
    plt.legend(fontsize=8, markerscale=1)
    var = pca_2D.explained_variance_ratio_
    var_txt = f"PC1= {var[0]*100:.0f}%, PC2= {var[1]*100:.0f}%"
    plt.title(f"K-Means on PCA 2D - variance: {var_txt}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    
def plot_cluster_heatmap(df, cluster_labels, k_kmeans, out_path: Path):
    """
    Summarize clusters with a heatmap of average CMC, CMC curve, and color distribution.
    """
    # Reduced Feature Mapping <- Having information of the results of other analyses
    features = [
        "avg_cmc",
        "cmc_0","cmc_1","cmc_2","cmc_3","cmc_4","cmc_5","cmc_6","cmc_7_plus",
        "color_W","color_U","color_B","color_R","color_G","color_C",
    ]

    # compute cluster means
    df_tmp = df[features].copy()
    df_tmp["cluster"] = cluster_labels
    cluster_means = df_tmp.groupby("cluster")[features].mean()

    # Normalization of values -> features comparable in same colormap
    cluster_means_norm = (cluster_means - cluster_means.min()) / (
        cluster_means.max() - cluster_means.min()
    )

    # plot heatmap
    plt.figure(figsize=(len(features) * 0.7, k_kmeans * 0.5 + 4))
    sns.heatmap(
        cluster_means_norm,
        annot=False,
        cmap="coolwarm",
        xticklabels=features,
        yticklabels=[f"Cluster {clbl}" for clbl in cluster_means.index],
    )
    plt.title(f"Cluster Profiles Heatmap - K={k_kmeans}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# ---------- Helpers ----------
def calculate_entropy_from_counts(counts):
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return -(p * np.log2(p)).sum() # Entropy

def compute_summary_df(df, cluster_labels):
    """
    df_full: columns deck_id, label; same length as labels
    labels: cluster ids
    Returns DataFrame with cluster metrics, including purity & entropy on labeled subset.
    """
    out_rows = []
    tmp = pd.DataFrame({"cluster": cluster_labels, "label": df["label"]})
    for c_lbl, g in tmp.groupby("cluster"):
        size_total = len(g)
        labeled_mask = g["label"].notna()
        size_labeled = int(labeled_mask.sum())
        size_unlabeled = int(size_total - size_labeled)

        row = {"cluster": int(c_lbl), "size_total": int(size_total),
               "size_labeled": size_labeled, "size_unlabeled": size_unlabeled}

        if size_labeled > 0:
            counts = {
                "Aggro": int((g.loc[labeled_mask, "label"] == "Aggro").sum()),
                "Midrange": int((g.loc[labeled_mask, "label"] == "Midrange").sum()),
                "Control": int((g.loc[labeled_mask, "label"] == "Control").sum()),
            }
            for name in BUCKET_NAMES:
                row[f"percentage_{name}"] = (counts[name] / size_labeled) if size_labeled > 0 else 0.0
            row["purity"] = max(row[f"percentage_{n}"] for n in BUCKET_NAMES)
            row["entropy"] = calculate_entropy_from_counts(np.array([counts[n] for n in BUCKET_NAMES], dtype=float))
        else:
            row["purity"] = np.nan
            row["entropy"] = np.nan
            for name in BUCKET_NAMES:
                row[f"percentage_{name}"] = np.nan

        out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values("cluster").reset_index(drop=True) # For safety: sort by clusterids and reindex. 

def print_cluster_summary_table(summary_df, k_kmeans):
    print(f"\nK={k_kmeans} - Cluster summary (label percentages use labeled rows only):")
    # Set column order
    column_order = [
    "cluster","size_total","size_labeled","size_unlabeled",
    "percentage_Aggro","percentage_Midrange","percentage_Control",
    "purity","entropy",
]
    summary_df_ordered = summary_df.filter(items=column_order)
    print(summary_df_ordered.to_string(index=False))

# ---------- Main ----------
def main(kmeans_range: Tuple[int, int] = (2, 15), k_kmeans:int= 8, seed:int= 42):
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

    k_min, k_max = kmeans_range
    k_list = list(range(k_min, k_max+1))
    
    inertias, silhouettes = [], []
    for k in k_list:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X_dense)
        inertias.append(float(kmeans.inertia_))
        try:
            silhouette = silhouette_score(X_dense, cluster_labels)
        except Exception:
            silhouette = float("nan")
        silhouettes.append(silhouette)

    plot_elbow_method(k_list, inertias, BASE_DIR / "kmeans_elbow.png")

    plot_silhouette_method(k_list, silhouettes, BASE_DIR / "kmeans_silhouette.png")
    
    print(f"Saved plots → {BASE_DIR / 'kmeans_elbow.png'}, {BASE_DIR / 'kmeans_silhouette.png'}")

    # ---- Final fit at chosen K ----
    kmeans = KMeans(n_clusters=k_kmeans, n_init=10, random_state=seed)
    cluster_labels = kmeans.fit_predict(X_dense)

    # PCA overlay
    plot_pca_clusters_dense(X_dense, cluster_labels, BASE_DIR / "pca_scatter_clusters.png", seed)

    plot_cluster_heatmap(df, cluster_labels, k_kmeans, BASE_DIR / "clusters_features_heatmap.png")
    
    print(f"Saved plots → {BASE_DIR / 'pca_scatter_clusters.png'}, {BASE_DIR / 'clusters_features_heatmap.png'}")
    
    summary_df = compute_summary_df(df, cluster_labels)
    print_cluster_summary_table(summary_df, k_kmeans)

if __name__ == "__main__":
    main()