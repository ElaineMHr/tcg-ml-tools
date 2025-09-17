# archetype_classification_scikit.py
# Canonical pipelines (ColumnTransformer) for Aggro/Midrange/Control:
#   - LogReg      : OHE(cats) + Scale(nums)        -> LogisticRegression
#   - LogReg+Poly : OHE(cats) + Poly->Scale(nums)  -> LogisticRegression
#   - RandomForest: OHE(cats) + Passthrough(nums)  -> RandomForest
#   - HistGB      : OHE(cats, dense) + Passthrough(nums, dense) -> HistGradientBoosting
#   - SVC-RBF     : OHE(cats, dense) + Scale(nums, dense) -> SVC
#
# Uses 5-fold CV (macro-F1) to pick a winner, evaluates on a test split,
# saves plots to PNGs (no GUI), and persists the best pipeline.

import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# Data utils
from torch.utils.data import DataLoader, Dataset

# Metrics & Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from mtg_tools.ml.plot_tools import plot_class_balance, plot_confusion_matrix

# --- Paths (no args) ---
BASE_DIR = Path(__file__).resolve().parent                  # .../mtg_tools/ml
DB_PATH  = (BASE_DIR.parent / "db" / "mtgcore_demo.db").resolve()
MODEL_OUT = (BASE_DIR / "best_archetype_model.joblib").resolve()

# --- Labels & features (canonical) ---
BUCKET_NAMES = ("Aggro", "Midrange", "Control")
SOURCE_PRIORITY = ("manual", "raw")

CATEGORICAL = ["dominant_type", "main_tribe"]
NUMERIC = [
    "avg_cmc",
    "cmc_0","cmc_1","cmc_2","cmc_3","cmc_4","cmc_5","cmc_6","cmc_7_plus",
    "color_W","color_U","color_B","color_R","color_G","color_C"
]

# String label -> integer id (for CrossEntropyLoss)
CLASS_TO_ID = {name: i for i, name in enumerate(BUCKET_NAMES)}
ID_TO_CLASS = {i: name for name, i in CLASS_TO_ID.items()}

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

    lab = pd.DataFrame({"deck_id": list(chosen.keys()), "label": list(chosen.values())})
    df = lab.merge(stats, on="deck_id", how="inner")
    df = df[df["label"].isin(BUCKET_NAMES)].reset_index(drop=True)

    # clean
    df[CATEGORICAL] = df[CATEGORICAL].fillna("None")
    df[NUMERIC] = df[NUMERIC].fillna(0.0)
    return df

# ---------- Simple feature-name helper (matches our preprocessors) ----------
def get_feature_names(ct, categorical=CATEGORICAL, numeric=NUMERIC):
    """
    Extract feature names for NN input.
    - "cat" = OneHotEncoder
    - "num" = passthrough or scaler
    (No polynomial expansion, since NN handles non-linearities itself)
    """
    names = []

    # Categorical OHE names
    ohe = ct.named_transformers_.get("cat")
    if ohe is not None and hasattr(ohe, "categories_"):
        for col, cats in zip(categorical, ohe.categories_):
            names.extend([f"{col}={c}" for c in cats])
    else:
        names.extend(list(categorical))

    # Numeric branch (no poly here)
    names.extend(list(numeric))

    return names

# Load

df = fetch_df()

# Dense OHE to be compatible with PyTorch tensors directly
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ],
    remainder="drop",
    sparse_threshold=0.0  # force dense
)

def evaluate_model(df: pd.DataFrame, seed: int = 42):
    X = df[CATEGORICAL + NUMERIC]   # keep as DataFrame so CT can select by name
    y = df["label"].values

    # Save class balance
    plot_class_balance(y, BASE_DIR / "class_balance_nn.png", title="Class balance")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )

def main():
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found at {DB_PATH}. Expected demo DB.")
    print(f"Using DB: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = fetch_df(conn)
    finally:
        conn.close()

    # Quick class counts
    print("Class counts:\n", df["label"].value_counts().to_string(), "\n")

    evaluate_model(df, seed=42)