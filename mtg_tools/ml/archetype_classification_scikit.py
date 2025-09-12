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

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

# Use non-interactive backend for plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib

# --- Paths (no args) ---
BASE_DIR = Path(__file__).resolve().parent                  # .../mtg_tools/ml
DB_PATH  = (BASE_DIR.parent / "db" / "mtgcore_demo.db").resolve()
MODEL_OUT = (BASE_DIR / "best_archetype_model.joblib").resolve()

# --- Labels & features (canonical) ---
BUCKET_NAMES = ("Aggro", "Midrange", "Control")
SOURCE_PRIORITY = ("manual", "model", "raw")

CATEGORICAL = ["dominant_type", "main_tribe"]
NUMERIC = [
    "avg_cmc",
    "cmc_0","cmc_1","cmc_2","cmc_3","cmc_4","cmc_5","cmc_6","cmc_7_plus",
    "color_W","color_U","color_B","color_R","color_G","color_C"
]

# ---------- Data access ----------
def fetch_df(conn: sqlite3.Connection) -> pd.DataFrame:
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
def get_feature_names_known(ct, categorical=CATEGORICAL, numeric=NUMERIC):
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

# ---------- Plot helpers (save to files, no GUI) ----------
def plot_class_balance(y, out_path: Path, title="Class balance"):
    labels, counts = np.unique(y, return_counts=True)
    plt.figure()
    plt.bar(labels, counts)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_confusion_matrix(cm, classes, out_path: Path, title="Confusion matrix"):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_rf_importances(pipeline, out_path: Path, top_k=20, title="RF feature importance (top)"):
    rf = pipeline.named_steps["clf"]
    pre = pipeline.named_steps["pre"]
    names = get_feature_names_known(pre)
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1][:top_k]
    plt.figure()
    plt.barh(range(len(order)), importances[order][::-1])
    plt.yticks(range(len(order)), [names[i] for i in order][::-1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_logreg_top_coefs(pipeline, out_path: Path, top_k=20, title="LogReg | top |coefficients|"):
    lr = pipeline.named_steps["clf"]
    pre = pipeline.named_steps["pre"]
    names = get_feature_names_known(pre)
    coefs = np.abs(lr.coef_)          # (n_classes, n_features)
    scores = coefs.sum(axis=0)        # global importance proxy
    order = np.argsort(scores)[::-1][:top_k]
    plt.figure()
    plt.barh(range(len(order)), scores[order][::-1])
    plt.yticks(range(len(order)), [names[i] for i in order][::-1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# ---------- Models (canonical transformers; only necessary changes) ----------
def build_models():
    # Sparse-friendly pieces for linear & RF
    ohe_sparse = OneHotEncoder(handle_unknown="ignore")  # sparse (default)
    scaler_sparse = StandardScaler()      # works with sparse

    # 1) Logistic Regression: OHE(cats) + Scale(nums)
    pre_linear = ColumnTransformer(
        transformers=[
            ("cat", ohe_sparse, CATEGORICAL),
            ("num", scaler_sparse, NUMERIC),
        ]
    )
    logreg = Pipeline([
        ("pre", pre_linear),
        ("clf", LogisticRegression(
            max_iter=2000, random_state=42
        )),
    ])

    # 2) Logistic Regression + PolynomialFeatures: OHE(cats) + Scale(Poly(nums))
    pre_poly = ColumnTransformer(
        transformers=[
            ("cat", ohe_sparse, CATEGORICAL),
            ("num_poly", Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
            ]), NUMERIC),
        ]
    )
    logreg_poly = Pipeline([
        ("pre", pre_poly),
        ("clf", LogisticRegression(
            max_iter=3000, random_state=42
        )),
    ])

    # 3) RandomForest: OHE(cats) + Passthrough(nums)
    pre_tree = ColumnTransformer(
        transformers=[
            ("cat", ohe_sparse, CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    rf = Pipeline([
        ("pre", pre_tree),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_split=2,
            n_jobs=-1, random_state=42
        )),
    ])

    # 4) HistGradientBoosting - requires DENSE input
    pre_hgbt = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", "passthrough", NUMERIC),  # dense DataFrame + dense OHE -> dense array
        ],
        sparse_threshold=0.0  # force dense
    )
    hgbt = Pipeline([
        ("pre", pre_hgbt),
        ("clf", HistGradientBoostingClassifier(random_state=42))
    ])

    # 5) SVC (RBF)
    pre_svc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(with_mean=True), NUMERIC),
        ],
        sparse_threshold= 1.0  # prefer sparse
    )
    svc = Pipeline([
        ("pre", pre_svc),
        ("clf", SVC(kernel="rbf", C=3.0, gamma="scale", probability=False, random_state=42)),
    ])

    return {
        "LogReg": logreg,
        "LogReg+Poly": logreg_poly,
        "RandomForest": rf,
        "HistGB": hgbt,
        "SVC-RBF": svc,
    }

# ---------- Evaluation ----------
def evaluate_models(df: pd.DataFrame, seed: int = 15):
    X = df[CATEGORICAL + NUMERIC]   # keep as DataFrame so CT can select by name
    y = df["label"].values

    # Save class balance
    plot_class_balance(y, BASE_DIR / "class_balance.png", title="Class balance (labeled)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )

    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scorer = "f1_macro"

    print(f"CV metric: macro-F1, classes={tuple(BUCKET_NAMES)}\n")
    header = f'{"Model":<14} {"F1 (CV)":>8}'
    print(header)
    print("-" * len(header))

    cv_scores = {}
    for name, pipe in models.items():
        score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scorer, n_jobs=-1).mean()
        cv_scores[name] = score
        print(f"{name:<14} {score:>8.3f}")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\nBest by CV macro-F1: {best_name}\n")

    # Fit best on train, evaluate on test
    best_model = models[best_name].fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

    header = f'{"Model":<14} {"Acc":>6} {"Prec":>6} {"Rec":>6} {"F1":>6}'
    print(header)
    print("-" * len(header))
    print(f"{best_name:<14} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}\n")

    print("Classification report:\n")
    print(classification_report(y_test, y_pred, labels=list(BUCKET_NAMES), digits=3))

    cm = confusion_matrix(y_test, y_pred, labels=list(BUCKET_NAMES))
    print("Confusion matrix (rows=true, cols=pred):")
    hdr = "            " + "  ".join(f"{c:>9}" for c in BUCKET_NAMES)
    print(hdr)
    for i, row in enumerate(cm):
        print(f"{BUCKET_NAMES[i]:>10}  " + "  ".join(f"{n:>9d}" for n in row))

    # Save CM plot
    plot_confusion_matrix(cm, classes=list(BUCKET_NAMES), out_path=BASE_DIR / "confusion_matrix.png")

    # Save importances/coeffs if available for the winner
    if best_name == "RandomForest":
        plot_rf_importances(best_model, BASE_DIR / "rf_importances_top.png")
    elif best_name in ("LogReg", "LogReg+Poly"):
        plot_logreg_top_coefs(best_model, BASE_DIR / "logreg_top_coeffs.png")

    # Save best pipeline
    joblib.dump({
        "pipeline": best_model,
        "classes": list(BUCKET_NAMES),
        "categorical": CATEGORICAL,
        "numeric": NUMERIC,
        "best_name": best_name,
    }, MODEL_OUT)
    print(f"\nSaved best model → {MODEL_OUT}")
    print(f"Saved plots → {BASE_DIR / 'class_balance.png'}, {BASE_DIR / 'confusion_matrix.png'}"
          + (", and feature plot" if best_name in ('RandomForest','LogReg','LogReg+Poly') else ""))

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

    evaluate_models(df, seed=15)

if __name__ == "__main__":
    main()
