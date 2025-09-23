from typing import Tuple
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
    f1_score, classification_report, confusion_matrix
)

from plot_tools import plot_class_balance, plot_confusion_matrix

# --- Paths (no args) ---
BASE_DIR = Path(__file__).resolve().parent                  # .../mtg_tools/ml
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

# String label -> integer id (for CrossEntropyLoss)
CLASS_TO_ID = {name: i for i, name in enumerate(BUCKET_NAMES)}
ID_TO_CLASS = {i: name for name, i in CLASS_TO_ID.items()}

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def preprocess_train_data(X_train_df):
    """
    Build & FIT preprocessing on TRAIN ONLY, then transform X_train.

    Returns:
      preprocessor  : fitted ColumnTransformer
      X_train_np    : dense numpy array after transform (for input_dim)
      feature_names : from get_feature_names()
    """
    # Dense OHE to be compatible with PyTorch tensors directly
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ],
        remainder="drop",
        sparse_threshold=0.0  # force dense
    )
    
    # Fit and then transform
    X_train_np = preprocessor.fit_transform(X_train_df)
    # Feature names
    feature_names = get_feature_names(preprocessor)
    
    return preprocessor, X_train_np, feature_names

def split_train_test(df, seed: int = 42, test_size: float = 0.10):
    """
    Stratified split
    Default split is train/val/test: 80/10/10

    Returns:
      (X_train_df, y_train), (X_test_df, y_test)
      - X_*_df: pandas DataFrame with CATEGORICAL + NUMERIC
      - y_*   : numpy int array in {0,1,2}
    """
    X = df[CATEGORICAL + NUMERIC]
    y = df["label"].map(CLASS_TO_ID).astype(int).values

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return (X_train_df, y_train), (X_test_df, y_test)

def split_train_val(X_train_df, y_train, seed: int = 42, test_size: float = 0.1, val_size: float = 0.1):
    """
    Stratified split
    Default split is train/val/test: 80/10/10
    from test_size -> rel_val_size calculation

    Returns:
      (X_train_df, y_train), (X_val_df, y_val)
      - X_*_df: pandas DataFrame with CATEGORICAL + NUMERIC
      - y_*   : numpy int array in {0,1,2}
    """
    rel_val_size = val_size/(1-test_size)
    
    X_tr_df, X_val_df, y_tr, y_val = train_test_split(
        X_train_df, y_train, test_size=rel_val_size, stratify=y_train, random_state=seed
    )
    return (X_tr_df, y_tr), (X_val_df, y_val)

def transform_test_data(preprocessor, X_test_df):
    """
    Apply the already-fitted preprocessor to the test split.
    Returns a dense numpy array shaped like X_train_np.
    """
    return preprocessor.transform(X_test_df)

def to_torch_tensors(X_np, y_np, device):
    """
    Convert numpy arrays to torch tensors with correct dtypes and device.
    """
    X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_np, dtype=torch.int64,    device=device)
    return X_tensor, y_tensor

class ArchetypeNN(nn.Module):
    """
    A classifier for tabular data.

    Structure:
      input_dim -> Linear(64) // ReLU -> Linear(32) // ReLU -> Linear(num_classes) # logits
    """
    
    def __init__(self, input_dim: int, hidden: Tuple[int, int] = (64, 32), num_classes: int = 3):
        super().__init__() # inherit classes of nn.Module (parent)
        h1, h2 = hidden
        self.net = nn.Sequential( # nn pipeline - stack layers as defined above
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_classes) # output logits z -> Crossentropy
        )
    
    def forward(self, x):
        # x: [batch, input_dim] float32
        return self.net(x)
    
class TabularDataset(Dataset):
    """
    Minimal dataset for (X, y) tensors after preprocessing.
    Returns one (x_row, y_label) pair per index.
    """
    
    def __init__(self, X_tensor, y_tensor):
        self.X = X_tensor   # float32 
        self.y = y_tensor   # int64
        
    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
          
def make_loaders(
    X_train_tensor,
    y_train_tensor,
    X_val_tensor,
    y_val_tensor,
    X_test_tensor,
    y_test_tensor,
    batch_size_train: int = 64,
    batch_size_val: int = 256,
    batch_size_test: int = 256,
    device: torch.device = DEVICE):
    """
    Wrap tensors in Datasets + DataLoaders.
    - Shuffle only training
    - If on GPU, pin_memory + non_blocking -> speed up
    """
    pin_switch = (device.type == "cuda")
    nw  = 2 if pin_switch else 0
    
    train_dataset = TabularDataset(X_train_tensor, y_train_tensor)
    val_dataset = TabularDataset(X_val_tensor, y_val_tensor)
    test_dataset = TabularDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=nw, pin_memory=pin_switch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=nw, pin_memory=pin_switch
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=nw, pin_memory=pin_switch
    )
    
    return train_loader, val_loader, test_loader

def make_criterion_and_optimiser(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_name: str = "adam"
):
    """
    Returns:
      - criterion: CrossEntropyLoss (no other available)
      - optimizer: Adam (default) or SGD+momentum if optimizer_name="sgd". Any other input -> Adam
      (instead of sgd -> sgd+momentum for smoothness and efficient convergence)
      - recommended learning_rate for sgd: 0.05
      
    """
    
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return criterion, optimizer
    
def train_single_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device = DEVICE
):
    """
    Train for a single epoch. Returns (avg_loss, avg_accuracy).
    """
    model.train()
    total_loss, total_correct, total_examples= 0.0, 0, 0
    
    for x_batch, y_batch in train_loader:
        
        # Redundant, just for safety (no overhead):
        x_batch = x_batch.to(device, non_blocking=True) # async
        y_batch = y_batch.to(device, non_blocking=True) # async
        
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        
        optimizer.zero_grad(set_to_none=True) # reset gradients before calling loss.backward() (set to None)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()*y_batch.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred==y_batch).sum().item()
        total_examples += y_batch.size(0)
        
    return total_loss / total_examples, total_correct / total_examples

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = DEVICE
):
    """
    Evaluation pass: no weight updates.
    Returns: avg_loss, avg_acc, macro_f1, y_true, y_pred
    """
    model.eval()
    total_loss, total_correct, total_examples = 0.0, 0, 0
    all_true, all_pred = [], []
    
    for x_batch, y_batch in data_loader:
        
        # Redundant, just for safety (no overhead):
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        preds = logits.argmax(dim=1)

        total_correct += (preds == y_batch).sum().item()
        total_examples += y_batch.size(0)
        total_loss += loss.item() * y_batch.size(0)
        
        all_true.append(y_batch.cpu().numpy())
        all_pred.append(preds.cpu().numpy())
        
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
    return total_loss / total_examples, total_correct / total_examples, f1, y_true, y_pred
    
def report_and_plot(y_true, y_pred, out_dir: Path):
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=list(BUCKET_NAMES), digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(BUCKET_NAMES))))
    print("Confusion matrix (rows=true, cols=pred):")
    hdr = "            " + "  ".join(f"{c:>9}" for c in BUCKET_NAMES)
    print(hdr)
    for i, row in enumerate(cm):
        print(f"{BUCKET_NAMES[i]:>10}  " + "  ".join(f"{n:>9d}" for n in row))

    plot_confusion_matrix(cm, classes=list(BUCKET_NAMES), out_path=BASE_DIR / "confusion_matrix_nn.png")
        
def run_epochs(
    model:nn.Module,
    train_loader:DataLoader,
    val_loader:DataLoader,
    criterion:nn.Module,
    optimizer:optim.Optimizer,
    device:torch.device = DEVICE,
    max_epochs:int = 30,
    patience:int = 10):
    """
    Train/eval loop with simple early stopping (patience) on best val f1.
    Returns: best_metrics (dict), best_state_dict (for saving), y_true, y_pred
    """

    best_f1 = -1.0
    best_state = None
    waiting = 0
    best_metrics = {}
    
    for epoch in range(max_epochs+1):
        train_loss, train_acc = train_single_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        
        print(f"epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            waiting = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1}
        else:
            waiting += 1
            if waiting >= patience:
                print(f"early stopping at epoch {epoch} (no val f1 improvement for {patience} epochs)")
                break
        
    # load best weights before returning (in case of early stop)
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return best_metrics, best_state, y_true, y_pred

def main(seed: int = 42, test_pct: float = 0.10, val_pct: float = 0.10,
         epochs: int = 30, patience: int = 10, optimizer_name: str = "adam"):
    # DB → DataFrame
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found at {DB_PATH}.")
    print(f"Using DB: {DB_PATH}")
    with sqlite3.connect(str(DB_PATH)) as conn:
        df = fetch_df(conn)

    # Quick class counts + plot balance
    print("Class counts:\n", df["label"].value_counts().to_string(), "\n")
    plot_class_balance(df["label"].values, BASE_DIR / "class_balance_nn.png")

    # Train/test split (stratified), then val from train
    (X_train_df, y_train), (X_test_df, y_test) = split_train_test(df, seed=seed, test_size=test_pct)

    (X_tr_df, y_tr), (X_val_df, y_val) = split_train_val(X_train_df, y_train, test_size=0.1, val_size=0.1, seed=seed)

    # Fit preprocessor on Train-only; transform train/val/test
    pre, X_tr_np, feature_names = preprocess_train_data(X_tr_df)
    X_val_np = transform_test_data(pre, X_val_df)
    X_te_np  = transform_test_data(pre, X_test_df)

    # Arrays -> tensors
    Xtr, ytr = to_torch_tensors(X_tr_np,  y_tr,   DEVICE)
    Xva, yva = to_torch_tensors(X_val_np, y_val,  DEVICE)
    Xte, yte = to_torch_tensors(X_te_np,  y_test, DEVICE)

    # Loaders
    train_loader, val_loader, test_loader = make_loaders(
        Xtr, ytr, Xva, yva, Xte, yte, batch_size_train=64, batch_size_val=256, batch_size_test=256, device=DEVICE
    )

    # Model (input_dim = columns after OHE+scale)
    input_dim = X_tr_np.shape[1]
    model = ArchetypeNN(input_dim, hidden=(64, 32), num_classes=len(BUCKET_NAMES)).to(DEVICE)

    # Loss & optimizer
    criterion, optimizer = make_criterion_and_optimiser(
        model, learning_rate=1e-3 if optimizer_name!="sgd" else 0.05,
        weight_decay=1e-4, optimizer_name=optimizer_name
    )

    # Train with validation each epoch with early stop on best val f1)
    best_metrics, best_state, _, _ = run_epochs(
        model, train_loader, val_loader, criterion, optimizer,
        device=DEVICE, max_epochs=epochs, patience=patience
    )

    # Final test evaluation (with the best weights from run_epochs)
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nFinal test — loss {test_loss:.4f} acc {test_acc:.3f} f1 {test_f1:.3f}")
    report_and_plot(y_true, y_pred, BASE_DIR)
    
if __name__ == "__main__":
    main(seed=42, test_pct=0.10, val_pct=0.10, epochs=30, patience=10, optimizer_name="adam")
    # main(seed=42, test_pct=0.10, val_pct=0.10, epochs=30, patience=10, optimizer_name="sgd")