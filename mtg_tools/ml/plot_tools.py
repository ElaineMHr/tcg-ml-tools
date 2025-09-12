# Matplotlib - Headless plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ---------- Plot helpers (save to files, no GUI) ----------

def plot_class_balance(labels, out_path: Path, title="Class balance"):
    labs, counts = np.unique(labels, return_counts=True)
    plt.figure()
    plt.bar(labs, counts)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, classes, out_path: Path, title="Confusion matrix"):
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
    
    