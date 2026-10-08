import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from scipy.io import savemat


def evaluate_predictions(y_true_map, y_pred_map, metrics_dir, figures_dir):
    """Evaluate labeled pixels and save metrics/confusion matrix."""
    y_true = y_true_map.reshape(-1)
    y_pred = y_pred_map.reshape(-1)
    mask = y_true > 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    cm = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    overall = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    metrics = {
        "overall_accuracy": float(overall),
        "kappa": float(kappa),
        "per_class_accuracy": {str(i + 1): float(v) for i, v in enumerate(per_class)},
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    savemat(metrics_dir / "y_true_1d_PU.mat", {"true_labels": y_true})
    savemat(metrics_dir / "y_pred_1d_PU.mat", {"pred_labels": y_pred})

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("PaviaU 1D CNN - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    return metrics


def save_training_history(history, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "training_history.png", dpi=300)
    plt.close()


def save_classification_map(pred_map, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.axis("off")
    plt.imshow(pred_map, interpolation="nearest")
    plt.colorbar(label="Predicted Class")
    plt.title("PaviaU 1D CNN Classification Map")
    plt.tight_layout()
    plt.savefig(figures_dir / "classification_map.png", dpi=300, bbox_inches="tight")
    plt.close()
