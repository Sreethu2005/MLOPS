"""
Model Evaluation Metrics and Diagnostic Plotting.
Calculates clinical screening metrics, calibration scores (Brier, ECE),
and saves artifact plots for MLflow tracking.
"""
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

class SigmoidCalibrator:
    """
    Platt Scaling (Sigmoid) Probability Calibrator.
    Fits a 1D logistic regression on model raw predicted probabilities.
    """
    def __init__(self):
        self.calibrator = LogisticRegression(solver="lbfgs", C=1.0)

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        eps = 1e-6
        clipped_p = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(clipped_p / (1 - clipped_p)).reshape(-1, 1)
        self.calibrator.fit(logits, y_true)
        return self

    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        eps = 1e-6
        clipped_p = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(clipped_p / (1 - clipped_p)).reshape(-1, 1)
        return self.calibrator.predict_proba(logits)[:, 1]

def compute_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """
    Computes true positive fractions and mean predicted probabilities per bin.
    Pure NumPy implementation to eliminate external C-extension dependencies.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    prob_true = []
    prob_pred = []

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        mask = (y_prob >= lower) & (y_prob < upper) if i < n_bins - 1 else (y_prob >= lower) & (y_prob <= upper)

        if np.sum(mask) > 0:
            prob_true.append(float(np.mean(y_true[mask])))
            prob_pred.append(float(np.mean(y_prob[mask])))

    return np.array(prob_true), np.array(prob_pred)

def calculate_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    Measures the weighted difference between predicted confidence and empirical accuracy.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)

    return float(ece)

def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Calculates comprehensive classification, screening, and calibration metrics at a given threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = calculate_expected_calibration_error(y_true, y_prob, n_bins=10)

    return {
        "threshold": float(threshold),
        "recall": round(float(recall), 4),
        "precision": round(float(precision), 4),
        "specificity": round(float(specificity), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "ece": round(float(ece), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.60, min_precision: float = 0.15) -> float:
    """
    Finds decision threshold suitable for a screening application:
    Prioritizes identifying at-risk patients (recall >= target_recall)
    subject to maintaining precision >= min_precision.
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    best_threshold = 0.50
    best_f1 = -1.0

    for t in thresholds:
        metrics = evaluate_predictions(y_true, y_prob, threshold=t)
        if metrics["recall"] >= target_recall and metrics["precision"] >= min_precision:
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = t

    # If constraint cannot be strictly satisfied, choose threshold maximizing F1
    if best_f1 < 0:
        for t in thresholds:
            metrics = evaluate_predictions(y_true, y_prob, threshold=t)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = t

    return round(float(best_threshold), 2)

def generate_evaluation_plots(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, model_name: str, output_dir: str = "reports/plots") -> dict:
    """
    Generates ROC curve, Calibration reliability curve, Precision-Recall curve,
    and Confusion Matrix heatmap. Returns dictionary of saved file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plots = {}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"{model_name} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    roc_file = out_path / f"{model_name}_roc_curve.png"
    fig.tight_layout()
    fig.savefig(roc_file, dpi=200)
    plt.close(fig)
    plots["roc_curve"] = str(roc_file)

    # 2. Calibration Curve
    prob_true, prob_pred = compute_calibration_curve(y_true, y_prob, n_bins=10)
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="s", color="#059669", lw=2, label=f"{model_name} (Brier = {brier:.3f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", label="Perfect Calibration")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of True Readmissions")
    ax.set_title(f"Calibration Curve - {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    calib_file = out_path / f"{model_name}_calibration_curve.png"
    fig.tight_layout()
    fig.savefig(calib_file, dpi=200)
    plt.close(fig)
    plots["calibration_curve"] = str(calib_file)

    # 3. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#7C3AED", lw=2, label=f"PR Curve (AP = {pr_auc:.3f})")
    baseline = np.mean(y_true)
    ax.axhline(baseline, color="#9CA3AF", linestyle="--", label=f"Baseline Rate ({baseline:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(f"Precision-Recall Curve - {model_name}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    pr_file = out_path / f"{model_name}_pr_curve.png"
    fig.tight_layout()
    fig.savefig(pr_file, dpi=200)
    plt.close(fig)
    plots["pr_curve"] = str(pr_file)

    # 4. Confusion Matrix Heatmap
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Readmit (0)", "Readmitted (1)"])
    ax.set_yticklabels(["No Readmit (0)", "Readmitted (1)"])
    ax.set_xlabel(f"Predicted Class (Threshold = {threshold:.2f})")
    ax.set_ylabel("Actual Class")
    ax.set_title(f"Confusion Matrix - {model_name}", pad=15)
    cm_file = out_path / f"{model_name}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(cm_file, dpi=200)
    plt.close(fig)
    plots["confusion_matrix"] = str(cm_file)

    return plots
