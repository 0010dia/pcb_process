"""원본 이미지 단위 정상/불량 평가. 패치 지표와 섞지 않는다."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve

from src.binary_evaluate import binary_scores, collect_binary_predictions


@plt.style.context("default")
def evaluate_full_model(root: Path, model, dataset, frame: pd.DataFrame, output: Path) -> dict:
    labels, scores = collect_binary_predictions(model, dataset)
    return save_full_predictions(root, frame, labels, scores, output)


@plt.style.context("default")
def save_full_predictions(root: Path, frame: pd.DataFrame, labels, scores, output: Path) -> dict:
    if not np.array_equal(labels, frame.target.values):
        raise ValueError("원본 평가 manifest와 라벨 순서가 다릅니다.")
    output.mkdir(parents=True, exist_ok=True)
    predictions = frame.copy()
    predictions["sigmoid_score"] = scores
    predictions["predicted_target"] = (np.asarray(scores) > .5).astype(int)
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    metrics = binary_scores(labels, scores)
    metrics.update(split="validation", unit="full_640_image", official_test_evaluated=False)
    (output / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
    report = classification_report(labels, predictions.predicted_target, labels=[0, 1],
                                   target_names=["normal", "defect"], output_dict=True, zero_division=0)
    (output / "classification_report.json").write_text(json.dumps(report, indent=2))
    table = pd.DataFrame({k: v for k, v in report.items() if isinstance(v, dict)}).T
    table.loc["accuracy", ["f1-score", "support"]] = [report["accuracy"], len(labels)]
    table.to_csv(output / "classification_report.csv")
    matrix = confusion_matrix(labels, predictions.predicted_target, labels=[0, 1])
    np.save(output / "confusion_matrix.npy", matrix)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(matrix[i, j]), ha="center", va="center",
                         color="white" if matrix[i, j] > matrix.max()/2 else "black")
    axes[0].set_xticks([0, 1], ["Normal", "Defect"])
    axes[0].set_yticks([0, 1], ["Normal", "Defect"])
    axes[0].set(xlabel="Predicted", ylabel="Actual", title="Full-image validation")
    fpr, tpr, _ = roc_curve(labels, scores)
    axes[1].plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.4f}")
    axes[1].set(xlabel="Normal false-positive rate", ylabel="Defect recall", title="ROC")
    axes[1].legend()
    precision, recall, _ = precision_recall_curve(labels, scores)
    axes[2].plot(recall, precision, label=f"AP={metrics['average_precision']:.4f}")
    axes[2].set(xlabel="Defect recall", ylabel="Defect precision", title="Precision / recall")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=150)
    plt.close(fig)
    errors = predictions[predictions.target != predictions.predicted_target]
    errors.to_csv(output / "misclassified.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, errors.head(8).itertuples()):
        with Image.open(root / row.image_path) as image:
            ax.imshow(np.asarray(image.convert("L")), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"True {row.label}; pred {'defect' if row.predicted_target else 'normal'}\n{Path(row.image_id).name}; score={row.sigmoid_score:.3f}", fontsize=7)
    if errors.empty:
        axes.flat[0].text(.5, .5, "No errors in this validation set", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output / "misclassified.png", dpi=140)
    plt.close(fig)
    return metrics


@plt.style.context("default")
def full_history_plot(history: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key in zip(axes[:2], ["loss", "accuracy"]):
        for col in [key, "val_"+key]:
            ax.plot(history.epoch+1, history[col], label=col)
        ax.set(xlabel="Epoch", ylabel=key)
        ax.legend()
    axes[2].plot(history.epoch+1, history.val_macro_f1, label="Macro F1")
    axes[2].plot(history.epoch+1, history.val_defect_recall, label="Defect recall")
    axes[2].plot(history.epoch+1, history.val_normal_fpr, label="Normal false-positive rate")
    axes[2].set(xlabel="Epoch", ylabel="Validation score", ylim=(0, 1))
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
