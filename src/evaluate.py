"""고정된 validation 결과를 저장한다. 공식 test 최종 평가와 구분한다."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from src.eda import CLASSES


def collect_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    probabilities, labels = [], []
    for images, targets in dataset:
        probabilities.append(model(images, training=False).numpy())
        labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def scores(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return dict(accuracy=float(accuracy_score(labels, predictions)),
                precision_macro=float(precision_score(labels, predictions, labels=range(6), average="macro", zero_division=0)),
                recall_macro=float(recall_score(labels, predictions, labels=range(6), average="macro", zero_division=0)),
                macro_f1=float(f1_score(labels, predictions, labels=range(6), average="macro", zero_division=0)),
                weighted_f1=float(f1_score(labels, predictions, labels=range(6), average="weighted", zero_division=0)))


@plt.style.context("default")
def save_evaluation(root: Path, model, dataset, frame: pd.DataFrame, output: Path) -> dict:
    labels, probabilities = collect_predictions(model, dataset)
    if not np.array_equal(labels, frame.target_id.values):
        raise ValueError("평가 manifest 순서와 예측 라벨 순서가 다릅니다.")
    predicted = probabilities.argmax(axis=1)
    metrics = scores(labels, predicted)
    metrics.update(split="validation", official_test_evaluated=False, samples=len(frame))
    names = [CLASSES[i][0] for i in CLASSES]
    report = classification_report(labels, predicted, labels=list(range(6)), target_names=names,
                                   output_dict=True, zero_division=0)
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "classification_report.json").write_text(json.dumps(report, indent=2))
    report_table = pd.DataFrame({name: values for name, values in report.items() if isinstance(values, dict)}).T
    report_table.loc["accuracy", ["f1-score", "support"]] = [report["accuracy"], len(labels)]
    report_table.to_csv(output / "classification_report.csv")
    result = frame.copy()
    result["predicted_target"] = predicted
    result["softmax_score"] = probabilities.max(axis=1)
    for i, name in enumerate(names):
        result[f"score_{i}"] = probabilities[:, i]
    result.to_csv(output / "validation_predictions.csv", index=False)
    result[result.target_id != result.predicted_target].to_csv(output / "misclassified.csv", index=False)
    matrix = confusion_matrix(labels, predicted, labels=list(range(6)))
    np.save(output / "confusion_matrix.npy", matrix)
    fig, ax = plt.subplots(figsize=(8, 7))
    heat = ax.imshow(matrix, cmap="Blues")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max()/2 else "black")
    ax.set_xticks(range(6), names, rotation=35, ha="right")
    ax.set_yticks(range(6), names)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Validation confusion matrix (whole group holdout)")
    fig.colorbar(heat, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    errors = result[result.target_id != result.predicted_target].sort_values("softmax_score", ascending=False).head(16)
    fig, axes = plt.subplots(4, 4, figsize=(10, 11))
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, errors.itertuples()):
        with Image.open(root / row.patch_path) as image:
            ax.imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"True: {CLASSES[row.target_id+1][0]}\nPred: {CLASSES[row.predicted_target+1][0]}\n"
                     f"score={row.softmax_score:.2f}, neighbors={row.neighbor_count}", fontsize=7)
    fig.suptitle("Validation errors — Softmax scores are uncalibrated")
    fig.tight_layout()
    fig.savefig(output / "misclassified.png", dpi=140)
    plt.close(fig)
    # 이웃/패딩 존재에 따른 진단은 val에만 보고하며 샘플을 성능 향상 목적으로 버리지 않는다.
    slices = []
    for name, mask in {"no_neighbor": result.neighbor_count.eq(0), "has_neighbor": result.neighbor_count.gt(0),
                       "no_padding": result.padding_fraction.eq(0), "has_padding": result.padding_fraction.gt(0)}.items():
        if mask.any():
            slices.append(dict(slice=name, samples=int(mask.sum()), **scores(labels[mask], predicted[mask])))
    pd.DataFrame(slices).to_csv(output / "validation_slices.csv", index=False)
    return metrics


@plt.style.context("default")
def plot_history(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key in zip(axes[:2], ["loss", "accuracy"]):
        for column in [key, "val_"+key]:
            ax.plot(history.epoch+1, history[column], label=column)
        ax.set(xlabel="Epoch", ylabel=key)
        ax.legend()
    axes[2].plot(history.epoch+1, history.val_macro_f1, label="Validation Macro F1")
    axes[2].set(xlabel="Epoch", ylabel="Macro F1", ylim=(0, 1))
    axes[2].axhline(.9, color="gray", linestyle="--", label="Target (not guaranteed)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
