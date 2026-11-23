"""이진 분류의 불량 미검출과 정상 오탐을 구분해 저장한다."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, precision_recall_curve)

from src.eda import CLASSES


def binary_scores(labels, probabilities, threshold: float = .5) -> dict:
    y = np.asarray(labels).astype(int).reshape(-1)
    p = np.asarray(probabilities).reshape(-1)
    if len(y) != len(p) or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("이진 라벨과 유한한 점수가 필요합니다.")
    predicted = (p > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(samples=len(y), accuracy=float(accuracy_score(y, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                defect_precision=float(precision_score(y, predicted, zero_division=0)),
                defect_recall=float(recall_score(y, predicted, zero_division=0)),
                defect_f1=float(f1_score(y, predicted, zero_division=0)),
                macro_f1=float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
                normal_false_positive_rate=float(fp/(tn+fp)) if tn+fp else None,
                defect_false_negative_rate=float(fn/(tp+fn)) if tp+fn else None,
                roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                average_precision=float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
                true_normal=int(tn), false_defect=int(fp), missed_defect=int(fn), true_defect=int(tp),
                threshold=threshold)


def collect_binary_predictions(model, dataset):
    labels, probabilities = [], []
    for images, targets in dataset:
        labels.append(targets.numpy().reshape(-1))
        probabilities.append(model(images, training=False).numpy().reshape(-1))
    return np.concatenate(labels).astype(int), np.concatenate(probabilities)


@plt.style.context("default")
def save_binary_evaluation(root: Path, model, dataset, frame: pd.DataFrame, output: Path) -> dict:
    labels, probabilities = collect_binary_predictions(model, dataset)
    if not np.array_equal(labels, frame.target.values):
        raise ValueError("평가 데이터 순서와 manifest가 다릅니다.")
    predicted = (probabilities > .5).astype(int)
    result = binary_scores(labels, probabilities)
    result.update(split="validation", official_test_evaluated=False)
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_metrics.json").write_text(json.dumps(result, indent=2))
    predictions = frame.copy()
    predictions["sigmoid_score"] = probabilities
    predictions["predicted_target"] = predicted
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    errors = predictions[predictions.target != predictions.predicted_target]
    errors.to_csv(output / "misclassified.csv", index=False)
    report = classification_report(labels, predicted, labels=[0, 1], target_names=["normal", "defect"], output_dict=True, zero_division=0)
    (output / "classification_report.json").write_text(json.dumps(report, indent=2))
    table = pd.DataFrame({name: values for name, values in report.items() if isinstance(values, dict)}).T
    table.loc["accuracy", ["f1-score", "support"]] = [report["accuracy"], len(labels)]
    table.to_csv(output / "classification_report.csv")
    slices = []
    for kind, subset in predictions.groupby("kind"):
        if subset.target.eq(0).all():
            slices.append(dict(kind=kind, samples=len(subset), normal_false_positive_rate=float(subset.predicted_target.mean())))
    pd.DataFrame(slices).to_csv(output / "normal_source_false_positives.csv", index=False)
    recalls = []
    for class_id, subset in predictions[predictions.target == 1].groupby("defect_class_id"):
        recalls.append(dict(class_id=class_id, defect_type=CLASSES[int(class_id)][0], samples=len(subset), recall=float(subset.predicted_target.mean())))
    pd.DataFrame(recalls).to_csv(output / "recall_by_defect_type.csv", index=False)
    # 같은 위치의 정상/불량 쌍을 모두 남긴 경우 점수의 대소 관계도 별도 진단한다.
    paired = predictions[predictions.kind.isin(["defect", "template_matched"])].pivot_table(
        index="anchor_id", columns="kind", values="sigmoid_score").dropna()
    paired.to_csv(output / "matched_pair_scores.csv")
    if len(paired):
        (output / "matched_pair_ranking.json").write_text(json.dumps(dict(
            pairs=len(paired), defect_score_higher_fraction=float((paired.defect > paired.template_matched).mean())), indent=2))

    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    np.save(output / "confusion_matrix.npy", matrix)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(matrix[i, j]), ha="center", va="center",
                         color="white" if matrix[i, j] > matrix.max()/2 else "black")
    axes[0].set_xticks([0, 1], ["Normal", "Defect"])
    axes[0].set_yticks([0, 1], ["Normal", "Defect"])
    axes[0].set(xlabel="Predicted", ylabel="Actual", title="Validation confusion matrix")
    fpr, tpr, _ = roc_curve(labels, probabilities)
    axes[1].plot(fpr, tpr, label=f"AUC={result['roc_auc']:.4f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set(xlabel="Normal false-positive rate", ylabel="Defect recall", title="Validation ROC")
    axes[1].legend()
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    axes[2].plot(recall, precision, label=f"AP={result['average_precision']:.4f}")
    axes[2].set(xlabel="Defect recall", ylabel="Defect precision", title="Validation precision / recall")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ax in axes.flat:
        ax.axis("off")
    hardest = errors.assign(confidence=lambda data: np.maximum(data.sigmoid_score, 1-data.sigmoid_score)).sort_values("confidence", ascending=False).head(12)
    for ax, row in zip(axes.flat, hardest.itertuples()):
        with Image.open(root / row.patch_path) as image:
            ax.imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"True: {row.label}; pred: {'defect' if row.predicted_target else 'normal'}\n"
                     f"score={row.sigmoid_score:.3f}; {row.kind}", fontsize=7)
    fig.suptitle("Validation errors — uncalibrated sigmoid scores")
    fig.tight_layout()
    fig.savefig(output / "misclassified.png", dpi=150)
    plt.close(fig)
    return result


@plt.style.context("default")
def plot_binary_history(frame: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes[:2], ["loss", "accuracy"]):
        ax.plot(frame.epoch+1, frame[metric], label="train")
        ax.plot(frame.epoch+1, frame["val_"+metric], label="validation")
        ax.set(xlabel="Epoch", ylabel=metric)
        ax.legend()
    axes[2].plot(frame.epoch+1, frame.val_macro_f1, label="Macro F1")
    axes[2].plot(frame.epoch+1, frame.val_defect_recall, label="Defect recall")
    axes[2].plot(frame.epoch+1, frame.val_normal_fpr, label="Normal false-positive rate")
    axes[2].set(xlabel="Epoch", ylabel="Validation score", ylim=(0, 1))
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
