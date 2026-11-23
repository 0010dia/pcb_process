"""정상/불량 분류의 실측 결과와 사용 범위를 별도 보고서로 정리한다."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


@plt.style.context("default")
def paired_example_figure(root: Path, frame: pd.DataFrame, output: Path):
    positives = frame.query("split == 'train' and kind == 'defect'").set_index("anchor_id")
    normals = frame.query("split == 'train' and kind == 'template_matched'").set_index("anchor_id")
    matched = positives.loc[positives.index.intersection(normals.index)]
    examples = matched.groupby("defect_class_id", sort=True).head(1)
    fig, axes = plt.subplots(len(examples), 2, figsize=(6, 2.4*len(examples)), squeeze=False)
    for axes_row, (anchor, bad) in zip(axes, examples.iterrows()):
        good = normals.loc[anchor]
        for ax, row, label in zip(axes_row, [good, bad], ["Normal template, same ROI", "Defect inspection, same ROI"]):
            with Image.open(root / row.patch_path) as image:
                ax.imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255)
            ax.set_title(label, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


@plt.style.context("default")
def write_binary_report(root: str | Path, run_name: str = "binary_cnn_001") -> Path:
    root = Path(root).resolve()
    reports = root / "results/binary"
    output = reports / run_name
    run = json.loads((output / "run_result.json").read_text())
    metrics = run["metrics"]
    summary = json.loads((reports / "data_summary.json").read_text())
    counts = pd.read_csv(reports / "data_counts.csv")
    baseline = json.loads((output / "majority_normal_baseline.json").read_text())
    normals = pd.read_csv(output / "normal_source_false_positives.csv")
    recalls = pd.read_csv(output / "recall_by_defect_type.csv")
    frame = pd.read_csv(root / "data/split/binary_v1/patches.csv")
    paired_example_figure(root, frame, reports / "matched_pair_examples.png")
    comparison = pd.DataFrame([
        dict(model="Always normal", split="validation", official_test_evaluated=False, **baseline),
        dict(model=run_name, **metrics)
    ])
    comparison.to_csv(reports / "baseline_comparison.csv", index=False)
    predictions = pd.read_csv(output / "validation_predictions.csv")
    worst_recall = recalls.sort_values("recall").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, subset in predictions.groupby("label"):
        axes[0].hist(subset.sigmoid_score, bins=np.linspace(0, 1, 31), density=True, histtype="step", label=label)
    axes[0].axvline(.5, color="gray", linestyle="--")
    axes[0].set(xlabel="Uncalibrated defect sigmoid score", ylabel="Density", title="Validation scores; threshold 0.5")
    axes[0].legend()
    axes[1].bar(comparison.model, comparison.macro_f1, color=["#999999", "#2a9d8f"])
    axes[1].set(ylim=(0, 1), ylabel="Validation Macro F1", title="Binary task baseline comparison")
    for i, value in enumerate(comparison.macro_f1):
        axes[1].text(i, value+.015, f"{value:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(reports / "score_distribution_and_baseline.png", dpi=150)
    plt.close(fig)
    data_rows = ['| 분할 | 종류 | 라벨 | 패치 수 |', '|---|---|---|---:|']
    data_rows += [f'| {row.split} | {row.kind} | {"불량" if row.target else "정상"} | {row.patches:,} |' for row in counts.itertuples()]
    normal_rows = ['| 정상 데이터 출처 | 표본 수 | 정상 오탐률 |', '|---|---:|---:|']
    normal_rows += [f'| {row.kind} | {row.samples:,} | {row.normal_false_positive_rate:.4%} |' for row in normals.itertuples()]
    recall_rows = ['| 원래 불량 유형 | 불량 표본 수 | 불량으로 검출한 Recall |', '|---|---:|---:|']
    recall_rows += [f'| {row.defect_type} | {row.samples:,} | {row.recall:.4%} |' for row in recalls.itertuples()]
    verification_path = reports / "inference_verification.json"
    verification_text = '독립 추론/CPU 대조는 노트북 마지막 검증 셀에서 확인한다.'
    if verification_path.exists():
        check = json.loads(verification_path.read_text())
        verification_text = (f"validation {check['cpu_samples']}개 전체의 CPU/Metal 예측 라벨 일치율은 "
                             f"{check['label_agreement']:.2%}, 최대 sigmoid 점수 차이는 {check['max_cpu_gpu_delta']:.3g}다. "
                             f"독립 추론 함수 점수 차이는 {check['max_inference_delta']:.3g}다.")
    text = f'''# PCB 개별 패치 정상/불량 분류 결과

## 목적 및 입력

`binary_classification.ipynb`에서 기존 6종 분류와 별도로 실행했다. 입력은 사용자가 선택한
128×128 개별 회로 패치다. `0=normal`, `1=defect`이며 기존 6종 불량을 모두 불량 1로 합쳤다.
정상 기준 영상을 추론 입력으로 요구하지 않는 단일 이미지 CNN이다. 전체 PCB의 불량 위치 탐지나
전체 이미지에 결함이 있는지의 성능은 이번에 평가하지 않았다.

## 데이터 생성과 누수 방지

기존 `sample.ipynb`, 원본, 6종 모델은 변경하지 않고 `binary_v1` 및 별도 모델/결과 폴더를 사용했다.
기존 train 800개/validation 200개 원본 그룹 분할과 검증 그룹 `group00041`을 그대로 유지했다.
같은 원본의 정상/불량/템플릿 패치는 분할을 넘지 않는다. 공식 test는 읽거나 Crop하지 않았다.

{chr(10).join(data_rows)}

정상 후보를 두 출처에서 만들었다. `template_matched`는 불량 bbox와 같은 위치의 정상 기준 Crop이다.
`tested_clear`는 검사 영상 안에서 모든 주석 bbox와 guard 8px를 피한 회로 Crop이다.
후자는 주석이 완전하다는 가정에 따른 정상 후보이며, 미주석 결함 가능성을 실제 생산 데이터에서
별도로 검증해야 한다. 검사/기준 영상의 저장 방식 차이만 배우는 문제를 줄이려고 두 출처를 섞었다.
이 구성만으로 모든 출처 단서를 제거했다고 단정하지 않고, 출처별 정상 오탐률을 함께 보고한다.

생성 후보 {summary['candidates']:,}개 중 {summary['excluded']:,}개를 격리하고 {summary['included']:,}개를 사용했다.
고정 규칙은 라벨이 충돌하는 동일 픽셀과 split 간 동일 픽셀을 격리하고 split 내부 반복은 한 번만 유지하는 것이다.
격리 내역은 `excluded_patches.csv`에 남겼고 생성 파일은 삭제하지 않았다. 성능을 본 후 표본을 제거하지 않았다.
조건을 만족하는 정상 회로 ROI를 찾지 못한 {summary['sampling_failures']}건은 `sampling_failures.csv`에 기록했다.

전처리는 원본 bbox + 16px 여백의 정사각형, edge padding, bilinear 128×128 grayscale PNG다.
검사 영상의 정상 회로 후보는 빈 배경만 학습하지 않도록 흑색 픽셀 비율 5~95%로 제한했다.
원본 좌표·부모 객체·픽셀 SHA256·출처·split을 모두 manifest에 저장했다.

## CNN 학습

Conv 블록(16→32→64; 블록마다 Conv2D 2회), BatchNormalization, MaxPooling, SpatialDropout,
Flatten, Dense64(L2), Dropout0.3, Sigmoid 1개를 직접 구성했다. 초기 가중치는 새로 생성했으며
기존 6종 CNN이나 ImageNet 가중치를 가져오지 않았다.
Python 3.11.9 / TensorFlow 2.18.1 / Metal 1.2.0 / Keras 3.8.0 환경을 사용했다.
기존에 검증한 Metal meta optimizer 회피 설정과 매 epoch graph/eager Accuracy 일치 검사를 적용했다.

train에만 90도 회전·좌우/상하 반전·약한 밝기 증강을 적용했다.
Adam 0.0005, batch 32, 최대 25 epoch, EarlyStopping patience 6, seed 42다.
클래스 가중치는 train 수량으로만 계산했다. 정규화는 저장 모델의 Rescaling(1/255)에서 한 번 수행한다.
train loss는 클래스 가중치가 적용된 손실이고 validation loss는 원래 표본 분포의 손실이다.
**판정 임계값은 0.5로 고정**하고 validation Macro F1로 checkpoint를 선택했다.
실제 {run['epochs_completed']} epoch를 실행했으며 최고 checkpoint는 {run['best_epoch']} epoch다.

## 실측 validation 성능

| 지표 | 값 |
|---|---:|
| Accuracy | {metrics['accuracy']:.4f} |
| Balanced Accuracy | {metrics['balanced_accuracy']:.4f} |
| Macro F1 | {metrics['macro_f1']:.4f} |
| 불량 Precision | {metrics['defect_precision']:.4f} |
| 불량 Recall | {metrics['defect_recall']:.4f} |
| 불량 F1 | {metrics['defect_f1']:.4f} |
| 정상 오탐률 | {metrics['normal_false_positive_rate']:.4%} |
| 불량 미검출률 | {metrics['defect_false_negative_rate']:.4%} |
| ROC AUC | {metrics['roc_auc']:.4f} |
| Average Precision | {metrics['average_precision']:.4f} |

정상 정답 {metrics['true_normal']}개, 정상 오탐 {metrics['false_defect']}개,
불량 정답 {metrics['true_defect']}개, 불량 미검출 {metrics['missed_defect']}개다.
항상 정상으로 판정하는 기준선은 Accuracy {baseline['accuracy']:.4f}, Macro F1 {baseline['macro_f1']:.4f},
불량 Recall {baseline['defect_recall']:.4f}다. 정상 표본이 더 많으므로 Accuracy 하나로 판단하지 않는다.

{chr(10).join(normal_rows)}

{chr(10).join(recall_rows)}

이번 검증에서 Recall이 가장 낮은 원래 불량 유형은 {worst_recall.defect_type}
({worst_recall.recall:.4%})였다. 오분류 시각 검토에서는 정상 패드의 구멍·작은 밝은 점과
불량 구멍이 비슷하게 보이는 사례, 정상 연결부와 단락이 비슷한 사례가 관측됐다.
이는 시각적 관찰이며 주석 오류를 확정한 것은 아니다. 평가 라벨과 표본은 수정하지 않았다.

{verification_text}

이 값은 모델 선택에 사용한 그룹 validation 결과이며 최종 test 성능이 아니다.
정상/불량 이진 분류와 기존 6종 불량 유형 분류는 라벨·표본 구성이 달라 F1을 직접 비교하지 않는다.
그룹/유사 타일 독립성 및 공식 trainval/test의 그룹 공유 제한은 기존 보고서와 같다.
sigmoid는 보정되지 않은 모델 점수이며 실제 불량 발생 확률이나 모델 정확도로 해석하지 않는다.
DeepPCB의 합성 불량과 제한된 기판 패턴에서 얻은 결과이며 실제 생산 데이터의 성능을 보장하지 않는다.

## 사용 및 저장 파일

```python
from PIL import Image
from src.binary_predict import predict_binary_patch
with Image.open("path/to/circuit_patch.png") as image:
    result = predict_binary_patch("models/{run_name}/best.keras", image)
print(result["korean"], result["sigmoid_score"])
```

- 모델: `models/{run_name}/best.keras`, `metadata.json`.
- 데이터: `data/processed/binary_v1/`, `data/split/binary_v1/`.
- 결과: `results/binary/{run_name}/`의 CSV 로그, 학습 곡선, 혼동행렬/ROC/PR, 예측 전체,
  오분류 이미지/CSV, 원래 불량 유형별 Recall, 정상 출처별 오탐률.
- 이 보고서 및 데이터 감사: `results/binary/`.

노트북 재실행은 동일한 설정의 완료 모델을 재사용한다. 다른 설정은 새 run_name을 사용한다.
공식 test 최종 평가는 아직 실행하지 않았다.
'''
    path = reports / "binary_report.md"
    path.write_text(text, encoding="utf-8")
    return path
