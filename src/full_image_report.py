"""단일 원본 이미지 모델의 비교·선택 및 사용 범위를 기록한다."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@plt.style.context("default")
def write_full_image_report(root: str | Path) -> pd.DataFrame:
    root = Path(root).resolve()
    reports = root / "results/full_image"
    rows = []
    for name in ["full_frozen_001", "full_finetuned_001"]:
        path = reports / name / "run_result.json"
        if path.exists():
            run = json.loads(path.read_text())
            rows.append(dict(run_name=name, best_epoch=run["best_epoch"], epochs=run["epochs_completed"], **run["metrics"]))
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise ValueError("완료된 원본 모델 실행이 없습니다.")
    # 동점이면 먼저 학습한 특징 고정 모델을 유지한다.
    best = comparison.sort_values("macro_f1", ascending=False, kind="stable").iloc[0]
    comparison.to_csv(reports / "model_comparison.csv", index=False)
    selected = dict(run_name=best.run_name, model_path=f"models/{best.run_name}/best.keras",
                    input_images=1, input_size=[640, 640, 1], reference_image_required=False,
                    crop=False, resize=False, validation_macro_f1=float(best.macro_f1),
                    threshold=.5, selection="validation Macro F1; frozen model wins ties", official_test_evaluated=False)
    (reports / "selected_model.json").write_text(json.dumps(selected, indent=2))
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(comparison))
    for offset, metric in [(-.25, "macro_f1"), (0, "defect_recall"), (.25, "normal_false_positive_rate")]:
        bars = ax.bar(x+offset, comparison[metric], width=.25, label=metric)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(x, comparison.run_name)
    ax.set(ylabel="Full-image validation score", ylim=(0, 1.1), title="Single 640x640 image; threshold 0.5")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(reports / "model_comparison.png", dpi=150)
    plt.close(fig)
    method_rows = ['| 원본 모델 | Accuracy | Macro F1 | 불량 Recall | 정상 오탐률 | 최고 epoch |',
                   '|---|---:|---:|---:|---:|---:|']
    method_rows += [f'| {row.run_name} | {row.accuracy:.4f} | {row.macro_f1:.4f} | {row.defect_recall:.4f} | {row.normal_false_positive_rate:.2%} | {row.best_epoch} |'
                    for row in comparison.itertuples()]
    check_path = reports / "inference_verification.json"
    check_text = "CPU 및 독립 추론 대조는 노트북의 마지막 검증 셀에서 수행한다."
    if check_path.exists():
        check = json.loads(check_path.read_text())
        check_text = (f"선택 모델의 validation {check['samples']}장 전체를 CPU에서 재추론했다. "
                      f"Metal과 판정 일치율 {check['label_agreement']:.2%}, 최대 점수 차이 {check['max_cpu_gpu_delta']:.3g}였다. "
                      f"단일 사진 독립 추론과 저장 평가 점수의 최대 차이는 {check['max_inference_delta']:.3g}였다.")
    text = f'''# 원본 PCB 사진 한 장 정상/불량 분류

## 실제 입력과 라벨

`full_image_classification.ipynb`에서 원본 640×640 사진 한 장을 입력받는 모델을 학습했다.
추론에는 정상 기준 사진, bbox 좌표, 미리 잘라 놓은 패치가 필요하지 않다.
학습은 원본 `_temp.jpg`를 정상 0, `_test.jpg`를 불량 1로 사용한다.
정상 기준 사진도 학습 표본으로서 각각 별도 한 장씩 입력하며 두 사진을 입력 채널로 합치지 않는다.

학습: 정상 800장 + 불량 800장. validation: 정상 200장 + 불량 200장.
같은 원본 쌍을 한 split에 배정했고 기존 `group00041` 검증 그룹을 유지했다.
원본 파일의 픽셀 해시와 640×640 크기를 검사했으며 train/validation의 그룹·픽셀 중복은 없다.
train 내부에 원래 데이터셋의 정상 템플릿 반복 2행은 있으나 다른 split으로 넘어가지 않는다.
공식 test는 raw 이미지/라벨을 읽거나 모델 선택에 사용하지 않았다.

## 방법 선택의 이유

원본 전체를 128×128로 줄이면 작은 결함 정보가 줄어든다. 입력 해상도를 640×640으로 유지하고,
기존 이진 패치 CNN에서 학습한 합성곱 가중치를 복제했다. 패치의 Flatten/Dense 분류기는 사용하지 않았다.
각 합성곱 블록의 공간 GlobalMax/GlobalAverage 통계를 결합해 224차원 전체 이미지 특징을 만들었다.
GlobalMax는 강한 국소 반응을, GlobalAverage는 전체 반응 통계를 제공한다. 이것이 결함을
잘 구별하는지는 이 데이터의 검증 결과로 판단했으며 일반적인 최적해라고 주장하지 않는다.

1차: 합성곱 특징을 고정하고 원본 전체 특징으로 Dense64 분류기를 학습했다.
train의 원본/90도 전체 회전 두 view로 3,200개 특징을 만들고 validation은 원본 400장만 사용했다.
픽셀 정규화는 encoder의 Rescaling(1/255)에서 한 번만 수행한다. 특징 평균/분산은 train으로만 계산했다.
원본 전체 합성곱 특징 추출은 GPU, 작은 분류기 학습은 CPU를 사용했다.
최종 `.keras`에는 encoder와 분류기가 함께 들어 있어 사용자 추론 시 특징 캐시 파일이 필요하지 않다.

2차: 1차 모델에서 시작해 원본 전체를 넣고 마지막 Conv2D 두 층과 분류기를 미세조정했다.
BN 통계는 고정하고 train에만 전체 회전/반전을 적용했다. batch 2, Adam 0.00001,
최대 6 epoch, EarlyStopping patience 3을 사용했다. 초기 모델을 epoch 0 checkpoint로 보존해
개선이 없으면 원래 모델을 유지한다. 모든 선택은 threshold 0.5의 validation Macro F1 기준이다.

특징 벡터만 작은 NPZ로 캐시했고 새 Crop PNG 데이터셋은 만들지 않았다.
원본 CNN 학습 입력/추론 입력을 축소하거나 잘라내지 않았다. 학습/추론 모두 Pillow grayscale 디코더를 사용한다.

## 실측 원본 단위 validation 결과

{chr(10).join(method_rows)}

현재 비교한 모델 중 선택 후보는 **{best.run_name} / Macro F1 {best.macro_f1:.4f}**다.
선택 모델은 정상 200장 중 {int(best.false_defect)}장을 불량으로 오탐했고,
불량 200장 중 {int(best.missed_defect)}장을 놓쳤다.
미세조정 최고 epoch가 0이면 미세조정보다 초기 모델이 좋아 초기 checkpoint를 유지했다는 의미다.

{check_text}

## 해석의 제한

- 모든 수치는 원본 400장 validation 결과다. 기존 패치 4,415개 분류 결과와 평가 단위가 달라 직접 비교할 수 없다.
- 같은 validation 그룹이 기존 패치 모델의 checkpoint 선택에도 사용됐다. 이번 값은 독립 최종 테스트 성능이 아니다.
- 정상 원본이 기준 영상, 불량 원본이 검사 영상으로 구분되는 데이터 출처 혼입 가능성이 있다.
  패치 사전학습은 검사 영상의 정상 영역도 사용했지만 원본 단계의 출처 단서가 완전히 제거됐다고 보장할 수 없다.
- DeepPCB의 합성 불량/제한된 기판 패턴과 공식 trainval/test 그룹 공유 제한이 남아 있다.
  실제 정상 검사 영상과 다른 기판에서의 추가 검증이 필요하다.
- sigmoid는 보정되지 않은 모델 점수다. 실제 불량 확률이나 모델 정확도로 해석하지 않는다.
- 이 모델은 전체 사진의 정상/불량만 판정한다. 불량 위치와 불량 6종은 출력하지 않는다.

## 한 장 추론 예시

```python
from PIL import Image
from src.full_image_predict import predict_full_image
with Image.open("path/to/pcb_640.png") as image:
    result = predict_full_image("models/{best.run_name}/best.keras", image)
print(result["korean"], result["sigmoid_score"])
```

현재 입력 계약은 640×640 단일 이미지다. 다른 크기를 임의로 줄이지 않고 명시적으로 오류를 반환한다.
선택 모델 경로는 `results/full_image/selected_model.json`에서 읽을 수 있다.

## 저장 산출물

- `data/split/full_image_v1/`: 원본 경로·라벨·그룹·split·픽셀 해시.
- `data/processed/full_image_v1/`: 원본 전체 특징 캐시 NPZ와 설정.
- `models/full_frozen_001/`, `models/full_finetuned_001/`: 단일 원본 모델과 입력 계약.
- `results/full_image/<run>/`: 학습 로그/곡선, 설정, 분류 보고서, 혼동행렬/ROC/PR,
  원본별 예측과 오분류 원본 예시.
- `results/full_image/`: 모델 비교·선택·이 보고서와 CPU/추론 대조.

참고: [Keras GlobalMaxPooling2D](https://keras.io/api/layers/pooling_layers/global_max_pooling2d/),
[Keras 전이학습/BN 미세조정 가이드](https://keras.io/guides/transfer_learning/).
'''
    (reports / "full_image_report.md").write_text(text, encoding="utf-8")
    return comparison
