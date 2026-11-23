"""원본 무작위 초기화 주 실험과 과거 전이학습 기록을 명확히 구분한다."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


@plt.style.context("default")
def write_scratch_report(root: str | Path):
    root = Path(root).resolve()
    folder = root / "results/full_scratch"
    rows = []
    for area, name, initialization in [
        ("full_image", "full_frozen_001", "patch_transfer_reference"),
        ("full_image", "full_finetuned_001", "patch_transfer_reference"),
        ("full_scratch", "full_scratch_aux_001", "random_from_scratch_primary")]:
        path = root / "results" / area / name / "run_result.json"
        if path.exists():
            result = json.loads(path.read_text())
            rows.append(dict(run_name=name, initialization=initialization, epochs=result["epochs_completed"],
                             best_epoch=result["best_epoch"], **result["metrics"]))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(folder / "model_comparison.csv", index=False)
    primary = comparison[comparison.initialization == "random_from_scratch_primary"]
    if primary.empty:
        raise ValueError("처음부터 학습한 주 실험이 아직 완료되지 않았습니다.")
    best = primary.sort_values("macro_f1", ascending=False).iloc[0]
    (folder / "selected_model.json").write_text(json.dumps(dict(
        run_name=best.run_name, model_path=f"models/{best.run_name}/best.keras", initialization="random",
        patch_weights_used=False, input_images=1, reference_image_required=False, mask_input_required=False,
        input_size=[640, 640, 1], threshold=.5, validation_macro_f1=float(best.macro_f1), official_test_evaluated=False), indent=2))
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(comparison))
    for offset, metric in [(-.25, "macro_f1"), (0, "defect_recall"), (.25, "normal_false_positive_rate")]:
        bars = ax.bar(x+offset, comparison[metric], width=.25, label=metric)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(x, comparison.run_name, rotation=10)
    ax.set(ylim=(0, 1.1), ylabel="Full-image validation score", title="Primary model trained from random initialization")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(folder / "model_comparison.png", dpi=150)
    plt.close(fig)
    manifest = pd.read_csv(root / "data/split/full_scratch_aux_v1/images.csv")
    fig, axes = plt.subplots(3, 2, figsize=(9, 12))
    for axes_row, row in zip(axes, manifest[manifest.target == 1].head(3).itertuples()):
        with Image.open(root / row.image_path) as image:
            axes_row[0].imshow(np.asarray(image.convert("L")), cmap="gray", vmin=0, vmax=255)
        with Image.open(root / row.mask_path) as mask:
            axes_row[1].imshow(np.asarray(mask), cmap="magma", vmin=0, vmax=255)
        axes_row[0].set_title("Only model input: original 640x640")
        axes_row[1].set_title("Training target only; never an input")
        for ax in axes_row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(folder / "training_target_examples.png", dpi=130)
    plt.close(fig)
    # 같은 기판 쌍의 점수를 함께 기록하되 추론에는 여전히 사진 한 장씩만 사용한다.
    predictions = pd.read_csv(folder / best.run_name / "validation_predictions.csv")
    pairs = predictions.pivot(index="source_id", columns="label", values="sigmoid_score")
    pairs["defect_score_higher"] = pairs.defect > pairs.normal
    pairs["both_correct"] = (pairs.normal <= .5) & (pairs.defect > .5)
    pairs["score_margin"] = pairs.defect-pairs.normal
    pairs.to_csv(folder / "paired_validation.csv")
    pair_summary = dict(pairs=len(pairs), both_correct=int(pairs.both_correct.sum()),
                        defect_score_higher=int(pairs.defect_score_higher.sum()),
                        median_score_margin=float(pairs.score_margin.median()))
    (folder / "paired_validation_summary.json").write_text(json.dumps(pair_summary, indent=2))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pairs.normal, pairs.defect, s=15, alpha=.6)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.axhline(.5, color="gray", linewidth=.5)
    ax.axvline(.5, color="gray", linewidth=.5)
    ax.set(xlabel="Normal reference score", ylabel="Defective image score",
           title="One-image inference, grouped afterward by source", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(folder / "paired_scores.png", dpi=150)
    plt.close(fig)
    table = ['| 모델 | 초기화 | Accuracy | Macro F1 | 불량 Recall | 정상 오탐률 |', '|---|---|---:|---:|---:|---:|']
    table += [f'| {row.run_name} | {row.initialization} | {row.accuracy:.4f} | {row.macro_f1:.4f} | {row.defect_recall:.4f} | {row.normal_false_positive_rate:.2%} |'
              for row in comparison.itertuples()]
    verification = folder / "inference_verification.json"
    verification_text = 'CPU/독립 단일 사진 추론은 노트북 마지막 셀에서 확인한다.'
    if verification.exists():
        check = json.loads(verification.read_text())
        verification_text = (f"원본 validation {check['samples']}장 전체의 CPU/Metal 판정 일치율은 "
                             f"{check['label_agreement']:.2%}, 최대 점수 차이는 {check['max_cpu_gpu_delta']:.3g}였다. "
                             f"독립 한 장 추론의 최대 점수 차이는 {check['max_inference_delta']:.3g}였다.")
    text = f'''# 원본 한 장 분류 — 처음부터 학습한 주 실험

## 확정된 요구사항

새 파일 `full_image_from_scratch.ipynb`에서 진행했다. 입력은 **원본 640×640 사진 한 장**이며
정상 기준 사진, bbox, 정답 마스크를 입력받지 않는다. 기존 패치 모델과 사전학습 가중치는 사용하지 않았다.
기존 `full_image_classification.ipynb`는 전이학습 비교 기록으로 보존했다.

학습 원본은 정상 800장/불량 800장, 검증 원본은 정상 200장/불량 200장이다.
같은 PCB 쌍을 한 split에 넣고 `group00041`을 고정 검증 그룹으로 유지했다.
원본을 Crop하거나 resize하지 않았다. 모든 입력 픽셀은 저장 모델에서 1/255 정규화한다.
공식 test는 학습·선택에 사용하지 않았다.

## 현재 방식

CNN의 모든 가중치를 새로 생성했다. 가벼운 encoder/decoder와 학습된 attention을 사용한다.
정상/불량 이미지 라벨뿐 아니라, 기존 위치 주석과 정렬된 원본 쌍의 차이를 학습 보조 정답으로 사용했다.
정상 기준과 검사 원본의 밝기 차이가 128 이상이고 bbox 내부인 픽셀을 결함 정답으로 만들고,
작은 변화도 남게 max pooling해 160×160 정답 맵을 저장했다. 이 맵은 **입력이 아닌 학습 target**이다.
정상 원본의 정답 맵은 0이다. 보조 loss의 픽셀 가중치는 train 양성 픽셀 비율로만 계산했다.

학습 입력 텐서는 하나이며 shape은 `(batch,640,640,1)`이다. 위치 감독은 추가 loss를 통해
결함 특징을 배우도록 돕는다. 배포 `.keras`에서는 정상/불량 출력 하나만 제공한다.
같은 입력으로 예측한 영역을 attention에 사용하며 정답 영역을 attention 입력으로 주입하지 않는다.

Adam 0.0003, clipnorm 1.0, batch 4, 최대 20 epoch, EarlyStopping patience 5, seed 42다.
학습할 때만 원본/정답 맵에 같은 회전·반전을 적용한다. 임계값은 0.5로 고정했다.
최고 모델은 이미지 단위 validation Macro F1로 저장 후 재로드했다.
초기 분류-only GroupNorm pilot은 5 epoch 후 중단했고 최종 비교에서 제외했다.
이 미완료 pilot은 초기 검증 Accuracy 0.95에 도달했으나 수렴 상한을 평가한 실험은 아니다.

## 실측 비교

{chr(10).join(table)}

현재 주 실험 `{best.run_name}`은 검증 원본 400장 중 정상 오탐 {int(best.false_defect)}장,
불량 미검출 {int(best.missed_defect)}장이었다. 실제 {int(best.epochs)} epoch를 실행했고
최고 checkpoint는 {int(best.best_epoch)} epoch다.

{verification_text}

같은 PCB 쌍을 추론 후 묶어 점수를 분석하면 {pair_summary['pairs']}쌍 중
{pair_summary['both_correct']}쌍에서 정상/불량 양쪽 판정이 모두 맞았다.
{pair_summary['defect_score_higher']}쌍에서 불량 점수가 정상 점수보다 높았다.
이 분석은 쌍을 입력받는 모델이라는 의미가 아니며 출처 혼입 가능성을 배제하는 증거도 아니다.
쌍별 점수와 판정은 `paired_validation.csv`, 산점도는 `paired_scores.png`에 저장했다.

## 해석과 후속 검증

- 수치는 고정 그룹 validation 결과이며 독립 최종 test 성능이 아니다.
- 이전 패치 모델의 4,415개 패치 지표와 원본 400장 지표는 직접 비교하지 않는다.
- 이번 주 실험의 가중치는 무작위 초기화이며, 과거 전이 모델은 참고 결과로만 표시했다.
- 정상 원본이 기준 사진, 불량 원본이 검사 사진이라는 출처 혼입 가능성은 남아 있다.
  보조 위치 감독을 썼다는 사실만으로 실제 결함만 배웠다고 단정하지 않는다.
- DeepPCB의 합성 불량과 제한된 기판 패턴, 공식 trainval/test의 그룹 공유 제한이 있다.
- 단일 사진의 정상/불량만 검증했다. 배포 출력의 sigmoid는 실제 불량 확률이나 정확도가 아니다.
- 160×160 보조 정답은 학습 감독에 사용했으며, 위치 탐지 성능을 최종 지표로 검증한 것은 아니다.

## 한 장 추론

```python
from PIL import Image
from src.full_image_predict import predict_full_image
with Image.open("path/to/pcb_640.png") as image:
    result = predict_full_image("models/{best.run_name}/best.keras", image)
print(result["korean"], result["sigmoid_score"])
```

결과 경로는 `results/full_scratch/selected_model.json`에 기록했다.
학습 로그, 원본 예측 CSV, 분류 보고서, 혼동행렬/ROC/PR, 오분류는 `results/full_scratch/{best.run_name}/`에 있다.
모델은 `models/{best.run_name}/best.keras`와 `metadata.json`이다.
학습용 정답 맵은 `data/processed/full_scratch_aux_v1/`, 목록은 `data/split/full_scratch_aux_v1/`에 있다.
원본 파일과 이전 노트북/모델은 변경하지 않았다.
'''
    (folder / "from_scratch_report.md").write_text(text, encoding="utf-8")
    return comparison
