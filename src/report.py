"""전처리 및 검증 실험의 현재 상태를 실제 저장 산출물로 보고한다."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.eda import CLASSES


@plt.style.context("default")
def class_metric_figure(output: Path) -> None:
    """기존 평가 JSON만 읽어 클래스별 Precision/Recall/F1을 시각화한다."""
    report = json.loads((output / "classification_report.json").read_text())
    names = [CLASSES[i][0] for i in CLASSES]
    report_table = pd.DataFrame({name: values for name, values in report.items() if isinstance(values, dict)}).T
    report_table.loc["accuracy", ["f1-score", "support"]] = [report["accuracy"], sum(report[name]["support"] for name in names)]
    report_table.to_csv(output / "classification_report.csv")
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(6)
    for offset, metric in [(-.25, "precision"), (0, "recall"), (.25, "f1-score")]:
        ax.bar(x+offset, [report[name][metric] for name in names], width=.25, label=metric)
    ax.set_xticks(x, names, rotation=15)
    ax.set(ylim=(0, 1.05), ylabel="Validation score", title="Per-class validation metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "class_metrics.png", dpi=150)
    plt.close(fig)


@plt.style.context("default")
def write_progress_report(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root).resolve()
    reports = root / "results/reports"
    pre = json.loads((reports / "preprocessing_summary.json").read_text())
    rows = []
    for path in sorted((root / "results/training").glob("*/run_result.json")):
        result = json.loads(path.read_text())
        signature = json.loads((path.parent / "config.json").read_text())
        # 그래프 출력 불일치가 있었던 초기 run은 명시적으로 비교에서 제외한다.
        safe = "execution" in signature
        if not safe:
            continue
        class_metric_figure(path.parent)
        rows.append(dict(run_name=result["run_name"], variant=signature["config"]["variant"],
                         augment=signature["config"]["augment"],
                         epochs=result["epochs_completed"], best_epoch=result["best_epoch"],
                         **result["metrics"]))
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise ValueError("완료된 안전 설정의 학습 결과가 없습니다.")
    comparison.to_csv(reports / "model_comparison_validation.csv", index=False)
    best = comparison.sort_values("macro_f1", ascending=False).iloc[0]
    selected = dict(run_name=best.run_name, selection_metric="validation macro_f1",
                    validation_macro_f1=float(best.macro_f1), official_test_evaluated=False,
                    model_path=f"models/{best.run_name}/best.keras", status="candidate; not final test result")
    (reports / "selected_model.json").write_text(json.dumps(selected, indent=2))
    predictions = pd.read_csv(root / "results/training" / best.run_name / "validation_predictions.csv")
    errors = predictions[predictions.target_id != predictions.predicted_target]
    pairs = errors.groupby(["class_name", "predicted_target"]).size().sort_values(ascending=False)
    error_lines = [f"- {name} → {CLASSES[int(target)+1][0]}: {int(count)}개"
                   for (name, target), count in pairs.head(5).items()]
    pairs.rename("count").to_csv(reports / "confusion_pairs_validation.csv")
    n_errors = len(errors)
    parity_path = reports / "cpu_metal_validation_parity.json"
    parity_text = '최종 후보의 전체 validation CPU/Metal 대조는 별도 실행 기록을 확인한다.'
    if parity_path.exists():
        parity = json.loads(parity_path.read_text())
        parity_text = (f"최종 B의 validation {parity['samples']}개 전체를 CPU에서 다시 추론했다. "
                       f"Metal 평가와 예측 라벨 일치율은 {parity['prediction_agreement']:.2%}, "
                       f"6개 점수의 최대 절대 차이는 {parity['max_score_delta']:.3g}였다. "
                       "실측 기록은 `cpu_metal_validation_parity.json`에 저장했다.")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(comparison))
    for offset, metric in [(-.25, "accuracy"), (0, "macro_f1"), (.25, "weighted_f1")]:
        bars = ax.bar(x+offset, comparison[metric], width=.25, label=metric)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(x, comparison.run_name, rotation=10)
    ax.set(ylim=(0, 1.1), ylabel="Validation score", title="Fixed group holdout — official test remains reserved")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(root / "results/figures/model_comparison_validation.png", dpi=150)
    plt.close(fig)
    count_rows = pd.read_csv(reports / "preprocessing_class_counts.csv")
    table = ['| 유형 | train | validation |', '|---|---:|---:|']
    table += [f'| {row.class_name} | {row.train} | {row.val} |' for row in count_rows.itertuples()]
    model_table = ['| 실행 | Accuracy | Macro F1 | Weighted F1 | 최고 epoch |', '|---|---:|---:|---:|---:|']
    model_table += [f'| {row.run_name} | {row.accuracy:.4f} | {row.macro_f1:.4f} | {row.weighted_f1:.4f} | {row.best_epoch} |'
                    for row in comparison.itertuples()]
    text = f'''# 2~3단계 전처리 및 CNN 검증 결과

## 실제 실행 범위

Python 3.11.9를 유지했다. TensorFlow 2.18.1 / Metal 1.2.0 / Keras 3.8.0 / NumPy 2.0.2를 설치했다.
M1 Pro GPU 열거와 실제 행렬 연산을 확인했고, CNN 학습과 저장 checkpoint 재로드 평가를 수행했다.
공식 test는 전처리와 모델 선택에 사용하지 않았으며 최종 성능을 아직 측정하지 않았다.

## 고정 분할 및 Crop

- train: {pre['source_counts']['train']}개 원본 / {pre['patch_counts']['train']}개 패치.
- validation: `group00041` 전체 {pre['source_counts']['val']}개 원본 / {pre['patch_counts']['val']}개 패치.
- test: 공식 {pre['source_counts']['test']}쌍 예약. 생성한 test 패치 {pre['test_patches_created']}개.
- 그룹은 성능을 보기 전 trainval의 이미지 수 20%에 가장 가까운 기준으로 고정했다.
- train/validation의 원본·그룹·정상/검사 영상 픽셀 중복 및 생성 패치의 동일 픽셀 중복을 검사했다.
- 원본 bbox 주변 16px 여백, 정사각형, edge padding, bilinear resize 128×128, grayscale PNG.
- README가 좌표 끝점의 포함 여부를 명시하지 않아 오른쪽/아래를 +1px 보수적으로 포함했다.
  원본 좌표와 반열린 ROI를 모두 manifest에 남겨 재현할 수 있다.
- 이웃 bbox가 ROI에 걸친 패치 {pre['neighbor_patches']}개, 경계 padding 패치 {pre['padded_patches']}개.
  해당 표본을 삭제하지 않았고 validation 진단 결과를 각 실행의 `validation_slices.csv`에 저장했다.
- 정규화는 저장 모델의 Rescaling에서 1/255로 한 번 수행한다. 로더/추론에서 이중 정규화하지 않는다.
- 증강 사용 실험은 train에만 90도 회전/좌우/상하 반전/약한 밝기 조절을 적용한다.
  epoch마다 온라인 적용하며 증강본을 validation 또는 디스크에 추가하지 않는다.

{chr(10).join(table)}

## Metal 실행 오류와 수정 근거

초기 `A_baseline_001`은 Keras graph Accuracy와 NumPy eager Accuracy가 달랐다.
같은 가중치·입력에서 CPU eager/graph 및 Metal eager는 약 1e-7 이내로 일치했으나,
Metal graph는 최대 약 0.415의 점수 차이가 있었다. 초기 실험은 성능 비교에서 제외했다.
`tf.config.optimizer.set_experimental_options({{"disable_meta_optimizer": True}})`를 적용한
후 진단 batch의 Metal graph/eager 출력 차이는 0이었다. GPU 사용은 유지했다.

이 결과는 현재 장비·설치 조합에서의 재현 관찰이며 다른 모든 TensorFlow/Metal 조합의 오류를 주장하지 않는다.
근거는 `metal_diagnostic.json`, `metal_workaround.json`, 초기 실행의 `history.csv`다.
수정 후 매 epoch마다 graph Keras Accuracy와 eager/NumPy Accuracy 차이가 1e-4보다 크면 즉시 실패하도록 했다.
학습/독립 추론 함수 모두 같은 회피 설정을 적용한다.

{parity_text}

## 검증 성능 (공식 test 성능 아님)

{chr(10).join(model_table)}

현재 validation Macro F1 기준 후보는 **{best.run_name} / {best.macro_f1:.4f}**다.
모든 지표는 동일 고정 그룹에서 측정했으며 최대 검증 F1의 checkpoint를 파일에서 다시 불러와 평가했다.
이 검증 집합은 모델 선택에도 쓰였으므로 독립적인 최종 일반화 성능으로 해석하지 않는다.
기판/인접 타일 유사도 전체를 배제한 것은 아니며 공식 trainval/test의 그룹 겹침 제한도 남아 있다.

현재 후보의 validation 오분류는 **{n_errors} / {len(predictions)}개**다. 많이 관측된 혼동은 다음과 같다.

{chr(10).join(error_lines) if error_lines else '- 오분류 없음.'}

train 지표는 증강과 Dropout이 켜진 상태, validation 지표는 둘 다 꺼진 상태이므로
validation 점수가 train보다 높다는 사실만으로 누수나 평가 오류로 단정하지 않는다.
검증 그룹 자체의 난이도도 다를 수 있다. 이웃/패딩 slice는 클래스 구성과 표본 수가 다르며
특히 작은 slice의 Macro F1 차이를 단독 원인으로 해석하지 않는다.
slice의 Macro F1은 고정된 6종 ID를 기준으로 계산해 그 slice에 없는 클래스도 0으로 포함한다.
따라서 4종만 존재하는 padding slice는 전부 정답이어도 Macro F1이 4/6이며,
이 값과 전체 validation Macro F1을 단순 비교해서는 안 된다.

오분류 시각 검토에서는 Mousebite→Pin Hole 사례의 흰색 영역이 Crop 안에서 구멍처럼 보였다.
이는 모양에 대한 관찰이며 라벨 오류를 확정한 것은 아니다. 원본/정상 기준 영상과 대조하는
수동 검토 후보로 기록하고, 현재 검증 라벨이나 평가 표본은 수정/삭제하지 않았다.

Model A는 요청된 기본 CNN(32→64→128, GAP, Dense64, Dropout0.3)이다.
Model B는 2회 합성곱 블록(24→48→96), BatchNormalization, SpatialDropout, Flatten,
Dense96의 L2 규제와 Dropout0.4를 사용해 중심 불량의 위치 관계를 유지한다.
ImageNet 가중치는 A/B에서 사용하지 않는다. 여러 요소를 동시에 바꾼 A/B 비교만으로
개별 요소의 인과 효과를 주장하지 않는다. Model C 전이학습 비교는 후속 단계다.

## 저장 파일과 후속 작업

- `data/split/`: 잠긴 설정, 원본별 split, train/val 패치 manifest, 예약된 test 메타데이터.
- `data/processed/roi128_v1/`: 실제 train/val PNG 패치.
- `models/<run>/best.keras`, `metadata.json`: 최고 checkpoint와 클래스/정규화 계약.
- `results/training/<run>/`: 설정, 실행 환경, epoch 로그, 학습 곡선, Accuracy/Precision/Recall/Macro·Weighted F1,
  클래스별 보고서, 혼동행렬, 전체 예측, 오분류 이미지/목록, 이웃/패딩 진단.
- `src/predict.py`: 학습 코드와 분리된 단일 패치 추론. 6종 결함의 상대 점수이며 정상 판별/불량 탐지가 아니다.

후속에는 A/B의 오분류와 표본 모호성을 검토하고, 필요 시 문맥 크기 한 가지씩 비교한다.
Model C와 추가 그룹 검증 후 설정을 확정한 뒤 공식 test를 최초 최종 평가한다.
Streamlit은 그 뒤에 실제 검증된 모델과 보고서를 연결한다.

## 참고 문서

- [Apple tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/)
- [TensorFlow 그래프 최적화 설정](https://www.tensorflow.org/api_docs/python/tf/config/optimizer/set_experimental_options)
- [DeepPCB 주석 형식](https://github.com/tangsanli5201/DeepPCB)
'''
    (reports / "training_progress_report.md").write_text(text, encoding="utf-8")
    return comparison
