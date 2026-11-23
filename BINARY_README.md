# PCB 패치 정상/불량 분류

새 노트북은 [binary_classification.ipynb](binary_classification.ipynb)다.
기존 `sample.ipynb`는 불량 6종 유형 분류이고, 이 실험은 **정상 0 / 불량 1** 분류다.
사용자가 선택한 입력은 잘라낸 개별 회로 패치다. 검사 대상 위치가 중심에 오도록 자른 패치를 사용한다.

## 실행

PyCharm에서 기존 Python 3.11.9 `venv` 커널로 새 노트북을 위에서부터 실행한다.
실제 학습은 노트북 커널 안에서 수행하며 완료된 같은 run은 재사용한다.
설정이 다른 실험에는 새 run_name을 지정한다.

```bash
cd /Users/yeongjunna/DNN_project
MPLBACKEND=Agg venv/bin/python -m src.binary_data
MPLBACKEND=Agg venv/bin/python -m src.binary_train --run-name binary_cnn_001 --epochs 25
MPLBACKEND=Agg venv/bin/python -m unittest discover -s tests -p test_binary.py -v
```

기존 설치 환경과 `requirements.txt`를 그대로 사용한다. 추가 패키지 설치는 필요하지 않다.
학습 최대 epoch는 25, batch 32, Adam 0.0005, EarlyStopping patience 6이다.
기존에 확인한 Metal 그래프 최적화 문제의 회피 설정과 graph/eager Accuracy 검사를 사용한다.

## 데이터

| 분할 | 정상 패치 | 불량 패치 | 원본 쌍 |
|---|---:|---:|---:|
| train | 9,998 | 5,338 | 800 |
| validation | 2,880 | 1,535 | 200 |

원본 그룹 분할은 기존 실험과 동일하며 `group00041` 전체를 validation으로 유지한다.
같은 원본의 정상/불량/템플릿 패치를 다른 split으로 나누지 않는다.
공식 test는 데이터 생성·학습·모델 선택에서 사용하지 않는다.

정상은 두 경로로 만든다.

1. **template_matched**: 불량 bbox와 같은 위치에서 자른 정상 기준 영상.
2. **tested_clear**: 검사 영상에서 모든 불량 bbox와 guard 8px를 피한 회로 영역.

검사 영상의 정상 후보는 주석이 완전하다는 가정에 따른 라벨이다. 흑색 픽셀 비율 5~95%의
회로 패턴을 선택해 빈 배경만 정상으로 학습하는 것을 줄였다. 출처 차이만 배우는 문제를
확인하기 위해 두 정상 출처의 오탐률을 별도로 저장한다.

후보 20,588개 중 split 사이 동일 픽셀 패치 837개를 격리해 19,751개를 사용했다.
정상 회로 ROI 샘플링 실패 31건과 모든 제외 사유를 기록했다.
격리 규칙은 학습 전에 고정했으며 성능을 보고 표본을 삭제하지 않는다.

Crop은 여백 16px 정사각형, edge padding, bilinear 128×128, grayscale PNG다.
모델 내부 Rescaling이 한 번 정규화하므로 모델에 넣는 픽셀 범위는 **0~255**다.
train에만 회전·반전·약한 밝기 증강을 적용하며 클래스 가중치는 train 수량으로 계산한다.

## 모델과 평가

Conv 블록 16→32→64(각 Conv2D 2회), BatchNormalization, MaxPooling, SpatialDropout,
Flatten, Dense64(L2), Dropout0.3, Sigmoid 1개를 직접 구성한다.
기존 6종 모델이나 ImageNet 가중치를 가져오지 않고 새로 학습한다.

임계값은 **0.5**로 고정하고 `sigmoid_score > 0.5`이면 불량으로 판정한다.
validation Macro F1로 최고 checkpoint를 저장하고 재로드해 평가한다.
Accuracy, Balanced Accuracy, Macro F1, 불량 Precision/Recall/F1, 정상 오탐률,
불량 미검출률, ROC AUC, Average Precision, 혼동행렬, 오분류를 저장한다.
항상 정상으로 예측하는 기준선과 원래 불량 6종별 Recall도 제공한다.

실측 결과는 [이진 분류 보고서](results/binary/binary_report.md)에서 확인한다.
이는 모델 선택에도 사용한 validation 성능이며 공식 test 성능과 구분한다.

| 실측 validation 지표 | 값 |
|---|---:|
| Accuracy | 0.9887 |
| Macro F1 | 0.9876 |
| 불량 Recall | 0.9896 |
| 정상 오탐률 | 1.18% |

18 epoch에서 조기 종료됐으며 최고 모델은 12 epoch다. 정상 오탐 34/2,880개,
불량 미검출 16/1,535개였다. CPU에서 validation 4,415개를 다시 추론했고 Metal과
모든 예측 라벨이 일치했다. 독립 추론, 노트북 재실행, 전체 테스트 18개도 통과했다.

## 독립 추론

```python
from PIL import Image
from src.binary_predict import predict_binary_patch

with Image.open('path/to/circuit_patch.png') as image:
    result = predict_binary_patch('models/binary_cnn_001/best.keras', image)
print(result['korean'], result['sigmoid_score'])
```

입력은 개별 회로 패치다. 전체 PCB 이미지의 불량 위치 탐지나 전수 검사 기능은 제공하지 않는다.
sigmoid 점수는 실제 불량 발생 확률이나 모델 정확도가 아니다.

```text
binary_classification.ipynb
src/binary_data.py               # 정상/불량 패치 생성·분할 감사
src/binary_model.py              # 직접 설계 이진 CNN
src/binary_train.py              # 온라인 로딩·학습
src/binary_evaluate.py           # 이진 지표·그래프·오분류
src/binary_predict.py            # 학습과 독립된 추론
src/binary_report.py             # 실측 보고서
tests/test_binary.py
data/processed/binary_v1/
data/split/binary_v1/
models/binary_cnn_001/
results/binary/
├── data_summary.json / data_counts.csv
├── excluded_patches.csv / sampling_failures.csv
├── data_examples.png / matched_pair_examples.png
├── inference_verification.json / binary_report.md
└── binary_cnn_001/
    ├── history.csv / learning_curves.png / run_result.json
    ├── validation_metrics.json / validation_predictions.csv
    ├── classification_report.csv / confusion_matrix.npy / evaluation.png
    ├── normal_source_false_positives.csv / recall_by_defect_type.csv
    └── misclassified.csv / misclassified.png
```
