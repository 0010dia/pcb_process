# PCB 개별 패치 정상/불량 분류 결과

## 목적 및 입력

`binary_classification.ipynb`에서 기존 6종 분류와 별도로 실행했다. 입력은 사용자가 선택한
128×128 개별 회로 패치다. `0=normal`, `1=defect`이며 기존 6종 불량을 모두 불량 1로 합쳤다.
정상 기준 영상을 추론 입력으로 요구하지 않는 단일 이미지 CNN이다. 전체 PCB의 불량 위치 탐지나
전체 이미지에 결함이 있는지의 성능은 이번에 평가하지 않았다.

## 데이터 생성과 누수 방지

기존 `sample.ipynb`, 원본, 6종 모델은 변경하지 않고 `binary_v1` 및 별도 모델/결과 폴더를 사용했다.
기존 train 800개/validation 200개 원본 그룹 분할과 검증 그룹 `group00041`을 그대로 유지했다.
같은 원본의 정상/불량/템플릿 패치는 분할을 넘지 않는다. 공식 test는 읽거나 Crop하지 않았다.

| 분할 | 종류 | 라벨 | 패치 수 |
|---|---|---|---:|
| train | defect | 불량 | 5,338 |
| train | template_matched | 정상 | 4,690 |
| train | tested_clear | 정상 | 5,308 |
| val | defect | 불량 | 1,535 |
| val | template_matched | 정상 | 1,346 |
| val | tested_clear | 정상 | 1,534 |

정상 후보를 두 출처에서 만들었다. `template_matched`는 불량 bbox와 같은 위치의 정상 기준 Crop이다.
`tested_clear`는 검사 영상 안에서 모든 주석 bbox와 guard 8px를 피한 회로 Crop이다.
후자는 주석이 완전하다는 가정에 따른 정상 후보이며, 미주석 결함 가능성을 실제 생산 데이터에서
별도로 검증해야 한다. 검사/기준 영상의 저장 방식 차이만 배우는 문제를 줄이려고 두 출처를 섞었다.
이 구성만으로 모든 출처 단서를 제거했다고 단정하지 않고, 출처별 정상 오탐률을 함께 보고한다.

생성 후보 20,588개 중 837개를 격리하고 19,751개를 사용했다.
고정 규칙은 라벨이 충돌하는 동일 픽셀과 split 간 동일 픽셀을 격리하고 split 내부 반복은 한 번만 유지하는 것이다.
격리 내역은 `excluded_patches.csv`에 남겼고 생성 파일은 삭제하지 않았다. 성능을 본 후 표본을 제거하지 않았다.
조건을 만족하는 정상 회로 ROI를 찾지 못한 31건은 `sampling_failures.csv`에 기록했다.

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
실제 18 epoch를 실행했으며 최고 checkpoint는 12 epoch다.

## 실측 validation 성능

| 지표 | 값 |
|---|---:|
| Accuracy | 0.9887 |
| Balanced Accuracy | 0.9889 |
| Macro F1 | 0.9876 |
| 불량 Precision | 0.9781 |
| 불량 Recall | 0.9896 |
| 불량 F1 | 0.9838 |
| 정상 오탐률 | 1.1806% |
| 불량 미검출률 | 1.0423% |
| ROC AUC | 0.9985 |
| Average Precision | 0.9968 |

정상 정답 2846개, 정상 오탐 34개,
불량 정답 1519개, 불량 미검출 16개다.
항상 정상으로 판정하는 기준선은 Accuracy 0.6523, Macro F1 0.3948,
불량 Recall 0.0000다. 정상 표본이 더 많으므로 Accuracy 하나로 판단하지 않는다.

| 정상 데이터 출처 | 표본 수 | 정상 오탐률 |
|---|---:|---:|
| template_matched | 1,346 | 1.3373% |
| tested_clear | 1,534 | 1.0430% |

| 원래 불량 유형 | 불량 표본 수 | 불량으로 검출한 Recall |
|---|---:|---:|
| Open Circuit | 278 | 99.2806% |
| Short Circuit | 203 | 96.5517% |
| Mousebite | 347 | 100.0000% |
| Spur | 254 | 98.8189% |
| Spurious Copper | 232 | 100.0000% |
| Pin Hole | 221 | 98.1900% |

이번 검증에서 Recall이 가장 낮은 원래 불량 유형은 Short Circuit
(96.5517%)였다. 오분류 시각 검토에서는 정상 패드의 구멍·작은 밝은 점과
불량 구멍이 비슷하게 보이는 사례, 정상 연결부와 단락이 비슷한 사례가 관측됐다.
이는 시각적 관찰이며 주석 오류를 확정한 것은 아니다. 평가 라벨과 표본은 수정하지 않았다.

validation 4415개 전체의 CPU/Metal 예측 라벨 일치율은 100.00%, 최대 sigmoid 점수 차이는 1.1e-06다. 독립 추론 함수 점수 차이는 3.67e-09다.

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
    result = predict_binary_patch("models/binary_cnn_001/best.keras", image)
print(result["korean"], result["sigmoid_score"])
```

- 모델: `models/binary_cnn_001/best.keras`, `metadata.json`.
- 데이터: `data/processed/binary_v1/`, `data/split/binary_v1/`.
- 결과: `results/binary/binary_cnn_001/`의 CSV 로그, 학습 곡선, 혼동행렬/ROC/PR, 예측 전체,
  오분류 이미지/CSV, 원래 불량 유형별 Recall, 정상 출처별 오탐률.
- 이 보고서 및 데이터 감사: `results/binary/`.

노트북 재실행은 동일한 설정의 완료 모델을 재사용한다. 다른 설정은 새 run_name을 사용한다.
공식 test 최종 평가는 아직 실행하지 않았다.
