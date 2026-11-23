# 2~3단계 전처리 및 CNN 검증 결과

## 실제 실행 범위

Python 3.11.9를 유지했다. TensorFlow 2.18.1 / Metal 1.2.0 / Keras 3.8.0 / NumPy 2.0.2를 설치했다.
M1 Pro GPU 열거와 실제 행렬 연산을 확인했고, CNN 학습과 저장 checkpoint 재로드 평가를 수행했다.
공식 test는 전처리와 모델 선택에 사용하지 않았으며 최종 성능을 아직 측정하지 않았다.

## 고정 분할 및 Crop

- train: 800개 원본 / 5338개 패치.
- validation: `group00041` 전체 200개 원본 / 1535개 패치.
- test: 공식 500쌍 예약. 생성한 test 패치 0개.
- 그룹은 성능을 보기 전 trainval의 이미지 수 20%에 가장 가까운 기준으로 고정했다.
- train/validation의 원본·그룹·정상/검사 영상 픽셀 중복 및 생성 패치의 동일 픽셀 중복을 검사했다.
- 원본 bbox 주변 16px 여백, 정사각형, edge padding, bilinear resize 128×128, grayscale PNG.
- README가 좌표 끝점의 포함 여부를 명시하지 않아 오른쪽/아래를 +1px 보수적으로 포함했다.
  원본 좌표와 반열린 ROI를 모두 manifest에 남겨 재현할 수 있다.
- 이웃 bbox가 ROI에 걸친 패치 1136개, 경계 padding 패치 264개.
  해당 표본을 삭제하지 않았고 validation 진단 결과를 각 실행의 `validation_slices.csv`에 저장했다.
- 정규화는 저장 모델의 Rescaling에서 1/255로 한 번 수행한다. 로더/추론에서 이중 정규화하지 않는다.
- 증강 사용 실험은 train에만 90도 회전/좌우/상하 반전/약한 밝기 조절을 적용한다.
  epoch마다 온라인 적용하며 증강본을 validation 또는 디스크에 추가하지 않는다.

| 유형 | train | validation |
|---|---:|---:|
| Open Circuit | 1005 | 278 |
| Short Circuit | 825 | 203 |
| Mousebite | 1032 | 347 |
| Spur | 888 | 254 |
| Spurious Copper | 778 | 232 |
| Pin Hole | 810 | 221 |

## Metal 실행 오류와 수정 근거

초기 `A_baseline_001`은 Keras graph Accuracy와 NumPy eager Accuracy가 달랐다.
같은 가중치·입력에서 CPU eager/graph 및 Metal eager는 약 1e-7 이내로 일치했으나,
Metal graph는 최대 약 0.415의 점수 차이가 있었다. 초기 실험은 성능 비교에서 제외했다.
`tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})`를 적용한
후 진단 batch의 Metal graph/eager 출력 차이는 0이었다. GPU 사용은 유지했다.

이 결과는 현재 장비·설치 조합에서의 재현 관찰이며 다른 모든 TensorFlow/Metal 조합의 오류를 주장하지 않는다.
근거는 `metal_diagnostic.json`, `metal_workaround.json`, 초기 실행의 `history.csv`다.
수정 후 매 epoch마다 graph Keras Accuracy와 eager/NumPy Accuracy 차이가 1e-4보다 크면 즉시 실패하도록 했다.
학습/독립 추론 함수 모두 같은 회피 설정을 적용한다.

최종 B의 validation 1535개 전체를 CPU에서 다시 추론했다. Metal 평가와 예측 라벨 일치율은 100.00%, 6개 점수의 최대 절대 차이는 4.72e-07였다. 실측 기록은 `cpu_metal_validation_parity.json`에 저장했다.

## 검증 성능 (공식 test 성능 아님)

| 실행 | Accuracy | Macro F1 | Weighted F1 | 최고 epoch |
|---|---:|---:|---:|---:|
| A_baseline_safe_002 | 0.6261 | 0.6315 | 0.6249 | 5 |
| B_improved_001 | 0.9948 | 0.9949 | 0.9948 | 36 |

현재 validation Macro F1 기준 후보는 **B_improved_001 / 0.9949**다.
모든 지표는 동일 고정 그룹에서 측정했으며 최대 검증 F1의 checkpoint를 파일에서 다시 불러와 평가했다.
이 검증 집합은 모델 선택에도 쓰였으므로 독립적인 최종 일반화 성능으로 해석하지 않는다.
기판/인접 타일 유사도 전체를 배제한 것은 아니며 공식 trainval/test의 그룹 겹침 제한도 남아 있다.

현재 후보의 validation 오분류는 **8 / 1535개**다. 많이 관측된 혼동은 다음과 같다.

- Mousebite → Pin Hole: 3개
- Mousebite → Open Circuit: 1개
- Pin Hole → Mousebite: 1개
- Short Circuit → Mousebite: 1개
- Spur → Short Circuit: 1개

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
