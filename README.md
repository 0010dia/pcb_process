# CNN-Based PCB Defect Classification

협력사별 납품 손익과 검사 성능은 [대시보드 사용 안내](pcb-dashboard/README.md)에서 확인할 수 있다.
저장소에는 코드, 모델, 검증 결과가 포함된다. 재학습이나 검증 사진 표시에는 별도의 [DeepPCB](https://github.com/tangsanli5201/DeepPCB) 데이터가 필요하다.

**원본 한 장으로 정상/불량을 판정하는 새 실험**은
[full_image_from_scratch.ipynb](full_image_from_scratch.ipynb)와
[원본 모델 실행 안내](FROM_SCRATCH_README.md)를 참고한다.
원본 640×640을 그대로 입력하고 기존 모델 가중치 없이 CNN을 새로 학습한다.
원본 모델 학습은 완료했다. 환경과 학습 이력은 [학습 인계 안내](TRAINING_HANDOFF.md)를 참고한다.

별도의 정상/불량 이진 분류는 [binary_classification.ipynb](binary_classification.ipynb)와
[이진 분류 실행 안내](BINARY_README.md)를 참고한다. 기존 6종 분류와 모델·데이터·결과를 분리했다.

DeepPCB의 불량 위치를 중심으로 Crop한 패치를 6종으로 분류하는 TensorFlow/Keras 프로젝트다.
Python **3.11.9**, MacBook Pro **M1 Pro / 16GB**, 실제 로컬 데이터를 사용한다.
EDA → 그룹 분할 및 전처리 → 직접 설계 CNN A/B의 1차 검증을 진행했다.
공식 test 평가, Model C와 Streamlit은 후속 단계다.

## 실행

PyCharm에서 [sample.ipynb](sample.ipynb)를 프로젝트 `venv` 커널로 위에서부터 실행한다.
노트북은 실제 출력, EDA, 패치 생성, Metal 진단, A/B 학습과 결과 비교를 포함한다.
학습 셀은 여러 분 소요되며 같은 run_name의 완료 결과는 재사용한다.

현재 Python은 유지했고 TensorFlow 호환성을 위해 NumPy를 **2.0.2**로 맞췄다.
TensorFlow **2.18.1**, Metal **1.2.0**, Keras **3.8.0**, Scikit-learn **1.6.1**을 설치했다.
직접 의존성은 `requirements.txt`, 전체 해석 결과는 `uv.lock`에 기록했다.

```bash
cd /Users/yeongjunna/DNN_project
# 새 환경에 설치할 때 대상 인터프리터 경로를 지정한다.
uv pip install --python venv/bin/python -r requirements.txt
# uv sync 사용 시 현재 venv를 지정한다.
UV_PROJECT_ENVIRONMENT=venv uv sync --locked --python venv/bin/python
```

실제 분석과 학습은 노트북 커널에서 함수를 직접 호출했다. CLI도 제공한다.

```bash
MPLBACKEND=Agg venv/bin/python -m src.eda --data-root data --output-root results
MPLBACKEND=Agg venv/bin/python -m src.preprocessing
MPLBACKEND=Agg venv/bin/python -m src.train --variant A --run-name A_baseline_safe_002 --epochs 30
MPLBACKEND=Agg venv/bin/python -m src.train --variant B --run-name B_improved_001 --epochs 40
MPLBACKEND=Agg venv/bin/python -m unittest discover -s tests -v
```

설정이 다른 실험에는 새로운 run_name을 사용한다. 같은 이름의 다른 설정이나 미완료 실행은 덮어쓰지 않는다.

## 데이터와 고정 분할

| 구분 | 원본 쌍 | 불량 패치/객체 |
|---|---:|---:|
| train | 800 | 5,338 패치 |
| validation (`group00041` 전체) | 200 | 1,535 패치 |
| 공식 test (예약, Crop 미생성) | 500 | 3,140 객체 |
| 합계 | 1,500 | 10,013 객체 |

`data/` 기존 파일을 사용했으며 다운로드하지 않았다. 공식 목록의 정상/불량 이미지는 3,000개다.
디스크에는 목록 외 기준 영상 1개를 포함해 3,001개가 있다. 추가 파일
`group90100/90100/90100034_temp.jpg`는 원본 그대로 두고 분석에서 제외했다.
라벨 형식·ID·좌표 검사 오류는 0건이었다.

```text
data/
├── trainval.txt / test.txt
├── group00041/
│   ├── 00041/00041000_test.jpg     # 불량 검사 영상
│   ├── 00041/00041000_temp.jpg     # 정렬된 정상 기준 영상
│   └── 00041_not/00041000.txt      # x1 y1 x2 y2 class_id
├── split/                          # 고정 설정/원본/패치 manifest
└── processed/roi128_v1/
    ├── train/{target_id}/*.png
    └── val/{target_id}/*.png
```

목록의 `00041000.jpg`는 `_test.jpg`/`_temp.jpg`의 공통 참조 이름이다.
파일명 `_test`는 검사 영상의 역할이며 공식 test 분할 여부는 목록으로 판단한다.
모든 영상은 640×640이고 L/RGB 저장 모드가 섞여 있어 grayscale로 통일한다.

| 원본 ID → target | 불량 유형 | train | validation | 예약 test |
|---|---|---:|---:|---:|
| 1 → 0 | Open Circuit / 단선 | 1,005 | 278 | 659 |
| 2 → 1 | Short Circuit / 단락 | 825 | 203 | 478 |
| 3 → 2 | Mousebite / 가장자리 결손 | 1,032 | 347 | 586 |
| 4 → 3 | Spur / 구리 돌출 | 888 | 254 | 483 |
| 5 → 4 | Spurious Copper / 잔여 구리 | 778 | 232 | 464 |
| 6 → 5 | Pin Hole / 내부 구멍 | 810 | 221 | 470 |

공식 test는 1단계 수량/무결성 감사에만 사용했다. 이후 Crop 생성, 전처리 조정,
학습과 모델 선택에는 사용하지 않았다. 현재 평가값은 모두 고정 그룹의 **validation 성능**이다.

## 전처리와 누수 방지

성능을 보기 전에 trainval 원본 수의 20%에 가장 가까운 `group00041`을 통째로 검증용으로
고정했다. 원본 ID, 그룹, 검사/기준 픽셀 해시가 train/val을 넘으면 거부한다.
생성한 패치의 동일 픽셀 중복도 없음을 확인했다. 성능에 따라 검증 그룹을 바꾸지 않는다.

객체 중심 정사각 ROI에 **16px 여백**, 경계 **edge padding**, **bilinear 128×128**, **L PNG**를 적용했다.
공식 문서는 오른쪽/아래 끝점 포함 여부가 명확하지 않아 +1px 보수적으로 포함하는 규약을 고정했다.
원본 좌표, 반열린 ROI, 패딩 비율, 이웃 bbox 수를 manifest에 저장한다.

이웃 bbox가 문맥에 걸친 패치는 **1,136개**, 패딩 패치는 **264개**다.
해당 표본을 제거하지 않았고 validation 평가에서 별도 slice로 분석한다.
B의 증강은 train에만 90도 회전/좌우/상하 반전/약한 밝기 변화를 온라인 적용한다.
정규화는 저장 모델의 `Rescaling(1/255)`에서 한 번 수행하므로 모델 입력은 **0~255**다.
경로만 shuffle하고 batch 16, map 병렬도 4, prefetch 1로 메모리를 제한한다.

공식 trainval/test는 **7개 폴더 그룹**을 공유한다. 이를 완전히 새로운 기판에 대한 성능으로
해석할 수 없다. 그룹 폴더가 실제 기판 ID라는 보장도 없으며 유사/인접 타일 독립성 검토는 남아 있다.

## Metal 실행 오류와 수정

GPU 열거와 실제 행렬 연산은 성공했으나 초기 CNN 실행에서 같은 입력·가중치의
Metal graph 추론과 CPU/Metal eager 추론이 달랐다. 최대 점수 차이는 약 **0.415**였다.
초기 `A_baseline_001`은 진단 기록으로 보존하고 성능 비교에서 제외했다.

`src/runtime.py`에서 `disable_meta_optimizer=True`를 적용하자 진단 batch의 Metal
 graph/eager 점수 차이는 **0**이 됐다. GPU 커널은 계속 사용한다. 수정 후 매 epoch
Keras graph Accuracy와 NumPy eager Accuracy가 다르면 학습을 즉시 실패시킨다.
독립 추론 함수도 같은 설정을 사용한다.

현재 장비·설치 조합에서 재현한 관찰이다. 근거는 `results/reports/metal_diagnostic.json`,
`metal_workaround.json`과 학습 로그다. 설정 API는
[TensorFlow 공식 문서](https://www.tensorflow.org/api_docs/python/tf/config/optimizer/set_experimental_options)를 참고한다.

## 모델과 저장 결과

- **A**: 직접 설계 Conv32→64→128, MaxPooling, GAP, Dense64, Dropout0.3. 증강 없음.
- **B**: Conv 2회 블록 24→48→96, BatchNormalization, SpatialDropout,
  Flatten→Dense96(L2)→Dropout0.4. train만 증강, Adam 학습률 0.0005.
- 최고 **validation Macro F1**로 checkpoint, EarlyStopping, ReduceLROnPlateau를 적용한다.
- 저장 checkpoint를 재로드해 Accuracy, Macro Precision/Recall/F1, Weighted F1,
  클래스별 보고서, 혼동행렬, 오분류, 이웃/패딩별 결과를 저장한다.

최신 실측 결과는 [학습 진행 보고서](results/reports/training_progress_report.md)와
`model_comparison_validation.csv`에 저장한다. 선택 후보는 `selected_model.json`에 기록한다.
검증으로 선택한 성능은 독립적인 최종 test 성능과 구분한다.

| 모델 | validation Accuracy | validation Macro F1 | 최고 epoch |
|---|---:|---:|---:|
| A (Metal 회피 설정 적용) | 0.6261 | 0.6315 | 5 / 12 |
| B (직접 설계 개선 CNN) | 0.9948 | 0.9949 | 36 / 40 |

B의 validation 오분류는 8/1,535개다. CPU에서 validation 전체를 다시 추론한 결과
Metal 평가와 모든 예측 라벨이 일치했고 최대 점수 차이는 약 4.72e-7이었다.
독립 추론 코드와 저장된 평가 점수도 일치했다. 테스트 12개, 노트북 전체 실행,
설치 패키지 호환성 및 uv.lock 일치 검사를 통과했다.

```text
sample.ipynb
src/
├── eda.py / preprocessing.py         # 감사, 그룹 분할, 패치 생성
├── dataset.py / model.py / train.py  # 온라인 로딩, 직접 설계 CNN, 학습
├── runtime.py                        # 실측한 Metal 회피 설정
├── evaluate.py / report.py           # 평가와 모델 비교
└── predict.py                        # 학습 코드와 분리된 패치 추론
models/<run>/
├── best.keras
└── metadata.json                     # 클래스·입력·정규화 계약
results/
├── figures/                          # EDA, 전처리 예시, 비교 그래프
├── reports/                          # 집계 CSV/JSON, 환경, 진행 보고서
└── training/<run>/
    ├── config.json / runtime.json / model_summary.txt
    ├── history.csv / learning_curves.png / run_result.json
    ├── validation_metrics.json / classification_report.csv
    ├── confusion_matrix.png / confusion_matrix.npy
    ├── validation_predictions.csv / validation_slices.csv
    └── misclassified.csv / misclassified.png
```

추론 예시:

```python
import json
from pathlib import Path
from PIL import Image
from src.predict import predict_patch

candidate = json.loads(Path('results/reports/selected_model.json').read_text())
with Image.open('path/to/defect_patch.png') as image:
    result = predict_patch(candidate['model_path'], image)
print(result)
```

입력은 불량 위치가 지정된 **개별 패치**여야 한다. 정상 클래스와 전체 PCB의 위치 탐지 기능은 없다.
Softmax는 6종 사이의 보정되지 않은 상대 점수이며 실제 불량 발생 확률이나 모델 정확도가 아니다.

다음에는 Model C 전이학습 비교, 추가 그룹 검증과 오분류 분석을 거쳐 설정을 확정하고
공식 test를 최종 평가한다. Streamlit은 실제 모델/보고서를 연결하는 별도 단계다.
[DeepPCB](https://github.com/tangsanli5201/DeepPCB)에는 합성 불량이 포함돼 실제 생산 데이터의 별도 검증이 필요하다.

