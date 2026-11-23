# 원본 PCB 한 장 정상/불량 분류

새 작업은 [full_image_from_scratch.ipynb](full_image_from_scratch.ipynb)에서 진행한다.
기존 `full_image_classification.ipynb`와 패치 실험은 이전 기록으로 보존한다.
학습은 Windows WSL2의 Python 3.11, TensorFlow 2.21.0, Keras 3.15.1 및 RTX 5060에서 완료했다.
17 epoch까지 학습했으며 최고 모델은 12 epoch (검증 Accuracy 98.75%, Macro F1 0.9875)이다.
검증 400장 CPU/GPU 판정 일치와 최종 보고서 생성을 확인했다. 공식 test는 평가하지 않았다.
노트북의 `START_TRAINING` 기본값은 `False`다. 이전 중단 이력은 [TRAINING_HANDOFF.md](TRAINING_HANDOFF.md)를 참고한다.

## 입력과 학습 방식

- 입력은 **640×640 원본 이미지 한 장**이다. 원본을 Crop하거나 resize하지 않는다.
- 정상 0 / 불량 1을 판정하며 정상 기준 이미지나 위치 주석을 추론 입력으로 요구하지 않는다.
- encoder/decoder CNN과 attention을 TensorFlow/Keras로 직접 구성하고 **모든 가중치를 새로 초기화**한다.
- 기존 패치 CNN이나 ImageNet 가중치를 불러오지 않는다.
- 학습에서만 정상/불량 라벨과 결함 위치 보조 정답을 함께 사용한다.

작은 결함이 전역 평균으로 사라지는 문제를 줄이기 위해, 원본 해상도를 유지하고
공간 특징을 복원하는 CNN을 사용한다. 정상 기준/검사 영상의 픽셀 차이와 bbox로 만든
160×160 정답 맵을 보조 loss에 사용한다. 이 맵은 모델에 제공하는 정답이며 입력이 아니다.
모델 내부의 attention은 입력 이미지에서 예측한 특징으로 계산한다.
배포 모델의 출력은 정상/불량 sigmoid 점수 한 개다.

## 데이터 분할

| 분할 | 정상 원본 | 불량 원본 | 합계 |
|---|---:|---:|---:|
| train | 800 | 800 | 1,600 |
| validation | 200 | 200 | 400 |

이전부터 고정한 `group00041` 전체를 validation으로 유지한다.
같은 정상/불량 쌍, 촬영 그룹, 동일 픽셀 원본이 train과 validation을 넘지 않도록 확인한다.
공식 test 500쌍은 이번 학습과 모델 선택에 사용하지 않는다.

## 실행

PyCharm에서 새 노트북을 프로젝트 `venv` 커널로 위에서부터 실행한다.
노트북이 환경 확인 → 원본 manifest → 보조 정답 생성 → CNN 학습 → 저장 모델 검증 → 보고서 순서로 구성돼 있다.
완료 결과가 있고 설정이 같으면 학습을 반복하지 않고 재사용한다.
중단된 실행이나 다른 설정을 같은 run 이름으로 덮어쓰지 않는다.

현재 주 실행은 `full_scratch_aux_001`이다. Adam 0.0003, batch 4, 최대 20 epoch,
EarlyStopping patience 5, seed 42를 사용한다. 회전/반전은 학습 원본과 보조 정답에 같이 적용한다.
픽셀 정규화는 저장 모델 안의 `Rescaling(1/255)`에서 수행한다.
모델 선택 기준은 임계값 0.5에서의 validation Macro F1이다.
M1 Pro 16GB에서 경로 단위 로딩, 병렬도 2, prefetch 1로 메모리 사용을 제한한다.
장비에서 재현한 Metal graph/eager 차이를 피하는 설정은 `src/runtime.py`에 공통 적용한다.

학습을 재개·완료한 후 새 노트북의 마지막 코드 셀에서 `IMAGE_PATH`에 원본 경로를 지정하면
한 장 추론을 실행할 수 있다. 저장 모델만 사용하는 예시는 다음과 같다.

```python
from PIL import Image
from src.full_image_predict import predict_full_image

with Image.open("path/to/original_640.png") as image:
    result = predict_full_image("models/full_scratch_aux_001/best.keras", image)
print(result["korean"], result["sigmoid_score"])
```

입력은 원본 640×640이어야 한다. RGB 저장 영상도 grayscale로 변환한다.
128×128 패치 등의 크기가 다른 입력은 오류로 알린다.
sigmoid 점수는 보정된 실제 불량 확률이나 모델 정확도가 아니다.

## 코드와 결과

| 파일/폴더 | 내용 |
|---|---|
| `src/full_supervised.py` | 보조 정답 생성, 온라인 로딩, 새 CNN 구성과 학습 |
| `src/full_image_data.py` | 원본 manifest와 분할 검증, 원본 한 장 로딩 |
| `src/full_image_evaluate.py` | 이미지 단위 평가와 오분류/학습 그래프 |
| `src/full_image_predict.py` | 학습 코드와 분리된 한 장 추론 |
| `src/full_scratch_report.py` | 새 주 실험과 이전 참고 실험의 결과 보고서 |
| `models/full_scratch_aux_001/` | 단일 입력·출력 `best.keras`, 입력 계약 `metadata.json` |
| `results/full_scratch/full_scratch_aux_001/` | 설정, epoch 로그, 지표, 혼동행렬, 예측 CSV와 그림 |
| `results/full_scratch/from_scratch_report.md` | 실제 완료 지표와 방법/해석 보고서 |
| `results/full_scratch/inference_verification.json` | validation 전체 CPU/Metal 및 독립 한 장 추론 검증 |
| `data/split/full_scratch_aux_v1/` | 원본과 보조 정답의 목록/설정 |
| `data/processed/full_scratch_aux_v1/` | 학습용 정답 맵 PNG. 원본 입력 패치가 아님 |

초기 분류 라벨만 사용한 `full_scratch_001`은 5 epoch에서 중단한 pilot이다.
최종 성능 비교에서 제외하며, 이 미완료 기록을 수렴 성능의 상한으로 해석하지 않는다.
과거 패치 전이학습 원본 모델도 참고 실험으로만 표시한다.

정상 데이터는 기준 영상, 불량 데이터는 검사 영상이라는 출처 차이가 남아 있다.
고정 validation 성능은 새로운 생산라인이나 촬영 환경의 성능을 보장하지 않는다.
DeepPCB의 합성 불량 특성과 반복한 validation 모델 선택을 고려해야 한다.
보조 정답을 사용했다는 사실만으로 실제 결함만 학습했다고 단정하지 않는다.
최종 공식 test와 실제 생산 데이터 검증은 별도로 남겨둔다.
