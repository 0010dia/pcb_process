# 원본 한 장 분류 — 처음부터 학습한 주 실험

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

| 모델 | 초기화 | Accuracy | Macro F1 | 불량 Recall | 정상 오탐률 |
|---|---|---:|---:|---:|---:|
| full_frozen_001 | patch_transfer_reference | 0.9525 | 0.9524 | 0.9900 | 8.50% |
| full_finetuned_001 | patch_transfer_reference | 0.9550 | 0.9549 | 0.9950 | 8.50% |
| full_scratch_aux_001 | random_from_scratch_primary | 0.9875 | 0.9875 | 0.9950 | 2.00% |

현재 주 실험 `full_scratch_aux_001`은 검증 원본 400장 중 정상 오탐 4장,
불량 미검출 1장이었다. 실제 17 epoch를 실행했고
최고 checkpoint는 12 epoch다.

원본 validation 400장 전체의 CPU/Metal 판정 일치율은 100.00%, 최대 점수 차이는 1.1e-06였다. 독립 한 장 추론의 최대 점수 차이는 1.92e-08였다.

같은 PCB 쌍을 추론 후 묶어 점수를 분석하면 200쌍 중
195쌍에서 정상/불량 양쪽 판정이 모두 맞았다.
200쌍에서 불량 점수가 정상 점수보다 높았다.
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
    result = predict_full_image("models/full_scratch_aux_001/best.keras", image)
print(result["korean"], result["sigmoid_score"])
```

결과 경로는 `results/full_scratch/selected_model.json`에 기록했다.
학습 로그, 원본 예측 CSV, 분류 보고서, 혼동행렬/ROC/PR, 오분류는 `results/full_scratch/full_scratch_aux_001/`에 있다.
모델은 `models/full_scratch_aux_001/best.keras`와 `metadata.json`이다.
학습용 정답 맵은 `data/processed/full_scratch_aux_v1/`, 목록은 `data/split/full_scratch_aux_v1/`에 있다.
원본 파일과 이전 노트북/모델은 변경하지 않았다.
