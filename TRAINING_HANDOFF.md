# 학습 완료 및 이전 인계 기록

**현재 상태:** 11 epoch 체크포인트에서 WSL2 RTX 5060으로 재개하여 17 epoch까지 학습을 완료했다.
최고 모델은 12 epoch이며 검증 정확도 98.75%, Macro F1 0.987499이다. 검증 400장의 CPU/GPU 판정이 모두 일치했고 공식 test는 사용하지 않았다.
배포 모델은 `models/full_scratch_aux_001/best.keras`, 상세 결과는 `results/full_scratch/full_scratch_aux_001/run_result.json`이다.
GPU 환경은 Python 3.11, TensorFlow 2.21.0, Keras 3.15.1이다. 아래 내용은 **이전 중단 시점의 인계 기록**이다.

---

## 이전 중단 시점

현재 학습은 사용자 요청으로 중지했으며 `full_image_from_scratch.ipynb` 커널도 종료했다.
이 컴퓨터에서 학습은 실행 중이 아니다. 노트북의 `START_TRAINING` 기본값은 `False`다.

## 저장 상태

- 실행: `full_scratch_aux_001`, 원본 640×640 사진 한 장 정상/불량 CNN.
- 로그: 12 epoch 완료. 최고 validation Macro F1 checkpoint는 **11 epoch**.
- 최고 checkpoint의 validation Accuracy **98.5%**, Macro F1 **0.9849996**.
- 체크포인트: `results/full_scratch/full_scratch_aux_001/best_training_model.keras`.
- 실제 파일을 CPU에서 로드해 **Adam, 4,400 step, 학습률 0.00015** 복원을 확인했다.
- SHA-256: `3281ce8c83f031a169e443f14c2410cd079f2d6791fe79f63cc6a6af6acd790f`.
- 자세한 중지 상태: 같은 폴더의 `pause_state.json`.

이 파일에는 CNN 가중치뿐 아니라 optimizer 상태도 있다.
재개는 저장된 11 epoch 직후부터 하므로 **12 epoch를 다시 학습**한다.
현재의 12 epoch 로그는 재개 시 `history_before_resume_<시각>.csv`로 보관한다.
best 모델보다 성능이 낮은 새 epoch가 best 파일을 덮어쓰지 않도록 callback의 최고값도 복원한다.
공식 test는 이번 학습이나 모델 선택에 사용하지 않았다.
전체 학습·최종 CPU/Metal 400장 검증은 아직 완료되지 않았다.

## 무엇을 옮기나

**`venv/`를 제외한 프로젝트 폴더 전체를 복사하는 것이 가장 간단하다.**
가상환경은 컴퓨터/운영체제의 경로와 바이너리에 의존하므로 새 컴퓨터에서 다시 만든다.
`.idea`, `__pycache__`, `.ipynb_checkpoints`는 없어도 된다.

특히 다음 파일은 빠지면 안 된다.

| 경로 | 이유 |
|---|---|
| `full_image_from_scratch.ipynb` | 재개 실행 노트북 |
| `src/` 전체 | 데이터/모델/학습/추론 코드와 사용자 정의 loss |
| `data/` 전체 | 원본 이미지, trainval/test 목록, 고정 분할, 보조 정답 |
| `results/` 전체 | 객체 목록, 실행 설정, epoch 로그와 **학습 checkpoint** |
| `requirements.txt`, `pyproject.toml`, `uv.lock` | 환경 재구성 |
| `models/` | 이전 모델 기록 보존. 이번 재개의 필수 checkpoint는 `results/` 안에 있음 |

현재 `data` 약 281MB, `results` 약 16MB, `models` 약 48MB다.
기존 `venv` 약 1.7GB는 옮기지 않아도 된다.
원본/마스크 manifest의 파일 경로는 프로젝트 기준 상대 경로여서 프로젝트 위치는 달라도 된다.
CSV의 split, 라벨, 마스크를 임의 수정하지 않는다.

## 다른 컴퓨터에서 재개

1. 프로젝트 폴더를 복사해 PyCharm에서 프로젝트로 연다.
2. **Python 3.11**로 새 가상환경을 만들고 `requirements.txt`의 패키지를 설치한다.
   현재 원본 환경은 Python 3.11.9, TensorFlow 2.18.1, Keras 3.8.0이다.
   `tensorflow-metal`은 설치 조건이 걸려 있어 macOS arm64에서만 설치된다.
   GPU 가속 구성은 장비에 맞게 설정해야 하며 CPU에서도 모델을 복원할 수 있다.
3. 새 가상환경을 노트북 커널로 선택한다. 작업 디렉터리는 프로젝트 최상위 폴더로 둔다.
4. `full_image_from_scratch.ipynb`를 위에서부터 실행한다. 학습 셀의 두 값만 다음처럼 바꾼다.

```python
START_TRAINING = True
RESUME_TRAINING = True
```

재개 시 `저장된 11 epoch 가중치/Adam 상태 복원 → 12 epoch부터 재개`가 출력돼야 한다.
총 최대 epoch는 20이며, 20 epoch를 추가 실행하는 것은 아니다.
최고 checkpoint 이후 5 epoch 동안 개선이 없으면 EarlyStopping이 적용된다.
재개 장비의 실행 정보는 `resume_runtime.json`에 저장한다.
학습 설정/데이터 manifest가 바뀌면 재개를 거부한다. 장비 실행 정보 차이만 허용한다.

노트북 대신 같은 환경의 커널에서 아래 함수를 직접 호출해도 된다.

```python
from pathlib import Path
from src.full_supervised import train_supervised_scratch

result = train_supervised_scratch(Path.cwd(), resume=True)
```

데이터 shuffle/증강 난수 흐름은 재시작되며 CPU/GPU 및 운영체제 차이로 연산 결과가 조금 달라질 수 있다.
따라서 같은 가중치와 Adam 상태로 학습을 이어가는 것이며, 중단하지 않은 실행과 bit 단위로 같다는 보장은 아니다.
재개 후 학습을 마치면 노트북의 CPU 비교·한 장 추론·최종 보고서 셀을 실행한다.

## 다시 이 컴퓨터에서 이어갈 때

다른 컴퓨터와 같은 재개 절차를 사용한다. 환경은 기존 `venv`를 그대로 선택하면 된다.
지금은 학습을 재시작하지 않았다.
