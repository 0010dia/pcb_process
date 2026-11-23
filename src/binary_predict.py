"""이진 분류 모델의 독립 추론. 6종 모델이나 학습 코드는 로드하지 않는다."""

import json
from pathlib import Path

from PIL import Image
from tensorflow import keras

from src.predict import prepare_input
from src.runtime import configure_runtime


def predict_binary_patch(model_path: str | Path, image: Image.Image) -> dict:
    configure_runtime()
    path = Path(model_path)
    contract = json.loads((path.parent / "metadata.json").read_text())
    if contract.get("task") != "binary_patch_classification":
        raise ValueError("정상/불량 이진 분류 모델 경로가 필요합니다.")
    model = keras.models.load_model(path, compile=False)
    score = float(model(prepare_input(image), training=False).numpy().reshape(-1)[0])
    target = int(score > contract["threshold"])
    return dict(target=target, label="defect" if target else "normal", korean="불량" if target else "정상",
                sigmoid_score=score, threshold=contract["threshold"],
                interpretation="보정되지 않은 모델 점수이며 실제 불량 발생 확률이나 정확도가 아닙니다.")
