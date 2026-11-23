"""원본 검사 사진 한 장만 입력받는 정상/불량 독립 추론."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

from src.runtime import configure_runtime


def prepare_full_input(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("L")
    if image.size != (640, 640):
        raise ValueError("현재 모델은 640×640 원본 한 장을 입력받습니다. 임의 축소/Crop은 수행하지 않습니다.")
    return np.asarray(image, dtype=np.float32)[None, ..., None]


def predict_full_image(model_path: str | Path, image: Image.Image) -> dict:
    configure_runtime()
    model_path = Path(model_path)
    metadata = json.loads((model_path.parent / "metadata.json").read_text())
    if metadata.get("task") != "full_image_single_binary":
        raise ValueError("원본 한 장용 모델 경로가 필요합니다.")
    model = keras.models.load_model(model_path, compile=False)
    score = float(model(prepare_full_input(image), training=False).numpy().reshape(-1)[0])
    target = int(score > metadata["threshold"])
    return dict(target=target, label="defect" if target else "normal", korean="불량" if target else "정상",
                sigmoid_score=score, threshold=metadata["threshold"], input_images=1,
                interpretation="보정되지 않은 모델 점수이며 실제 불량 확률이나 모델 정확도가 아닙니다.")
