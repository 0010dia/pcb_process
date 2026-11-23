"""학습 코드를 import하지 않고 저장 모델로 불량 패치 유형을 추론한다."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras
from src.runtime import configure_runtime


def prepare_input(image: Image.Image, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    # 임의 종횡비 업로드 패치는 중앙 edge padding으로 비율을 보존한다.
    pixels = np.asarray(ImageOps.exif_transpose(image).convert("L"))
    height, width = pixels.shape
    side = max(height, width)
    top, left = (side-height)//2, (side-width)//2
    pixels = np.pad(pixels, ((top, side-height-top), (left, side-width-left)), mode="edge")
    resized = Image.fromarray(pixels).resize(size, Image.Resampling.BILINEAR)
    # 저장 모델 자체에 정규화가 있으므로 여기서는 255로 나누지 않는다.
    return np.asarray(resized, dtype=np.float32)[None, ..., None]


def predict_patch(model_path: str | Path, image: Image.Image) -> dict:
    configure_runtime()
    path = Path(model_path)
    contract = json.loads((path.parent / "metadata.json").read_text())
    model = keras.models.load_model(path, compile=False)
    probabilities = model(prepare_input(image), training=False).numpy()[0]
    index = int(probabilities.argmax())
    return dict(prediction=contract["classes"][index], softmax_score=float(probabilities[index]),
                class_scores={c["name"]: float(probabilities[c["target_id"]]) for c in contract["classes"]},
                interpretation=contract["softmax"])
