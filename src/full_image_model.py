"""패치 CNN의 합성곱 특징을 원본 640×640 단일 입력 CNN으로 확장한다."""

from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from src.runtime import configure_runtime


def build_full_encoder(patch_model_path: str | Path) -> keras.Model:
    configure_runtime()
    patch_model = keras.models.load_model(patch_model_path, compile=False)
    if patch_model.input_shape[1:] != (128, 128, 1) or patch_model.output_shape[-1] != 1:
        raise ValueError("이진 패치 CNN checkpoint가 필요합니다.")
    inputs = keras.Input((640, 640, 1), name="full_grayscale_0_to_255")
    x = inputs
    pooled = []
    # 레이어를 복제하고 가중치만 복사해 원래 패치 모델을 변경하지 않는다.
    for original in patch_model.layers[1:]:
        if isinstance(original, layers.Flatten):
            break
        if isinstance(original, layers.SpatialDropout2D):
            continue  # 추론 시 identity인 layer는 특징 추출기에서 생략한다.
        config = original.get_config()
        config["name"] = "full_"+original.name
        clone = original.__class__.from_config(config)
        x = clone(x)
        clone.set_weights(original.get_weights())
        clone.trainable = False
        if isinstance(original, layers.MaxPooling2D):
            # 국소적으로 강한 결함 신호와 이미지 전반의 통계를 함께 보존한다.
            pooled.extend([layers.GlobalMaxPooling2D()(x), layers.GlobalAveragePooling2D()(x)])
    if len(pooled) != 6:
        raise ValueError("예상한 3개 합성곱 block 구조가 아닙니다.")
    output = layers.Concatenate(name="multiscale_full_features")(pooled)
    encoder = keras.Model(inputs, output, name="full_encoder")
    encoder.trainable = False
    return encoder


def build_full_head(train_features: np.ndarray) -> keras.Model:
    # 통계는 train 원본 특징으로만 계산한다. validation 특징은 adapt에 사용하지 않는다.
    mean = train_features.mean(axis=0).astype("float32")
    variance = np.maximum(train_features.var(axis=0), 1e-6).astype("float32")
    inputs = keras.Input((train_features.shape[1],))
    x = layers.Normalization(mean=mean, variance=variance, name="train_feature_normalization")(inputs)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(.3)(x)
    output = layers.Dense(1, activation="sigmoid", name="full_defect_score")(x)
    return keras.Model(inputs, output, name="full_head")


def assemble_full_model(encoder: keras.Model, head: keras.Model) -> keras.Model:
    inputs = keras.Input((640, 640, 1), name="single_full_image")
    # BN은 고정된 inference 모드를 유지하면서 선택한 Conv 커널의 gradient는 허용한다.
    features = encoder(inputs, training=False)
    return keras.Model(inputs, head(features), name="PCB_Full_Image_Binary")


def unfreeze_last_convolutions(model: keras.Model, count: int = 2):
    encoder = model.get_layer("full_encoder")
    encoder.trainable = True
    convs = [layer for layer in encoder.layers if isinstance(layer, layers.Conv2D)]
    selected = set(layer.name for layer in convs[-count:])
    for layer in encoder.layers:
        layer.trainable = layer.name in selected
    return sorted(selected)
