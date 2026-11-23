"""ImageNet 가중치를 사용하지 않는 직접 설계 CNN A/B."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.runtime import configure_runtime


def build_model(variant: str = "A", learning_rate: float = 1e-3) -> keras.Model:
    configure_runtime()
    inputs = keras.Input((128, 128, 1), name="grayscale_0_to_255")
    # 원본 PNG와 업로드 영상 모두 같은 정규화 경로를 거친다.
    x = layers.Rescaling(1/255.0, name="normalize")(inputs)
    if variant == "A":
        for i, filters in enumerate([32, 64, 128]):
            x = layers.Conv2D(filters, 3, activation="relu", padding="valid")(x)
            if i < 2:
                x = layers.MaxPooling2D(2)(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(.3)(x)
    elif variant == "B":
        for filters in [24, 48, 96]:
            for _ in range(2):
                x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.SpatialDropout2D(.1)(x)
        # 불량은 패치 중심에 있으므로 위치 관계를 유지해 단선/구멍의 연결 구조를 학습한다.
        # 세 번의 pooling 이후 16×16×96이므로 16GB 환경에서도 head 크기를 제한할 수 있다.
        x = layers.Flatten()(x)
        x = layers.Dense(96, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = layers.Dropout(.4)(x)
    else:
        raise ValueError("variant는 A 또는 B여야 합니다.")
    outputs = layers.Dense(6, activation="softmax", name="class_scores")(x)
    model = keras.Model(inputs, outputs, name=f"PCB_CNN_{variant}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"],
                  jit_compile=False)
    return model
