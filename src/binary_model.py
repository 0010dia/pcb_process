"""정상/불량 패치만 판별하는 별도의 직접 설계 CNN."""

from tensorflow import keras
from tensorflow.keras import layers

from src.runtime import configure_runtime


def build_binary_model(learning_rate: float = .0005) -> keras.Model:
    configure_runtime()
    inputs = keras.Input((128, 128, 1), name="grayscale_0_to_255")
    x = layers.Rescaling(1/255.0, name="normalize")(inputs)
    for filters in [16, 32, 64]:
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.SpatialDropout2D(.1)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="defect_score")(x)
    model = keras.Model(inputs, outputs, name="PCB_Binary_CNN")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy(name="accuracy", threshold=.5)], jit_compile=False)
    return model
