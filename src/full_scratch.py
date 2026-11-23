"""원본 640×640만 사용하는 무작위 초기화 Residual CNN. 기존 모델 가중치를 읽지 않는다."""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.binary_evaluate import binary_scores, collect_binary_predictions
from src.full_image_data import assert_full_split, full_image_dataset
from src.full_image_evaluate import evaluate_full_model, full_history_plot
from src.preprocessing import digest
from src.runtime import configure_runtime


@dataclass(frozen=True)
class ScratchConfig:
    run_name: str = "full_scratch_001"
    epochs: int = 40
    batch_size: int = 4
    learning_rate: float = .0003
    patience: int = 8
    seed: int = 42


def build_scratch_model(learning_rate=.0003) -> keras.Model:
    configure_runtime()
    inputs = keras.Input((640, 640, 1), name="single_original_image")
    x = layers.Rescaling(1/255.0, name="pixel_normalization")(inputs)
    for stage, filters in enumerate([8, 16, 32, 64]):
        shortcut = layers.Conv2D(filters, 1, padding="same", name=f"stage{stage}_shortcut")(x)
        for index in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                              kernel_initializer="he_normal", name=f"stage{stage}_conv{index}")(x)
            # 작은 batch에서도 train/validation 실행이 같은 통계를 사용하는 GroupNorm을 쓴다.
            x = layers.GroupNormalization(groups=4, axis=-1, epsilon=1e-3,
                                           name=f"stage{stage}_norm{index}")(x)
            if index == 0:
                x = layers.Activation("relu")(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
    # 공간 관계를 학습한 뒤 pooling한다. 원본의 작은 결함과 주변 회로 문맥을 함께 처리한다.
    for index in range(2):
        x = layers.Conv2D(64, 3, dilation_rate=2, padding="same", use_bias=False,
                          kernel_initializer="he_normal", name=f"context_conv{index}")(x)
        x = layers.GroupNormalization(groups=4, axis=-1, epsilon=1e-3)(x)
        x = layers.Activation("relu")(x)
    x = layers.Concatenate()([layers.GlobalMaxPooling2D()(x), layers.GlobalAveragePooling2D()(x)])
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(.25)(x)
    output = layers.Dense(1, activation="sigmoid", name="defect_score")(x)
    model = keras.Model(inputs, output, name="Native640_Residual_FromScratch")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate, clipnorm=1.0),
                  loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
                  jit_compile=False)
    return model


class ScratchValidation(keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.started = time.monotonic()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        labels, probabilities = collect_binary_predictions(self.model, self.dataset)
        metrics = binary_scores(labels, probabilities)
        if abs(metrics["accuracy"]-logs["val_accuracy"]) > 1e-4:
            raise RuntimeError("graph/eager 원본 Accuracy 불일치입니다.")
        logs.update(val_macro_f1=metrics["macro_f1"], val_defect_recall=metrics["defect_recall"],
                    val_normal_fpr=metrics["normal_false_positive_rate"], val_numpy_accuracy=metrics["accuracy"],
                    learning_rate=float(keras.backend.get_value(self.model.optimizer.learning_rate)))
        print(f"epoch={epoch+1:02d} train_loss={logs['loss']:.4f} train_acc={logs['accuracy']:.4f} "
              f"val_acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
              f"defect_recall={metrics['defect_recall']:.4f} normal_fpr={metrics['normal_false_positive_rate']:.4f} "
              f"elapsed={time.monotonic()-self.started:.0f}s", flush=True)


def train_from_scratch(root: str | Path, config: ScratchConfig = ScratchConfig()) -> dict:
    root = Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", config.run_name):
        raise ValueError("올바른 run 이름을 사용하세요.")
    if not 1 <= config.epochs <= 100 or config.batch_size < 1 or config.patience < 1 or config.learning_rate <= 0:
        raise ValueError("학습 설정이 올바르지 않습니다.")
    execution = configure_runtime()
    split = root / "data/split/full_image_v1"
    frame = pd.read_csv(split / "images.csv")
    assert_full_split(frame)
    for name in ["train", "val"]:
        if not pd.read_csv(split / f"{name}.csv").equals(frame[frame.split == name].reset_index(drop=True)):
            raise ValueError("원본 split manifest가 다릅니다.")
    signature = dict(config=asdict(config), initialization="random_he_normal_no_pretrained_weights",
                     pixel_input="native_640_no_crop_no_resize", execution=execution,
                     manifest_sha256={name: digest(split / name) for name in ["images.csv", "train.csv", "val.csv", "config.json"]})
    output = root / "results/full_scratch" / config.run_name
    model_dir = root / "models" / config.run_name
    if (output / "run_result.json").exists():
        if json.loads((output / "config.json").read_text()) != signature:
            raise ValueError("기존 run 설정이 다릅니다. 새 이름을 사용하세요.")
        print(f"완료된 처음부터 학습한 원본 모델을 재사용합니다: {config.run_name}", flush=True)
        return json.loads((output / "run_result.json").read_text())
    if (output / "history.csv").exists():
        raise ValueError("미완료 로그를 보존하고 새 run 이름으로 시작하세요.")
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(signature, indent=2))
    keras.backend.clear_session()
    keras.utils.set_random_seed(config.seed)
    # 여기서는 모델 파일을 전혀 로드하지 않고 모든 학습 가중치를 새로 생성한다.
    model = build_scratch_model(config.learning_rate)
    lines = []
    model.summary(print_fn=lambda line, **kwargs: lines.append(line))
    (output / "model_summary.txt").write_text("\n".join(lines))
    contract = dict(task="full_image_single_binary", input_size=[640, 640, 1], input_images=1,
                    reference_image_required=False, crop=False, resize=False, threshold=.5,
                    classes={"0": "normal", "1": "defect"}, input_range=[0, 255], calibrated=False,
                    initialization="random", pretrained_weights_used=False, patch_weights_used=False,
                    normalization="Rescaling(1/255) inside saved model", validation_group="group00041",
                    official_test_evaluated=False)
    (model_dir / "metadata.json").write_text(json.dumps(contract, indent=2))
    runtime = dict(parameters=model.count_params(), python=platform.python_version(),
                   packages={name: metadata.version(name) for name in ["tensorflow", "keras", "numpy", "pandas", "scikit-learn"]},
                   execution=execution, pixel_resolution=[640, 640], batch_size=config.batch_size,
                   all_model_weights_new=True, nondeterminism="fixed seed; GPU and parallel augmentation may vary")
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    train, train_frame = full_image_dataset(root, "train", config.batch_size, augment=True, seed=config.seed)
    val, val_frame = full_image_dataset(root, "val", config.batch_size, augment=False)
    print(f"원본 처음부터 학습: {len(train_frame)} train / {len(val_frame)} val; {model.count_params():,} parameters", flush=True)
    callbacks = [ScratchValidation(val),
        keras.callbacks.ModelCheckpoint(model_dir / "best.keras", monitor="val_macro_f1", mode="max", save_best_only=True),
        keras.callbacks.CSVLogger(output / "history.csv"),
        keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=config.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", patience=3, factor=.5, min_lr=1e-6),
        keras.callbacks.TerminateOnNaN()]
    started = time.monotonic()
    model.fit(train, validation_data=val, epochs=config.epochs, callbacks=callbacks, verbose=0)
    best = keras.models.load_model(model_dir / "best.keras", compile=False)
    metrics = evaluate_full_model(root, best, val, val_frame, output)
    history = pd.read_csv(output / "history.csv")
    full_history_plot(history, output / "learning_curves.png")
    result = dict(run_name=config.run_name, status="completed", metrics=metrics, epochs_completed=len(history),
                  best_epoch=int(history.val_macro_f1.idxmax())+1, initialization="random", patch_weights_used=False,
                  model_path=str(model_dir / "best.keras"), elapsed_seconds=time.monotonic()-started)
    (output / "run_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="full_scratch_001")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    train_from_scratch(Path(__file__).resolve().parents[1], ScratchConfig(run_name=args.run_name, epochs=args.epochs))
