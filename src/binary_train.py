"""6종 분류와 독립된 정상/불량 CNN 학습. test 접근을 차단한다."""

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

from src.binary_data import assert_binary_split
from src.binary_evaluate import binary_scores, collect_binary_predictions, plot_binary_history, save_binary_evaluation
from src.binary_model import build_binary_model
from src.preprocessing import digest
from src.runtime import configure_runtime


@dataclass(frozen=True)
class BinaryTrainingConfig:
    run_name: str = "binary_cnn_001"
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = .0005
    patience: int = 6
    seed: int = 42
    augment: bool = True
    threshold: float = .5


def make_binary_dataset(root: str | Path, split: str, batch_size: int = 32,
                        augment: bool = False, seed: int = 42):
    if split not in {"train", "val"}:
        raise ValueError("이진 학습 로더는 train/val만 허용합니다.")
    if augment and split != "train":
        raise ValueError("증강은 train에만 적용할 수 있습니다.")
    root = Path(root).resolve()
    frame = pd.read_csv(root / "data/split/binary_v1" / f"{split}.csv")
    if not frame.split.eq(split).all() or not frame.official_split.eq("trainval").all():
        raise ValueError("분할 메타데이터가 올바르지 않습니다.")
    if not set(frame.target) == {0, 1}:
        raise ValueError("정상/불량 두 클래스가 필요합니다.")
    paths = [str(root / path) for path in frame.patch_path]
    labels = frame.target.to_numpy(dtype="float32").reshape(-1, 1)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if split == "train":
        dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        image = tf.io.decode_png(tf.io.read_file(path), channels=1)
        image = tf.cast(tf.ensure_shape(image, [128, 128, 1]), tf.float32)
        if augment:
            image = tf.image.rot90(image, tf.random.uniform((), 0, 4, dtype=tf.int32))
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.clip_by_value(tf.image.random_brightness(image, 5.0), 0.0, 255.0)
        return image, label  # 저장 모델 내부에서만 1/255 정규화한다.

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 4
    return dataset.with_options(options).map(load, num_parallel_calls=4).batch(batch_size).prefetch(1), frame


class BinaryValidation(keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.started = time.monotonic()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        labels, probabilities = collect_binary_predictions(self.model, self.dataset)
        values = binary_scores(labels, probabilities)
        if abs(values["accuracy"]-logs["val_accuracy"]) > 1e-4:
            raise RuntimeError("Keras graph Accuracy와 독립 eager Accuracy가 다릅니다.")
        logs.update(val_macro_f1=values["macro_f1"], val_defect_recall=values["defect_recall"],
                    val_normal_fpr=values["normal_false_positive_rate"], val_numpy_accuracy=values["accuracy"],
                    learning_rate=float(keras.backend.get_value(self.model.optimizer.learning_rate)))
        print(f"epoch={epoch+1:02d} loss={logs['loss']:.4f} val_acc={values['accuracy']:.4f} "
              f"macro_f1={values['macro_f1']:.4f} defect_recall={values['defect_recall']:.4f} "
              f"normal_fpr={values['normal_false_positive_rate']:.4f} elapsed={time.monotonic()-self.started:.0f}s", flush=True)


def train_binary_model(project_root: str | Path, config: BinaryTrainingConfig = BinaryTrainingConfig()) -> dict:
    root = Path(project_root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", config.run_name):
        raise ValueError("run_name은 영문/숫자/밑줄/하이픈만 사용할 수 있습니다.")
    if not 1 <= config.epochs <= 100 or config.batch_size < 1 or config.learning_rate <= 0 or config.patience < 1:
        raise ValueError("학습 설정이 올바르지 않습니다.")
    if config.threshold != .5:
        raise ValueError("이번 실험은 비교를 위해 임계값 0.5를 고정합니다.")
    execution = configure_runtime()
    split_dir = root / "data/split/binary_v1"
    summary = json.loads((root / "results/binary/data_summary.json").read_text())
    if digest(split_dir / "patches.csv") != summary["manifest_sha256"]:
        raise ValueError("이진 전처리 manifest가 변경되었습니다.")
    combined = pd.read_csv(split_dir / "patches.csv")
    assert_binary_split(combined)
    for split in ["train", "val"]:
        if not pd.read_csv(split_dir / f"{split}.csv").equals(combined[combined.split == split].reset_index(drop=True)):
            raise ValueError(f"{split}.csv가 전체 manifest와 다릅니다.")
    output = root / "results/binary" / config.run_name
    model_dir = root / "models" / config.run_name
    signature = dict(config=asdict(config), execution=execution,
                     data_sha256={name: digest(split_dir / name) for name in ["config.json", "patches.csv", "train.csv", "val.csv"]})
    if (output / "run_result.json").exists():
        if json.loads((output / "config.json").read_text()) != signature:
            raise ValueError("완료 run 설정과 다릅니다. 새 run_name을 사용하세요.")
        print(f"완료된 이진 분류 결과를 재사용합니다: {config.run_name}", flush=True)
        return json.loads((output / "run_result.json").read_text())
    if (output / "history.csv").exists():
        raise ValueError("미완료 run 기록을 보존하고 새 이름으로 시작하세요.")
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(signature, indent=2))
    keras.backend.clear_session()
    keras.utils.set_random_seed(config.seed)
    train_data, train_frame = make_binary_dataset(root, "train", config.batch_size, config.augment, config.seed)
    val_data, val_frame = make_binary_dataset(root, "val", config.batch_size, False, config.seed)
    counts = train_frame.target.value_counts()
    class_weight = {int(target): len(train_frame)/(2*int(count)) for target, count in counts.items()}
    model = build_binary_model(config.learning_rate)
    lines = []
    model.summary(print_fn=lambda line, **kwargs: lines.append(line))
    (output / "model_summary.txt").write_text("\n".join(lines))
    # 모델을 쓰지 않고 항상 정상으로 예측하는 기준선도 같은 검증 데이터에서 기록한다.
    baseline = binary_scores(val_frame.target.values, np.zeros(len(val_frame)))
    (output / "majority_normal_baseline.json").write_text(json.dumps(baseline, indent=2))
    runtime = dict(python=platform.python_version(), packages={name: metadata.version(name) for name in ["tensorflow", "tensorflow-metal", "keras", "numpy", "pandas", "scikit-learn"]},
                   execution=execution, parameters=model.count_params(), class_weight=class_weight,
                   train_samples=len(train_frame), validation_samples=len(val_frame),
                   fresh_weights=True, transferred_six_class_weights=False,
                   nondeterminism="fixed seed; parallel augmentation and GPU may not be bitwise reproducible")
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    contract = dict(task="binary_patch_classification", classes={"0": "normal", "1": "defect"},
                    threshold=.5, decision="defect when sigmoid_score > threshold", input_size=[128, 128, 1],
                    input_range=[0, 255], normalization="Rescaling(1/255) inside model",
                    input_scope="single PCB circuit patch; not a complete PCB image", calibrated=False,
                    preprocessing_version="binary_v1", validation_group="group00041", official_test_evaluated=False)
    (model_dir / "metadata.json").write_text(json.dumps(contract, indent=2))
    callbacks = [BinaryValidation(val_data),
                 keras.callbacks.ModelCheckpoint(model_dir / "best.keras", monitor="val_macro_f1", mode="max", save_best_only=True),
                 keras.callbacks.CSVLogger(output / "history.csv"),
                 keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=config.patience, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", patience=3, factor=.5, min_lr=1e-6),
                 keras.callbacks.TerminateOnNaN()]
    print(f"이진 CNN 학습: {len(train_frame)} train / {len(val_frame)} val; {model.count_params():,} parameters; weights={class_weight}", flush=True)
    started = time.monotonic()
    model.fit(train_data, validation_data=val_data, epochs=config.epochs, class_weight=class_weight, callbacks=callbacks, verbose=0)
    best = keras.models.load_model(model_dir / "best.keras")
    metrics = save_binary_evaluation(root, best, val_data, val_frame, output)
    history = pd.read_csv(output / "history.csv")
    plot_binary_history(history, output / "learning_curves.png")
    result = dict(run_name=config.run_name, status="completed", metrics=metrics,
                  best_epoch=int(history.val_macro_f1.idxmax())+1, epochs_completed=len(history),
                  elapsed_seconds=time.monotonic()-started, model_path=str(model_dir / "best.keras"))
    (output / "run_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--run-name", default="binary_cnn_001")
    args = parser.parse_args()
    train_binary_model(Path(__file__).resolve().parents[1], BinaryTrainingConfig(run_name=args.run_name, epochs=args.epochs))
