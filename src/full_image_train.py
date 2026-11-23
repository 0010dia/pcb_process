"""원본 특징 전이학습과 native 640 이미지 미세조정을 별도 실행으로 비교한다."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.binary_evaluate import binary_scores, collect_binary_predictions
from src.full_image_data import assert_full_split, full_image_dataset
from src.full_image_evaluate import evaluate_full_model, full_history_plot
from src.full_image_model import assemble_full_model, build_full_encoder, build_full_head, unfreeze_last_convolutions
from src.preprocessing import digest
from src.runtime import configure_runtime


class FullValidation(keras.callbacks.Callback):
    def __init__(self, features=None, labels=None, dataset=None):
        super().__init__()
        self.features, self.labels, self.dataset = features, labels, dataset
        self.start = time.monotonic()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        if self.dataset is not None:
            labels, scores = collect_binary_predictions(self.model, self.dataset)
        else:
            labels = self.labels
            scores = self.model(self.features, training=False).numpy().reshape(-1)
        values = binary_scores(labels, scores)
        if abs(values["accuracy"]-logs["val_accuracy"]) > 1e-4:
            raise RuntimeError("원본 모델의 graph/eager Accuracy가 다릅니다.")
        logs.update(val_macro_f1=values["macro_f1"], val_defect_recall=values["defect_recall"],
                    val_normal_fpr=values["normal_false_positive_rate"], val_numpy_accuracy=values["accuracy"],
                    learning_rate=float(keras.backend.get_value(self.model.optimizer.learning_rate)))
        print(f"epoch={epoch+1:02d} val_acc={values['accuracy']:.4f} macro_f1={values['macro_f1']:.4f} "
              f"defect_recall={values['defect_recall']:.4f} normal_fpr={values['normal_false_positive_rate']:.4f} "
              f"elapsed={time.monotonic()-self.start:.0f}s", flush=True)


def _assert_inputs(root):
    split = root / "data/split/full_image_v1"
    frame = pd.read_csv(split / "images.csv")
    assert_full_split(frame)
    expected = json.loads((root / "results/full_image/data_summary.json").read_text())["manifest_sha256"]
    if digest(split / "images.csv") != expected:
        raise ValueError("원본 이미지 manifest가 변경됐습니다.")
    for name in ["train", "val"]:
        if not pd.read_csv(split / f"{name}.csv").equals(frame[frame.split == name].reset_index(drop=True)):
            raise ValueError("split별 원본 목록이 전체 manifest와 다릅니다.")


def _prepare_run(root, run_name, config):
    _assert_inputs(root)
    output = root / "results/full_image" / run_name
    model_dir = root / "models" / run_name
    if (output / "run_result.json").exists():
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("완료 run 설정과 다릅니다. 새 run 이름을 사용하세요.")
        print(f"원본 모델의 완료 결과를 재사용합니다: {run_name}", flush=True)
        return output, model_dir, json.loads((output / "run_result.json").read_text())
    if (output / "history.csv").exists():
        raise ValueError("미완료 run이 있습니다. 기존 로그를 보존하고 새 이름으로 실행하세요.")
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (model_dir / "metadata.json").write_text(json.dumps(dict(task="full_image_single_binary",
        input_size=[640, 640, 1], input_images=1, reference_image_required=False,
        crop=False, resize=False, input_range=[0, 255], threshold=.5,
        classes={"0": "normal", "1": "defect"}, calibrated=False,
        pixel_normalization="Rescaling(1/255) inside encoder", feature_normalization="train feature mean/variance",
        validation_group="group00041", official_test_evaluated=False), indent=2))
    return output, model_dir, None


def _cache_features(root, encoder):
    cache = root / "data/processed/full_image_v1"
    cache.mkdir(parents=True, exist_ok=True)
    signature = dict(manifest_sha256=digest(root / "data/split/full_image_v1/images.csv"),
                     patch_model_sha256=digest(root / "models/binary_cnn_001/best.keras"),
                     train_views=["original_640", "rotate90_640"], val_views=["original_640"],
                     pixel_normalization="inside encoder only", feature_dimensions=224)
    if (cache / "features.npz").exists() and (cache / "config.json").exists():
        if json.loads((cache / "config.json").read_text()) != signature:
            raise ValueError("원본 특징 캐시 설정이 다릅니다.")
        with np.load(cache / "features.npz") as data:
            return {name: data[name].copy() for name in data.files}
    result = {}
    for split in ["train", "val"]:
        dataset, frame = full_image_dataset(root, split, batch_size=2, augment=False)
        features, labels = [], []
        for index, (images, targets) in enumerate(dataset):
            features.append(encoder(images, training=False).numpy())
            labels.append(targets.numpy())
            if split == "train":
                features.append(encoder(tf.image.rot90(images, 1), training=False).numpy())
                labels.append(targets.numpy())
            if (index+1) % 100 == 0:
                print(f"{split} 원본 특징 추출: {min((index+1)*2, len(frame))}/{len(frame)}장", flush=True)
        result[split+"_x"], result[split+"_y"] = np.concatenate(features), np.concatenate(labels)
    np.savez_compressed(cache / "features.npz", **result)
    (cache / "config.json").write_text(json.dumps(signature, indent=2))
    return result


def _callbacks(validation, checkpoint, history_path, patience, initial_threshold=None):
    return [validation,
        keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_macro_f1", mode="max", save_best_only=True,
                                         initial_value_threshold=initial_threshold),
        keras.callbacks.CSVLogger(history_path),
        keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", patience=2, factor=.5, min_lr=1e-6),
        keras.callbacks.TerminateOnNaN()]


def train_full_frozen(root: str | Path) -> dict:
    root = Path(root).resolve()
    execution = configure_runtime()
    run_name = "full_frozen_001"
    config = dict(method="patch_conv_transfer_frozen", native_size=640, head_epochs=40, head_batch=32,
                  learning_rate=.001, patience=6, seed=42, train_views=2, execution=execution,
                  input_manifest_sha256=digest(root / "data/split/full_image_v1/images.csv"),
                  patch_model_sha256=digest(root / "models/binary_cnn_001/best.keras"))
    output, model_dir, completed = _prepare_run(root, run_name, config)
    if completed:
        return completed
    keras.utils.set_random_seed(42)
    encoder = build_full_encoder(root / "models/binary_cnn_001/best.keras")
    encoder.save(output / "frozen_encoder.keras")
    started = time.monotonic()
    features = _cache_features(root, encoder)
    print(f"원본 전체 특징: train={features['train_x'].shape}, validation={features['val_x'].shape}", flush=True)
    # 작은 분류기는 CPU에서 학습하고, 원본 합성곱 특징 추출은 Metal을 사용한다.
    with tf.device("/CPU:0"):
        head = build_full_head(features["train_x"])
        head.compile(optimizer=keras.optimizers.Adam(.001), loss="binary_crossentropy",
                     metrics=[keras.metrics.BinaryAccuracy(name="accuracy")], jit_compile=False)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        train = tf.data.Dataset.from_tensor_slices((features["train_x"], features["train_y"])).shuffle(len(features["train_x"]), seed=42).batch(32).with_options(options)
        val = tf.data.Dataset.from_tensor_slices((features["val_x"], features["val_y"])).batch(32).with_options(options)
        head.fit(train, validation_data=val, epochs=40, callbacks=_callbacks(
            FullValidation(features=features["val_x"], labels=features["val_y"]),
            output / "best_feature_head.keras", output / "history.csv", 6), verbose=0)
        best_head = keras.models.load_model(output / "best_feature_head.keras")
    model = assemble_full_model(encoder, best_head)
    model.save(model_dir / "best.keras")
    validation, frame = full_image_dataset(root, "val", batch_size=2)
    metrics = evaluate_full_model(root, model, validation, frame, output)
    cached_scores = best_head(features["val_x"], training=False).numpy().reshape(-1)
    fresh_scores = pd.read_csv(output / "validation_predictions.csv").sigmoid_score.values
    delta = float(abs(cached_scores-fresh_scores).max())
    if delta > 1e-4:
        raise RuntimeError("캐시 특징 분류와 원본 end-to-end 추론이 다릅니다.")
    history = pd.read_csv(output / "history.csv")
    full_history_plot(history, output / "learning_curves.png")
    result = dict(run_name=run_name, metrics=metrics, epochs_completed=len(history),
                  best_epoch=int(history.val_macro_f1.idxmax())+1, elapsed_seconds=time.monotonic()-started,
                  full_model_parameters=model.count_params(), trainable_parameters=head.count_params(),
                  cached_vs_end_to_end_max_score_delta=delta, model_path=str(model_dir / "best.keras"), status="completed")
    (output / "run_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def train_full_finetuned(root: str | Path) -> dict:
    root = Path(root).resolve()
    execution = configure_runtime()
    run_name = "full_finetuned_001"
    initial_path = root / "models/full_frozen_001/best.keras"
    config = dict(method="native640_last_two_conv_finetune", native_size=640, epochs=6, batch_size=2,
                  learning_rate=.00001, patience=3, seed=42, augment="whole_image_rot90_and_flips",
                  execution=execution, input_manifest_sha256=digest(root / "data/split/full_image_v1/images.csv"),
                  initialization_sha256=digest(initial_path))
    output, model_dir, completed = _prepare_run(root, run_name, config)
    if completed:
        return completed
    keras.utils.set_random_seed(42)
    model = keras.models.load_model(initial_path, compile=False)
    selected_layers = unfreeze_last_convolutions(model, 2)
    model.compile(optimizer=keras.optimizers.Adam(.00001), loss="binary_crossentropy",
                  metrics=[keras.metrics.BinaryAccuracy(name="accuracy")], jit_compile=False)
    baseline = json.loads((root / "results/full_image/full_frozen_001/validation_metrics.json").read_text())["macro_f1"]
    # 미세조정이 나빠지면 epoch 0의 초기 모델을 그대로 유지한다.
    model.save(model_dir / "best.keras")
    train, train_frame = full_image_dataset(root, "train", 2, augment=True)
    val, val_frame = full_image_dataset(root, "val", 2, augment=False)
    print(f"원본 640×640 미세조정 시작: train={len(train_frame)}, val={len(val_frame)}, Conv={selected_layers}", flush=True)
    started = time.monotonic()
    model.fit(train, validation_data=val, epochs=6, callbacks=_callbacks(
        FullValidation(dataset=val), model_dir / "best.keras", output / "history.csv", 3, initial_threshold=baseline), verbose=0)
    best = keras.models.load_model(model_dir / "best.keras", compile=False)
    metrics = evaluate_full_model(root, best, val, val_frame, output)
    history = pd.read_csv(output / "history.csv")
    full_history_plot(history, output / "learning_curves.png")
    best_epoch = int(history.val_macro_f1.idxmax())+1 if history.val_macro_f1.max() > baseline else 0
    result = dict(run_name=run_name, metrics=metrics, epochs_completed=len(history), best_epoch=best_epoch,
                  baseline_macro_f1=baseline, trainable_convolutions=selected_layers,
                  elapsed_seconds=time.monotonic()-started, model_path=str(model_dir / "best.keras"), status="completed")
    (output / "run_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result
