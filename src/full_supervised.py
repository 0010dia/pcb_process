"""무작위 초기화 원본 CNN. 학습 때만 결함 위치 정답을 보조 감독으로 사용한다."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

from src.binary_evaluate import binary_scores, collect_binary_predictions
from src.full_image_data import assert_full_split, full_image_dataset
from src.full_image_evaluate import evaluate_full_model, full_history_plot
from src.preprocessing import digest
from src.runtime import configure_runtime


def change_target(template, inspected, boxes, threshold=128):
    """정렬된 원본 쌍의 큰 차이 중 bbox 내부만 정답으로 사용한다. 모델 입력에는 넣지 않는다."""
    if template.shape != (640, 640) or inspected.shape != template.shape:
        raise ValueError("640×640 정렬 원본 쌍이 필요합니다.")
    allowed = np.zeros((640, 640), dtype=bool)
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=int):
        allowed[max(y1, 0):min(y2+1, 640), max(x1, 0):min(x2+1, 640)] = True
    changed = (np.abs(template.astype(np.int16)-inspected.astype(np.int16)) >= threshold) & allowed
    # max pooling으로 작은 1픽셀 변화도 160×160 보조 정답에서 사라지지 않게 한다.
    return changed.reshape(160, 4, 160, 4).max(axis=(1, 3)).astype(np.uint8)


def prepare_supervision(root: str | Path):
    root = Path(root).resolve()
    frame = pd.read_csv(root / "data/split/full_image_v1/images.csv")
    assert_full_split(frame)
    objects = pd.read_csv(root / "results/reports/objects.csv")
    folder = root / "data/processed/full_scratch_aux_v1"
    folder.mkdir(parents=True, exist_ok=True)
    output = root / "data/split/full_scratch_aux_v1"
    output.mkdir(parents=True, exist_ok=True)
    config = dict(raw_manifest_sha256=digest(root / "data/split/full_image_v1/images.csv"),
                  annotation_sha256=digest(root / "results/reports/objects.csv"),
                  teacher_threshold=128, teacher_downsample="max_pool_4", input_size=640,
                  masks_are_training_targets_only=True, test_images_read=0)
    if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != config:
        raise ValueError("보조 감독 설정이 변경됐습니다. 새 버전이 필요합니다.")
    rows = []
    for source, pair in frame.groupby("source_id", sort=False):
        good = pair[pair.target == 0].iloc[0]
        bad = pair[pair.target == 1].iloc[0]
        with Image.open(root / good.image_path) as image:
            template = np.asarray(image.convert("L"))
        with Image.open(root / bad.image_path) as image:
            inspected = np.asarray(image.convert("L"))
        boxes = objects.loc[objects.source_id == source, ["x1", "y1", "x2", "y2"]].values
        target = change_target(template, inspected, boxes)
        if not target.any():
            raise ValueError(f"보조 정답에 결함 픽셀이 없습니다: {source}")
        for row in pair.to_dict(orient="records"):
            mask = target if row["target"] else np.zeros((160, 160), dtype=np.uint8)
            name = hashlib.sha256(row["image_id"].encode()).hexdigest()+".png"
            relative = Path("data/processed/full_scratch_aux_v1") / name
            Image.fromarray(mask*255).save(root / relative)
            rows.append(dict(**row, mask_path=relative.as_posix(), positive_cells=int(mask.sum())))
    result = pd.DataFrame(rows)
    result.to_csv(output / "images.csv", index=False)
    for split in ["train", "val"]:
        result[result.split == split].to_csv(output / f"{split}.csv", index=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    train = result[result.split == "train"]
    positive = int(train.positive_cells.sum())
    total = len(train)*160*160
    weight = float((total-positive)/positive)
    summary = dict(train_images=len(train), validation_images=int(result.split.eq("val").sum()),
                   train_positive_fraction=positive/total, auxiliary_positive_weight=weight,
                   input_images=1, masks_are_not_inputs=True, old_model_weights_used=False,
                   manifest_sha256=digest(output / "images.csv"))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def supervised_dataset(root, split, batch_size=4, augment=False, seed=42):
    if split not in {"train", "val"} or (augment and split != "train"):
        raise ValueError("train/val만 허용하며 증강은 train에만 적용합니다.")
    root = Path(root).resolve()
    frame = pd.read_csv(root / "data/split/full_scratch_aux_v1" / f"{split}.csv")
    paths = [str(root / p) for p in frame.image_path]
    masks = [str(root / p) for p in frame.mask_path]
    dataset = tf.data.Dataset.from_tensor_slices((paths, masks, frame.target.to_numpy(dtype="float32").reshape(-1, 1)))
    if augment:
        dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)

    def decode(path):
        if isinstance(path, np.ndarray):
            path = path.item()
        with Image.open(path.decode()) as image:
            return np.asarray(image.convert("L"))[..., None].copy()

    def load(path, mask_path, label):
        image = tf.cast(tf.ensure_shape(tf.numpy_function(decode, [path], tf.uint8), [640, 640, 1]), tf.float32)
        target = tf.cast(tf.io.decode_png(tf.io.read_file(mask_path), channels=1), tf.float32)/255.0
        target = tf.ensure_shape(target, [160, 160, 1])
        if augment:
            k = tf.random.uniform((), 0, 4, dtype=tf.int32)
            image, target = tf.image.rot90(image, k), tf.image.rot90(target, k)
            horizontal = tf.random.uniform(()) > .5
            vertical = tf.random.uniform(()) > .5
            image = tf.cond(horizontal, lambda: tf.image.flip_left_right(image), lambda: image)
            target = tf.cond(horizontal, lambda: tf.image.flip_left_right(target), lambda: target)
            image = tf.cond(vertical, lambda: tf.image.flip_up_down(image), lambda: image)
            target = tf.cond(vertical, lambda: tf.image.flip_up_down(target), lambda: target)
        return image, {"decision": label, "mask_logits": target}

    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.experimental_deterministic = True
    return dataset.with_options(options).map(load, num_parallel_calls=2).batch(batch_size).prefetch(1), frame


@keras.utils.register_keras_serializable(package="PCB")
class WeightedMaskLoss(keras.losses.Loss):
    def __init__(self, positive_weight, name="weighted_mask_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.positive_weight = float(positive_weight)

    def call(self, y_true, logits):
        loss = tf.nn.weighted_cross_entropy_with_logits(y_true, logits, self.positive_weight)
        denominator = tf.reduce_mean(1+y_true*(self.positive_weight-1), axis=[1, 2, 3])
        return tf.reduce_mean(loss, axis=[1, 2, 3])/denominator

    def get_config(self):
        return dict(**super().get_config(), positive_weight=self.positive_weight)


def build_supervised_model(positive_weight):
    """모든 Conv 가중치를 새로 만든다. 정답 마스크를 받는 입력층은 없다."""
    configure_runtime()
    inputs = keras.Input((640, 640, 1), name="single_original_image")
    x = layers.Rescaling(1/255.0, name="pixel_normalization")(inputs)

    def block(x, filters, name, repetitions=2):
        for i in range(repetitions):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal", name=f"{name}_conv{i}")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        return x

    x = block(x, 8, "stem", repetitions=1)
    x = layers.MaxPooling2D()(x)  # 320
    x = block(x, 16, "encoder320")
    x = layers.MaxPooling2D()(x)  # 160
    at160 = block(x, 32, "encoder160")
    x = layers.MaxPooling2D()(at160)  # 80
    at80 = block(x, 64, "encoder80")
    x = layers.MaxPooling2D()(at80)  # 40
    context = block(x, 96, "context40")
    x = layers.UpSampling2D(interpolation="bilinear")(context)
    x = block(layers.Concatenate()([x, at80]), 32, "decoder80")
    x = layers.UpSampling2D(interpolation="bilinear")(x)
    x = block(layers.Concatenate()([x, at160]), 16, "decoder160")
    logits = layers.Conv2D(1, 1, name="mask_logits")(x)
    attention = layers.Activation("sigmoid", name="learned_attention")(logits)
    attended = layers.Multiply()([x, attention])
    features = layers.Concatenate()([layers.GlobalMaxPooling2D()(attended),
        layers.GlobalAveragePooling2D()(attended), layers.GlobalMaxPooling2D()(logits),
        layers.GlobalAveragePooling2D()(attention), layers.GlobalAveragePooling2D()(context)])
    features = layers.Dense(32, activation="relu")(features)
    features = layers.Dropout(.2)(features)
    decision = layers.Dense(1, activation="sigmoid", name="decision")(features)
    model = keras.Model(inputs, {"decision": decision, "mask_logits": logits}, name="Native640_Auxiliary_FromScratch")
    model.compile(optimizer=keras.optimizers.Adam(.0003, clipnorm=1.0),
        loss={"decision": keras.losses.BinaryCrossentropy(), "mask_logits": WeightedMaskLoss(positive_weight)},
        loss_weights={"decision": 1.0, "mask_logits": .5},
        metrics={"decision": [keras.metrics.BinaryAccuracy(name="accuracy")]}, jit_compile=False)
    return model


class SupervisedValidation(keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.started = time.monotonic()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        labels, probabilities = [], []
        for images, target in self.dataset:
            labels.append(target.numpy().reshape(-1))
            probabilities.append(self.model(images, training=False)["decision"].numpy().reshape(-1))
        scores = binary_scores(np.concatenate(labels), np.concatenate(probabilities))
        graph_accuracy = logs["val_decision_accuracy"]
        if abs(graph_accuracy-scores["accuracy"]) > 1e-4:
            raise RuntimeError("원본 graph/eager Accuracy 불일치입니다.")
        logs.update(accuracy=logs["decision_accuracy"], val_accuracy=graph_accuracy,
                    val_macro_f1=scores["macro_f1"], val_defect_recall=scores["defect_recall"],
                    val_normal_fpr=scores["normal_false_positive_rate"],
                    learning_rate=float(keras.backend.get_value(self.model.optimizer.learning_rate)))
        print(f"epoch={epoch+1:02d} train_acc={logs['decision_accuracy']:.4f} val_acc={scores['accuracy']:.4f} "
              f"macro_f1={scores['macro_f1']:.4f} recall={scores['defect_recall']:.4f} "
              f"normal_fpr={scores['normal_false_positive_rate']:.4f} elapsed={time.monotonic()-self.started:.0f}s", flush=True)


def resume_history(history: pd.DataFrame):
    """최고 checkpoint 이후의 로그는 보관하고 해당 epoch 다음부터 다시 실행한다."""
    if history.empty or history.epoch.tolist() != list(range(len(history))):
        raise ValueError("재개할 epoch 로그가 비어 있거나 연속적이지 않습니다.")
    best_index = int(history.val_macro_f1.idxmax())
    initial_epoch = int(history.loc[best_index, "epoch"])+1
    return initial_epoch, float(history.loc[best_index, "val_macro_f1"]), history.iloc[:initial_epoch].copy()


class RestoreBestCallbackState(keras.callbacks.Callback):
    """재개 시 새 callback이 기존 최고값을 잊어 checkpoint를 덮어쓰지 않게 한다."""
    def __init__(self, checkpoint, early_stopping, reduce_lr, best, initial_epoch):
        super().__init__()
        self.checkpoint, self.early_stopping, self.reduce_lr = checkpoint, early_stopping, reduce_lr
        self.best, self.initial_epoch = best, initial_epoch

    def on_train_begin(self, logs=None):
        self.checkpoint.best = self.best
        self.early_stopping.best = self.best
        self.early_stopping.best_weights = self.model.get_weights()
        self.early_stopping.best_epoch = self.initial_epoch-1
        self.reduce_lr.best = self.best


def train_supervised_scratch(root: str | Path, resume=False):
    root = Path(root).resolve()
    execution = configure_runtime()
    run_name = "full_scratch_aux_001"
    output = root / "results/full_scratch" / run_name
    model_dir = root / "models" / run_name
    summary = json.loads((root / "data/split/full_scratch_aux_v1/summary.json").read_text())
    config = dict(initialization="random", patch_weights_used=False, input_images=1, native_size=640,
                  epochs=20, batch_size=4, learning_rate=.0003, patience=5, seed=42,
                  auxiliary_weight=.5, positive_weight=summary["auxiliary_positive_weight"], execution=execution,
                  manifest_sha256=digest(root / "data/split/full_scratch_aux_v1/images.csv"))
    if config["manifest_sha256"] != summary["manifest_sha256"]:
        raise ValueError("보조 정답 manifest가 변경되었습니다.")
    if (output / "run_result.json").exists():
        saved_config = json.loads((output / "config.json").read_text())
        if {k: v for k, v in saved_config.items() if k != "execution"} != {k: v for k, v in config.items() if k != "execution"}:
            raise ValueError("저장된 실행 설정이 다릅니다.")
        print("완료된 원본 무작위 초기화 모델을 재사용합니다.", flush=True)
        return json.loads((output / "run_result.json").read_text())
    has_history = (output / "history.csv").exists()
    if has_history and not resume:
        raise ValueError("미완료 run이 있습니다. 새 이름을 사용하세요.")
    if resume and not has_history:
        raise ValueError("재개할 history/checkpoint가 없습니다. 처음 학습할 때는 resume=False를 사용하세요.")
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    initial_epoch, previous_best = 0, None
    if resume:
        saved_config = json.loads((output / "config.json").read_text())
        if {k: v for k, v in saved_config.items() if k != "execution"} != {k: v for k, v in config.items() if k != "execution"}:
            raise ValueError("장비 실행 정보를 제외한 학습 설정이 다릅니다.")
        initial_epoch, previous_best, retained = resume_history(pd.read_csv(output / "history.csv"))
        # compile=True로 Adam의 moment/iteration/현재 학습률도 함께 복원한다.
        model = keras.models.load_model(output / "best_training_model.keras")
        train_count = summary["train_images"]
        expected_steps = initial_epoch*((train_count+config["batch_size"]-1)//config["batch_size"])
        if int(model.optimizer.iterations.numpy()) != expected_steps:
            raise ValueError("checkpoint optimizer step과 최고 epoch 기록이 일치하지 않습니다.")
        if model.input_shape != (None, 640, 640, 1):
            raise ValueError("원본 단일 이미지 checkpoint가 아닙니다.")
        archive = output / f"history_before_resume_{time.time_ns()}.csv"
        shutil.copy2(output / "history.csv", archive)
        retained.to_csv(output / "history.csv", index=False)
        (output / "resume_runtime.json").write_text(json.dumps(dict(
            initial_epoch=initial_epoch, optimizer_iterations=expected_steps, execution=execution,
            previous_log_archive=archive.name, augmentation_random_stream_restarted=True), indent=2))
        print(f"저장된 {initial_epoch} epoch 가중치/Adam 상태 복원 → {initial_epoch+1} epoch부터 재개", flush=True)
    else:
        (output / "config.json").write_text(json.dumps(config, indent=2))
        model = build_supervised_model(config["positive_weight"])
    lines = []
    model.summary(print_fn=lambda line, **kwargs: lines.append(line))
    (output / "model_summary.txt").write_text("\n".join(lines))
    train, train_frame = supervised_dataset(root, "train", 4, augment=True)
    val, _ = supervised_dataset(root, "val", 4, augment=False)
    decision_val, decision_frame = full_image_dataset(root, "val", 4, augment=False)
    checkpoint = keras.callbacks.ModelCheckpoint(output / "best_training_model.keras", monitor="val_macro_f1", mode="max", save_best_only=True)
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=5, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", patience=2, factor=.5, min_lr=1e-6)
    callbacks = [SupervisedValidation(decision_val), checkpoint,
        keras.callbacks.CSVLogger(output / "history.csv", append=resume), early_stopping, reduce_lr,
        keras.callbacks.TerminateOnNaN()]
    if resume:
        callbacks.append(RestoreBestCallbackState(checkpoint, early_stopping, reduce_lr, previous_best, initial_epoch))
    print(f"원본 보조 감독 CNN: {len(train_frame)}장, {'저장된 자체 모델 재개' if resume else '모든 가중치 새 초기화'}, 입력 사진 1장", flush=True)
    started = time.monotonic()
    model.fit(train, validation_data=val, initial_epoch=initial_epoch, epochs=20, callbacks=callbacks, verbose=0)
    best = keras.models.load_model(output / "best_training_model.keras", compile=False)
    # 배포 파일에는 단일 사진→정상/불량 출력만 남긴다. 정답 마스크/기준 사진 입력은 없다.
    classifier = keras.Model(best.input, best.get_layer("decision").output, name="Native640_SingleImage_FromScratch")
    classifier.save(model_dir / "best.keras")
    (model_dir / "metadata.json").write_text(json.dumps(dict(task="full_image_single_binary", input_size=[640, 640, 1],
        input_images=1, reference_image_required=False, mask_input_required=False, threshold=.5,
        initialization="random", patch_weights_used=False, pretrained_weights_used=False,
        auxiliary_targets_training_only=True, crop=False, resize=False, calibrated=False,
        classes={"0": "normal", "1": "defect"}, official_test_evaluated=False), indent=2))
    scores = evaluate_full_model(root, classifier, decision_val, decision_frame, output)
    history = pd.read_csv(output / "history.csv")
    full_history_plot(history, output / "learning_curves.png")
    result = dict(run_name=run_name, metrics=scores, initialization="random", patch_weights_used=False,
                  input_images=1, epochs_completed=len(history), best_epoch=int(history.val_macro_f1.idxmax())+1,
                  elapsed_seconds=time.monotonic()-started, parameters=classifier.count_params(),
                  resumed_from_epoch=initial_epoch if resume else None, elapsed_seconds_scope="this_session",
                  model_path=str(model_dir / "best.keras"), status="completed")
    (output / "run_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result
