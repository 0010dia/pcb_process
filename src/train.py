"""validation Macro F1로 CNN을 선택한다. 공식 test는 학습 코드에서 읽지 않는다."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.dataset import make_dataset
from src.eda import CLASSES
from src.evaluate import collect_predictions, plot_history, save_evaluation, scores
from src.model import build_model
from src.preprocessing import assert_separation, digest
from src.runtime import configure_runtime


@dataclass(frozen=True)
class TrainingConfig:
    run_name: str = "A_baseline_safe_002"
    variant: str = "A"
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = .001
    patience: int = 7
    augment: bool = False
    seed: int = 42


class ValidationMacroF1(keras.callbacks.Callback):
    """고정된 validation 전체로 지표를 집계해 checkpoint보다 먼저 logs에 추가한다."""
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.started = time.monotonic()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        labels, probabilities = collect_predictions(self.model, self.dataset)
        metric = scores(labels, probabilities.argmax(axis=1))
        logs["val_numpy_accuracy"] = metric["accuracy"]
        if abs(metric["accuracy"] - logs["val_accuracy"]) > 1e-4:
            raise RuntimeError("graph의 Keras Accuracy와 eager 추론 Accuracy가 다릅니다. 실행 백엔드를 확인하세요.")
        logs["val_macro_f1"] = metric["macro_f1"]
        logs["learning_rate"] = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"epoch={epoch+1:02d} loss={logs['loss']:.4f} acc={logs['accuracy']:.4f} "
              f"val_loss={logs['val_loss']:.4f} val_acc={logs['val_accuracy']:.4f} "
              f"val_macro_f1={metric['macro_f1']:.4f} elapsed={time.monotonic()-self.started:.0f}s", flush=True)


def train_model(project_root: str | Path, config: TrainingConfig = TrainingConfig()) -> dict:
    root = Path(project_root).resolve()
    execution = configure_runtime()
    if not 1 <= config.epochs <= 100 or config.variant not in {"A", "B"}:
        raise ValueError("epochs는 1~100, variant는 A/B여야 합니다.")
    if Path(config.run_name).name != config.run_name:
        raise ValueError("run_name에 경로를 넣을 수 없습니다.")
    output = root / "results/training" / config.run_name
    model_dir = root / "models" / config.run_name
    split_dir = root / "data/split"
    assert_separation(pd.read_csv(split_dir / "sources.csv"))
    expected = json.loads((root / "results/reports/preprocessing_summary.json").read_text())["manifest_sha256"]
    if digest(split_dir / "patches.csv") != expected:
        raise ValueError("전처리 manifest가 변경되었습니다.")
    signature = dict(config=asdict(config), execution=execution,
                     split_sha256={name: digest(split_dir / name) for name in ["train.csv", "val.csv", "sources.csv", "preprocessing_config.json"]})
    if (output / "run_result.json").exists():
        if json.loads((output / "config.json").read_text()) != signature:
            raise ValueError("기존 run 설정과 다릅니다. 새로운 run_name을 지정하세요.")
        print(f"기존 완료 결과를 재사용합니다: {config.run_name}", flush=True)
        return dict(run_name=config.run_name, status="reused_completed_run",
                    metrics=json.loads((output / "validation_metrics.json").read_text()), model_path=str(model_dir / "best.keras"))
    if (output / "history.csv").exists():
        raise ValueError("미완료 run이 있습니다. 기록을 보존하고 새 run_name으로 재실행하세요.")
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(signature, indent=2))
    keras.backend.clear_session()
    keras.utils.set_random_seed(config.seed)
    train_data, train_frame = make_dataset(root, "train", config.batch_size, config.augment, config.seed)
    val_data, val_frame = make_dataset(root, "val", config.batch_size, False, config.seed)
    # 개별 split 파일을 변조해 원본이나 그룹이 섞이는 것을 추가로 확인한다.
    train_ids, val_ids = set(train_frame.source_id), set(val_frame.source_id)
    if train_ids & val_ids or set(train_frame.group) & set(val_frame.group):
        raise ValueError("train/val manifest에 원본/그룹 누수가 있습니다.")
    model = build_model(config.variant, config.learning_rate)
    lines = []
    model.summary(print_fn=lambda line, **kwargs: lines.append(line))
    (output / "model_summary.txt").write_text("\n".join(lines))
    versions = {name: metadata.version(name) for name in ["tensorflow", "keras", "numpy", "pandas", "scikit-learn"]}
    runtime = dict(python=platform.python_version(), machine=platform.machine(), packages=versions,
                   physical_gpus=[str(gpu) for gpu in tf.config.list_physical_devices("GPU")],
                   parameters=model.count_params(), train_samples=len(train_frame), validation_samples=len(val_frame),
                   official_test_evaluated=False, reproducibility="fixed seed; GPU kernels may be nondeterministic")
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    contract = dict(classes=[dict(target_id=i-1, name=CLASSES[i][0], korean=CLASSES[i][1]) for i in CLASSES],
                    image_size=[128, 128], channels=1, input_range=[0, 255],
                    normalization="Rescaling(1/255) inside saved model", input_type="defect-centered patch",
                    softmax="uncalibrated relative class scores; not probability of defect presence",
                    crop_config=json.loads((split_dir / "preprocessing_config.json").read_text())["config"])
    (model_dir / "metadata.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2))
    callbacks = [ValidationMacroF1(val_data),
                 keras.callbacks.ModelCheckpoint(model_dir / "best.keras", monitor="val_macro_f1", mode="max", save_best_only=True),
                 keras.callbacks.CSVLogger(output / "history.csv"),
                 keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=config.patience, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", patience=3, factor=.5, min_lr=1e-6),
                 keras.callbacks.TerminateOnNaN()]
    print(f"학습 시작: {config.run_name}, {len(train_frame)} train / {len(val_frame)} val, {model.count_params():,} parameters", flush=True)
    started = time.monotonic()
    model.fit(train_data, validation_data=val_data, epochs=config.epochs, callbacks=callbacks, verbose=0)
    # 메모리의 마지막 상태가 아니라 실제 저장된 최고 F1 checkpoint를 재로드한다.
    best = keras.models.load_model(model_dir / "best.keras")
    history = pd.read_csv(output / "history.csv")
    plot_history(history, output / "learning_curves.png")
    metrics = save_evaluation(root, best, val_data, val_frame, output)
    result = dict(run_name=config.run_name, status="completed", metrics=metrics,
                  epochs_completed=len(history), best_epoch=int(history.val_macro_f1.idxmax())+1,
                  elapsed_seconds=time.monotonic()-started, model_path=str(model_dir / "best.keras"))
    (output / "run_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["A", "B"], default="A")
    parser.add_argument("--run-name", default="A_baseline_safe_002")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train_model(Path(__file__).resolve().parents[1], TrainingConfig(
        run_name=args.run_name, variant=args.variant, epochs=args.epochs, augment=args.variant == "B",
        batch_size=args.batch_size, learning_rate=args.learning_rate or (.0005 if args.variant == "B" else .001),
        patience=args.patience if args.patience is not None else (10 if args.variant == "B" else 7)))
