"""원본 640×640 한 장을 읽는 정상/불량 분류 데이터. Crop/resize를 하지 않는다."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.preprocessing import assert_separation, digest


def assert_full_split(frame: pd.DataFrame) -> None:
    if set(frame.split) != {"train", "val"} or not frame.official_split.eq("trainval").all():
        raise ValueError("공식 trainval의 train/val만 사용할 수 있습니다.")
    for key in ["source_id", "group", "pixel_sha256"]:
        if frame.groupby(key).split.nunique().gt(1).any():
            raise ValueError(f"원본 전체 이미지의 {key}가 train/val에 걸쳐 있습니다.")
    if frame.image_id.duplicated().any():
        raise ValueError("원본 image_id 중복입니다.")
    if frame.groupby("pixel_sha256").target.nunique().gt(1).any():
        raise ValueError("동일 픽셀에 정상/불량 라벨이 충돌합니다.")
    for split in ["train", "val"]:
        if set(frame.loc[frame.split == split, "target"]) != {0, 1}:
            raise ValueError("각 분할에 정상/불량 원본이 모두 필요합니다.")


def prepare_full_image_data(root: str | Path) -> dict:
    root = Path(root).resolve()
    sources = pd.read_csv(root / "data/split/sources.csv")
    assert_separation(sources)
    output = root / "data/split/full_image_v1"
    reports = root / "results/full_image"
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    config = dict(version="full_image_v1", native_size=[640, 640], channels=1,
                  crop=False, resize=False, source_manifest_sha256=digest(root / "data/split/sources.csv"),
                  raw_manifest_sha256={name: digest(root / "data" / name) for name in ["trainval.txt", "test.txt"]},
                  class_mapping={"0": "normal_template", "1": "defective_inspection"},
                  test_policy="official test raw images/labels not read; no test evaluation")
    lock = output / "config.json"
    if lock.exists() and json.loads(lock.read_text()) != config:
        raise ValueError("full_image_v1 설정이 변경되었습니다. 새 버전을 사용하세요.")
    expected = json.loads((root / "data/split/preprocessing_config.json").read_text())["official_manifest_sha256"]
    if config["raw_manifest_sha256"] != expected:
        raise ValueError("공식 원본 목록이 이전 데이터 감사 이후 변경됐습니다.")
    rows = []
    for source in sources[sources.split.isin(["train", "val"])].itertuples():
        for role, target in [("temp", 0), ("test", 1)]:
            relative = Path("data") / getattr(source, f"{role}_path")
            with Image.open(root / relative) as image:
                pixels = np.asarray(image.convert("L"))
            if pixels.shape != (640, 640):
                raise ValueError(f"640×640 원본이 필요합니다: {relative}")
            sha = hashlib.sha256(b"(640, 640):L:"+pixels.tobytes()).hexdigest()
            if sha != getattr(source, f"{role}_pixel_sha256"):
                raise ValueError(f"이전 감사 이후 원본이 변경됐습니다: {relative}")
            rows.append(dict(image_id=source.source_id+"_"+role, source_id=source.source_id,
                             group=source.group, split=source.split, official_split=source.official_split,
                             target=target, label="defect" if target else "normal", role=role,
                             image_path=relative.as_posix(), pixel_sha256=sha))
    frame = pd.DataFrame(rows)
    assert_full_split(frame)
    lock.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    frame.to_csv(output / "images.csv", index=False, lineterminator="\n")
    for split in ["train", "val"]:
        frame[frame.split == split].to_csv(output / f"{split}.csv", index=False, lineterminator="\n")
    counts = frame.groupby(["split", "label"]).size().rename("images").reset_index()
    counts.to_csv(reports / "data_counts.csv", index=False, lineterminator="\n")
    summary = dict(counts=counts.to_dict(orient="records"), rows=len(frame),
                   duplicate_pixel_rows_within_train=int(frame[frame.split == "train"].pixel_sha256.duplicated().sum()),
                   train_sources=int(frame[frame.split == "train"].source_id.nunique()),
                   validation_sources=int(frame[frame.split == "val"].source_id.nunique()),
                   cross_split_pixel_overlap=0, cross_split_group_overlap=0,
                   test_images_read=0, native_resolution=[640, 640], cropped=False,
                   manifest_sha256=digest(output / "images.csv"))
    (reports / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def full_image_dataset(root: str | Path, split: str, batch_size: int = 2,
                       augment: bool = False, seed: int = 42):
    if split not in {"train", "val"}:
        raise ValueError("공식 test는 현재 학습/선택용 로더에서 접근할 수 없습니다.")
    if augment and split != "train":
        raise ValueError("증강은 train에만 허용합니다.")
    root = Path(root).resolve()
    frame = pd.read_csv(root / "data/split/full_image_v1" / f"{split}.csv")
    if not frame.split.eq(split).all() or not frame.official_split.eq("trainval").all():
        raise ValueError("원본 분할 manifest가 올바르지 않습니다.")
    dataset = tf.data.Dataset.from_tensor_slices(([str(root / path) for path in frame.image_path],
                                                 frame.target.to_numpy(dtype="float32").reshape(-1, 1)))
    if augment:
        dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)

    def decode(path):
        # 학습/독립 추론/원본 SHA 감사에 동일한 Pillow 디코더를 사용한다.
        if isinstance(path, np.ndarray):
            path = path.item()
        with Image.open(path.decode("utf-8")) as image:
            return np.asarray(image.convert("L"))[..., None].copy()

    def load(path, target):
        image = tf.numpy_function(decode, [path], tf.uint8)
        image = tf.ensure_shape(image, [640, 640, 1])
        if augment:
            # 90도 회전/반전은 원본의 결함을 자르거나 보간으로 없애지 않는다.
            image = tf.image.rot90(image, tf.random.uniform((), 0, 4, dtype=tf.int32))
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
        return tf.cast(image, tf.float32), target

    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.experimental_deterministic = True
    return dataset.with_options(options).map(load, num_parallel_calls=2).batch(batch_size).prefetch(1), frame


if __name__ == "__main__":
    prepare_full_image_data(Path(__file__).resolve().parents[1])
