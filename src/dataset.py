"""디스크에서 batch 단위로 PNG를 읽는다. 학습 시 test 접근은 허용하지 않는다."""

from pathlib import Path

import pandas as pd
import tensorflow as tf


def make_dataset(root: str | Path, split: str, batch_size: int = 16,
                 augment: bool = False, seed: int = 42) -> tuple[tf.data.Dataset, pd.DataFrame]:
    if split not in {"train", "val"}:
        raise ValueError("학습 로더는 train/val만 허용합니다. test는 최종 평가 단계에서 개방합니다.")
    if augment and split != "train":
        raise ValueError("증강은 train에만 적용할 수 있습니다.")
    root = Path(root).resolve()
    frame = pd.read_csv(root / "data/split" / f"{split}.csv")
    if frame.empty or not frame.split.eq(split).all() or not frame.official_split.eq("trainval").all():
        raise ValueError("분할 manifest가 올바르지 않습니다.")
    paths = [str(root / p) for p in frame.patch_path]
    dataset = tf.data.Dataset.from_tensor_slices((paths, frame.target_id.to_numpy(dtype="int32")))
    if split == "train":
        # 영상 배열 대신 경로만 섞어 큰 shuffle 버퍼의 메모리 사용을 피한다.
        dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)

    def load(path, target):
        image = tf.io.decode_png(tf.io.read_file(path), channels=1)
        image = tf.ensure_shape(image, [128, 128, 1])
        image = tf.cast(image, tf.float32)
        if augment:
            # 보간 없이 90도 단위로 돌려 작은 구멍/단선의 형태를 유지한다.
            image = tf.image.rot90(image, tf.random.uniform((), 0, 4, dtype=tf.int32))
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.clip_by_value(tf.image.random_brightness(image, max_delta=5.0), 0.0, 255.0)
        # 정규화는 저장 모델 안의 Rescaling에서 딱 한 번 수행한다.
        return image, target

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 4
    dataset = dataset.with_options(options).map(load, num_parallel_calls=4)
    return dataset.batch(batch_size).prefetch(1), frame
