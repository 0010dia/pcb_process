"""공식 test를 잠근 상태에서 그룹 분할과 단일 불량 패치를 만든다."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.eda import CLASSES


@dataclass(frozen=True)
class CropConfig:
    size: int = 128
    margin: int = 16
    validation_group: str = "group00041"
    seed: int = 42
    endpoint: str = "inclusive_conservative"
    padding: str = "edge"
    interpolation: str = "bilinear"

    def __post_init__(self):
        if self.size < 16 or self.margin < 0:
            raise ValueError("size는 16 이상, margin은 0 이상이어야 합니다.")
        if self.endpoint != "inclusive_conservative" or self.padding != "edge" or self.interpolation != "bilinear":
            raise ValueError("지원하지 않는 Crop 규약입니다.")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_separation(sources: pd.DataFrame) -> None:
    """train/val의 원본·그룹·검사/기준 픽셀 중복을 모두 거부한다."""
    if sources.source_id.duplicated().any():
        raise ValueError("중복 원본 ID가 있습니다.")
    tv = sources[sources.split.isin(["train", "val"])]
    if not set(tv.split) == {"train", "val"}:
        raise ValueError("train과 val이 모두 필요합니다.")
    for key in ["group", "test_pixel_sha256", "temp_pixel_sha256"]:
        if tv.groupby(key).split.nunique().gt(1).any():
            raise ValueError(f"train/val 사이에 {key} 중복이 있습니다.")
    if not (sources.loc[sources.official_split == "test", "split"] == "test").all():
        raise ValueError("공식 test가 train/val에 배정되었습니다.")


def square_roi(box: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    """오른쪽/아래 끝 픽셀을 보수적으로 포함하고 정사각형 반열린 ROI로 변환한다."""
    x1, y1, x2, y2 = map(int, box)
    if not (x1 < x2 and y1 < y2) or margin < 0:
        raise ValueError("잘못된 bbox 또는 margin입니다.")
    # 공식 README는 끝점 포함 여부가 명확하지 않다. +1로 경계 픽셀 유실을 피한다.
    side = max(x2-x1+1, y2-y1+1) + 2*margin
    left = math.floor((x1+x2+1-side)/2)
    top = math.floor((y1+y2+1-side)/2)
    return left, top, left+side, top+side


def crop_patch(pixels: np.ndarray, roi: tuple[int, int, int, int], size: int) -> tuple[Image.Image, float]:
    """비율을 유지한 정사각형 Crop. 경계 밖은 가장자리 픽셀로 채운다."""
    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise ValueError("uint8 grayscale 영상이 필요합니다.")
    left, top, right, bottom = roi
    height, width = pixels.shape
    clipped = (max(left, 0), max(top, 0), min(right, width), min(bottom, height))
    a, b, c, d = clipped
    if c <= a or d <= b:
        raise ValueError("영상과 ROI가 겹치지 않습니다.")
    tile = pixels[b:d, a:c]
    tile = np.pad(tile, ((b-top, bottom-d), (a-left, right-c)), mode="edge")
    padded_fraction = 1 - (c-a)*(d-b)/((right-left)*(bottom-top))
    patch = Image.fromarray(tile).resize((size, size), Image.Resampling.BILINEAR)
    return patch, padded_fraction


def neighbor_count(boxes: np.ndarray, current: int, roi: tuple[int, int, int, int]) -> int:
    """ROI에 조금이라도 들어온 이웃 bbox 수를 보수적으로 기록한다."""
    left, top, right, bottom = roi
    hit = ((boxes[:, 0] < right) & (boxes[:, 2]+1 > left)
           & (boxes[:, 1] < bottom) & (boxes[:, 3]+1 > top))
    hit[current] = False
    return int(hit.sum())


def prepare_patches(project_root: str | Path, config: CropConfig = CropConfig()) -> dict:
    """분할을 먼저 고정한 후 train/val만 생성한다. 기존 다른 설정은 덮어쓰지 않는다."""
    root = Path(project_root).resolve()
    data = root / "data"
    reports = root / "results" / "reports"
    summary = json.loads((reports / "dataset_summary.json").read_text())
    if summary["errors"]:
        raise ValueError("EDA 무결성 오류를 먼저 해결하세요.")
    for name, expected in summary["manifest_sha256"].items():
        if digest(data / name) != expected:
            raise ValueError(f"공식 목록이 EDA 이후 변경되었습니다: {name}")
    sources = pd.read_csv(reports / "images.csv")
    objects = pd.read_csv(reports / "objects.csv")
    if not objects.valid.all() or objects.target_id.isna().any():
        raise ValueError("유효하지 않은 객체가 있습니다.")
    sources["split"] = np.where(sources.official_split == "test", "test",
                                 np.where(sources.group == config.validation_group, "val", "train"))
    assert_separation(sources)
    objects = objects.merge(sources[["source_id", "split"]], on="source_id", validate="many_to_one")
    for split in ["train", "val"]:
        if set(objects.loc[objects.split == split, "class_id"]) != set(CLASSES):
            raise ValueError(f"{split}에 6개 클래스가 모두 필요합니다.")

    split_dir = data / "split"
    processed = data / "processed" / "roi128_v1"
    payload = dict(config=asdict(config),
                   input_sha256={name: digest(reports/name) for name in ["images.csv", "objects.csv"]},
                   official_manifest_sha256=summary["manifest_sha256"],
                   test_policy="reserved_not_cropped",
                   selection_rule="closest to 20% image count among whole trainval groups, before training")
    split_dir.mkdir(parents=True, exist_ok=True)
    lock = split_dir / "preprocessing_config.json"
    if lock.exists() and json.loads(lock.read_text()) != payload:
        raise ValueError("기존 분할/Crop 설정이 다릅니다. 새 버전 디렉터리를 사용하세요.")
    lock.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    sources.to_csv(split_dir / "sources.csv", index=False)
    # test의 메타데이터만 별도 보존하고 test 영상/라벨은 여기서 읽지 않는다.
    sources[sources.split == "test"].to_csv(split_dir / "test_reserved_sources.csv", index=False)
    objects[objects.split == "test"].to_csv(split_dir / "test_reserved_objects.csv", index=False)
    rows = []
    for source in sources[sources.split != "test"].itertuples():
        subset = objects[objects.source_id == source.source_id].reset_index(drop=True)
        label = data / source.label_path
        # 좌표 CSV와 원본 주석을 재비교해 오래된 EDA 결과의 사용을 막는다.
        actual = [list(map(int, line.split())) for line in label.read_text().splitlines() if line.strip()]
        expected = subset[["x1", "y1", "x2", "y2", "class_id"]].values.tolist()
        if actual != expected:
            raise ValueError(f"EDA 이후 주석이 변경되었습니다: {label}")
        with Image.open(data / source.test_path) as image:
            pixels = np.asarray(image.convert("L"))
        current_hash = hashlib.sha256(f"{(pixels.shape[1], pixels.shape[0])}:L:".encode()+pixels.tobytes()).hexdigest()
        if current_hash != source.test_pixel_sha256:
            raise ValueError(f"EDA 이후 영상이 변경되었습니다: {source.source_id}")
        boxes = subset[["x1", "y1", "x2", "y2"]].values
        for index, obj in enumerate(subset.itertuples()):
            roi = square_roi(tuple(boxes[index]), config.margin)
            patch, fraction = crop_patch(pixels, roi, config.size)
            patch_id = f"{source.group}_{Path(source.source_id).name}_{obj.line_number:02d}"
            relative = Path("data/processed/roi128_v1") / source.split / str(obj.target_id) / f"{patch_id}.png"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            patch.save(path)
            rows.append(dict(patch_id=patch_id, source_id=source.source_id, group=source.group,
                             split=source.split, official_split=source.official_split,
                             class_id=obj.class_id, target_id=obj.target_id, class_name=obj.class_name,
                             line_number=obj.line_number, x1=obj.x1, y1=obj.y1, x2=obj.x2, y2=obj.y2,
                             roi_left=roi[0], roi_top=roi[1], roi_right=roi[2], roi_bottom=roi[3],
                             padding_fraction=fraction, neighbor_count=neighbor_count(boxes, index, roi),
                             patch_path=relative.as_posix(), pixel_sha256=hashlib.sha256(patch.tobytes()).hexdigest()))
    patches = pd.DataFrame(rows)
    # Crop를 만든 뒤에도 동일 픽셀 패치가 train/val을 넘나드는지 검사한다.
    overlap = patches.groupby("pixel_sha256").split.nunique().gt(1)
    if overlap.any():
        patches[patches.pixel_sha256.isin(overlap[overlap].index)].to_csv(
            reports / "cross_split_patch_duplicates.csv", index=False)
        raise ValueError("train/val에 동일 픽셀 패치가 있습니다. 해당 보고서를 검토하세요.")
    if patches.patch_id.duplicated().any():
        raise ValueError("patch ID 중복입니다.")
    patches.to_csv(split_dir / "patches.csv", index=False)
    for split in ["train", "val"]:
        patches[patches.split == split].to_csv(split_dir / f"{split}.csv", index=False)
    counts = patches.groupby(["class_id", "class_name", "split"]).size().unstack(fill_value=0)
    counts.to_csv(reports / "preprocessing_class_counts.csv")
    result = dict(config=asdict(config),
                  source_counts={str(k): int(v) for k, v in sources.split.value_counts().items()},
                  patch_counts={str(k): int(v) for k, v in patches.split.value_counts().items()},
                  neighbor_patches=int(patches.neighbor_count.gt(0).sum()),
                  padded_patches=int(patches.padding_fraction.gt(0).sum()),
                  cross_split_exact_patch_duplicates=0,
                  cross_split_group_overlap=0,
                  test_patches_created=0,
                  manifest_sha256=digest(split_dir / "patches.csv"))
    (reports / "preprocessing_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    preprocessing_figure(root, patches)
    print(f"전처리 완료: {result}", flush=True)
    return result


@plt.style.context("default")
def preprocessing_figure(root: Path, patches: pd.DataFrame) -> None:
    """실제 저장된 학습 패치와 패딩/이웃 통계를 확인한다."""
    fig, axes = plt.subplots(6, 4, figsize=(9, 13))
    for class_id in CLASSES:
        examples = patches.query("split == 'train' and class_id == @class_id").sample(4, random_state=42)
        for ax, row in zip(axes[class_id-1], examples.itertuples()):
            with Image.open(root / row.patch_path) as image:
                ax.imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{row.class_name}\nneighbors={row.neighbor_count}, pad={row.padding_fraction:.0%}", fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(root / "results/figures/processed_patches_train.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    prepare_patches(Path(__file__).resolve().parents[1])
