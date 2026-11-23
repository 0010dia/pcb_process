"""기존 그룹 분할을 보존해 정상/불량 단일 패치 데이터를 별도로 만든다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.preprocessing import assert_separation, crop_patch, digest, square_roi


def overlaps_defect(roi, boxes: np.ndarray, guard: int = 8) -> bool:
    """주석 bbox를 guard만큼 확장해 정상 후보에 불량 경계가 들어오는 것을 막는다."""
    left, top, right, bottom = roi
    return bool(((boxes[:, 0]-guard < right) & (boxes[:, 2]+1+guard > left)
                 & (boxes[:, 1]-guard < bottom) & (boxes[:, 3]+1+guard > top)).any())


def normal_roi(pixels: np.ndarray, boxes: np.ndarray, side: int,
               rng: np.random.Generator, guard: int = 8):
    """같은 검사 영상에서 불량과 겹치지 않는 회로 영역을 뽑는다."""
    height, width = pixels.shape
    if side > min(height, width):
        return None
    for _ in range(200):
        left, top = int(rng.integers(0, width-side+1)), int(rng.integers(0, height-side+1))
        roi = (left, top, left+side, top+side)
        if overlaps_defect(roi, boxes, guard):
            continue
        # 빈 배경만 정상으로 학습하지 않도록 흑/백 패턴이 함께 있는 후보를 사용한다.
        black_fraction = float((pixels[top:top+side, left:left+side] < 128).mean())
        if .05 <= black_fraction <= .95:
            return roi
    return None


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """라벨 충돌·분할 간 동일 픽셀은 격리하고 분할 내부 반복은 한 개만 유지한다."""
    result = frame.copy()
    result["exclusion_reason"] = ""
    conflicts = result.groupby("pixel_sha256").target.nunique().gt(1)
    cross_split = result.groupby("pixel_sha256").split.nunique().gt(1)
    result.loc[result.pixel_sha256.isin(conflicts[conflicts].index), "exclusion_reason"] = "identical_pixels_conflicting_labels"
    result.loc[result.pixel_sha256.isin(cross_split[cross_split].index)
               & result.exclusion_reason.eq(""), "exclusion_reason"] = "identical_pixels_across_splits"
    eligible = result.exclusion_reason.eq("")
    duplicates = result.loc[eligible].duplicated(["split", "pixel_sha256"], keep="first")
    result.loc[duplicates[duplicates].index, "exclusion_reason"] = "repeated_pixels_within_split"
    return (result[result.exclusion_reason.eq("")].drop(columns="exclusion_reason").reset_index(drop=True),
            result[result.exclusion_reason.ne("")].reset_index(drop=True))


def assert_binary_split(frame: pd.DataFrame) -> None:
    """한 원본의 정상/불량/템플릿 패치가 분할을 넘지 않도록 검증한다."""
    if set(frame.split) != {"train", "val"} or not frame.official_split.eq("trainval").all():
        raise ValueError("이진 학습 데이터는 공식 trainval의 train/val만 허용합니다.")
    for key in ["source_id", "group", "pixel_sha256"]:
        if frame.groupby(key).split.nunique().gt(1).any():
            raise ValueError(f"이진 데이터 분할 간 {key} 중복이 있습니다.")
    if frame.patch_id.duplicated().any():
        raise ValueError("patch_id 중복입니다.")
    for split in ["train", "val"]:
        if set(frame.loc[frame.split == split, "target"]) != {0, 1}:
            raise ValueError(f"{split}에 정상/불량 두 클래스가 필요합니다.")


def prepare_binary_data(project_root: str | Path) -> dict:
    """원본과 6종 분류 산출물은 변경하지 않고 binary_v1에만 저장한다."""
    root = Path(project_root).resolve()
    split_dir = root / "data/split/binary_v1"
    reports = root / "results/binary"
    split_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    config = dict(version="binary_v1", size=128, margin=16, guard=8, seed=42,
                  negative_sources=["template_matched", "tested_clear"],
                  normal_sampling_attempts=200, tested_normal_black_fraction=[.05, .95],
                  dedup_policy="quarantine conflicting/cross-split hashes; keep one within split",
                  input_sha256={name: digest(root / "data/split" / name) for name in ["sources.csv", "patches.csv"]},
                  object_table_sha256=digest(root / "results/reports/objects.csv"),
                  existing_crop_config_sha256=digest(root / "data/split/preprocessing_config.json"),
                  test_policy="reserved; not read or cropped")
    lock = split_dir / "config.json"
    if lock.exists() and json.loads(lock.read_text()) != config:
        raise ValueError("binary_v1 설정이 변경됐습니다. 새 버전 이름을 사용하세요.")
    sources = pd.read_csv(root / "data/split/sources.csv")
    assert_separation(sources)
    objects = pd.read_csv(root / "results/reports/objects.csv")
    if not objects.valid.all():
        raise ValueError("유효하지 않은 주석이 있습니다.")
    crop_cfg = json.loads((root / "data/split/preprocessing_config.json").read_text())["config"]
    if crop_cfg["size"] != 128 or crop_cfg["margin"] != 16:
        raise ValueError("기존 Crop 규약과 다릅니다.")
    lock.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    rows, pair_differences, sampling_failures = [], [], []
    rng = np.random.default_rng(42)
    for source in sources[sources.split.isin(["train", "val"])].itertuples():
        subset = objects[objects.source_id == source.source_id]
        expected_labels = subset[["x1", "y1", "x2", "y2", "class_id"]].values.tolist()
        actual_labels = [list(map(int, line.split())) for line in (root / "data" / source.label_path).read_text().splitlines() if line.strip()]
        if actual_labels != expected_labels:
            raise ValueError(f"원본 주석이 변경되었습니다: {source.source_id}")
        pixels = {}
        for role in ["test", "temp"]:
            with Image.open(root / "data" / getattr(source, f"{role}_path")) as image:
                pixels[role] = np.asarray(image.convert("L"))
            checksum = hashlib.sha256(f"{(pixels[role].shape[1], pixels[role].shape[0])}:L:".encode()+pixels[role].tobytes()).hexdigest()
            if checksum != getattr(source, f"{role}_pixel_sha256"):
                raise ValueError(f"원본 {role} 영상이 변경되었습니다: {source.source_id}")
        boxes = subset[["x1", "y1", "x2", "y2"]].values
        for obj in subset.itertuples():
            anchor = f"{source.group}_{Path(source.source_id).name}_{obj.line_number:02d}"
            roi = square_roi((obj.x1, obj.y1, obj.x2, obj.y2), 16)
            template, _ = crop_patch(pixels["temp"], roi, 128)
            defective, _ = crop_patch(pixels["test"], roi, 128)
            pair_differences.append(dict(anchor_id=anchor, split=source.split,
                mean_absolute_difference=float(np.abs(np.asarray(template).astype(float)-np.asarray(defective)).mean())))
            clear = normal_roi(pixels["test"], boxes, roi[2]-roi[0], rng)
            candidates = [("defect", 1, "test", roi), ("template_matched", 0, "temp", roi)]
            if clear is not None:
                candidates.append(("tested_clear", 0, "test", clear))
            else:
                sampling_failures.append(dict(anchor_id=anchor, source_id=source.source_id, reason="no_clear_pattern_roi_in_200_attempts"))
            for kind, target, role, region in candidates:
                patch, padding = crop_patch(pixels[role], region, 128)
                patch_id = f"{anchor}_{kind}"
                relative = Path("data/processed/binary_v1") / source.split / str(target) / f"{patch_id}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                patch.save(path)
                rows.append(dict(patch_id=patch_id, anchor_id=anchor, source_id=source.source_id,
                    group=source.group, split=source.split, official_split=source.official_split,
                    target=target, label="defect" if target else "normal", kind=kind,
                    defect_class_id=int(obj.class_id) if target else 0,
                    anchor_defect_class_id=int(obj.class_id), source_role=role,
                    roi_left=region[0], roi_top=region[1], roi_right=region[2], roi_bottom=region[3],
                    padding_fraction=padding, patch_path=relative.as_posix(),
                    pixel_sha256=hashlib.sha256(patch.tobytes()).hexdigest()))
    all_candidates = pd.DataFrame(rows)
    frame, excluded = deduplicate(all_candidates)
    assert_binary_split(frame)
    frame.to_csv(split_dir / "patches.csv", index=False)
    for split in ["train", "val"]:
        frame[frame.split == split].to_csv(split_dir / f"{split}.csv", index=False)
    excluded.to_csv(reports / "excluded_patches.csv", index=False)
    pd.DataFrame(sampling_failures, columns=["anchor_id", "source_id", "reason"]).to_csv(reports / "sampling_failures.csv", index=False)
    pd.DataFrame(pair_differences).to_csv(reports / "matched_pair_differences.csv", index=False)
    counts = frame.groupby(["split", "kind", "target"]).size().rename("patches").reset_index()
    counts.to_csv(reports / "data_counts.csv", index=False)
    summary = dict(candidates=len(all_candidates), included=len(frame), excluded=len(excluded),
                   exclusions={str(k): int(v) for k, v in excluded.exclusion_reason.value_counts().items()},
                   counts=counts.to_dict(orient="records"), sampling_failures=len(sampling_failures),
                   train_sources=int(frame.loc[frame.split == "train", "source_id"].nunique()),
                   val_sources=int(frame.loc[frame.split == "val", "source_id"].nunique()),
                   test_patches_created=0, cross_split_group_overlap=0, cross_split_pixel_overlap=0,
                   manifest_sha256=digest(split_dir / "patches.csv"))
    (reports / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    binary_examples(root, frame, reports / "data_examples.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


@plt.style.context("default")
def binary_examples(root: Path, frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 6, figsize=(12, 7))
    for i, kind in enumerate(["template_matched", "tested_clear", "defect"]):
        candidates = frame.query("split == 'train' and kind == @kind")
        selected = candidates.sample(min(6, len(candidates)), random_state=42)
        for ax in axes[i]:
            ax.axis("off")
        for ax, row in zip(axes[i], selected.itertuples()):
            with Image.open(root / row.patch_path) as image:
                ax.imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{row.label}\n{kind}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    prepare_binary_data(Path(__file__).resolve().parents[1])
