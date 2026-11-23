"""DeepPCB 1단계 감사와 EDA. 원본은 읽기만 하며 이미지 한 쌍씩 처리한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image


# 공식 ID 순서를 고정한다. 문서의 나열 순서와 혼동해 copper와 pin-hole을 바꾸면 안 된다.
CLASSES = {
    1: ("Open Circuit", "회로 단선"),
    2: ("Short Circuit", "회로 단락"),
    3: ("Mousebite", "패턴 가장자리 결손"),
    4: ("Spur", "구리 돌출"),
    5: ("Spurious Copper", "잔여 구리"),
    6: ("Pin Hole", "패턴 내부 구멍"),
}
COLORS = ["#e76f51", "#f4a261", "#2a9d8f", "#457b9d", "#9b5de5", "#e63986"]
OBJECT_COLUMNS = [
    "source_id", "group", "official_split", "label_path", "line_number",
    "x1", "y1", "x2", "y2", "class_id", "class_name", "target_id",
    "bbox_width", "bbox_height", "bbox_area", "valid",
]


@dataclass
class Audit:
    root: Path
    images: pd.DataFrame
    objects: pd.DataFrame
    issues: pd.DataFrame
    summary: dict
    pixel_histograms: dict[str, np.ndarray]


def _safe_path(root: Path, relative: str) -> Path:
    """목록의 경로가 데이터 루트 밖을 가리키면 거부한다."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"데이터 루트 외부 경로: {relative}")
    return path


def audit_dataset(data_root: str | Path) -> Audit:
    """모든 split의 수량/무결성만 감사하고 상세 EDA는 trainval에서 수행한다."""
    root = Path(data_root).resolve()
    image_rows, object_rows, issues = [], [], []
    hist = {role: np.zeros(256, dtype=np.int64) for role in ("test", "temp")}
    seen_sources: dict[str, str] = {}
    seen_labels: set[str] = set()
    manifest_hashes = {}

    def problem(kind: str, source: str, detail: str, severity: str = "error") -> None:
        issues.append(dict(severity=severity, kind=kind, source_id=source, detail=detail))

    for split, filename in (("trainval", "trainval.txt"), ("test", "test.txt")):
        manifest = root / filename
        if not manifest.is_file():
            raise FileNotFoundError(f"공식 분할 목록이 없습니다: {manifest}")
        manifest_hashes[filename] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        for line_no, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2:
                problem("manifest_format", filename, f"{line_no}행: 경로 2개 필요")
                continue
            image_ref, label_ref = fields
            try:
                base = _safe_path(root, image_ref)
                label = _safe_path(root, label_ref)
            except ValueError as exc:
                problem("unsafe_path", filename, str(exc))
                continue
            source = base.relative_to(root).with_suffix("").as_posix()
            if source in seen_sources:
                problem("duplicate_source", source, f"{seen_sources[source]} 및 {split}")
                continue
            seen_sources[source] = split
            if label_ref in seen_labels:
                problem("reused_label", source, label_ref)
            seen_labels.add(label_ref)
            group = Path(image_ref).parts[0]
            row = dict(source_id=source, group=group, official_split=split, label_path=label_ref)

            # 목록에는 00041000.jpg, 디스크에는 00041000_test/temp.jpg가 존재한다.
            for role in ("test", "temp"):
                path = base.with_name(f"{base.stem}_{role}{base.suffix}")
                row[f"{role}_path"] = path.relative_to(root).as_posix()
                try:
                    with Image.open(path) as image:
                        image.load()  # 파일 헤더뿐 아니라 실제 디코딩도 확인한다.
                        row[f"{role}_width"], row[f"{role}_height"] = image.size
                        row[f"{role}_mode"] = image.mode
                        if split == "trainval" and image.mode == "RGB":
                            rgb = np.asarray(image).astype(np.int16)
                            row[f"{role}_rgb_channel_spread"] = int(
                                (rgb.max(axis=2) - rgb.min(axis=2)).max())
                        pixels = np.asarray(image.convert("L"))
                        row[f"{role}_pixel_sha256"] = hashlib.sha256(
                            f"{image.size}:L:".encode() + pixels.tobytes()
                        ).hexdigest()
                        # 테스트 영상은 무결성만 검사하며 픽셀 통계에 합산하지 않는다.
                        if split == "trainval":
                            hist[role] += np.bincount(pixels.ravel(), minlength=256)
                except (OSError, ValueError) as exc:
                    problem("image_read", source, f"{role}: {exc}")
            if row.get("test_width") is not None and row.get("temp_width") is not None:
                if (row["test_width"], row["test_height"]) != (row["temp_width"], row["temp_height"]):
                    problem("pair_size", source, "검사/기준 영상 해상도 불일치")

            n_objects = 0
            seen_boxes = set()
            try:
                label_lines = label.read_text(encoding="utf-8-sig").splitlines()
            except OSError as exc:
                problem("label_read", source, str(exc))
                label_lines = []
            for number, annotation in enumerate(label_lines, 1):
                if not annotation.strip():
                    continue
                n_objects += 1
                try:
                    values = tuple(map(int, annotation.split()))
                    if len(values) != 5:
                        raise ValueError("정수 5개 필요")
                except ValueError as exc:
                    problem("label_format", source, f"{number}행: {exc}")
                    continue
                x1, y1, x2, y2, class_id = values
                valid = True
                if values in seen_boxes:
                    valid = False
                    problem("duplicate_annotation", source, f"{number}행: {values}")
                seen_boxes.add(values)
                if class_id not in CLASSES:
                    valid = False
                    problem("class_id", source, f"{number}행: {class_id}")
                if not (0 <= x1 < x2 <= row.get("test_width", -1)
                        and 0 <= y1 < y2 <= row.get("test_height", -1)):
                    valid = False
                    problem("bbox_bounds", source, f"{number}행: {values}")
                object_rows.append(dict(
                    source_id=source, group=group, official_split=split,
                    label_path=label_ref, line_number=number,
                    x1=x1, y1=y1, x2=x2, y2=y2, class_id=class_id,
                    class_name=CLASSES.get(class_id, ("UNKNOWN", ""))[0], target_id=class_id - 1,
                    bbox_width=x2-x1, bbox_height=y2-y1, bbox_area=(x2-x1)*(y2-y1), valid=valid,
                ))
            row["annotation_lines"] = n_objects
            if n_objects == 0:
                problem("empty_label", source, "불량 객체가 없는 라벨", "warning")
            image_rows.append(row)
        print(f"{split}: 목록 확인 완료", flush=True)

    images = pd.DataFrame(image_rows)
    objects = pd.DataFrame(object_rows, columns=OBJECT_COLUMNS)
    if images.empty:
        raise ValueError("읽을 수 있는 이미지 쌍이 없습니다.")
    listed_images = set(images["test_path"]) | set(images["temp_path"])
    # 실제 디스크 파일과 공식 목록을 대조해 누락/추가 파일을 놓치지 않는다.
    disk_images = {p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() and p.relative_to(root).parts[0] not in {"processed", "split"}
                   and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}}
    disk_labels = {p.relative_to(root).as_posix() for p in root.rglob("*.txt")
                   if p.name not in {"trainval.txt", "test.txt"}
                   and p.relative_to(root).parts[0] not in {"processed", "split"}}
    for path in sorted(disk_images - listed_images):
        problem("unlisted_image", path, "공식 목록에 없는 이미지", "warning")
    for path in sorted(disk_labels - seen_labels):
        problem("unlisted_label", path, "공식 목록에 없는 라벨", "warning")

    shared_groups = sorted(set(images.loc[images.official_split == "trainval", "group"])
                           & set(images.loc[images.official_split == "test", "group"]))
    duplicate_summary = {}
    for role in ("test", "temp"):
        key = f"{role}_pixel_sha256"
        if key not in images:
            continue
        hashes = images.dropna(subset=[key]).groupby(key)
        mixed = hashes.official_split.nunique()
        cross = mixed[mixed > 1].index
        duplicate_summary[role] = dict(
            repeated_hash_groups=int((hashes.size() > 1).sum()),
            cross_split_hash_groups=len(cross),
            cross_split_images=int(images[key].isin(cross).sum()),
        )
        # 일치가 있으면 분할 위험을 기록하되 템플릿 반복과 검사 영상 반복을 구분한다.
        if len(cross):
            problem(f"cross_split_{role}_pixels", "official_split",
                    f"동일 L 픽셀 해시 {len(cross)}개; 상세 CSV 확인", "warning")

    issues_frame = pd.DataFrame(issues, columns=["severity", "kind", "source_id", "detail"])
    valid_objects = objects.loc[objects.valid]
    counts = valid_objects.groupby(["official_split", "class_id"]).size()
    summary = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), data_root=str(root),
        manifest_sha256=manifest_hashes,
        physical_image_files=len(disk_images), physical_annotation_files=len(disk_labels),
        listed_image_files=len(listed_images), unlisted_images=sorted(disk_images-listed_images),
        unlisted_labels=sorted(disk_labels-seen_labels),
        manifest_pairs=len(images), parsed_objects=len(objects), valid_objects=len(valid_objects),
        annotation_lines=int(images.annotation_lines.sum()),
        invalid_objects=int((~objects.valid).sum()),
        split_pairs={str(k): int(v) for k, v in images.official_split.value_counts().items()},
        split_objects={str(k): int(v) for k, v in valid_objects.official_split.value_counts().items()},
        class_counts={s: {str(c): int(counts.get((s, c), 0)) for c in CLASSES}
                      for s in ("trainval", "test")},
        group_count=int(images.group.nunique()), shared_official_groups=shared_groups,
        pixel_duplicates=duplicate_summary,
        image_shapes={role: {str(k): int(v) for k, v in images.groupby(
            [f"{role}_width", f"{role}_height", f"{role}_mode"]).size().items()}
                      for role in ("test", "temp") if f"{role}_width" in images},
        errors=int((issues_frame.severity == "error").sum()),
        warnings=int((issues_frame.severity == "warning").sum()),
        test_policy="counts_and_integrity_only; all detailed EDA uses official trainval",
        bbox_convention="EDA extent=x2-x1,y2-y1; inclusive endpoint not asserted",
        class_mapping={str(k): dict(name=v[0], korean=v[1], target_id=k-1) for k, v in CLASSES.items()},
    )
    return Audit(root, images, objects, issues_frame, summary, hist)


def environment_snapshot() -> dict:
    """TensorFlow를 import하지 않고 설치 상태를 기록한다. GPU 실행 검증과 구분한다."""
    versions = {}
    for name in ("numpy", "pandas", "matplotlib", "Pillow", "ipykernel", "tensorflow", "tensorflow-metal"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, executable=sys.executable, machine=platform.machine(),
                platform=platform.platform(), packages=versions, gpu_runtime_checked_by_eda=False,
                gpu_verification_reports=["hardware_probe.json", "metal_workaround.json"])


def save_tables(audit: Audit, output_root: str | Path) -> None:
    """재현 가능한 목록, 문제 행, 집계표를 UTF-8 CSV/JSON으로 저장한다."""
    reports = Path(output_root) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    audit.images.to_csv(reports / "images.csv", index=False)
    audit.objects.to_csv(reports / "objects.csv", index=False)
    audit.issues.to_csv(reports / "integrity_issues.csv", index=False)
    (reports / "dataset_summary.json").write_text(
        json.dumps(audit.summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports / "environment.json").write_text(
        json.dumps(environment_snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
    counts = pd.DataFrame([
        dict(class_id=c, target_id=c-1, class_name=CLASSES[c][0], korean=CLASSES[c][1],
             trainval=audit.summary["class_counts"]["trainval"][str(c)],
             test=audit.summary["class_counts"]["test"][str(c)]) for c in CLASSES
    ])
    counts["total"] = counts.trainval + counts.test
    counts.to_csv(reports / "class_counts.csv", index=False)
    audit.images.groupby(["group", "official_split"]).agg(
        image_pairs=("source_id", "size"), objects=("annotation_lines", "sum")
    ).to_csv(reports / "group_split_counts.csv")
    train = audit.objects.loc[audit.objects.valid & (audit.objects.official_split == "trainval")]
    train.groupby("class_name")[["bbox_width", "bbox_height", "bbox_area"]].describe().to_csv(
        reports / "bbox_statistics_trainval.csv")
    train.groupby(["group", "class_name"]).size().unstack(fill_value=0).to_csv(
        reports / "group_class_counts_trainval.csv")
    for role in ("test", "temp"):
        key = f"{role}_pixel_sha256"
        if key in audit.images:
            duplicates = audit.images[audit.images[key].notna() & audit.images[key].duplicated(keep=False)]
            duplicates[["source_id", "official_split", "group", key]].to_csv(
                reports / f"duplicate_pixels_{role}.csv", index=False)
    pd.DataFrame({"pixel_value": np.arange(256), **audit.pixel_histograms}).to_csv(
        reports / "pixel_histogram_trainval.csv", index=False)


def _gray(audit: Audit, relative: str) -> np.ndarray:
    with Image.open(audit.root / relative) as image:
        return np.asarray(image.convert("L"))


def _box(ax, obj, offset=(0, 0), label=True) -> None:
    color = COLORS[int(obj.class_id)-1]
    x, y = obj.x1-offset[0], obj.y1-offset[1]
    ax.add_patch(Rectangle((x, y), obj.bbox_width, obj.bbox_height,
                           fill=False, edgecolor=color, linewidth=1.4))
    if label:
        ax.text(x, max(0, y-4), f"{obj.class_id}: {obj.class_name}", fontsize=6, color=color,
                bbox=dict(facecolor="black", alpha=.7, edgecolor="none", pad=1))


@plt.style.context("default")
def make_figures(audit: Audit, output_root: str | Path, seed: int = 42) -> list[Path]:
    """전체 클래스 개수 외의 그래프와 예시는 trainval로만 만든다."""
    if audit.summary["errors"]:
        raise ValueError("무결성 오류가 있습니다. integrity_issues.csv를 먼저 확인하세요.")
    figures = Path(output_root) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    saved = []

    def save(fig, name):
        path = figures / name
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    objects = audit.objects.query("valid and official_split == 'trainval'")
    images = audit.images.query("official_split == 'trainval'").set_index("source_id")
    names = [CLASSES[c][0] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(6)
    for shift, split, color in ((-.2, "trainval", "#2a9d8f"), (.2, "test", "#457b9d")):
        values = [audit.summary["class_counts"][split][str(c)] for c in CLASSES]
        bars = ax.bar(x+shift, values, .4, label=f"Official {split} (inventory)", color=color)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylabel("Defect objects")
    ax.set_title("Class inventory — counts only; test is reserved for final evaluation")
    ax.legend()
    save(fig, "class_distribution.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(axes, ["bbox_width", "bbox_height", "bbox_area"],
                              ["Bounding-box width", "Bounding-box height", "Bounding-box area"]):
        ax.hist(objects[col], bins=40, color="#2a9d8f")
        ax.set_title(title + " (trainval)")
        ax.set_xlabel("pixels" if col != "bbox_area" else "pixels squared")
        ax.set_ylabel("Objects")
    save(fig, "bbox_sizes_trainval.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for role, label in (("temp", "Template"), ("test", "Defective")):
        h = audit.pixel_histograms[role]
        axes[0].plot(np.arange(256), h/h.sum(), label=label)
    axes[0].set(title="Pixel histogram (trainval)", xlabel="Grayscale value", ylabel="Pixel fraction")
    axes[0].set_yscale("log")
    axes[0].legend()
    frequencies = objects.groupby("source_id").size().reindex(images.index, fill_value=0).value_counts().sort_index()
    axes[1].bar(frequencies.index, frequencies.values, color="#457b9d")
    axes[1].set(title="Defects per image (trainval)", xlabel="Objects in image", ylabel="Images")
    save(fig, "pixels_and_objects_trainval.png")

    group_classes = objects.groupby(["group", "class_id"]).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    heat = ax.imshow(group_classes.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(6), names, rotation=20, ha="right")
    ax.set_yticks(range(len(group_classes)), group_classes.index)
    ax.set_title("Group / class inventory (trainval)")
    for i in range(len(group_classes)):
        for j in range(6):
            ax.text(j, i, str(group_classes.iloc[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if group_classes.iloc[i, j] > group_classes.values.max()*.55 else "black")
    fig.colorbar(heat, ax=ax, label="Objects")
    save(fig, "group_class_distribution_trainval.png")

    # 각 클래스에서 고정 seed로 대표 객체를 고르고 16px 여백을 보여준다. 학습 패치는 저장하지 않는다.
    fig, axes = plt.subplots(6, 3, figsize=(10, 18))
    for i, class_id in enumerate(CLASSES):
        subset = objects[objects.class_id == class_id]
        examples = subset.sample(n=min(3, len(subset)), random_state=seed)
        for ax in axes[i]:
            ax.axis("off")
        for ax, obj in zip(axes[i], examples.itertuples()):
            row = images.loc[obj.source_id]
            pixels = _gray(audit, row.test_path)
            left, top = max(0, obj.x1-16), max(0, obj.y1-16)
            right, bottom = min(pixels.shape[1], obj.x2+16), min(pixels.shape[0], obj.y2+16)
            ax.imshow(pixels[top:bottom, left:right], cmap="gray", vmin=0, vmax=255)
            _box(ax, obj, offset=(left, top), label=False)
            ax.set_title(f"{class_id}: {obj.class_name}\n{Path(obj.source_id).name} / line {obj.line_number}", fontsize=8)
    save(fig, "class_examples_trainval.png")

    # 클래스가 다양한 trainval 이미지 중 사전순 첫 파일을 택해 선택 기준을 재현한다.
    source = objects.groupby("source_id").class_id.nunique().sort_values(ascending=False, kind="stable").index[0]
    row = images.loc[source]
    template, defective = _gray(audit, row.temp_path), _gray(audit, row.test_path)
    difference = np.abs(defective.astype(np.int16) - template.astype(np.int16))
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, pixels, title in zip(axes, [template, defective, difference],
                                 ["Defect-free template", "Defective + annotations", "Absolute difference (diagnostic)"]):
        ax.imshow(pixels, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    for obj in objects[objects.source_id == source].itertuples():
        _box(axes[1], obj)
    fig.suptitle(f"Official trainval: {source}", fontsize=10)
    save(fig, "template_defect_annotations_trainval.png")
    return saved


def write_report(audit: Audit, output_root: str | Path) -> Path:
    """측정한 값만 보고서로 내보낸다. 모델 성능이나 분할 독립성을 가정하지 않는다."""
    s = audit.summary
    train = audit.objects.query("valid and official_split == 'trainval'")
    counts = [s["class_counts"]["trainval"][str(c)] for c in CLASSES]
    nonzero = [n for n in counts if n]
    ratio = max(nonzero)/min(nonzero) if nonzero else float("nan")
    rows = ["| ID | 유형 | trainval 객체 | test 객체 (수량 감사) | 전체 |",
            "|---:|---|---:|---:|---:|"]
    for c, (name, kr) in CLASSES.items():
        a, b = s["class_counts"]["trainval"][str(c)], s["class_counts"]["test"][str(c)]
        rows.append(f"| {c} | {name} ({kr}) | {a:,} | {b:,} | {a+b:,} |")
    sizes = train[["bbox_width", "bbox_height", "bbox_area"]].quantile([0, .25, .5, .75, .95, 1])
    size_text = sizes.to_string(float_format=lambda x: f"{x:.1f}")
    per_image = train.groupby("source_id").size()
    extra = ', '.join(f"`{p}`" for p in s['unlisted_images']) or '없음'
    h = audit.pixel_histograms['test']
    extreme_fraction = (int(h[:16].sum()) + int(h[240:].sum())) / max(int(h.sum()), 1)
    train_images = audit.images.query("official_split == 'trainval'")
    rgb_n = int((train_images.test_mode == 'RGB').sum())
    spread_key = 'test_rgb_channel_spread'
    spread = int(train_images[spread_key].max()) if spread_key in train_images and rgb_n else 0
    groups = train.groupby(['group', 'class_id']).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
    complete_groups = list(groups.index[(groups > 0).all(axis=1)])
    candidate_counts = train_images.groupby('group').size().reindex(complete_groups)
    closest = (candidate_counts/len(train_images)-.2).abs().sort_values().index[:2]
    candidate_text = ', '.join(f"`{group}` ({candidate_counts[group]}쌍)" for group in closest)
    text = f"""# DeepPCB 데이터 확보 및 EDA 결과

생성 시각(UTC): {s['created_at_utc']}

## 1. 실제 파일과 라벨

- 데이터 루트: `{audit.root}`. 다운로드는 수행하지 않았고 기존 파일을 사용했다.
- 디스크 이미지 **{s['physical_image_files']:,}개**, 주석 파일 **{s['physical_annotation_files']:,}개**.
- 목록의 이미지 쌍 **{s['manifest_pairs']:,}개** = 검사 영상 + 정상 기준 영상.
- 목록에 연결된 실제 이미지 **{s['listed_image_files']:,}개**. 목록 외 이미지: {extra}.
  목록 외 파일은 자동 포함/삭제하지 않고 학습 후보에서 제외한다.
- 공식 trainval **{s['split_pairs'].get('trainval', 0):,}쌍**, test **{s['split_pairs'].get('test', 0):,}쌍**.
- 라벨 비어 있지 않은 행 **{s['annotation_lines']:,}개**, 파싱 객체 **{s['parsed_objects']:,}개**, 유효 객체 **{s['valid_objects']:,}개**.
- 오류 **{s['errors']}건**, 경고 **{s['warnings']}건**. 세부 내역은 `integrity_issues.csv` 참조.
- 검사 영상 해상도/모드: `{s['image_shapes'].get('test')}`.
- 기준 영상 해상도/모드: `{s['image_shapes'].get('temp')}`.
- trainval 검사 영상 중 RGB 저장 형식은 **{rgb_n}개**이며 이 파일의 최대 채널 간 차이는
  **{spread}**이다. 파일 모드가 혼재하므로 로딩 시 grayscale 변환 규칙을 명시한다.

```text
data/
  trainval.txt / test.txt
  group00041/
    00041/00041000_test.jpg   # 불량 검사 영상
    00041/00041000_temp.jpg   # 정렬된 정상 기준 영상
    00041_not/00041000.txt    # 검사 영상 안의 모든 객체
  ...
```

목록 한 행은 이미지 참조 경로와 라벨 경로 2개다. 참조 `00041000.jpg`는
실제 `_test.jpg`와 `_temp.jpg`에 연결해야 한다. 주석 한 행은 공백으로 구분된
`x1 y1 x2 y2 class_id`이며 YOLO 정규화 좌표 형식이 아니다.
원본 ID `1~6`은 모델 학습 시 `0~5`로 변환하되 매핑을 모델과 함께 저장해야 한다.

## 2. 클래스 분포

{chr(10).join(rows)}

trainval의 최대/최소 클래스 수 비율: **{ratio:.3f}**.
이 비율과 클래스별 검증 Recall을 바탕으로 이후 class weight 필요성을 판단한다.
공식 test의 라벨은 수량/무결성 감사에만 읽었으며, 예시 이미지·크기·픽셀 분석 및
하이퍼파라미터 선택에는 사용하지 않았다.

## 3. 학습 후보(trainval)의 객체 크기

아래는 `x2-x1`, `y2-y1` 기준 좌표 범위 통계다. 원본 주석의 끝점 포함 여부는
아직 확정하지 않았으며 실제 Crop 규약은 다음 단계에서 고정한다.

```text
quantile       width    height      area
{size_text}
```

이미지별 객체 수: 최소 {per_image.min() if len(per_image) else 0},
중앙값 {per_image.median() if len(per_image) else 0}, 최대 {per_image.max() if len(per_image) else 0}.
픽셀값 분포는 `pixel_histogram_trainval.csv` 및 그래프로 저장했다. 이미 이진화에 가까운
영상이므로 밝기 증강은 일반 사진과 다르게 소량만 검증하고 과도한 재이진화는 피한다.
trainval 검사 영상에서 0~15 또는 240~255 범위인 픽셀은 **{extreme_fraction:.2%}**다.

## 4. 분할과 누수 위험

- 디렉터리 그룹 **{s['group_count']}개**. 공식 trainval/test 공통 그룹 **{len(s['shared_official_groups'])}개**:
  `{', '.join(s['shared_official_groups'])}`.
- L 변환 픽셀 해시 감사: `{s['pixel_duplicates']}`.
- 상세한 그룹별 수량은 `group_split_counts.csv`, 중복 픽셀 목록은 `duplicate_pixels_*.csv`에 저장했다.
- 동일 검사 이미지의 patch를 무작위 분할하면 누수가 발생한다. **원본 이미지 단위로 먼저 분할**해야 한다.
- 디렉터리 그룹은 촬영/기판 그룹의 후보이며 실제 기판 식별자라는 보장은 없다.
  공식 test는 그대로 보존하되 공통 그룹이 있으면 '미관측 기판 일반화' 성능으로 해석할 수 없다.
- trainval 내부에서 그룹 전체를 검증용으로 홀드아웃하는 방식을 우선 검토한다.
  모든 클래스가 포함되는지와 검증 규모를 trainval만으로 확인한 뒤 seed/그룹 목록을 고정한다.
- trainval에서 6종 객체가 모두 관측된 그룹: `{', '.join(complete_groups)}`.
  trainval의 20%에 가장 가까운 검증용 그룹 후보는 {candidate_text}이며,
  최종 그룹은 템플릿 유사도와 클래스 분포를 확인한 후 선택한다. 아직 분할하지 않았다.
- 정확한 픽셀 해시는 유사/인접 타일의 중복을 증명하거나 배제하지 않는다.
  다음 단계에서 trainval의 그룹/템플릿 유사도를 조사하고 필요 시 연결 그룹을 함께 분할한다.

## 5. 다음 단계 전처리 계획 (미실행)

1. 무결성 오류가 있으면 원본을 자동 수정하지 말고 원인과 처리 방침부터 기록한다.
2. trainval 원본에 train/validation을 먼저 배정하고 공식 test를 잠근 manifest를 저장한다.
3. 각 객체 중심 주변에 여백을 둔 정사각 ROI를 만든다. 우선 16px 또는 bbox 비례 여백을
   trainval 예시로 비교하고, 영상 경계와 주변 불량이 함께 포함되는 비율을 조사한다.
   이웃 불량이 들어오면 주 불량 ID와 포함 여부를 남겨 단일 라벨 모호성을 관리한다.
4. 좌표 규약을 명시하고 영상 밖 여백은 padding한다. 종횡비를 보존해 128×128로 맞춘다.
   PNG 등 무손실로 저장하며 source_id, group, split, class_id, 좌표, 여백 설정을 함께 저장한다.
5. Model A/B는 단일 grayscale 채널을 우선 사용한다. 0~1 정규화는 로딩 시 수행하고
   데이터셋 전체를 float32 배열로 메모리에 올리지 않는다. tf.data에서 batch 16으로 시작한다.
6. 증강은 train만 적용한다. 90도 회전/좌우/상하 반전부터 시작하고 미세 회전/밝기/확대는
   불량이 잘리거나 보간으로 사라지지 않는지 확인한다. validation/test는 변형하지 않는다.
7. A/B 구조와 검증 성능을 먼저 비교한다. Model C는 grayscale을 RGB로 복제하고
   선택한 백본의 preprocess_input 규약을 사용해 이중 정규화를 피한다.
8. 분할과 모델 선택은 validation Macro F1로 수행하고 공식 test는 최종 확정 후 한 번 평가한다.
   Macro F1 0.90은 목표이며 현재 측정한 성능은 없다.

이 보고서는 1단계 EDA 범위만 다룬다. 후속 전처리/학습 실행 상태는
`preprocessing_summary.json`과 모델별 학습 보고서에서 확인한다.
여기서 제시한 대표 이미지의 Crop은 시각화 전용이며 학습 데이터 파일이 아니다.

## 6. 환경

사용자 결정에 따라 기존 Python **{platform.python_version()}** 환경을 유지했다.
실제 설치 버전과 아키텍처는 `environment.json`에 기록했다. EDA 함수에서는
GPU 런타임을 검증하지 않는다. 학습 단계의 실제 검증은 별도 보고서를 확인한다.
Apple Silicon의 TensorFlow 가속은 `tensorflow-metal`
PluggableDevice를 사용하며 PyTorch의 `mps` 설정과 구분한다.
학습 단계에서 패키지 호환성을 확인하고 작은 tensor 연산 및 GPU 열거를 실제 실행해야 한다.

## 출처

- [DeepPCB 공식 README: 구조·ID·생성 방식](https://github.com/tangsanli5201/DeepPCB)
- [TensorFlow 설치 및 Python 지원](https://www.tensorflow.org/install/pip)
- [Apple tensorflow-metal 설치/제한](https://developer.apple.com/metal/tensorflow-plugin/)

DeepPCB에는 인공적으로 추가한 불량이 포함된다. 본 Crop 분류 결과를 실제 생산 데이터의
전수 검사 성능이나 원본 영상의 자동 탐지 성능으로 해석할 수 없다. 공식 문서의 연구 목적
사용 안내도 후속 활용 전에 확인한다.
"""
    path = Path(output_root) / "reports" / "eda_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    audit = audit_dataset(args.data_root)
    save_tables(audit, args.output_root)
    report = write_report(audit, args.output_root)
    if audit.summary["errors"]:
        raise SystemExit(f"감사 실패: {audit.summary['errors']}개 오류. {report} 참고")
    paths = make_figures(audit, args.output_root)
    print(f"완료: {report}; 그래프 {len(paths)}개")


if __name__ == "__main__":
    main()
