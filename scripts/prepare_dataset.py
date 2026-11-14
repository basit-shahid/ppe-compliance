"""Audit an extracted Construction-PPE source and prepare three YOLO classes.

Run from the repository root. Raw data is never modified. Reusing an existing
destination is refused so prior prepared data cannot be silently overwritten.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import yaml


TARGETS = {"person": 0, "helmet": 1, "vest": 2}
NAMES = {value: key for key, value in TARGETS.items()}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("data/processed/ppe-three-class"))
    parser.add_argument("--exclusions", type=Path, help="JSON mapping split/filename to reviewed exclusion reason")
    args = parser.parse_args()
    exclusions = json.loads(args.exclusions.read_text(encoding="utf-8")) if args.exclusions else {}
    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        parser.error(f"Destination already exists: {destination}")
    configs = list(source.rglob("*.yaml")) + list(source.rglob("*.yml"))
    candidates = []
    for config in configs:
        data = yaml.safe_load(config.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "names" in data:
            candidates.append((config, data))
    if len(candidates) != 1:
        parser.error(f"Expected one source YAML containing names, found {len(candidates)}")
    config, metadata = candidates[0]
    names = metadata["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    mapping = {key: TARGETS[value.lower()] for key, value in names.items() if value.lower() in TARGETS}
    if set(mapping.values()) != set(NAMES):
        parser.error(f"Required classes missing: {names}")
    root = config.parent
    if not (root / "images").is_dir():
        parser.error(f"Expected images/ beside {config}")
    records, issues, raw_counts = [], [], {}
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for split in ("val", "test", "train"):
        image_dir = root / "images" / split
        if not image_dir.is_dir():
            continue
        images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in extensions)
        counts = Counter()
        for image_path in images:
            relative = image_path.relative_to(image_dir)
            label_path = (root / "labels" / split / relative).with_suffix(".txt")
            image = cv2.imread(str(image_path))
            errors, labels = [], []
            if image is None:
                errors.append("unreadable image")
            if not label_path.is_file():
                errors.append("missing label file")
            else:
                for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        values = list(map(float, line.split()))
                        if len(values) != 5 or not all(np.isfinite(values)):
                            raise ValueError("expected five finite values")
                        cls, x, y, width, height = values
                        if not cls.is_integer() or int(cls) not in names:
                            raise ValueError("unknown class ID")
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                            raise ValueError("invalid normalized coordinates")
                        if min(x-width/2, y-height/2) < -1e-5 or max(x+width/2, y+height/2) > 1+1e-5:
                            raise ValueError("box extends beyond image")
                        counts[names[int(cls)]] += 1
                        if int(cls) in mapping:
                            labels.append((mapping[int(cls)], x, y, width, height))
                    except ValueError as error:
                        errors.append(f"line {line_number}: {error}")
            if errors:
                issues.append({"image": str(image_path.relative_to(root)), "errors": errors})
                continue
            height, width = image.shape[:2]
            gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
            dhash = int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), "big")
            records.append({"split": split, "relative": relative.as_posix(), "width": width,
                            "height": height, "sha256": digest(image_path.read_bytes()),
                            "pixels_sha256": digest(str(image.shape).encode() + image.tobytes()),
                            "dhash": dhash, "labels": sorted(set(labels)), "source": image_path})
        raw_counts[split] = {"images": len(images), "instances": dict(counts)}
    destination.mkdir(parents=True)
    audit = destination / "audit"
    audit.mkdir()
    seen, kept, duplicates = {}, [], []
    for record in records:
        fingerprint = record["pixels_sha256"]
        if fingerprint in seen:
            first = seen[fingerprint]
            duplicates.append({"removed": f"{record['split']}/{record['relative']}",
                               "kept": f"{first['split']}/{first['relative']}",
                               "cross_split": record["split"] != first["split"],
                               "labels_agree": record["labels"] == first["labels"]})
            continue
        seen[fingerprint] = record
        kept.append(record)
    near = []
    for i, left in enumerate(kept):
        for right in kept[i+1:]:
            if left["split"] == right["split"]:
                continue
            distance = (left["dhash"] ^ right["dhash"]).bit_count()
            if distance <= 4:
                near.append({"left": f"{left['split']}/{left['relative']}",
                             "right": f"{right['split']}/{right['relative']}", "distance": distance})
    known_paths = {f"{r['split']}/{r['relative']}" for r in kept}
    if set(exclusions) - known_paths:
        raise ValueError(f"Unknown reviewed exclusion paths: {set(exclusions) - known_paths}")
    kept = [r for r in kept if f"{r['split']}/{r['relative']}" not in exclusions]
    prepared = {}
    for record in kept:
        split = record["split"]
        target_image = destination / "images" / split / record["relative"]
        target_label = (destination / "labels" / split / record["relative"]).with_suffix(".txt")
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["source"], target_image)
        text = "".join(f"{label[0]} " + " ".join(f"{v:.8f}" for v in label[1:]) + "\n" for label in record["labels"])
        target_label.write_text(text, encoding="utf-8")
        stats = prepared.setdefault(split, {"images": 0, "empty_target_labels": 0, "instances": Counter()})
        stats["images"] += 1
        stats["empty_target_labels"] += not record["labels"]
        stats["instances"].update(NAMES[label[0]] for label in record["labels"])
    report = {"source_config": str(config), "source_names": names, "class_mapping": mapping,
              "reviewed_exclusions": exclusions,
              "raw_counts": raw_counts, "prepared_counts": prepared, "invalid_images": issues,
              "exact_pixel_duplicates_removed": duplicates, "cross_split_near_duplicate_candidates": near,
              "near_duplicate_method": "64-bit difference hash, Hamming distance <=4; candidates require review",
              "scene_leakage_status": "not established by hashes; human review required",
              "test_available": "test" in prepared}
    (audit / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (audit / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["split", "relative", "width", "height", "sha256", "pixels_sha256", "dhash"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    for split in prepared:
        samples = [r for r in kept if r["split"] == split]
        indices = np.linspace(0, len(samples)-1, min(12, len(samples)), dtype=int)
        sheet = np.full((3 * 260, 4 * 340, 3), 245, dtype=np.uint8)
        for tile, index in enumerate(indices):
            record = samples[index]
            image = cv2.imread(str(record["source"]))
            h, w = image.shape[:2]
            for cls, x, y, bw, bh in record["labels"]:
                p1 = (int((x-bw/2)*w), int((y-bh/2)*h))
                p2 = (int((x+bw/2)*w), int((y+bh/2)*h))
                color = [(220, 100, 0), (0, 180, 0), (0, 100, 240)][cls]
                cv2.rectangle(image, p1, p2, color, max(1, w//500))
                cv2.putText(image, NAMES[cls], (p1[0], max(12, p1[1])), cv2.FONT_HERSHEY_SIMPLEX, max(.4, w/1800), color, max(1, w//700))
            scale = min(330/w, 220/h)
            thumb = cv2.resize(image, (round(w*scale), round(h*scale)))
            top, left = (tile//4)*260, (tile%4)*340
            sheet[top:top+thumb.shape[0], left:left+thumb.shape[1]] = thumb
            cv2.putText(sheet, f"{split} sample {int(index)}", (left+3, top+245), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,0,0), 1)
        cv2.imwrite(str(audit / f"{split}-contact-sheet.jpg"), sheet)
    dataset = {"path": destination.as_posix(), "train": "images/train", "val": "images/val", "names": NAMES}
    if "test" in prepared:
        dataset["test"] = "images/test"
    (destination / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"invalid_images", "exact_pixel_duplicates_removed", "cross_split_near_duplicate_candidates"}}, indent=2))
    print(f"Invalid images: {len(issues)}; removed duplicates: {len(duplicates)}; near-duplicate candidates: {len(near)}")


if __name__ == "__main__":
    main()
