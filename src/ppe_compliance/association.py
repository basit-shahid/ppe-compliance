"""Conservative, deterministic geometry for candidate PPE ownership.

Region matches are evidence of plausible placement, not proof of wearing state.
Coordinates are original-image pixels; scale gates use model-input pixels.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class AssociationRules:
    min_confidence: float = 0.5
    min_person_height: float = 80.0
    min_person_width: float = 20.0
    min_equipment_size: float = 5.0
    min_region_coverage: float = 0.65
    ownership_margin: float = 0.15
    person_overlap_iou: float = 0.35
    border_margin: float = 2.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            if key in {"min_confidence", "min_region_coverage", "person_overlap_iou"}:
                if not 0 < value <= 1:
                    raise ValueError(f"{key} must be in (0, 1]")
            elif key == "ownership_margin":
                if not 0 <= value <= 1:
                    raise ValueError("ownership_margin must be in [0, 1]")
            elif value < 0:
                raise ValueError(f"{key} must be nonnegative")


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou(a, b):
    overlap = intersection(a, b)
    union = area(a) + area(b) - overlap
    return overlap / union if union else 0.0


def _candidate_score(person, equipment, kind, rules):
    left, top, right, bottom = person
    width, height = right - left, bottom - top
    x1, y1, x2, y2 = equipment
    cx, cy = (x1+x2)/2, (y1+y2)/2
    if kind == "helmet":
        region = [left-.05*width, top-.08*height, right+.05*width, top+.32*height]
        if y2-y1 > .35*height or x2-x1 > .85*width:
            return None
    else:
        region = [left-.02*width, top+.16*height, right+.02*width, top+.72*height]
        if y2-y1 > .75*height or x2-x1 > 1.1*width:
            return None
    if not (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
        return None
    coverage = intersection(region, equipment) / area(equipment)
    if coverage < rules.min_region_coverage:
        return None
    horizontal = max(0.0, 1-abs(cx-(left+right)/2)/(width/2))
    return .8*coverage + .2*horizontal


def associate_ppe(detections, image_width, image_height, image_size=640, rules=None):
    """Return candidate worker records, unmatched equipment, and all detected IDs."""
    rules = rules or AssociationRules()
    if min(image_width, image_height, image_size) <= 0:
        raise ValueError("Image dimensions and image_size must be positive")
    items = []
    for index, detection in enumerate(detections):
        box = detection["bbox_xyxy"]
        confidence = detection["confidence"]
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or area(box) <= 0:
            raise ValueError(f"Invalid bounding box for detection {index}")
        if not (0 <= box[0] < box[2] <= image_width and 0 <= box[1] < box[3] <= image_height):
            raise ValueError(f"Out-of-image bounding box for detection {index}")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"Invalid confidence for detection {index}")
        items.append({**detection, "detection_id": index, "class_name": detection["class_name"].casefold()})
    people = sorted((d for d in items if d["class_name"] == "person"),
                    key=lambda d: (*d["bbox_xyxy"], d["detection_id"]))
    scale = image_size / max(image_width, image_height)
    workers = []
    for index, person in enumerate(people):
        box = person["bbox_xyxy"]
        problems = []
        if person["confidence"] < rules.min_confidence:
            problems.append("low person confidence")
        if (box[3]-box[1])*scale < rules.min_person_height or (box[2]-box[0])*scale < rules.min_person_width:
            problems.append("person too small at model input scale")
        margin = rules.border_margin / scale
        if min(box[0], box[1], image_width-box[2], image_height-box[3]) <= margin:
            problems.append("person touches image border; visibility uncertain")
        if any(other is not person and iou(box, other["bbox_xyxy"]) >= rules.person_overlap_iou for other in people):
            problems.append("overlapping workers; occlusion risk")
        workers.append({"worker_id": f"W{index+1}", "person_detection_id": person["detection_id"],
                        "bbox_xyxy": list(box), "person_confidence": person["confidence"],
                        "quality_issues": problems,
                        "helmet": {"state": "not_detected", "detection_id": None},
                        "vest": {"state": "not_detected", "detection_id": None}})
    matches, uncertain, ambiguous = {}, set(), set()
    unmatched = []
    for equipment in (d for d in items if d["class_name"] in {"helmet", "vest"}):
        kind = equipment["class_name"]
        candidates = []
        for index, person in enumerate(people):
            score = _candidate_score(person["bbox_xyxy"], equipment["bbox_xyxy"], kind, rules)
            if score is not None:
                candidates.append((score, index))
        candidates.sort(key=lambda pair: (-pair[0], pair[1]))
        reason = None
        box = equipment["bbox_xyxy"]
        weak = equipment["confidence"] < rules.min_confidence or min(box[2]-box[0], box[3]-box[1])*scale < rules.min_equipment_size
        if not candidates:
            reason = "no plausible head/torso region"
        elif len(candidates) > 1 and candidates[0][0]-candidates[1][0] <= rules.ownership_margin:
            reason = "ambiguous ownership"
            ambiguous.update((index, kind) for score, index in candidates
                             if candidates[0][0]-score <= rules.ownership_margin)
        elif weak:
            reason = "low-confidence or small equipment"
            uncertain.add((candidates[0][1], kind))
        else:
            score, index = candidates[0]
            matches.setdefault((index, kind), []).append((score, equipment["confidence"], equipment["detection_id"]))
        if reason:
            unmatched.append({"detection_id": equipment["detection_id"], "class_name": kind, "reason": reason})
    for index, worker in enumerate(workers):
        for kind in ("helmet", "vest"):
            key = (index, kind)
            candidates = sorted(matches.get(key, []), reverse=True)
            if key in ambiguous:
                worker[kind]["state"] = "ambiguous"
                reason = "worker has competing PPE ownership"
            elif candidates:
                score, confidence, detection_id = candidates[0]
                worker[kind] = {"state": "matched", "detection_id": detection_id,
                                "confidence": confidence, "association_score": round(score, 6)}
                reason = "additional candidate for the same worker/item"
                candidates = candidates[1:]
            elif key in uncertain:
                worker[kind]["state"] = "uncertain"
                reason = None
            else:
                reason = None
            if reason:
                for _, _, detection_id in candidates:
                    unmatched.append({"detection_id": detection_id, "class_name": kind, "reason": reason})
    return workers, sorted(unmatched, key=lambda item: item["detection_id"])
