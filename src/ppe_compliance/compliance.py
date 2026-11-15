"""Worker-level evidence policy; no absence claim from missed detections alone."""

from collections import Counter
from dataclasses import asdict
import math

from ppe_compliance.association import AssociationRules, associate_ppe


def visibility_template(workers, source_sha256):
    return {"source_sha256": source_sha256,
            "instructions": "Human review only: set visibility to true only if that body region is clearly visible. Null means unknown. Do not infer visibility from missing detections.",
            "workers": {w["worker_id"]: {"bbox_xyxy": w["bbox_xyxy"], "head_visible": None,
                                        "torso_visible": None} for w in workers}}


def _validate_visibility(visibility, workers, source_sha256):
    if visibility is None:
        return {}
    if not isinstance(visibility, dict) or visibility.get("source_sha256") != source_sha256:
        raise ValueError("Visibility file must match the exact source image SHA-256")
    entries = visibility.get("workers")
    if not isinstance(entries, dict):
        raise ValueError("Visibility file must contain a workers object")
    known = {w["worker_id"]: w for w in workers}
    if set(entries) - set(known):
        raise ValueError("Visibility file contains unknown worker IDs; regenerate the template")
    for worker_id, entry in entries.items():
        if not isinstance(entry, dict) or set(entry) != {"bbox_xyxy", "head_visible", "torso_visible"}:
            raise ValueError(f"Invalid visibility fields for {worker_id}")
        box = entry["bbox_xyxy"]
        if not isinstance(box, list) or len(box) != 4 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box
        ) or any(abs(a-b) > 2 for a, b in zip(box, known[worker_id]["bbox_xyxy"])):
            raise ValueError(f"Visibility box for {worker_id} does not match current detections; regenerate the template")
        if any(entry[key] is not None and type(entry[key]) is not bool for key in ("head_visible", "torso_visible")):
            raise ValueError(f"Visibility values for {worker_id} must be true, false or null")
    return entries


def assess_detections(report, source_sha256, rules=None, visibility=None):
    """Assess detections without I/O. Optional visibility is explicit human input."""
    rules = rules or AssociationRules()
    names = {name.casefold() for name in report["model_classes"].values()}
    if not {"person", "helmet", "vest"} <= names:
        raise ValueError("Assessment requires model classes person, helmet and vest; use PPE-trained weights")
    workers, unmatched = associate_ppe(report["detections"], report["image_width"], report["image_height"],
                                       report["image_size"], rules)
    entries = _validate_visibility(visibility, workers, source_sha256)
    for worker in workers:
        entry = entries.get(worker["worker_id"], {})
        reasons = list(worker["quality_issues"])
        unconfirmed = []
        for kind, region in (("helmet", "head_visible"), ("vest", "torso_visible")):
            evidence = worker[kind]
            evidence["region_visible"] = entry.get(region)
            state = evidence["state"]
            if entry.get(region) is False:
                reasons.append(f"{kind}: region marked not clearly visible by reviewer")
            if state in {"ambiguous", "uncertain"}:
                reasons.append(f"{kind}: {state} association evidence")
            elif state == "not_detected":
                unconfirmed.append(kind)
                if entry.get(region) is not True:
                    reasons.append(f"{kind}: no match and region visibility unknown")
        if reasons:
            worker["status"] = "review required"
        elif unconfirmed:
            worker["status"] = "potential violation"
            reasons = [f"{kind} not confirmed in a human-reviewed visible region; absence is not proven" for kind in unconfirmed]
        else:
            worker["status"] = "PPE detected"
            reasons = ["Both required items have plausible region matches; wearing state is not proven"]
        worker["reasons"] = reasons
    counts = Counter(w["status"] for w in workers)
    return {"schema_version": 1, "source": report["source"], "source_sha256": source_sha256,
            "weights": report["weights"], "image_width": report["image_width"],
            "weights_sha256": report.get("weights_sha256"), "device": report.get("device"),
            "image_height": report["image_height"], "image_size": report["image_size"],
            "detection_confidence": report["confidence_threshold"], "rules": asdict(rules),
            "visibility_source": "human-reviewed file" if visibility is not None else "not supplied",
            "worker_id_scope": "frame-local, ordered by person bounding box; not tracking IDs",
            "detection_id_scope": "zero-based index into companion detections.json detections list",
            "workers": workers, "unmatched_equipment": unmatched,
            "detection_validation": report.get("detection_validation"),
            "summary": {"workers": len(workers), **{status: counts[status] for status in
                        ("PPE detected", "potential violation", "review required")}},
            "note": "Heuristic visible-PPE assessment, not a safety guarantee. Missing detections do not prove absence. No detected workers does not establish an empty or safe scene."}
