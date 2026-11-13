"""Single-image YOLO inference. No PPE compliance claims are made here."""

from pathlib import Path


def detect_image(
    source: Path,
    weights: str,
    confidence: float = 0.25,
    image_size: int = 640,
    device: str = "cpu",
) -> tuple[object, dict]:
    """Return an annotated-result object and JSON-compatible detections.

    YOLO is imported lazily so CLI help does not require loading PyTorch.
    Class names come from the weights, never from hard-coded PPE class IDs.
    """
    import cv2
    from ultralytics import YOLO

    frame = cv2.imread(str(source))
    if frame is None:
        raise ValueError(f"Cannot decode image: {source}")
    model = YOLO(weights)
    result = model.predict(
        source=frame, conf=confidence, imgsz=image_size, device=device, verbose=False
    )[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": float(box.conf.item()),
                "bbox_xyxy": [float(value) for value in box.xyxy[0].tolist()],
            })
    return result, {
        "source": str(source.resolve()),
        "weights": weights,
        "image_width": int(frame.shape[1]),
        "image_height": int(frame.shape[0]),
        "confidence_threshold": confidence,
        "device": device,
        "detections": detections,
        "note": "Object detections only. General pretrained weights are not a PPE compliance model.",
    }
