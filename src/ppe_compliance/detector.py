"""Single-image YOLO inference. No PPE compliance claims are made here."""

from pathlib import Path
import hashlib


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrameDetector:
    """Load a model once and reuse it across image/video frames."""

    def __init__(self, weights, confidence=0.25, image_size=640, device="cpu"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.weights = str(weights)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.model_classes = dict(self.model.names)
        self.weights_sha256 = file_sha256(Path(weights)) if Path(weights).is_file() else None

    def detect(self, frame, source):
        result = self.model.predict(source=frame, conf=self.confidence, imgsz=self.image_size,
                                    device=self.device, verbose=False)[0]
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append({"class_id": class_id, "class_name": result.names[class_id],
                                   "confidence": float(box.conf.item()),
                                   "bbox_xyxy": [float(value) for value in box.xyxy[0].tolist()]})
        return result, {"source": str(source), "weights": self.weights,
                        "weights_sha256": self.weights_sha256,
                        "image_width": int(frame.shape[1]), "image_height": int(frame.shape[0]),
                        "image_size": self.image_size, "model_classes": self.model_classes,
                        "confidence_threshold": self.confidence, "device": self.device,
                        "detections": detections,
                        "note": "Object detections only. Use ppe assess with PPE-trained weights for worker-level evidence; this output contains no compliance decision."}


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

    frame = cv2.imread(str(source))
    if frame is None:
        raise ValueError(f"Cannot decode image: {source}")
    detector = FrameDetector(weights, confidence, image_size, device)
    return detector.detect(frame, str(source.resolve()))
