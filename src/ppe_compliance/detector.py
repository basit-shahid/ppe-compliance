"""Single-image YOLO inference. No PPE compliance claims are made here."""

from pathlib import Path
import hashlib
import math


def validate_inference_options(confidence, image_size):
    """Reject invalid public API settings before importing/loading a model."""
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 < confidence <= 1):
        raise ValueError("Confidence must be a finite number in (0, 1]")
    if type(image_size) is not int or image_size <= 0:
        raise ValueError("Image size must be a positive integer")


def validated_model_detections(candidates, width, height):
    """Reject unusable model outputs before strict downstream association.

    Degenerate boxes can survive model postprocessing, for example when a box
    collapses at an image boundary. Do not swap coordinates, inflate boxes or
    fabricate scores to make them pass. Keep accepted coordinates unchanged.
    """
    accepted, rejected = [], []

    def finite_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    for index, candidate in enumerate(candidates):
        box = candidate.get('bbox_xyxy', [])
        confidence = candidate.get('confidence')
        reason = None
        if len(box) != 4 or not all(finite_number(value) for value in box):
            reason = 'non-finite or malformed bounding box'
        elif box[2] <= box[0] or box[3] <= box[1]:
            reason = 'zero-area or inverted bounding box'
        elif not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            reason = 'bounding box outside source image'
        elif not finite_number(confidence) or not 0 <= confidence <= 1:
            reason = 'invalid detection confidence'
        if reason:
            rejected.append({'model_detection_index': index, 'class_id': candidate.get('class_id'),
                             'class_name': candidate.get('class_name'),
                             'confidence': confidence if finite_number(confidence) else None,
                             'bbox_xyxy': [value if finite_number(value) else None for value in box],
                             'reason': reason})
        else:
            accepted.append({**candidate, 'model_detection_index': index})
    return accepted, {'model_count': len(candidates), 'accepted_count': len(accepted),
                      'rejected_count': len(rejected), 'rejected_detections': rejected,
                      'policy': 'Reject invalid model predictions; preserve all source frames and valid boxes. '
                                'Non-finite diagnostic values are null. Accepted-list indices remain the report detection IDs.'}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrameDetector:
    """Load a model once and reuse it across image/video frames."""

    def __init__(self, weights, confidence=0.25, image_size=640, device="cpu"):
        validate_inference_options(confidence, image_size)
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.weights = str(weights)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.model_classes = dict(self.model.names)
        self.weights_sha256 = file_sha256(Path(weights)) if Path(weights).is_file() else None

    def detect(self, frame, source):
        try:
            result = self.model.predict(source=frame, conf=self.confidence, imgsz=self.image_size,
                                        device=self.device, verbose=False)[0]
        except AssertionError as error:
            # Ultralytics uses assertions for unavailable CUDA/MPS devices.
            raise ValueError(f"Inference settings were rejected on device {self.device!r}. "
                             f"Try device cpu and check the selected weights. Details: {error}") from error
        candidates = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                candidates.append({"class_id": class_id, "class_name": result.names[class_id],
                                   "confidence": float(box.conf.item()),
                                   "bbox_xyxy": [float(value) for value in box.xyxy[0].tolist()]})
        detections, validation = validated_model_detections(candidates, frame.shape[1], frame.shape[0])
        if validation['rejected_count']:
            # Keep result.save()/plot() aligned with the accepted detection report.
            result = result[[item['model_detection_index'] for item in detections]]
        return result, {"source": str(source), "weights": self.weights,
                        "weights_sha256": self.weights_sha256,
                        "image_width": int(frame.shape[1]), "image_height": int(frame.shape[0]),
                        "image_size": self.image_size, "model_classes": self.model_classes,
                        "confidence_threshold": self.confidence, "device": self.device,
                        "detections": detections,
                        "detection_validation": validation,
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
