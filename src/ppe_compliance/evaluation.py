"""Reproducible worker-level evaluation, including missed and spurious workers."""

from collections import Counter
import json
import math
from pathlib import Path
from time import perf_counter

from ppe_compliance.association import iou
from ppe_compliance.detector import file_sha256

STATUSES = ('PPE detected', 'review required', 'potential violation')


def score_workers(expected, predicted, threshold=.5):
    """Greedy descending-IoU one-to-one matching, independent of status labels."""
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0, 1]')
    for worker in [*expected, *predicted]:
        box = worker.get('bbox_xyxy', [])
        if (worker.get('status') not in STATUSES or len(box) != 4 or
                not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in box) or
                not 0 <= box[0] < box[2] or not 0 <= box[1] < box[3]):
            raise ValueError('Each worker needs a valid bbox_xyxy and a recognized status')
    pairs = sorted(((iou(a['bbox_xyxy'], b['bbox_xyxy']), ai, bi)
                    for ai, a in enumerate(expected) for bi, b in enumerate(predicted)), reverse=True)
    used_a, used_b, outcomes = set(), set(), []
    for overlap, ai, bi in pairs:
        if overlap < threshold:
            break
        if ai not in used_a and bi not in used_b:
            used_a.add(ai)
            used_b.add(bi)
            outcomes.append({'expected': expected[ai]['status'], 'predicted': predicted[bi]['status'],
                             'iou': overlap, 'expected_index': ai, 'predicted_index': bi})
    outcomes.extend({'expected': a['status'], 'predicted': 'missed', 'expected_index': ai}
                    for ai, a in enumerate(expected) if ai not in used_a)
    outcomes.extend({'expected': 'spurious', 'predicted': b['status'], 'predicted_index': bi}
                    for bi, b in enumerate(predicted) if bi not in used_b)
    return outcomes


def aggregate_scores(rows):
    outcomes = [item for row in rows for item in row['outcomes']]
    truth = sum(o['expected'] != 'spurious' for o in outcomes)
    predictions = sum(o['predicted'] != 'missed' for o in outcomes)
    matched = [o for o in outcomes if o['expected'] != 'spurious' and o['predicted'] != 'missed']
    correct = sum(o['expected'] == o['predicted'] for o in matched)
    review = sum(o['predicted'] == 'review required' for o in outcomes)
    confusion = {}
    for o in outcomes:
        confusion.setdefault(o['expected'], Counter())[o['predicted']] += 1
    metrics = {}
    for status in STATUSES:
        tp = sum(o['expected'] == status and o['predicted'] == status for o in outcomes)
        fp = sum(o['expected'] != status and o['predicted'] == status for o in outcomes)
        fn = sum(o['expected'] == status and o['predicted'] != status for o in outcomes)
        precision, recall = tp/(tp+fp) if tp+fp else None, tp/(tp+fn) if tp+fn else None
        metrics[status] = {'precision': precision, 'recall': recall,
                           'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                           'support': tp+fn}
    return {'images': len(rows), 'expected_workers': truth, 'predicted_workers': predictions,
            'matched_workers': len(matched), 'missed_workers': truth-len(matched),
            'spurious_workers': predictions-len(matched),
            'worker_detection_recall': len(matched)/truth if truth else None,
            'worker_detection_precision': len(matched)/predictions if predictions else None,
            'status_accuracy_on_matches': correct/len(matched) if matched else None,
            'correct_status_per_expected_worker': correct/truth if truth else None,
            'review_rate': review/predictions if predictions else None,
            'per_status': metrics, 'confusion': confusion}


def load_manifest(path, exclude_dirs=()):
    """Verify frozen pixels, human provenance and exact train/validation overlap.

    Hash checks cannot establish scene independence; the declaration is supplied
    by the dataset reviewer. No training or validation data is used for tuning.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('independent') is not True or not data.get('dataset_id') or not data.get('annotation_notes'):
        raise ValueError('Manifest needs dataset_id, annotation_notes and independent=true from its reviewer')
    if not isinstance(data.get('items'), list) or not data['items']:
        raise ValueError('Manifest items must be a nonempty list')
    excluded = set()
    for folder in exclude_dirs:
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f'Exclusion directory does not exist: {folder}')
        excluded.update(file_sha256(p) for p in folder.rglob('*')
                        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'})
    seen = set()
    for item in data['items']:
        if not item.get('source_group') or not isinstance(item.get('workers'), list):
            raise ValueError('Each item needs source_group and workers (empty list for a background)')
        image = (path.parent/item['image']).resolve()
        digest = file_sha256(image)
        if item.get('sha256') != digest:
            raise ValueError(f'Frozen image hash mismatch: {image}')
        if digest in seen or digest in excluded:
            raise ValueError(f'Duplicate evaluation image or exact training/validation overlap: {image}')
        seen.add(digest)
        score_workers(item['workers'], [])
        item['resolved_image'] = str(image)
    return data


def evaluate_manifest(manifest, output, weights, image_size=640, confidence=.25,
                      device='cpu', threshold=.5, exclude_dirs=()):
    import cv2
    from ppe_compliance.detector import FrameDetector
    from ppe_compliance.compliance import assess_detections
    from ppe_compliance.rendering import draw_assessment

    data = load_manifest(manifest, exclude_dirs)
    output = Path(output)
    if output.exists():
        raise ValueError('Evaluation output already exists; choose a new directory')
    # Validate before loading model or creating output artifacts.
    score_workers([], [], threshold)
    detector = FrameDetector(weights, confidence, image_size, device)
    output.mkdir(parents=True, exist_ok=False)
    (output/'manifest.json').write_bytes(Path(manifest).read_bytes())
    rows = []
    started = perf_counter()
    for index, item in enumerate(data['items']):
        frame = cv2.imread(item['resolved_image'])
        if frame is None:
            raise ValueError(f"Cannot decode {item['resolved_image']}")
        height, width = frame.shape[:2]
        if any(w['bbox_xyxy'][2] > width or w['bbox_xyxy'][3] > height for w in item['workers']):
            raise ValueError('Ground truth box is outside its image')
        before = perf_counter()
        _, raw = detector.detect(frame, item['resolved_image'])
        assessment = assess_detections(raw, item['sha256'])
        elapsed = perf_counter()-before
        outcomes = score_workers(item['workers'], assessment['workers'], threshold)
        row = {'image': item['image'], 'source_group': item['source_group'], 'sha256': item['sha256'],
               'assessment_seconds': elapsed, 'outcomes': outcomes}
        rows.append(row)
        (output/f'{index:05d}.json').write_text(json.dumps({'detections': raw, 'assessment': assessment, **row}, indent=2), encoding='utf-8')
        if not cv2.imwrite(str(output/f'{index:05d}.png'), draw_assessment(frame, assessment, raw['detections'])):
            raise OSError('Could not write evaluation preview')
    seconds = perf_counter()-started
    summary = {'schema_version': 1, 'dataset_id': data['dataset_id'], 'metrics': aggregate_scores(rows),
               'by_source_group': {group: aggregate_scores([r for r in rows if r['source_group'] == group])
                                   for group in sorted({r['source_group'] for r in rows})},
               'weights_sha256': detector.weights_sha256, 'manifest_sha256': file_sha256(Path(manifest)),
               'image_size': image_size, 'confidence': confidence, 'device': device,
               'iou_threshold': threshold, 'wall_seconds_excluding_model_load': seconds,
               'images_per_second_including_exports': len(rows)/seconds if seconds else None,
               'excluded_directories': [str(Path(p).resolve()) for p in exclude_dirs],
               'note': 'Greedy descending-IoU matching. Misses count against recall. Review is an abstention. No manual visibility fed to predictions. Dataset independence is reviewer-declared; hashes detect exact duplicates only. These metrics are worker outcomes, not detection mAP.'}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
