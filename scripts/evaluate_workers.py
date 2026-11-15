"""Evaluate independently annotated workers without changing model thresholds."""

import argparse
import json
from pathlib import Path

from ppe_compliance.evaluation import evaluate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', default='models/best.pt')
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--exclude-dir', type=Path, action='append', required=True,
                        help='Training and validation images, repeat for each directory')
    args = parser.parse_args()
    try:
        summary = evaluate_manifest(args.manifest, args.output, args.weights, args.image_size,
                                    args.confidence, args.device, args.iou, args.exclude_dir)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f'Evaluation failed: {error}\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
