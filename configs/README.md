# Configuration

- `dataset.example.yaml`: template only. IDs and paths must match an audited dataset.
- `dataset-exclusions.json`: reviewed source-image exclusions used in preparation.
- `training.example.yaml`: baseline training example; performs real training.
- `compliance.example.yaml`: validated association heuristics for `ppe assess`;
  not calibrated accuracy guarantees.
- `dataset.yaml`: generated local configuration, intentionally ignored by Git.

After preparing the dataset, copy its generated `dataset.yaml` here. It contains
an absolute root specific to that machine. The Colab notebook uses its own
generated configuration with Colab paths, rather than the Windows file.
