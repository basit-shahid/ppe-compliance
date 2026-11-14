# Experiment evidence

`../experiments.csv` indexes recorded experiments. `baseline-001/` preserves:

- `args.yaml`: original Colab training configuration, including historical paths.
- `results.csv`: original per-epoch training measurements.
- `validation-metrics.json`: separately reported final validation metrics.
- `provenance.json`: hardware, versions, class mapping and input hashes.
- `dataset-audit.json`: preparation audit returned from Colab.
- `environment-colab.txt`: historical installed package snapshot.
- `local-verification.json`: weight/archive fingerprints and CPU integration checks.

These are original compact evidence files, not current machine configuration.
Weights, full exports and routine images remain in ignored local artifact folders.
Interpretation and caveats are in [the training review](../phase-3-results.md).
