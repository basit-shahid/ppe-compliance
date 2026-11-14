# Project documentation

## Start here

- [Project checkpoint](project-status.md): current milestone and next steps.
- [Comprehensive report (PDF)](PPE_Project_Comprehensive_Report.pdf): full project,
  proposal alignment, all phases, acceptance gates and suggested team roles.
- [Report source](project-report.md): editable Markdown used to generate the PDF.
- [Phase plan](plan.md): current gates and original relative schedule.

## Completed work

- [Phase 1 verification](phase-1.md): environment, baseline and error checks.
- [Dataset audit](dataset-audit.md): preparation, actual mapping, counts and limitations.
- [Dataset attribution](dataset-attribution.md): source, licence and modifications.
- [Phase 3 results](phase-3-results.md): training review and local model checks.
- [Experiment log](experiments.csv) and [baseline evidence](experiments/baseline-001/).

## Reproduction

- [Colab training guide](phase-3-colab.md).
- [Scripts](../scripts/README.md) for data preparation, notebook packaging and report generation.
- Environment snapshots: [Windows CPU](environment-windows.txt),
  [historical Linux CPU](environment-cpu.txt),
  [Colab GPU](experiments/baseline-001/environment-colab.txt).

Snapshots and original experiment configs can contain historical paths. They are
evidence, not portable installation lockfiles. Large data, weights, exported ZIPs
and routine generated outputs stay in ignored local artifact directories.
Reviewed report/README figures are in [assets](assets/README.md).
