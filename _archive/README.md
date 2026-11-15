# Archive

Historical code kept for provenance, not maintained and not part of the
current pipeline/evaluation path.

- `collecting/` -- the original, pre-pipeline per-source data collection
  scripts (predates `pipeline/stage0_prepare_originals.py`).
- `evaluation_auroc_v1/` -- an earlier evaluation design (AUROC over a
  `<im_start>`/BOI first-token probability signal, comparing 3 models
  including Show-o2 as a primary evaluated model) that the project moved
  away from. The released results use the MSA / TP / C-TP methodology in
  `evaluation/` and `docs/Experiments.md` instead.
