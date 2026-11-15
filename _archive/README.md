# Archive

Historical code kept for provenance, not maintained and not part of the
current pipeline/evaluation path.

- `collecting/` -- the original, pre-pipeline per-source data collection
  scripts (predates `pipeline/stage0_prepare_originals.py`).
- `auroc_modality_decision_v1/` -- an earlier inference + evaluation design
  (AUROC over a `<im_start>`/BOI first-token probability signal as the
  modality-decision proxy, comparing 3 models including Show-o2 as a
  primary evaluated model; `v3.json`-format data) that the project moved
  away from. The released results use free-choice generation + the MSA /
  TP / C-TP methodology in `inference/` and `evaluation/` instead -- not a
  first-token-probability proxy, but the model's actual generated output.
