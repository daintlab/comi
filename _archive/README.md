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
  TP+ / COMI-score methodology in `inference/` and `evaluation/` instead --
  not a first-token-probability proxy, but the model's actual generated
  output.
- `assets/` -- `comi_pipeline.png` (predates Stage 5; shows Stage 3's
  4-5-variant output as the final dataset, not the released 2-variant one)
  and `dataset_composition.png` (unlabeled, was never referenced anywhere).
