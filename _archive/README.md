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
- `stage2_pipeline_combined/` -- `stage2_pipeline.py`, the original combined
  Stage 2a+2b+3 script, superseded once those were split into
  `stage2a_fix.py` / `stage2b_generate.py` / `stage3_validate.py`. Not used
  for the released dataset.
- `stage4_finalize_unused/` -- `stage4_finalize.py` and the two decision
  files it reads (`drop_decisions.json`, `qa_decisions.json`): an earlier,
  separate human-curation pass over Stage 3's output that predates Stage 5
  and was not used to produce the released dataset. See `pipeline/README.md`
  ("Human verification") for what it actually did and why Stage 5's own
  review (kept live, in `pipeline/decisions/`) is the one that matters.
- `old_docs/` -- design docs for two earlier, now-superseded pipeline/
  inference designs, neither matching the current `pipeline/` or
  `inference/`:
  - `PIPELINE.mermaid.md` describes an even earlier "aug_bench_v2" pipeline
    naming (`1_check_expected` / `2_refine` / `3_qc_sampling` / `4_evaluate`,
    scripts like `expected_5.2.py` / `refine.py` / `score_variants.py`) that
    predates the current `stage0`-`stage5` design entirely.
  - `anole_run_inputs_single_load.md` / `tar_run_inputs_single_load.md`
    document the same `boi_prob` / `im_start_prob` first-token-probability
    scripts archived in `auroc_modality_decision_v1/` above (under an
    earlier name, `run_inputs_single_load.py`, at this-machine-specific
    paths that no longer apply).
