# Stage 0 sub-pipelines

Reproduction scripts/artifacts for the source-specific reworks that
`stage0_prepare_originals.py` assembles (it reads their frozen outputs from
`data/stage0_inputs/` at runtime, not these scripts).

## A) `mmlu_openstyle/` -- fully reproducible

`sample_mmlu_openstyle_700.py` pulls `Open-Style/Open-LLM-Benchmark` and
`cais/mmlu` from the HF Hub only, no API key or local data needed.

## B) `scienceqa_filtering/` -- partially reproducible

Visual-grounding filter (GPT-5.2, 3x without-image, see root
`pipeline/README.md`). Needs `OPENAI_API_KEY` + ScienceQA images (not
bundled here -- `derek-thomas/ScienceQA` on HF, or `/nas/datahub/ScienceQA/`).
The script that built the frozen `gpt_audit_full.json` (2,017-item pool)
wasn't preserved; treat that file as the source of truth. Output:
`blind_test_results_full.json`, 653 survived.

## C) `vqav2_filtering/` -- reproducible given the official VQAv2 files

Chain: `prepare_pool.py` (random 2,400) → `blind_test_full.py` (same
GPT-5.2 filter as B) → `finalize_survivors.py` (1,541 survived, downloads
COCO images) → `select_final_991.py` (991 final, `SEED=42`). Needs
`OPENAI_API_KEY` + official VQAv2 questions/annotations
(<https://visualqa.org/download.html>).

## D) `scienceqa_openstyle/` -- the MC-option-free rewrite actually used

`normalize_scienceqa_409()` keeps the source question's multiple-choice
options in `prompt` -- a surface-form crutch this benchmark otherwise
controls for. The released dataset uses the option-free rewrite frozen in
this folder instead (applied to `data/stage3_2variant_final.json` as a
post-process). The scripts that produced it weren't preserved; **only a
60/409-item pilot went through full Stage-3-equivalent LLM validation** --
the rest were spot-checked by hand at integration time, which
`stage3_2variant_final.json`'s `checks[].reason` notes per item.

## What feeds `stage0_prepare_originals.py`

`data/stage0_inputs/`: `mmlu_openstyle_700.json` (A), `scienceqa_final_409.json`
(B), `vqav2_final_ti2t_991.json` (C), plus `comi_bench_full_refined.json`
(the 5 untouched sources, from an earlier pipeline not included here).
Re-run A/B/C only to regenerate those inputs from scratch.
