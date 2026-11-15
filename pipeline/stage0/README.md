# Stage 0 sub-pipelines (A/B/C)

Reproduction scripts for the three source-specific reworks that
`stage0_prepare_originals.py` assembles. Not all of them are runnable from a
fresh clone — this file says exactly what is and isn't.

## A) `mmlu_openstyle/sample_mmlu_openstyle_700.py` — fully reproducible

Pulls `Open-Style/Open-LLM-Benchmark` and `cais/mmlu` from the HF Hub only.
No API key, no local raw data. Run as-is:

```bash
cd mmlu_openstyle && python sample_mmlu_openstyle_700.py
```

## B) `scienceqa_filtering/` — partially reproducible

- `blind_test_full.py` needs `gpt_audit_full.json` (the 2,017-item pool, kept
  in this folder) and each item's `image_path` pointing at a real ScienceQA
  image. **The images themselves are not in this repo** (thousands of files).
  Re-download the test split with images from HF (`derek-thomas/ScienceQA`,
  filter `image is not None`) and point `image_path` at wherever you save
  them, or ask for a copy of `/nas/datahub/ScienceQA/`.
- The step that built `gpt_audit_full.json` from the raw ScienceQA test split
  was never saved as a script — only its output survived. Reconstructing it
  is straightforward (HF `derek-thomas/ScienceQA`, test split, `image is not
  None` → 2,017 rows) but hasn't been re-verified byte-for-byte against the
  frozen file, so treat the frozen file as the source of truth for now.
- `run_gpt_audit_v5_full.py` derives its 653-item input by filtering
  `blind_test_results_full.json` for `survived == True` (this used to be a
  separate file, `gpt_audit_survived_653.json`, that was never persisted —
  the filter is now inlined in `main()`). Same image-path caveat as above.
- Needs `OPENAI_API_KEY` (see root `pipeline/README.md`) and `gpt-5.2` access.

Frozen intermediate artifacts kept in this folder so the two scripts can at
least be inspected/re-run against them: `gpt_audit_full.json` (2,017),
`blind_test_results_full.json` (Step 1 output, 653 survived).

## C) `vqav2_filtering/` — reproducible given the official VQAv2 files

All four scripts (`prepare_pool.py`, `blind_test_full.py`,
`finalize_survivors.py`, `select_final_991.py`) take paths via CLI args /
env vars now — no hardcoded machine-specific paths left. What's missing is
the raw data:

- Official VQAv2 val2014 `questions`/`annotations` JSON — download from
  <https://visualqa.org/download.html> and point `prepare_pool.py` at them
  via `VQAV2_QUESTIONS_PATH` / `VQAV2_ANNOTATIONS_PATH` (defaults to a
  `vqav2_official/` folder next to the script).
- COCO val2014 images — `finalize_survivors.py` downloads these itself from
  `images.cocodataset.org` given a `--dest-dir`/`--staged-dir`.

Chain: `prepare_pool.py` (pilot 200, then random 2,400) → `blind_test_full.py`
→ `finalize_survivors.py` (1,541 survived, downloads images) →
`select_final_991.py` (991 final, SEED=42).

Needs `OPENAI_API_KEY` and `gpt-5.2` access (same model/temperature/prompt as
B's Step 1 — see root README's Human verification section... actually see
`blind_test_full.py`'s own docstring, which documents every deliberate
difference from ScienceQA's version).

## What actually feeds `stage0_prepare_originals.py`

It does **not** call any of the above at runtime — it reads their frozen
outputs from `data/stage0_inputs/`: `mmlu_openstyle_700.json` (A),
`scienceqa_final_409.json` (B), `vqav2_final_ti2t_991.json` (C), plus
`comi_bench_full_refined.json` (D — the 5 untouched sources, from an earlier,
separate pipeline not included here). Re-run A/B/C only if you need to
regenerate those inputs from scratch; otherwise `stage0_prepare_originals.py`
works directly off a fresh clone.
