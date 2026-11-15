# ScienceQA open-style conversion (MC options removed)

`stage0_prepare_originals.py`'s `normalize_scienceqa_409()` keeps the source
ScienceQA question's multiple-choice options in `prompt` (e.g. "Will these
magnets attract or repel each other? A. attract B. repel"). That is a surface-
form crutch of exactly the kind this benchmark otherwise controls for
(see root `pipeline/README.md`, "Why existing data cannot be used as is") --
with the options visible, the output modality is trivial to guess without
reading the question at all. The released dataset (`data/stage3_2variant_final.json`)
uses the MC-option-free rewrite in this folder instead.

## What's here

- `scienceqa_openstyle_409_stage2b.json` -- all 409 items, MC options stripped
  from `source_prompt`, 5 variants generated per item (same schema as
  `stage2b_generate.py`'s output).
- `scienceqa_openstyle_409_stage3_pilot60.json` -- a 60-item pilot run through
  the same 3-axis LLM validation as `stage3_validate.py` (`val_model=gpt-5.5`),
  used to check the conversion approach before scaling it up. 299/300 variants
  passed (`variant_pass_rate: 0.9967`).
- `scienceqa_openstyle_2variant_bench.json` -- the released final form: all
  409 items reduced to 2 variants (1 interrogative + 1 non-interrogative).

## What's not here, and the actual caveat for the released data

The scripts that produced these three files were not preserved -- same
situation as `scienceqa_filtering/`'s `gpt_audit_full.json` (see that folder's
README). Treat these frozen files as the source of truth.

More importantly: **only the 60-item pilot went through real per-axis LLM
validation.** The other 349 items' variants were taken directly from the
5-variant generation step without an equivalent validation pass. They were
manually spot-checked (a random sample, cross-referenced against the original
MC-option source text for meaning preservation) at the time this was
integrated into `data/stage3_2variant_final.json`, and looked consistent with
the pilot's quality, but that is not the same guarantee the rest of the
benchmark's variants have. `data/stage3_2variant_final.json`'s `checks[].reason`
field says this explicitly, per item, for anyone auditing the released data.
