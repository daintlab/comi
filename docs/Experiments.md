# Experiments

## Setup

### Models

Three unified multimodal models (UMMs) are evaluated under COMI-Bench's free
modality-choice protocol (no external modality flag):

- **Tar-7B** (Han et al., 2025) -- Qwen2.5-7B-Instruct with a text-aligned
  visual tokenizer (TA-Tok). Emits `<im_start>` to switch into image
  generation as part of ordinary next-token prediction.
- **Anole-7B** (Chern et al., 2024) -- Chameleon-7B, an early-fusion model
  that interleaves text and image tokens in one shared vocabulary. Emits a
  begin-of-image (BOI) token to switch to image generation.
- **Emu3.5** (BAAI) -- a unified autoregressive model with its own native
  BOI-emission mechanism for free modality choice, used with its official
  decoding settings throughout.

BAGEL, Janus-Pro, Emu3, Show-o and Show-o2 are used only for the qualitative
comparison figure, not for the quantitative tables below -- most either
require an external modality flag (so they cannot be evaluated under free
choice at all) or were not part of the main quantitative comparison.

### Decoding settings

Free-choice generation uses **each model's own official, released decoding
settings** (sampling parameters, resolution, image-token budget) rather than
a single shared greedy configuration, so that a model's reported MSA/TP/C-TP
reflects how it is actually meant to be run. For Tar specifically, free
choice needs settings for both an initial text-or-BOI decision and (if BOI is
emitted) continued image-token decoding; these are applied as two decoding
stages over one continuous generation, not two independent samples, so
neither stage's sampling parameters contaminate the other's token
distribution. See `inference/` for the per-model settings actually used.

### Dataset

**COMI-Bench**: 5,638 samples / 11,276 prompts (1 interrogative + 1
non-interrogative variant per sample) across 8 sources:

| Task type | Input | Expected output | Sources | Samples |
|-----------|-------|------------------|---------|--------:|
| t2t | text | text | GSM8K, MMLU | 699 + 691 = 1,390 |
| ti2t | text + image | text | VQAv2, ScienceQA | 985 + 409 = 1,394 |
| t2i | text | image | GenEval, DPG-Bench | 553 + 1,056 = 1,609 |
| ti2i | text + image | image | MagicBrush, ImgEdit-Bench | 510 + 735 = 1,245 |
| **Total** | | | | **5,638** |

Construction pipeline, the "why" behind each stage, and exact reproduction
counts are in `pipeline/README.md`.

### Metrics

- **MSA (Modality Selection Accuracy, strict)**: for a text-gold item,
  success iff the response's text content is non-empty (independent of
  whether the response also contains an image). For an image-gold item,
  success iff the response's modality is image.
- **TP (Task Performance)**: mean per-item correctness score (0-100), over
  MSA-passing items only.
- **C-TP (Conditional Task Performance)**: mean per-item score over *all*
  items; an MSA-failing item is scored as `s_fail` (50 for MagicBrush's
  0-100 CLIP-I scale, 0 for every other source) instead of being scored on
  its (nonexistent or wrong-modality) output.

Implementation: `evaluation/compute_msa_tp.py`; per-source scorers:
`evaluation/rule_based_scorers.py` (GSM8K, VQAv2 exactly; MMLU, ScienceQA as
a cheap local proxy only -- the paper's numbers use a GPT-5.5 judge for
these two, see `evaluation/README.md`) plus each benchmark's own official
evaluator or a VLM/CLIP judge for the four image-output sources.

### Baseline (forced modality)

For comparison, each model is also run on the **original** (pre-rewrite)
prompt with its output modality **hard-forced** to the gold modality, using
that model's own official single-modality inference path (not COMI-Bench's
free-choice generation). This isolates how well a model can do the task at
all from whether it chooses the right modality unprompted; Table 2 reports
this as `Baseline`, denominator = full prompt count (an empty forced-text
response scores 0).


## Table 1: MSA (%), official decoding, free choice

"COMI" = the released 2-variant prompts, averaged over both variants.
"Original" = the pre-rewrite prompt (free choice, no forcing), reported only
for the 5 sources where the pipeline produces this condition
(GenEval/DPG-Bench/MagicBrush/ImgEdit-Bench/ScienceQA -- see
`pipeline/README.md`).

| Model | Source | Condition | n | MSA |
|---|---|---|---:|---:|
| Emu3.5 | gsm8k | COMI | 1398 | 100.0 |
| Tar | gsm8k | COMI | 1398 | 97.6 |
| Anole | gsm8k | COMI | 1398 | 100.0 |
| Emu3.5 | mmlu | COMI | 1382 | 99.6 |
| Tar | mmlu | COMI | 1382 | 89.6 |
| Anole | mmlu | COMI | 1382 | 100.0 |
| Emu3.5 | ScienceQA | COMI | 818 | 82.4 |
| Tar | ScienceQA | COMI | 818 | 99.9 |
| Anole | ScienceQA | COMI | 818 | 84.0 |
| Emu3.5 | VQAv2 | COMI | 1970 | 68.7 |
| Tar | VQAv2 | COMI | 1970 | 97.7 |
| Anole | VQAv2 | COMI | 1970 | 84.1 |
| Emu3.5 | geneval | COMI | 1106 | 100.0 |
| Tar | geneval | COMI | 1106 | 100.0 |
| Anole | geneval | COMI | 1106 | 100.0 |
| Emu3.5 | dpg_bench | COMI | 2112 | 99.9 |
| Tar | dpg_bench | COMI | 2112 | 100.0 |
| Anole | dpg_bench | COMI | 2112 | 100.0 |
| Emu3.5 | MagicBrush | COMI | 1020 | 100.0 |
| Tar | MagicBrush | COMI | 1020 | 99.0 |
| Anole | MagicBrush | COMI | 1020 | 18.2 |
| Emu3.5 | ImgEditBench | COMI | 1470 | 100.0 |
| Tar | ImgEditBench | COMI | 1470 | 99.3 |
| Anole | ImgEditBench | COMI | 1470 | 26.6 |
| Emu3.5 | geneval | Original | 553 | 100.0 |
| Tar | geneval | Original | 553 | 100.0 |
| Anole | geneval | Original | 553 | 97.6 |
| Emu3.5 | dpg_bench | Original | 1056 | 99.9 |
| Tar | dpg_bench | Original | 1056 | 100.0 |
| Anole | dpg_bench | Original | 1056 | 92.8 |
| Emu3.5 | MagicBrush | Original | 510 | 100.0 |
| Tar | MagicBrush | Original | 510 | 99.4 |
| Anole | MagicBrush | Original | 510 | 7.6 |
| Emu3.5 | ImgEditBench | Original | 735 | 100.0 |
| Tar | ImgEditBench | Original | 735 | 99.3 |
| Anole | ImgEditBench | Original | 735 | 11.7 |
| Emu3.5 | ScienceQA | Original | 409 | 99.8 |
| Tar | ScienceQA | Original | 409 | 100.0 |
| Anole | ScienceQA | Original | 409 | 65.3 |


## Table 2: Baseline / TP / C-TP / Δ, COMI condition

Δ = C-TP − Baseline. ScienceQA's Δ is reported as `--`: COMI's TP/C-TP here
use a GPT-5.5 judge while Baseline uses a different scoring path for this
source, so the two are not on a directly comparable basis for a Δ (see
`evaluation/README.md`).

| Model | Source | n | Baseline | TP | C-TP | Δ |
|---|---|---:|---:|---:|---:|---:|
| Emu3.5 | gsm8k | 1398 | 78.7 | 76.6 | 76.6 | -2.1 |
| Tar | gsm8k | 1398 | 56.1 | 55.5 | 54.2 | -1.9 |
| Anole | gsm8k | 1398 | 20.9 | 23.1 | 23.1 | 2.2 |
| Emu3.5 | mmlu | 1382 | 51.7 | 48.1 | 47.9 | -3.8 |
| Tar | mmlu | 1382 | 25.5 | 31.1 | 27.9 | 2.4 |
| Anole | mmlu | 1382 | 23.3 | 28.1 | 28.1 | 4.8 |
| Emu3.5 | ScienceQA | 818 | 33.0 | 32.8 | 27.0 | -- |
| Tar | ScienceQA | 818 | 52.6 | 25.9 | 25.9 | -- |
| Anole | ScienceQA | 818 | 9.0 | 13.5 | 11.4 | -- |
| Emu3.5 | VQAv2 | 1970 | 40.3 | 50.5 | 34.7 | -5.6 |
| Tar | VQAv2 | 1970 | 67.3 | 58.5 | 57.1 | -10.2 |
| Anole | VQAv2 | 1970 | 19.3 | 37.5 | 31.5 | 12.2 |
| Emu3.5 | geneval | 1106 | 69.3 | 68.4 | 68.4 | -0.9 |
| Tar | geneval | 1106 | 85.4 | 70.6 | 70.6 | -14.7 |
| Anole | geneval | 1106 | 36.3 | 28.7 | 28.7 | -7.7 |
| Emu3.5 | dpg_bench | 2112 | 85.8 | 84.1 | 84.1 | -1.7 |
| Tar | dpg_bench | 2112 | 80.8 | 76.0 | 76.0 | -4.8 |
| Anole | dpg_bench | 2112 | 56.8 | 54.7 | 54.7 | -2.1 |
| Emu3.5 | MagicBrush | 1020 | 84.5 | 82.3 | 82.3 | -2.1 |
| Tar | MagicBrush | 1020 | 77.2 | 76.4 | 76.1 | -1.1 |
| Anole | MagicBrush | 1020 | 59.6 | 61.4 | 52.1 | -7.6 |
| Emu3.5 | ImgEditBench | 1470 | 69.3 | 69.4 | 69.4 | 0.1 |
| Tar | ImgEditBench | 1470 | 66.9 | 64.3 | 63.9 | -3.0 |
| Anole | ImgEditBench | 1470 | 51.2 | 48.9 | 10.0 | -41.2 |


## Table 3: before/after comparisons

- **+Instr.**: samples whose original prompt was classified as
  *underspecified* (a rule-based classifier over explicit-instruction
  markers, `scripts/underspec_rule.py` in the evaluation codebase) --
  before = original prompt (free choice, no forcing), after = that same
  sample's COMI non-interrogative (statement-form) variant, which carries
  an explicit instruction the original lacked. (T2I): GenEval/DPG-Bench
  samples. (TI2I): MagicBrush/ImgEdit-Bench samples.
- **Q→N**: within COMI, interrogative variant (before) vs. non-interrogative
  variant (after) of the same sample, for the two sources where this
  comparison is reported (ScienceQA, VQAv2).
- **−Opt.** (ScienceQA only): before = original prompt (keeps the source
  question's multiple-choice options), after = COMI interrogative variant
  (options removed, see `pipeline/stage0/scienceqa_openstyle/`).

`n_paired` = number of samples with both a before- and an after-condition
MSA-passing response (required for a TP delta); below 20, TP is reported as
`N/A`.

| Row | Model | n | MSA before | MSA after | ΔMSA | TP before | TP after | ΔTP | n_paired |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| +Instr. (T2I) | Emu3.5 | 1604 | 99.9 | 99.9 | 0.0 | 80.8 | 78.1 | -2.6 | 1603 |
| +Instr. (T2I) | Tar | 1604 | 100.0 | 100.0 | 0.0 | 74.2 | 73.9 | -0.4 | 1604 |
| +Instr. (T2I) | Anole | 1604 | 94.5 | 99.9 | 5.5 | 49.1 | 45.8 | -3.3 | 1515 |
| +Instr. (TI2I) | Emu3.5 | 83 | 100.0 | 100.0 | 0.0 | 84.5 | 82.8 | -1.7 | 83 |
| +Instr. (TI2I) | Tar | 83 | 96.4 | 100.0 | 3.6 | 79.0 | 76.9 | -2.1 | 80 |
| +Instr. (TI2I) | Anole | 83 | 2.4 | 28.9 | 26.5 | N/A | N/A | N/A | 2 |
| Q→N (ScienceQA) | Emu3.5 | 409 | 94.9 | 69.9 | -24.9 | 34.5 | 34.9 | 0.4 | 278 |
| Q→N (ScienceQA) | Tar | 409 | 100.0 | 99.8 | -0.2 | 28.9 | 23.0 | -5.9 | 408 |
| Q→N (ScienceQA) | Anole | 409 | 84.8 | 83.1 | -1.7 | 13.2 | 12.3 | -0.9 | 312 |
| Q→N (VQAv2) | Emu3.5 | 985 | 65.7 | 71.8 | 6.1 | 52.4 | 48.9 | -3.4 | 509 |
| Q→N (VQAv2) | Tar | 985 | 97.6 | 97.8 | 0.2 | 59.7 | 58.4 | -1.3 | 943 |
| Q→N (VQAv2) | Anole | 985 | 73.4 | 94.7 | 21.3 | 41.4 | 35.5 | -5.9 | 697 |
| −Opt. (ScienceQA) | Emu3.5 | 409 | 99.8 | 94.9 | -4.9 | N/A | N/A | N/A | - |
| −Opt. (ScienceQA) | Tar | 409 | 100.0 | 100.0 | 0.0 | N/A | N/A | N/A | - |
| −Opt. (ScienceQA) | Anole | 409 | 65.3 | 84.8 | 19.6 | N/A | N/A | N/A | - |

A LaTeX (`\begin{tabular}`) version of all three tables is generated
alongside these by the same reporting scripts referenced in the paper.
