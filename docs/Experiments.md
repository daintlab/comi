# Experiments

This follows *A Blind Spot in Evaluating Unified Multimodal Models:
Contextual Output Modality Inference* (Kim, Kim, Hwang). Section/table
references below are to that paper.

## Setup

### Models

Two groups, matching the paper's Sec. V-A:

**Group 1 -- external output modality control** (BAGEL, Janus-Pro, Emu3,
Show-o, Show-o2): the output modality is selected by a user-specified mode,
flag, or task-token, not inferred by the model. Each is evaluated under
*both* a forced-text and a forced-image configuration, so Table 1 reports
two rows (`Mode = Text` / `Mode = Image`) per model. `N/A` marks an
input-output pair the model's architecture doesn't support at all (e.g.
Show-o lacks TI2I).

**Group 2 -- autonomous output modality selection** (Anole-7B, Tar-7B,
Emu3.5): the model emits a special token (BOI for Anole/Emu3.5,
`<im_start>` for Tar) to switch into image generation as part of ordinary
next-token prediction, so a single free-choice inference (no external flag)
is run per prompt.

- **Tar-7B** (Han et al., 2025) -- Qwen2.5-7B-Instruct with a text-aligned
  visual tokenizer (TA-Tok).
- **Anole-7B** (Chern et al., 2024) -- Chameleon-7B, an early-fusion model
  that interleaves text and image tokens in one shared vocabulary.
- **Emu3.5** (BAAI, 2025) -- a unified autoregressive model with its own
  native BOI-emission mechanism.

Only Group 2 appears in Table 2 (Baseline vs. COMI-score) and Table 3 /
Fig. 4 (prompt-refinement ablations), since Group 1 has no COMI capability
to isolate -- its MSA is 100% or 0% by construction of which mode was
specified.

### Decoding settings

Free-choice generation uses **each model's own official, released decoding
settings** (sampling parameters, resolution, image-token budget) rather than
a single shared greedy configuration, so that a model's reported MSA /
TP+ / COMI-score reflects how it is actually meant to be run. For Tar
specifically, free choice needs settings for both an initial text-or-BOI
decision and (if `<im_start>` is emitted) continued image-token decoding;
these are applied as two decoding stages over one continuous generation,
not two independent samples, so neither stage's sampling parameters
contaminate the other's token distribution. See `inference/` for the
per-model settings actually used, and Sec. V-C for why Tar's free-choice
image generation defaults to the lowest available resolution (and the
resulting TP+/COMI-score effect on its image-output sources).

### Dataset

**COMI-Bench**: 5,638 samples / 11,276 prompts (1 interrogative + 1
non-interrogative variant per sample) across 8 sources:

| Task type | Input | Expected output | Sources | Samples |
|-----------|-------|------------------|---------|--------:|
| T2T | text | text | GSM8K, OSQ-bench (MMLU) | 699 + 691 = 1,390 |
| TI2T | text + image | text | ScienceQA, VQAv2 | 409 + 985 = 1,394 |
| T2I | text | image | GenEval, DPG-Bench | 553 + 1,056 = 1,609 |
| TI2I | text + image | image | MagicBrush, ImgEdit-Bench | 510 + 735 = 1,245 |
| **Total** | | | | **5,638** |

"OSQ-bench" is the paper's table label for MMLU converted to open-ended
questions via the OSQ-bench procedure (Myrzakhan et al., 2024); the
underlying source dataset is MMLU (see `pipeline/stage0/mmlu_openstyle/`).
ScienceQA similarly has its multiple-choice options removed (see
`pipeline/stage0/scienceqa_openstyle/`).

Construction pipeline, the "why" behind each stage, and exact reproduction
counts are in `pipeline/README.md`.

### Metrics

- **MSA (Modality Selection Accuracy)**: for a text-gold item, success iff
  the response's text content is non-empty (independent of whether the
  response also contains an image). For an image-gold item, success iff
  the response's modality is image. Formally (paper Sec. IV-B1):
  `MSA = (1/N) * sum_i 1[m_i in g(f(q_i))]`, where `g` detects the set of
  modalities present in a response (non-empty text and/or a BOI-type
  special token, as independent signals) and a response is correct if the
  gold modality `m_i` is *among* the detected ones.
- **TP+ (task performance on correct-modality responses)**: mean per-item
  correctness score (0-100), over MSA-passing items only (`I+` in the
  paper).
- **COMI-score (COMI-aware task score)**: mean per-item score over *all*
  items; an MSA-failing item is scored as `s_min` (50 for MagicBrush's
  0-100 CLIP-I scale -- the paper sets this empirically rather than at
  CLIP-I's theoretical floor, since unrelated image pairs average ~0.52
  cosine similarity; 0 for every other source) instead of being scored on
  its (nonexistent or wrong-modality) output.

Implementation: `evaluation/compute_msa_tp.py`; per-source scorers:
`evaluation/rule_based_scorers.py` (GSM8K, VQAv2 exactly; MMLU/OSQ-bench,
ScienceQA as a cheap local proxy only -- the paper's numbers use a GPT-5.5
judge for these two, see `evaluation/README.md`) plus each benchmark's own
official evaluator or a VLM/CLIP judge for the four image-output sources.

### Baseline (forced modality)

For Group 2 models, Table 2 also reports a **Baseline** condition: the
original (pre-rewrite) prompt, with the model's output modality
**hard-forced** to the gold modality via each model's own official
single-modality inference path (not COMI-Bench's free-choice generation).
This isolates how well a model can do the task at all from whether it
chooses the right modality unprompted. Blank responses are treated as
failures (scored `s_min`) rather than excluded.


## Table 1: MSA (%)

Group 1 (external control) is forced into each mode separately; Group 2
(autonomous) runs once per prompt, free choice. All values are on
COMI-Bench's released 2-variant prompts.

| Model | Mode | GSM8K | OSQ-bench | ScienceQA | VQAv2 | GenEval | DPG-Bench | MagicBrush | ImgEdit-Bench |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BAGEL | Text | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| BAGEL | Image | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Janus-Pro | Text | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Janus-Pro | Image | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Emu3 | Text | N/A | N/A | 100.0 | 100.0 | N/A | N/A | 0.0 | 0.0 |
| Emu3 | Image | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Show-o | Text | N/A | N/A | 100.0 | 100.0 | N/A | N/A | N/A | N/A |
| Show-o | Image | N/A | N/A | N/A | N/A | 100.0 | 100.0 | N/A | N/A |
| Show-o2 | Text | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Show-o2 | Image | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Emu3.5 | -- | 100.0 | 99.6 | 82.4 | 68.7 | 100.0 | 99.9 | 100.0 | 100.0 |
| Tar | -- | 97.6 | 89.6 | 99.9 | 97.7 | 100.0 | 100.0 | 99.0 | 99.3 |
| Anole | -- | 100.0 | 100.0 | 84.0 | 84.1 | 100.0 | 100.0 | 18.2 | 26.6 |


## Table 2: Baseline / COMI-score / Δ (Group 2 only)

Δ = COMI-score − Baseline. Red/blue shading in the paper figure marks
decreases/increases; here, negative Δ = decrease.

| Model | Source | Baseline | COMI-score | Δ |
|---|---|---:|---:|---:|
| Emu3.5 | GSM8K | 78.7 | 76.6 | -2.1 |
| Emu3.5 | OSQ-bench | 51.7 | 47.9 | -3.8 |
| Emu3.5 | ScienceQA | 33.0 | 27.0 | -6.0 |
| Emu3.5 | VQAv2 | 40.3 | 34.7 | -5.6 |
| Emu3.5 | GenEval | 69.3 | 68.4 | -0.9 |
| Emu3.5 | DPG-Bench | 85.8 | 84.1 | -1.7 |
| Emu3.5 | MagicBrush | 84.5 | 82.3 | -2.1 |
| Emu3.5 | ImgEdit-Bench | 69.3 | 69.4 | +0.1 |
| Tar | GSM8K | 56.1 | 54.2 | -1.9 |
| Tar | OSQ-bench | 25.5 | 27.9 | +2.4 |
| Tar | ScienceQA | 52.6 | 25.9 | -26.7 |
| Tar | VQAv2 | 67.3 | 57.1 | -10.2 |
| Tar | GenEval | 85.4 | 70.6 | -14.7 |
| Tar | DPG-Bench | 80.8 | 76.0 | -4.8 |
| Tar | MagicBrush | 77.2 | 76.1 | -1.1 |
| Tar | ImgEdit-Bench | 66.9 | 63.9 | -3.0 |
| Anole | GSM8K | 20.9 | 23.1 | +2.2 |
| Anole | OSQ-bench | 23.3 | 28.1 | +4.8 |
| Anole | ScienceQA | 9.0 | 11.4 | +2.4 |
| Anole | VQAv2 | 19.3 | 31.5 | +12.2 |
| Anole | GenEval | 36.3 | 28.7 | -7.7 |
| Anole | DPG-Bench | 56.8 | 54.7 | -2.1 |
| Anole | MagicBrush | 59.6 | 52.1 | -7.6 |
| Anole | ImgEdit-Bench | 51.2 | 10.0 | -41.2 |

ScienceQA uses a GPT-5.5 judge for both Baseline and COMI-score (its
original multiple-choice evaluation metric doesn't apply once questions are
open-ended), following Myrzakhan et al. (2024).

Notable points from Sec. V-C: Tar's image-output Δ (e.g. -14.7 on GenEval)
is largely explained by a resolution difference -- Tar's free-choice image
generation defaults to its lowest available resolution, while its official
single-modality path defaults to the highest; scoring Tar's Baseline at the
low resolution reproduces a comparable decline. Anole's large ImgEdit-Bench
drop (-41.2) reflects near-zero MSA on TI2I (Table 1) rather than output
quality -- forced into image mode, Anole scores 51.2 on the same task
(Table 2's Baseline), well above the `s_min` floor its free-choice COMI-score
is anchored to.


## Table 3 / Fig. 4: effect of individual prompt refinements (Group 2 only)

Each row isolates one refinement, applied to the *original* prompt, holding
everything else fixed; TP+ is computed only over the shared index set of
samples where both the before- and after-condition response selected the
correct modality (`I+_shared`), so a TP+ delta reflects quality, not a
change in which samples are included. Below `n_paired = 20`, TP+ is `N/A`.

- **Underspec. → Spec.**: before = original prompt (output-modality
  underspecified), after = the same sample's COMI non-interrogative variant,
  which adds an explicit instruction (generation directive for T2I; an
  explicit edit instruction for TI2I) the original lacked.
- **Interr. → Non-interr.**: within COMI, interrogative variant (before) vs.
  non-interrogative variant (after) of the same sample (ScienceQA, VQAv2).
- **Choice → Open-set** (ScienceQA only): before = original prompt (keeps
  the source question's multiple-choice options), after = COMI
  interrogative variant (options removed).

| Row | Model | n | MSA before | MSA after | ΔMSA | TP+ before | TP+ after | ΔTP+ | n_paired |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Underspec. → Spec. (T2I) | Emu3.5 | 1604 | 99.9 | 99.9 | 0.0 | 80.8 | 78.1 | -2.6 | 1603 |
| Underspec. → Spec. (T2I) | Tar | 1604 | 100.0 | 100.0 | 0.0 | 74.2 | 73.9 | -0.4 | 1604 |
| Underspec. → Spec. (T2I) | Anole | 1604 | 94.5 | 99.9 | +5.5 | 49.1 | 45.8 | -3.3 | 1515 |
| Underspec. → Spec. (TI2I) | Emu3.5 | 83 | 100.0 | 100.0 | 0.0 | 84.5 | 82.8 | -1.7 | 83 |
| Underspec. → Spec. (TI2I) | Tar | 83 | 96.4 | 100.0 | +3.6 | 79.0 | 76.9 | -2.1 | 80 |
| Underspec. → Spec. (TI2I) | Anole | 83 | 2.4 | 28.9 | +26.5 | N/A | N/A | N/A | 2 |
| Interr. → Non-interr. (VQAv2) | Emu3.5 | 985 | 65.7 | 71.8 | +6.1 | 52.4 | 48.9 | -3.4 | 509 |
| Interr. → Non-interr. (VQAv2) | Tar | 985 | 97.6 | 97.8 | +0.2 | 59.7 | 58.4 | -1.3 | 943 |
| Interr. → Non-interr. (VQAv2) | Anole | 985 | 73.4 | 94.7 | +21.3 | 41.4 | 35.5 | -5.9 | 697 |
| Interr. → Non-interr. (ScienceQA) | Emu3.5 | 409 | 94.9 | 69.9 | -24.9 | 34.5 | 34.9 | +0.4 | 278 |
| Interr. → Non-interr. (ScienceQA) | Tar | 409 | 100.0 | 99.8 | -0.2 | 28.9 | 23.0 | -5.9 | 408 |
| Interr. → Non-interr. (ScienceQA) | Anole | 409 | 84.8 | 83.1 | -1.7 | 13.2 | 12.3 | -0.9 | 312 |
| Choice → Open-set (ScienceQA) | Emu3.5 | 409 | 99.8 | 94.9 | -4.9 | N/A | N/A | N/A | - |
| Choice → Open-set (ScienceQA) | Tar | 409 | 100.0 | 100.0 | 0.0 | N/A | N/A | N/A | - |
| Choice → Open-set (ScienceQA) | Anole | 409 | 65.3 | 84.8 | +19.6 | N/A | N/A | N/A | - |

TP+ for Choice → Open-set is `N/A` for all models: the two conditions use
different evaluation metrics (multiple-choice exact-match vs. open-ended
judge), so a TP+ delta isn't meaningful there (ΔMSA still is).

Across refinements, ΔMSA is generally larger in magnitude than ΔTP+ --
prompt form affects *which* modality a model picks more than how well it
then performs the task in that modality.
