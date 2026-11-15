# Evaluation

Computes the paper's three metrics -- **MSA** (Modality Selection Accuracy),
**TP+** (task performance on correct-modality responses), **COMI-score**
(COMI-aware task score) -- from a model's COMI-Bench outputs. Definitions
and the rationale for each are in `compute_msa_tp.py`'s docstring and
`docs/Experiments.md`.

## Pipeline

1. **Generate** -- run the model over `data/stage3_2variant_final.json` (free
   modality choice, no forcing) with `inference/`. Each model's output
   records one JSON object per prompt variant:
   `{variant_id, source, gold_modality, pred_modality, text_output,
   generated_image_paths}`.
2. **Score** -- add a `score` field (0-100) to each record, per source:
   - `gsm8k`, `VQAv2`: `rule_based_scorers.py` (exact rules, no judge needed).
   - `mmlu` (= **OSQ-bench** in the paper's tables), `ScienceQA`:
     `rule_based_scorers.py`'s `score_open_style` is a cheap local proxy
     (loose containment, under-counts paraphrases); the paper uses a
     GPT-5.5 judge instead (binary correct/incorrect, independent of modality).
   - `geneval`, `dpg_bench`: each benchmark's own official evaluator
     (object-detection checker; VQA-based question answering) on the
     generated image.
   - `MagicBrush`: CLIP-I similarity to the target image, scaled to 0-100.
   - `ImgEditBench`: a VLM judge (paper uses Qwen2.5-VL) rating prompt
     compliance / visual seamlessness / content preservation 1-5 each,
     rescaled via `(avg_1to5 - 1) / 4 * 100`.
   These four judge-based scorers need external models/API access not
   bundled here; `compute_msa_tp.py` only consumes the scored jsonl, not
   how it was scored.
3. **Aggregate** -- `python compute_msa_tp.py --input <scored>.jsonl --out table.md`.

## Baseline comparison (forced-modality)

The paper also reports a *baseline* condition: the original (pre-rewrite)
prompt, with the model's output modality hard-forced to the gold modality
using each model's own official single-modality inference path (not COMI's
free-choice generation). This isolates "how well would the model do this
task if told the right modality" from "does it pick the right modality on
its own." See `docs/Experiments.md` for how Baseline compares to the
free-choice COMI-score above, and `inference/` for per-model baseline-forcing
notes.
