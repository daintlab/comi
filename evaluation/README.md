# Evaluation

Computes the paper's three metrics -- **MSA** (Modality Selection Accuracy,
strict), **TP** (Task Performance), **C-TP** (Conditional Task Performance)
-- from a model's COMI-Bench outputs. Definitions and the rationale for each
are in `compute_msa_tp.py`'s docstring and `docs/Experiments.md`.

## Pipeline

1. **Generate** -- run the model over `data/stage3_2variant_final.json` (free
   modality choice, no forcing) with `inference/`. Each model's output
   records one JSON object per prompt variant:
   `{variant_id, source, gold_modality, pred_modality, text_output,
   generated_image_paths}`.
2. **Score** -- add a `score` field (0-100) to each record, per source:
   - `gsm8k`, `VQAv2`: `rule_based_scorers.py` (exact rules, no judge needed).
   - `mmlu`, `ScienceQA`: `rule_based_scorers.py`'s `score_open_style` is a
     cheap local proxy, but it under-counts paraphrased correct answers
     (loose text containment, not semantic match). The paper's reported
     numbers use a GPT-5.5 judge instead -- grade the response against the
     gold answer, asking only "does this response state the correct
     answer" (binary, strict), independent of modality.
   - `geneval`, `dpg_bench`: score with each benchmark's own official
     evaluator against the generated image (geneval's object-detection
     checker; dpg_bench's VQA-based question answering over the generated
     image, following each benchmark's own released code).
   - `MagicBrush`, `ImgEditBench`: `MagicBrush` uses CLIP-I similarity
     between the generated image and the target image, scaled to 0-100.
     `ImgEditBench` uses a VLM judge (the paper uses Qwen2.5-VL) rating
     prompt compliance / visual seamlessness / content preservation each
     1-5, averaged and rescaled to 0-100 via `(avg_1to5 - 1) / 4 * 100`.
   These four judge-based scorers need external models/API access not
   bundled in this repo; wire up whichever judge you have access to, as
   long as it outputs a 0-100 `score` per item matching the rubric above --
   `compute_msa_tp.py` only consumes the scored jsonl, not how it was scored.
3. **Aggregate** -- `python compute_msa_tp.py --input <scored>.jsonl --out table.md`.

## Baseline comparison (forced-modality)

The paper also reports a *baseline* condition: the original (pre-rewrite)
prompt, with the model's output modality hard-forced to the gold modality
using each model's own official single-modality inference path (not COMI's
free-choice generation). This isolates "how well would the model do this
task if told the right modality" from "does it pick the right modality on
its own." See `docs/Experiments.md` for how baseline TP/C-TP compare to the
free-choice numbers above, and `inference/` for per-model baseline-forcing
notes.
