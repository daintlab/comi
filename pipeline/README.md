# COMI-Bench

**C**ontextual **O**utput **M**odality **I**nference benchmark.

Unified multimodal models (UMMs) can respond with either text or an image.
Existing benchmarks always tell the model which mode to use, so they cannot
measure whether a model would have chosen correctly on its own. COMI-Bench
removes that instruction and asks a single question: given a prompt, does the
model produce the modality the prompt actually calls for?

The benchmark covers four task types — T2T, TI2T, T2I, TI2I — built from
existing datasets with minimal refinement.

![pipeline](assets/comi_pipeline.png)

## Why existing data cannot be used as is

Prompts in text-to-image datasets are typically bare captions (`a photo of a
dog`) or statements of a desired result (`the sky should be clearer`). These
read as image requests only by convention of the dataset they came from. A UMM
can always continue in text, so on their own such prompts do not settle what
the model should do — and scoring a text continuation as wrong would be unfair.

A second problem is surface form. In source datasets, question marks correlate
almost perfectly with the text modality: VQA prompts are questions, image
prompts are not. A model could score well by looking at punctuation alone. The
construction pipeline controls for this.

## Pipeline

| Stage | What it does | Applies to |
|---|---|---|
| 0 | Preprocess: strip MCQ options, drop image-independent TI samples | subset of sources |
| 1 | Judge whether the prompt leads to its ground-truth modality | all |
| 1.5 | Gate: `keep` or `fix` (code, no LLM call) | all |
| 2a | Rewrite prompts that failed Stage 1 | `fix` only |
| 2b | Generate 5 variants (2 interrogative, 3 non-interrogative) | all |
| 3 | Validate each variant on modality / meaning / naturalness | all |

Stage 1 does not produce repair instructions. Its judgement criterion would
become the prescription, and every image prompt would end up rewritten as
"Generate an image: …". Stage 2a follows a fixed rewriting policy instead.

## Layout

```
pipeline/
  stage1_judge.py       Stage 1 only
  stage2_pipeline.py    Stage 2a → 2b → 3
  prompts/              system prompts, one txt per stage
data/
  samples.jsonl         input samples
assets/
  comi_pipeline.png
```

## Usage

```bash
pip install openai python-dotenv
echo 'OPENAI_API_KEY=sk-...' > .env

cd pipeline
python stage1_judge.py --input ../data/samples.jsonl --output stage1.json
python stage2_pipeline.py --input stage1.json --output stage2.json
```

Models are set through environment variables. The validator is deliberately a
different model from the generator (Panickssery et al., NeurIPS 2024).

```bash
JUDGE_MODEL=gpt-5.5 GEN_MODEL=gpt-5.1 VAL_MODEL=gpt-5.5 python stage2_pipeline.py ...
```

System prompts live in `prompts/` as plain text so they can be edited without
touching the scripts. `PROMPT_DIR` overrides the location.

### Input format

```json
{"id": "t2i_007", "prompt": "an astronaut riding a horse on Mars", "task": "T2I", "source": "DrawBench"}
```

`task` is one of `T2T`, `TI2T`, `T2I`, `TI2I`. Ground-truth modality and input
image presence are derived from it.

## What to check in the output

- **`gate_per_task`** — bare captions and bare goal states should route to
  `fix`; explicit directives and plain VQA should stay `keep`.
- **`interrogative_ratio_by_modality`** — text and image should be close. A gap
  means question marks still predict the answer.
- **`axis_fail_rate`** and **`fail_combinations`** — which of the three
  validation checks is doing the rejecting.
- **`pass_rate_by_route`** — a low rate on `fix` points at Stage 2a; a low rate
  on `keep` points at Stage 2b.

## Human verification

LLM judgements are spot-checked by hand: Stage 1 decisions for agreement with
human labels, and a sample of Stage 3 survivors for meaning preservation.
Details in the paper.

## References

- Panickssery et al. *LLM Evaluators Recognize and Favor Their Own
  Generations.* NeurIPS 2024.
- Mizrahi et al. *State of What Art? A Call for Multi-Prompt LLM Evaluation.*
  TACL 2024.
- Chen et al. *Are We on the Right Way for Evaluating Large Vision-Language
  Models?* (MMStar) NeurIPS 2024.
- Wang et al. *MMLU-Pro.* NeurIPS 2024 Datasets and Benchmarks.
