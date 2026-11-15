# COMI-Bench

**C**ontextual **O**utput **M**odality **I**nference benchmark.

Unified multimodal models (UMMs) can respond with either text or an image.
Existing benchmarks always tell the model which mode to use, so they cannot
measure whether a model would have chosen correctly on its own. COMI-Bench
removes that instruction: given a prompt, does the model produce the
modality the prompt actually calls for?

## Why existing data cannot be used as is

Prompts in text-to-image datasets are typically bare captions (`a photo of a
dog`) or result statements (`the sky should be clearer`) -- a UMM can always
continue in text, so these don't settle what the model should do. Separately,
question marks correlate almost perfectly with the text modality across
source datasets, so a model could score well on punctuation alone. The
pipeline below controls for both.

## Pipeline

| Stage | What it does | Applies to |
|---|---|---|
| 0 | Preprocess: strip MCQ options, visual-grounding filter (see below) | subset of sources |
| 1 | Judge whether the prompt leads to its ground-truth modality | all |
| 1.5 | Gate: `keep` or `fix` (code, no LLM call) | all |
| 2a | Rewrite prompts that failed Stage 1 | `fix` only |
| 2b | Generate 5 variants (2 interrogative, 3 non-interrogative) | all |
| 3 | Validate each variant on modality / meaning / naturalness | all |
| 5 | Reduce 5 variants to 2 (1 interrogative + 1 non-interrogative); human review for the rest | all |

Stage 1 doesn't produce repair instructions -- its judgement would become the
prescription (every image prompt rewritten as "Generate an image: ..."), so
Stage 2a follows a fixed rewriting policy instead.

**Visual grounding** (VQAv2, ScienceQA): each TI2T sample is answered by
GPT-5.2 three times without the image; dropped if ≥2/3 responses are still
correct. 1,541/2,400 VQAv2 and 653/2,017 ScienceQA samples survive -- see
`stage0/vqav2_filtering/`, `stage0/scienceqa_filtering/`.

**ScienceQA MCQ-option stripping** is a post-process on
`data/stage3_2variant_final.json`, not inside `stage0_prepare_originals.py` --
see `stage0/README.md` (D) (also notes a validation-coverage caveat: only
60/409 items got full Stage-3-equivalent LLM validation).

A Stage 4 (`_archive/stage4_finalize_unused/`) predates Stage 5 and is **not**
part of the released dataset's path -- see Human verification below.

## Layout

```
pipeline/
  stage0_prepare_originals.py    Stage 0
  stage0/                        per-source candidate-pool sub-pipelines
  stage1_judge.py                Stage 1
  stage2a_fix.py                 Stage 2a
  stage2b_generate.py            Stage 2b
  stage3_validate.py             Stage 3
  stage5_reduce_to_2variants.py  Stage 5
  prompts/                       system prompts, one txt per stage
  decisions/                     Stage 5's human-review decisions
  logs/                          full-run logs (stage1/2a/2b/3)
data/
  stage0_originals.jsonl ... stage3_full_5682.json   per-stage outputs
  stage3_2variant_final.json                         released dataset
```

## Usage

```bash
pip install openai python-dotenv
echo 'OPENAI_API_KEY=sk-...' > .env

cd pipeline
python stage1_judge.py --input ../data/stage0_originals.jsonl --output stage1_out.json
python stage2a_fix.py --input stage1_out.json --output stage2a_out.json
python stage2b_generate.py --input stage2a_out.json --output stage2b_out.json
python stage3_validate.py --input stage2b_out.json --output stage3_out.json
python stage5_reduce_to_2variants.py --input stage3_out.json --output ../data/stage3_2variant_final.json
```

Models are set via env vars (the validator is deliberately a different model
from the generator, Panickssery et al. NeurIPS 2024):

```bash
JUDGE_MODEL=gpt-5.5 GEN_MODEL=gpt-5.5 VAL_MODEL=gpt-5.5 python stage1_judge.py ...
```

System prompts live in `prompts/`; `PROMPT_DIR` overrides the location.

### Input format

```json
{"id": "t2i_007", "prompt": "an astronaut riding a horse on Mars", "task": "T2I", "source": "DrawBench"}
```

`task` is one of `T2T`/`TI2T`/`T2I`/`TI2I`; ground-truth modality and input
image presence are derived from it.

## What to check in the output

- `gate_per_task` -- bare captions/goal states should route to `fix`.
- `interrogative_ratio_by_modality` -- should be close across modalities.
- `axis_fail_rate` / `fail_combinations` -- which validation axis is rejecting.
- `pass_rate_by_route` -- low `fix` rate points at Stage 2a, low `keep` at 2b.

## Human verification

Stage 1 decisions and a sample of Stage 3 survivors are spot-checked by hand
for agreement with human labels (details in the paper).

**Stage 5** (`stage5_reduce_to_2variants.py`, `random.Random(42)`) is what
produced the released dataset: 74 samples couldn't be auto-reduced (33
all-failed, 23 no non-interrogative survivor, 18 no interrogative survivor);
30 were rescued by hand, 44 dropped (25 via an automatic drop registry, the
rest manual) -- 5,682 → 5,638 samples.

**Stage 4** (`_archive/stage4_finalize_unused/`) was an earlier, separate
curation pass (36 reviewed, 7 dropped, 5,682 → 5,675) that predates Stage 5
and is not part of the released path -- Stage 5 ran directly on Stage 3's
output instead. Kept only so this history stays legible.

## References

- Panickssery et al. *LLM Evaluators Recognize and Favor Their Own
  Generations.* NeurIPS 2024.
- Mizrahi et al. *State of What Art? A Call for Multi-Prompt LLM Evaluation.*
  TACL 2024.
- Chen et al. *Are We on the Right Way for Evaluating Large Vision-Language
  Models?* (MMStar) NeurIPS 2024. -- basis for the visual-grounding filter.
- Myrzakhan et al. *Open-LLM-Leaderboard: From Multi-Choice to Open-Style
  Questions for LLMs Evaluation, Benchmark, and Arena.* arXiv:2406.07545,
  2024. -- the OSQ-bench conversion procedure, see `stage0/mmlu_openstyle/`.
