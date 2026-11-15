# COMI-Bench

**C**ontextual **O**utput **M**odality **I**nference benchmark.

Unified multimodal models (UMMs) can respond with either text or an image.
Existing benchmarks always tell the model which mode to use, so they cannot
measure whether a model would have chosen correctly on its own. COMI-Bench
removes that instruction and asks a single question: given a prompt, does
the model produce the modality the prompt actually calls for?

## Dataset

5,638 samples / 11,276 prompts (1 interrogative + 1 non-interrogative
variant per sample), built from 8 existing datasets across 4 task types:

| Task type | Input | Expected output | Sources | Samples |
|-----------|-------|------------------|---------|--------:|
| t2t | text | text | GSM8K, MMLU | 1,390 |
| ti2t | text + image | text | VQAv2, ScienceQA | 1,394 |
| t2i | text | image | GenEval, DPG-Bench | 1,609 |
| ti2i | text + image | image | MagicBrush, ImgEdit-Bench | 1,245 |
| **Total** | | | | **5,638** |

Prompts in text-to-image datasets are typically bare captions or result
statements, and question marks correlate almost perfectly with the text
modality across source datasets -- the construction pipeline below controls
for both. See `data/README.md` for the full datasheet (per-file contents,
per-source licenses, dataset license).

## Repository structure

```
.
├── pipeline/        # Benchmark construction, Stage 0-5 (see below)
│   ├── stage0_prepare_originals.py ... stage5_reduce_to_2variants.py
│   ├── stage0/       # per-source candidate-pool sub-pipelines
│   ├── prompts/      # system prompts, one txt per stage
│   └── decisions/    # Stage 5's human-review decisions
├── inference/       # Single-load batch inference runners (Tar, Anole, Emu3.5)
├── evaluation/      # MSA / TP+ / COMI-score computation (see below)
├── docs/
│   └── Experiments.md   # methodology + the paper's reported tables
├── data/            # Benchmark data + datasheet (data/README.md)
└── _archive/        # superseded code, not maintained: collecting/,
                      # auroc_modality_decision_v1/, stage2_pipeline_combined/,
                      # stage4_finalize_unused/, old_docs/, assets/
```

## Construction pipeline

| Stage | What it does | Applies to |
|---|---|---|
| 0 | Preprocess: strip MCQ options, visual-grounding filter | subset of sources |
| 1 | Judge whether the prompt leads to its ground-truth modality | all |
| 1.5 | Gate: `keep` or `fix` (code, no LLM call) | all |
| 2a | Rewrite prompts that failed Stage 1 | `fix` only |
| 2b | Generate 5 variants (2 interrogative, 3 non-interrogative) | all |
| 3 | Validate each variant on modality / meaning / naturalness | all |
| 5 | Reduce 5 variants to 2 (1 interrogative + 1 non-interrogative); human review for the rest | all |

Stage 1 doesn't produce repair instructions -- its judgement would become the
prescription (every image prompt rewritten as "Generate an image: ...") -- so
Stage 2a follows a fixed rewriting policy instead.

**Visual grounding** (VQAv2, ScienceQA): each TI2T sample is answered by
GPT-5.2 three times without the image; dropped if >=2/3 responses are still
correct. 1,541/2,400 VQAv2 and 653/2,017 ScienceQA samples survive.

**ScienceQA open-style rewrite**: the released dataset's ScienceQA items have
their source question's multiple-choice options stripped from `prompt`, as a
post-process on `data/stage3_2variant_final.json` rather than inside
`stage0_prepare_originals.py` (whose own `normalize_scienceqa_409()` keeps the
options -- a surface-form crutch this benchmark otherwise controls for). Only
a 60/409-item pilot went through full Stage-3-equivalent LLM validation; the
rest were spot-checked by hand at integration time (noted per item in
`stage3_2variant_final.json`'s `checks[].reason`). Frozen artifacts for this
rewrite live in `pipeline/stage0/scienceqa_openstyle/`; the scripts that
produced them were not preserved.

**Stage 0 sub-pipelines** (`pipeline/stage0/`, read by
`stage0_prepare_originals.py` from `data/stage0_inputs/`, not run live):
- `mmlu_openstyle/` -- fully reproducible: pulls `Open-Style/Open-LLM-Benchmark`
  and `cais/mmlu` from the HF Hub only (OSQ-bench conversion, Myrzakhan et al.,
  arXiv:2406.07545).
- `vqav2_filtering/` -- the visual-grounding filter above, fully reproducible
  from scratch given the official VQAv2 files (`prepare_pool.py` samples the
  starting 2,400-item pool itself, `SEED=42`); needs `OPENAI_API_KEY` and
  <https://visualqa.org/download.html>.
- `scienceqa_filtering/` -- the same filter, but only reproducible from its
  frozen `gpt_audit_full.json` (the starting 2,017-item pool) onward; the
  script that built that pool from raw ScienceQA wasn't preserved, so treat
  the frozen file as the source of truth. Needs `OPENAI_API_KEY` and
  ScienceQA images (HF `derek-thomas/ScienceQA` or `/nas/datahub/ScienceQA/`).
- `scienceqa_openstyle/` -- frozen output of the rewrite above.

A Stage 4 (`_archive/stage4_finalize_unused/`) predates Stage 5 and is **not**
part of the released dataset's path -- see Human verification below.

### Human verification

Stage 1 decisions and a sample of Stage 3 survivors are spot-checked by hand
for agreement with human labels (details in the paper).

**Stage 5** (`stage5_reduce_to_2variants.py`, `random.Random(42)`) is what
produced the released dataset: 74/5,682 samples couldn't be auto-reduced (33
all-failed, 23 no non-interrogative survivor, 18 no interrogative survivor);
30 were rescued by hand, 44 dropped -- 5,682 -> 5,638 samples.

**Stage 4** (`_archive/stage4_finalize_unused/`) was an earlier, separate
curation pass (36 reviewed, 7 dropped, 5,682 -> 5,675) that predates Stage 5
and is not part of the released path -- Stage 5 ran directly on Stage 3's
output instead. Kept only so this history stays legible.

### Construct the benchmark from scratch

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

Models are set via env vars (default `gpt-5.5` for all three roles). The
paper's own construction run mixed three GPT-5 versions (5.1/5.2/5.5) across
stages -- not a deliberate judge/generator separation, just changes to the
available model during the project (paper, footnote 5):

```bash
JUDGE_MODEL=gpt-5.5 GEN_MODEL=gpt-5.5 VAL_MODEL=gpt-5.5 python stage1_judge.py ...
```

System prompts live in `pipeline/prompts/`; `PROMPT_DIR` overrides the location.

Input format:

```json
{"id": "t2i_007", "prompt": "an astronaut riding a horse on Mars", "task": "T2I", "source": "DrawBench"}
```

`task` is one of `T2T`/`TI2T`/`T2I`/`TI2I`; ground-truth modality and input
image presence are derived from it.

What to check in the output: `gate_per_task` (bare captions/goal states
should route to `fix`), `interrogative_ratio_by_modality` (should be close
across modalities), `axis_fail_rate`/`fail_combinations` (which validation
axis is rejecting), `pass_rate_by_route` (low `fix` rate points at Stage 2a,
low `keep` at 2b).

## Evaluation

Computes the paper's three metrics -- **MSA** (Modality Selection Accuracy),
**TP+** (task performance on correct-modality responses), **COMI-score**
(COMI-aware task score) -- from a model's COMI-Bench outputs. Full
definitions and rationale are in `evaluation/compute_msa_tp.py`'s docstring
and `docs/Experiments.md`.

1. **Generate** -- run the model over `data/stage3_2variant_final.json` (free
   modality choice, no forcing) with `inference/`. Each model's output
   records one JSON object per prompt variant:
   `{variant_id, source, gold_modality, pred_modality, text_output,
   generated_image_paths}`.
2. **Score** -- add a `score` field (0-100) to each record, per source:
   - `gsm8k`, `VQAv2`: `evaluation/rule_based_scorers.py` (exact rules, no
     judge needed).
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

   The four judge-based scorers need external models/API access not bundled
   here; `compute_msa_tp.py` only consumes the scored jsonl, not how it was
   scored.
3. **Aggregate**:

   ```bash
   cd evaluation
   python compute_msa_tp.py --input <scored_results>.jsonl --out table.md
   ```

**Baseline comparison**: the paper also reports a baseline condition -- the
original (pre-rewrite) prompt, with the model's output modality hard-forced
to the gold modality using each model's own official single-modality
inference path (not COMI's free-choice generation). This isolates "how well
would the model do this task if told the right modality" from "does it pick
the right modality on its own." See `docs/Experiments.md` for how Baseline
compares to the free-choice COMI-score above, and `inference/` for per-model
baseline-forcing notes.

## Citation

```bibtex
@article{comibench,
  title   = {A Blind Spot in Evaluating Unified Multimodal Models:
             Contextual Output Modality Inference},
  author  = {Kim, Yeonsu and Kim, Jihyo and Hwang, Sangheum},
  journal = {TODO},
  year    = {2026},
}
```

## License

- **Code** (`pipeline/`, `evaluation/`, `inference/`): MIT, see `LICENSE`.
- **Dataset** (`data/`): CC BY-NC-SA 4.0, see `data/README.md` for per-source
  license details (ScienceQA's CC BY-NC-SA 4.0 is the binding constraint).
