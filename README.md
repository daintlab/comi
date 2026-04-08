# CAMO-Bench

**Contextual Autonomous Modality Output Benchmark**

Evaluating whether unified multimodal models (UMMs) can infer the intended output modality from context alone.

## Structure

```
CAMO-Bench/
├── data/                   # Benchmark data
│   ├── camo_bench_v3.json  # 496 questions, 4 task types, 5 variants each
│   └── original_prompts.json
├── pipeline/               # Benchmark construction (Steps 1-3)
│   ├── check_expected.py   # Step 1: instruction-completeness gate
│   ├── refine.py           # Step 2: generate prompt variants
│   ├── score_variants.py   # Step 3: cosine similarity QC
│   └── ...
├── inference/              # Model inference scripts
│   ├── tar_inference.py    # Tar-7B (single-load batch runner)
│   ├── anole_inference.py  # Anole-7B (Chameleon-based)
│   ├── convert_to_inputs.py
│   ├── convert_to_original.py
│   └── setup_tar_env.sh
├── evaluation/             # Analysis & metrics
│   ├── analyze_experiments.py  # Main: Exp 1-4 tables
│   ├── compute_msa.py
│   └── analyze_anole.py
├── docs/                   # Documentation
│   └── Experiments.md
└── _archive/               # Raw dataset collection (not polished)
    └── collecting/
```

## Task Types

| Type | Input | Expected Output | Sources |
|------|-------|----------------|---------|
| t2t  | text | text | GSM8K, MMLU |
| ti2t | text + image | text | VQAv2, ScienceQA |
| t2i  | text | image | GenEval, DPG-Bench |
| ti2i | text + image | image | MagicBrush, ImgEditBench |

## Quick Start

```bash
# Run analysis on existing results
python evaluation/analyze_experiments.py \
  --v3 data/camo_bench_v3.json \
  --models "tar:<tar_output_dir>,anole:<anole_output_dir>" \
  --mock show-o2 \
  --out results/
```

## License

TBD
