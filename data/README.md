# COMI-Bench data

## Files

| File | What it is |
|---|---|
| `stage0_originals.jsonl` | Stage 0 output: 5,682 candidate items pooled from the 8 source datasets, before any LLM judging/rewriting. |
| `stage0_inputs/` | Frozen per-source inputs that `stage0_prepare_originals.py` assembles from (see root `README.md`, "Stage 0 sub-pipelines"). |
| `stage1_full_5682.json` | Stage 1 output: adds `needs_refinement`/`route` per item. |
| `stage2a_full_5682.json` | Stage 2a output: adds `fixed_prompt`/`source_prompt` for `route=="fix"` items. |
| `stage2b_full_5682.json` | Stage 2b output: adds 5 variants per item (2 interrogative, 3 non-interrogative). |
| `stage3_full_5682.json` | Stage 3 output: adds per-variant validation (`modality_ok`/`meaning_ok`/`naturalness_ok`). |
| `stage3_2variant_final.json` | **The released dataset.** 5,638 samples, 2 variants each -- 11,276 prompts total. ScienceQA's variants have their source question's MC options stripped (see root `README.md`, "ScienceQA open-style rewrite"); this differs from the other files above. |
| `comi_bench_final_original_5682.json` | The untouched original prompts (pre-Stage-1), for the "original prompt, no forcing" baseline comparison reported in the paper. |

## Source datasets and their licenses

COMI-Bench's prompts and gold answers/images are built from eight existing
benchmarks. Obtain each from its official source; this repo does not bundle
any source images.

| Source | License | Official source |
|---|---|---|
| GSM8K | MIT | <https://github.com/openai/grade-school-math> |
| MMLU | MIT | <https://github.com/hendrycks/test> |
| ScienceQA | CC BY-NC-SA 4.0 | <https://github.com/lupantech/ScienceQA> |
| VQAv2 | CC BY 4.0 (annotations); images are MS COCO, under COCO's own terms | <https://visualqa.org/> |
| GenEval | MIT | <https://github.com/djghosh13/geneval> |
| DPG-Bench | Apache 2.0 | <https://github.com/TencentQQGYLab/ELLA> |
| MagicBrush | CC BY 4.0 | <https://github.com/OSU-NLP-Group/MagicBrush> |
| ImgEdit-Bench | not stated by the source repo as of this writing -- obtain directly and check its terms before redistributing anything derived from it | <https://github.com/PKU-YuanGroup/ImgEdit> |

## License of the released COMI-Bench dataset

The dataset files in this directory (`stage3_2variant_final.json` and the
other per-stage/original-prompt files) are released under **CC BY-NC-SA 4.0**.
This is the strictest constraint among the eight source licenses above
(ScienceQA's CC BY-NC-SA 4.0, share-alike), applied to the dataset as a whole
rather than tracked per-source-subset.

Fields derived from a source dataset (`original_question`/`prompt`,
`gold_answer`, `metadata`) remain subject to that source's own license in
addition to CC BY-NC-SA 4.0 -- if you plan to redistribute a derivative that
includes these fields, check the corresponding source's license in the table
above, not just COMI-Bench's own license.

The code in this repository (construction pipeline, evaluation, inference
scripts) is licensed separately under MIT -- see the repository root `LICENSE`.
