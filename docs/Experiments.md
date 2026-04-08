# Experiments

## Setup

### Models

We evaluate three representative UMMs that differ in how they handle output modality selection:

- **Tar-7B** (Han et al., 2025): Based on Qwen2.5-7B-Instruct with a text-aligned visual tokenizer (TA-Tok). The model can autonomously emit an `<im_start>` token to initiate image generation as part of its standard next-token prediction. We use the first-token probability of `<im_start>` as the modality decision signal.

- **Anole-7B** (Chern et al., 2024): Based on Chameleon-7B, an early-fusion model that interleaves text and image tokens in a shared vocabulary. The model generates a begin-of-image (BOI) token to switch to image generation. We use the first-token BOI probability as the modality decision signal.

- **Show-o2** (Xie et al., 2025): A unified model that supports both understanding and generation, but requires an external modality flag to select the output type. When no flag is provided, the model defaults to text-only output, making it unable to produce images through contextual inference alone.

### Data

We use CAMO-Bench, constructed by reformulating 496 questions from eight existing datasets spanning four task types:

| Task type | Input | Expected output | Sources | N |
|-----------|-------|----------------|---------|---|
| t2t | text | text | GSM8K, MMLU | 121 |
| ti2t | text + image | text | VQAv2, ScienceQA | 125 |
| t2i | text | image | GenEval, DPG-Bench | 125 |
| ti2i | text + image | image | MagicBrush, ImgEditBench | 125 |

Each question has up to 5 semantically equivalent prompt variants produced by the CAMO-Bench reformulation pipeline, yielding 2,472 items in total. The reformulation removes surface-form cues (e.g., "generate an image of") that could trivially signal the intended output modality, while preserving task semantics.

### Metrics

We frame modality selection as a binary classification problem: given a prompt, the model must decide whether to produce an image (positive) or text (negative). Each model exposes a modality decision signal---the probability assigned to the image-initiating token at the first generation step (`im_start_prob` for Tar, `boi_prob` for Anole). For Show-o2, which lacks such a signal, we assign a constant probability of 0.

Our primary metric is **AUROC** (Area Under the Receiver Operating Characteristic curve), which measures how well the modality decision signal separates image-expected from text-expected prompts. AUROC = 1.0 indicates perfect separation; AUROC = 0.5 indicates no discriminative ability (random baseline). We also report FPR and TPR at the optimal threshold (Youden's J statistic).

For accuracy-based evaluation, we adopt an **inclusion criterion**: a response is correct if it contains the expected modality, regardless of whether it also contains additional modalities. For example, a response that produces both text and an image is counted as correct for a text-expected prompt.


## Experiment 1: Modality Decision

**Goal.** Can models select the correct output modality when it is not externally specified?

**Setup.** We run each model on 496 questions (variant 0) without any modality flag, and measure how well the modality decision signal discriminates image-expected from text-expected prompts.

### Results

**Table 1.** Overall modality decision performance.

| Model | FPR | TPR | AUROC |
|-------|-----|-----|-------|
| Tar-7B | 0.004 | 0.992 | **0.9997** |
| Anole-7B | 0.000 | 0.540 | 0.7711 |
| Show-o2 | 0.000 | 0.000 | 0.5000 |

**Table 2.** Modality decision by input modality. T-in: text-only input (t2t + t2i). I-in: text + image input (ti2t + ti2i).

| Model | Input | FPR | TPR | AUROC |
|-------|-------|-----|-----|-------|
| Tar-7B | T-in | 0.000 | 1.000 | 1.0000 |
| Tar-7B | I-in | 0.008 | 0.984 | 0.9988 |
| Anole-7B | T-in | 0.000 | 0.960 | 0.9800 |
| Anole-7B | I-in | 0.000 | 0.128 | 0.5605 |
| Show-o2 | T-in | 0.000 | 0.000 | 0.5000 |
| Show-o2 | I-in | 0.000 | 0.000 | 0.5000 |

**Table 3.** Modality decision accuracy (supplementary). Inclusion criterion: correct if expected modality is present in output.

| Model | Overall | t2t | ti2t | t2i | ti2i |
|-------|---------|-----|------|-----|------|
| Tar-7B | 98.4% | 98.4% | 96.8% | 100% | 98.4% |
| Anole-7B | 75.8% | 100% | 100% | 92.0% | 12.0% |
| Show-o2 | 49.6% | 100% | 100% | 0.0% | 0.0% |

### Analysis

Three distinct patterns emerge.

**Show-o2** achieves AUROC = 0.5 (random), confirming that models relying on external modality control have no contextual modality inference capability. Its 100% accuracy on text tasks (t2t, ti2t) is trivial: it always outputs text.

**Anole-7B** shows a striking asymmetry between input modalities. With text-only input (T-in), the model achieves AUROC = 0.98, successfully distinguishing between prompts that call for text vs. image responses. However, when images are present in the input (I-in), AUROC drops to 0.56---near random. This manifests as a catastrophic failure on ti2i tasks (12.0% accuracy), where the model almost never initiates image generation despite the task clearly requiring it. This asymmetry may reflect training data composition rather than an architectural limitation, as the model's BOI token mechanism is in principle capable of signaling image generation.

**Tar-7B** achieves near-perfect modality decision (AUROC = 0.9997) across both input modalities, with TPR > 0.98 and FPR < 0.01. This suggests that when the image-initiating token is integrated into the model's standard generation vocabulary and the model is trained on diverse multimodal tasks, contextual modality inference can be highly reliable.


## Experiment 2: Prompt Variant Sensitivity

**Goal.** Do models make consistent modality decisions when the same question is rephrased in semantically equivalent ways?

**Setup.** For each of the 496 questions, we use all available prompt variants (up to 5) and measure whether the model's modality decision remains stable across variants. We define:
- **Consistency rate**: fraction of questions where all variants produce the same modality decision.
- **Flip rate**: fraction of questions where at least one variant yields a different modality.
- **Mean entropy**: average binary entropy of per-question modality distributions (0 = fully consistent, 1 = maximally split).

### Results

**Table 4.** Prompt variant sensitivity. (Tar results pending full variant inference.)

| Model | Consistency | Flip Rate | Mean Entropy | t2t | ti2t | t2i | ti2i |
|-------|-------------|-----------|--------------|-----|------|-----|------|
| Anole-7B | 82.3% | 17.7% | 0.139 | 100% | 96.8% | 83.2% | 49.6% |
| Show-o2 | 100% | 0.0% | 0.000 | 100% | 100% | 100% | 100% |

### Analysis

**Show-o2** is trivially consistent: it always outputs text regardless of prompt form.

**Anole-7B** reveals a notable sensitivity to prompt form. Overall, 17.7% of questions have at least one variant that flips the modality decision. This sensitivity is not uniform across task types:

- **t2t and ti2t** (text-expected): near-perfect consistency (100%, 96.8%). The model reliably decides on text output.
- **t2i**: 83.2% consistency. For roughly 1 in 6 image-generation questions, at least one rephrasing causes the model to not generate an image.
- **ti2i**: only 49.6% consistency---effectively a coin flip. For more than half of image-editing questions, the modality decision depends on how the prompt is worded rather than what it means.

This pattern suggests that Anole's modality decision mechanism is more sensitive to surface-form variation when the expected output is an image, particularly when images are also present in the input.


## Experiment 3: Sensitivity and Output Quality

**Goal.** Does prompt variant sensitivity affect not only modality selection but also the quality of correct outputs?

**Setup.** For text-expected tasks (t2t, ti2t) where the model correctly outputs text, we measure answer accuracy (gold answer containment) across variants. For image-expected tasks, we measure variance in modality decision probability across variants. Higher variance indicates that the model's confidence in its modality choice fluctuates with prompt form.

### Results

**Table 5.** Output quality sensitivity across prompt variants.

| Model | Text Mean Acc | Text Acc Variance | Questions w/ Variance | Image Prob Variance |
|-------|---------------|-------------------|-----------------------|---------------------|
| Anole-7B | 32.7% | 0.1246 | 151 / 246 | 0.0593 |

### Analysis

Among the 246 text-expected questions, 151 (61.4%) show non-zero variance in answer correctness across prompt variants. This means that for a majority of questions, whether the model answers correctly depends on how the prompt is worded, even when the semantics are held constant.

The image probability variance (0.059) for image-expected tasks corroborates the Exp 2 finding: the model's confidence in generating images fluctuates across semantically equivalent prompts.

These results demonstrate that prompt variant sensitivity is not merely a modality selection issue---it propagates to output quality. A model that inconsistently selects the correct modality also inconsistently produces correct answers within that modality.


## Experiment 4: Surface-Form Cue Analysis

**Goal.** Do original benchmark prompts contain surface-form cues that inflate modality decision accuracy compared to CAMO-Bench reformulated prompts?

**Setup.** We compare modality decision accuracy between original prompts (from the source datasets) and CAMO-Bench reformulated prompts on the same questions using the same model.

### Results

*(Pending: original prompt inference runs for Tar and Anole.)*

### Expected Analysis

If original prompts yield substantially higher modality accuracy than CAMO-Bench prompts, this would confirm that existing benchmarks contain surface-form cues that trivially signal the expected output modality. The gap between original and CAMO accuracy would quantify the degree to which current evaluation overestimates contextual modality inference capability.


## Summary of Findings

1. **Models that rely on external modality control (Show-o2) have zero contextual modality inference capability** (AUROC = 0.5), confirming that this is a non-trivial capability that must be explicitly supported.

2. **Modality inference is asymmetric**: even models with autonomous modality selection (Anole) are markedly weaker at deciding to generate images than text. This asymmetry is particularly severe when images are present in the input (AUROC drops from 0.98 to 0.56 for Anole).

3. **Modality decisions are sensitive to prompt form**: 17.7% of questions trigger modality flips across semantically equivalent variants in Anole, concentrated in image-expected tasks (ti2i: 49.6% consistency).

4. **Prompt sensitivity propagates to output quality**: over 60% of text-expected questions show answer accuracy variance across prompt variants, indicating that the fragility is not limited to modality selection.

5. **Not all models are equally affected**: Tar-7B achieves AUROC = 0.9997, demonstrating that near-perfect contextual modality inference is achievable when the model architecture and training adequately support it.
