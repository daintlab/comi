## aug_bench_v2 파이프라인 다이어그램 (ape 제외)

아래 Mermaid 블록을 그대로 복사해서 사용하면 됩니다.

```mermaid
flowchart TB
  %% =========================
  %% 0) Inputs
  %% =========================
  subgraph IN["0) Input dataset"]
    A["`aug_bench_v2/1_check_expected/merged_as_t2t.json`\n(t2t-style JSON: {results:[...]})"]
  end

  %% =========================
  %% 1) Check expected
  %% =========================
  subgraph S1["1) 1_check_expected (instruction-completeness gate)"]
    B["`1_check_expected/expected_5.2.py`\nadds: expected / expected_reason / refine_comment\nmodel: gpt-5.2 (default)\nAPI: OPENAI_API_KEY"]
    C["`1_check_expected/outputs/expected.json`\n(results[] enriched)"]
    D["`1_check_expected/analyze.py`\n(optional) plot expected vs not_expected by source"]
    E["`1_check_expected/analyze/*.png`\n(optional plots)"]
  end

  %% =========================
  %% 2) Refine
  %% =========================
  subgraph S2["2) 2_refine (generate diverse refined variants)"]
    F["`2_refine/refine.py`\nadds: refined_variants: [v1..vN]\nmodel: gpt-5.2 (default)\nreads: expected + refine_comment\nAPI: OPENAI_API_KEY (.env supported)"]
    G["`2_refine/outputs/refined*.json`\n(e.g., refined.json, refined_all.json,\nrefined_with_examples.json)"]
  end

  %% =========================
  %% 3) QC & Sampling
  %% =========================
  subgraph S3["3) 3_qc_sampling (quality control & sampling)"]
    H["`3_qc_sampling/score_variants.py`\nadds: refined_variants_scores\nemb_model: text-embedding-3-small (default)\ncompare_field: original_question|refined_prompt|...\nAPI: OPENAI_API_KEY (.env supported)"]
    I["`3_qc_sampling/outputs/refined.scored.json`\n(scored variants)"]
    J["`3_qc_sampling/outputs/refined.scored.tuned_*.json`\n(optional thresholded kept/dropped)"]
    K["`3_qc_sampling/evaluate.sh`\n(example CLI wrapper)"]
  end

  %% =========================
  %% 4) Evaluate
  %% =========================
  subgraph S4["4) 4_evaluate (baseline model evaluation)"]
    L["`4_evaluate/inference/run_tar_aug_bench_inference.py`\nruns model on benchmark samples\noutput: model_outputs per sample"]
    M["`4_evaluate/metrics/compute_msa.py`\ncomputes: B-MSA, hard-MSA, soft-MSA"]
    N["`4_evaluate/outputs/msa_report.*.json`\n(evaluation report)"]
  end

  %% =========================
  %% Connections
  %% =========================
  A --> B --> C
  C -. optional .-> D --> E

  C --> F --> G

  G --> H --> I
  I -. optional: --reuse_scores + --threshold .-> J
  K -. runs .-> H

  J --> L --> M --> N
```

---

## 단계별 "무엇이 추가되는지" (코드 기준)

- **1_check_expected (`expected_5.2.py`)**  
  - 입력: `{ "results": [ {task_id, source, original_question, ...}, ... ] }`  
  - 출력(각 item에 추가): `expected`, `expected_reason`, `refine_comment`

- **2_refine (`refine.py`)**  
  - 입력: expected gate 결과(`expected.json`)  
  - 출력(각 item에 추가): `refined_variants` (기본 5개, `--n_variants`로 조절)
  - 기본 동작: `expected != "expected"`만 변형 생성, `--include_expected`로 전체 생성 가능

- **3_qc_sampling (`score_variants.py` 등)**  
  - 입력: `refined_variants`가 들어있는 JSON  
  - 출력(각 item에 추가): `refined_variants_scores` (base field 대비 cosine)  
  - 옵션: `--threshold`로 `refined_variants_kept/dropped` 생성, 또는 `--reuse_scores`로 재임계값 적용(재임베딩 없음)
  - 추가 QC: `filter_empty_refined_variants_kept.py` (빈 variant 제거), `make_minimal_random3_kept.py` (3개 랜덤 샘플링), `drop_sources_from_results.py` (source 제거)

- **4_evaluate (`compute_msa.py`)**  
  - 입력: 모델 추론 결과 JSON (`model_outputs` 포함)
  - 추론: `4_evaluate/inference/run_tar_aug_bench_inference.py` (Tar 모델, modality 강제 없음)
  - 지표:
    - **B-MSA**: Text macro-MSA(t2t+ti2t)와 Image macro-MSA(t2i+ti2i)의 조화평균
    - **hard-MSA**: 샘플당 3개 variant 모두 정답인 비율
    - **soft-MSA**: 샘플당 1개 이상 정답인 비율
