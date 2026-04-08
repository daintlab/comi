"""
Common configuration for dataset collection scripts.

IMPORTANT: 실행 전 반드시 HF_HOME 경로를 본인 NAS 경로로 수정하세요.
기본값은 /nas/datahub/hf_cache 로 설정되어 있습니다.
"""

import os

# HF 캐시 — 반드시 컨테이너 내부로 설정 (NAS 사용 금지)
HF_HOME = "/data/ys_hf_cache"
os.environ["HF_HOME"] = HF_HOME

# 가공된 결과 저장 경로 — NAS
RAW_ROOT = "/nas/home/ysk1207/cams-benchmark/aug_bench_v2/0_collecting_dataset/raw"

T2T_DIR  = os.path.join(RAW_ROOT, "t2t")
TI2T_DIR = os.path.join(RAW_ROOT, "ti2t")
T2I_DIR  = os.path.join(RAW_ROOT, "t2i")
TI2I_DIR = os.path.join(RAW_ROOT, "ti2i")

# ──────────────────────────────────────────────
# MMLU subjects to collect
# ──────────────────────────────────────────────
MMLU_SUBJECTS = [
    # STEM
    "conceptual_physics",
    "high_school_biology",      # 문항 단위 필터링 적용
    "machine_learning",
    # Humanities
    "philosophy",               # 문항 단위 필터링 적용
    "high_school_world_history",
    "moral_disputes",
    # Social Sciences
    "sociology",
    "high_school_psychology",   # 문항 단위 필터링 적용
    "high_school_government_and_politics",
    # Other / Professional
    "management",               # business_ethics 대체
    "human_sexuality",          # virology 대체
    "human_aging",
]

# ──────────────────────────────────────────────
# ScienceQA filter criteria
# ──────────────────────────────────────────────
SCIENCEQA_SUBJECTS = {
    "natural science": ["biology", "physics", "earth science"],
    "social science": ["geography", "history"],
}

# ──────────────────────────────────────────────
# ImgEditBench subtasks
# ──────────────────────────────────────────────
IMGEDIT_SUBTASKS = [
    "add",
    "remove",
    "replace",
    "alter",
    "motion_change",
    "object_extraction",
    "background_replacement",
    "style_tone_transfer",
]
