import pandas as pd
import json
import os
import ast
from tqdm import tqdm

SRC_CSV = "/nas/datahub/ti2t/vqav2/validation.csv"
OUT_DIR = "/nas/home/ysk1207/cams-benchmark/aug_bench_v2/0_collecting_dataset/raw/ti2t/vqav2"
IMG_DIR = os.path.join(OUT_DIR, "images")
OUT_JSONL = os.path.join(OUT_DIR, "validation.jsonl")

os.makedirs(IMG_DIR, exist_ok=True)

print("Loading CSV...")
df = pd.read_csv(SRC_CSV)
print(f"Total rows: {len(df)}")

# answer_type == "other" 필터링
df_filtered = df[df["answer_type"] == "other"].reset_index(drop=True)
print(f"After filter (answer_type=other): {len(df_filtered)}")

count = 0
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        # 이미지 저장
        img_name = f"{int(row['image_id']):012d}.jpg"
        img_path = os.path.join(IMG_DIR, img_name)

        if not os.path.exists(img_path):
            try:
                img_dict = ast.literal_eval(row["image"])
                img_bytes = img_dict["bytes"]
                with open(img_path, "wb") as img_f:
                    img_f.write(img_bytes)
            except Exception as e:
                print(f"[warn] image_id={row['image_id']}: {e}")
                continue

        # jsonl 저장 (image 컬럼 제외, 경로 추가)
        record = {
            "question_id": row["question_id"],
            "image_id": row["image_id"],
            "question": row["question"],
            "answer_type": row["answer_type"],
            "question_type": row["question_type"],
            "multiple_choice_answer": row["multiple_choice_answer"],
            "answers": row["answers"],
            "image_path": os.path.join("images", img_name),
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

print(f"Done: {count} samples -> {OUT_JSONL}")