import json
import os
import shutil
from tqdm import tqdm

SRC_JSON = "/nas/datahub/ScienceQA/problems.json"
SRC_IMG_DIR = "/nas/datahub/ScienceQA/test"
OUT_DIR = "/nas/home/ysk1207/cams-benchmark/aug_bench_v2/0_collecting_dataset/raw/ti2t/scienceqa"
IMG_DIR = os.path.join(OUT_DIR, "images")
OUT_JSONL = os.path.join(OUT_DIR, "test.jsonl")

ALLOWED_SUBJECTS = {"natural science", "social science"}
ALLOWED_TOPICS = {"biology", "physics", "earth science", "geography", "history"}

os.makedirs(IMG_DIR, exist_ok=True)

print("Loading problems.json...")
with open(SRC_JSON, encoding="utf-8") as f:
    problems = json.load(f)
print(f"Total samples: {len(problems)}")

# 필터링
filtered = {
    pid: v for pid, v in problems.items()
    if v.get("split") == "test"
    and v.get("image") is not None
    and v.get("subject") in ALLOWED_SUBJECTS
    and v.get("topic") in ALLOWED_TOPICS
}
print(f"After filter (test / image / subject / topic): {len(filtered)}")

count = 0
skipped = 0
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for pid, v in tqdm(filtered.items()):
        # 이미지 복사
        src_img = os.path.join(SRC_IMG_DIR, str(pid), "image.png")
        img_name = f"{pid}.png"
        dst_img = os.path.join(IMG_DIR, img_name)

        if not os.path.exists(src_img):
            skipped += 1
            continue

        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)

        record = {
            "problem_id": pid,
            "question": v["question"],
            "choices": v["choices"],
            "answer": v["answer"],
            "hint": v.get("hint", ""),
            "subject": v["subject"],
            "topic": v["topic"],
            "category": v.get("category", ""),
            "skill": v.get("skill", ""),
            "grade": v.get("grade", ""),
            "lecture": v.get("lecture", ""),
            "solution": v.get("solution", ""),
            "image_path": os.path.join("images", img_name),
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

print(f"Done: {count} samples -> {OUT_JSONL}")
if skipped:
    print(f"Skipped (image not found): {skipped}")

# 분포 확인
print("\n--- Subject/Topic 분포 ---")
from collections import Counter
records = [json.loads(l) for l in open(OUT_JSONL)]
topic_counter = Counter(f"{r['subject']} / {r['topic']}" for r in records)
for k, v in sorted(topic_counter.items()):
    print(f"  {k}: {v}")