"""blind test 결과(survived==True인 문항)만 골라 comi-bench 포맷
(task_id, source, original_question, task, gold_answer, gold_modality, image_path, metadata)
으로 저장하고, 필요한 COCO 이미지를 다운로드한다.

사용법:
    python finalize_survivors.py --src blind_test_results_random2400.json \\
        --out-data vqav2_blindtest_survived_1541.json \\
        --out-report vqav2_blindtest_survived_1541_report.json \\
        --dest-dir /path/to/vqav2_validation_images --staged-dir /path/to/staging
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import time
import urllib.request

CONCURRENCY = 6
MAX_RETRIES = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="prepare_pool.py + blind_test_full.py 산출물")
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--dest-dir", required=True, help="최종 이미지가 놓일 디렉토리")
    ap.add_argument("--staged-dir", required=True, help="다운로드 스테이징(임시) 디렉토리")
    args = ap.parse_args()

    SRC, OUT_DATA, OUT_REPORT = args.src, args.out_data, args.out_report
    DEST_DIR, STAGED_DIR = args.dest_dir, args.staged_dir

    results = json.load(open(SRC))
    survived = [r for r in results if r["survived"]]
    print(f"생존 {len(survived)}/{len(results)}건")

    records = []
    for r in survived:
        qid = r["question_id"]
        records.append({
            "task_id": f"vqav2bt_{qid}",
            "source": "VQAv2",
            "original_question": r["question"],
            "task": "ti2t",
            "gold_answer": r["multiple_choice_answer"],
            "gold_modality": "text",
            "image_path": os.path.join(DEST_DIR, f"{qid}.jpg"),
            "metadata": {
                "question_id": qid,
                "image_id": r["image_id"],
                "answer_type": r["answer_type"],
                "question_type": r["question_type"],
                "answers": r["answers"],
                "accepted_answers": r["accepted_answers"],
                "blind_picks_raw": r["blind_picks_raw"],
                "blind_picks_normalized": r["blind_picks_normalized"],
                "blind_correct_count": r["blind_correct_count"],
            },
        })

    # ---- 이미지 다운로드 ----
    os.makedirs(STAGED_DIR, exist_ok=True)
    needed_image_ids = sorted({rec["metadata"]["image_id"] for rec in records})
    already_staged = {int(f[:-4]) for f in os.listdir(STAGED_DIR) if f.endswith(".jpg")}
    to_download = [iid for iid in needed_image_ids if iid not in already_staged]
    print(f"필요한 고유 이미지: {len(needed_image_ids)}개, 이미 보유: "
          f"{len(needed_image_ids) - len(to_download)}개, 신규 다운로드: {len(to_download)}개", flush=True)

    def fetch_one(image_id):
        url = f"http://images.cocodataset.org/val2014/COCO_val2014_{image_id:012d}.jpg"
        dest = os.path.join(STAGED_DIR, f"{image_id}.jpg")
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "comi-bench-dataset-repair/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
                    shutil.copyfileobj(resp, f)
                if os.path.getsize(dest) == 0:
                    raise IOError("downloaded 0 bytes")
                return image_id, "downloaded", None
            except Exception as ex:
                last_err = str(ex)
                if os.path.exists(dest):
                    os.remove(dest)
                time.sleep(2 ** attempt)
        return image_id, "failed", last_err

    fetch_status = {}
    if to_download:
        done = 0
        with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futs = {ex.submit(fetch_one, iid): iid for iid in to_download}
            for fut in cf.as_completed(futs):
                iid, status, err = fut.result()
                fetch_status[iid] = (status, err)
                done += 1
                if done % 100 == 0 or done == len(to_download):
                    print(f"  다운로드 {done}/{len(to_download)}", flush=True)
    n_downloaded = sum(1 for s, _ in fetch_status.values() if s == "downloaded")
    n_failed = sum(1 for s, _ in fetch_status.values() if s == "failed")
    print(f"다운로드 결과: 성공={n_downloaded} 실패={n_failed}", flush=True)

    os.makedirs(DEST_DIR, exist_ok=True)
    copy_failed = []
    copied = 0
    for rec in records:
        iid = rec["metadata"]["image_id"]
        src = os.path.join(STAGED_DIR, f"{iid}.jpg")
        if not os.path.exists(src):
            copy_failed.append({"task_id": rec["task_id"], "image_id": iid, "reason": "never staged"})
            continue
        try:
            if not (os.path.exists(rec["image_path"]) and os.path.getsize(rec["image_path"]) > 0):
                shutil.copyfile(src, rec["image_path"])
            copied += 1
        except Exception as ex:
            copy_failed.append({"task_id": rec["task_id"], "image_id": iid, "reason": f"copy failed: {ex}"})
    print(f"복사 결과: {copied}/{len(records)}, 실패 {len(copy_failed)}", flush=True)

    from PIL import Image
    valid, invalid = 0, []
    for rec in records:
        p = rec["image_path"]
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            invalid.append({"task_id": rec["task_id"], "path": p, "reason": "missing_or_zero_bytes"})
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            valid += 1
        except Exception as ex:
            invalid.append({"task_id": rec["task_id"], "path": p, "reason": f"invalid_jpeg: {ex}"})
    print(f"검증 결과: valid={valid}/{len(records)}, invalid={len(invalid)}", flush=True)

    json.dump({"total_questions": len(records), "datasets": ["VQAv2"], "results": records},
               open(OUT_DATA, "w"), ensure_ascii=False, indent=2)

    report = {
        "n_survived": len(records),
        "n_unique_images": len(needed_image_ids),
        "n_downloaded": n_downloaded, "n_download_failed": n_failed,
        "n_copy_failed": len(copy_failed), "copy_failed": copy_failed,
        "n_valid_images": valid, "n_invalid_images": len(invalid), "invalid_images": invalid,
        "out_data_path": OUT_DATA,
    }
    json.dump(report, open(OUT_REPORT, "w"), ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT_DATA}")
    print(f"리포트: {OUT_REPORT}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
