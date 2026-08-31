import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import os

# .env 경로를 이 저장소 밖(예: /nas/home/xxx/.env)에 두고 있다면 COMI_DOTENV_PATH로 지정할 것.
# 지정 안 하면 python-dotenv가 CWD부터 위로 탐색한다 -- 여러 개의 .env가 있는 환경이면
# 엉뚱한(오래된 키가 든) .env를 집을 수 있으니 주의.
load_dotenv(os.getenv("COMI_DOTENV_PATH"))
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-5.2"
N_RUNS = 3
WORKERS = 16

SYSTEM_PROMPT = """You will be given a multiple-choice question with its answer choices, but no
image. Answer the question using only the text given. Choose the single best option.

Respond with exactly the letter of your chosen option (A, B, C, D, or E), nothing else."""


def ask_blind(item):
    letters = ["A", "B", "C", "D", "E"]
    choice_lines = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(item["choices"]))
    user_text = f"QUESTION: {item['question']}\n\n{choice_lines}"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        max_completion_tokens=5,
    )
    text = (resp.choices[0].message.content or "").strip().upper()
    letters_used = ["A", "B", "C", "D", "E"][: len(item["choices"])]
    for L in letters_used:
        if text.startswith(L):
            return L
    return "UNPARSED"


def process_item(item):
    letters = ["A", "B", "C", "D", "E"][: len(item["choices"])]
    gold_letter = letters[item["choices"].index(item["answer"])]
    picks = []
    for _ in range(N_RUNS):
        for attempt in range(3):
            try:
                picks.append(ask_blind(item))
                break
            except Exception as e:
                if attempt == 2:
                    picks.append(f"ERROR")
                time.sleep(1)
    n_correct = sum(1 for p in picks if p == gold_letter)
    blind_majority_correct = n_correct > N_RUNS / 2
    item["blind_picks"] = picks
    item["blind_gold_letter"] = gold_letter
    item["blind_correct_count"] = n_correct
    item["blind_majority_correct"] = blind_majority_correct
    item["survived"] = not blind_majority_correct
    return item


def main():
    src = Path(__file__).parent / "gpt_audit_full.json"
    items = json.loads(src.read_text())
    print(f"Running BLIND test ({N_RUNS} runs/item, {WORKERS} workers) on {len(items)} items...")

    results = [None] * len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process_item, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % 100 == 0 or done == len(items):
                n_surv = sum(1 for r in results if r and r.get("survived"))
                print(f"  {done}/{len(items)} done, survived so far: {n_surv}")

    out = Path(__file__).parent / "blind_test_results_full.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    n_survived = sum(1 for x in results if x["survived"])
    print(f"\nTOTAL: {len(results)}, survived = {n_survived}, filtered out = {len(results) - n_survived}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
