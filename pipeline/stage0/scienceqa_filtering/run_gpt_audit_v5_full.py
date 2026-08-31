import json
import base64
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
WORKERS = 12

# Approved v4 system prompt, unchanged. User message omits CHOICES (v5 fix: judge
# never sees the choices, only question + image).
SYSTEM_PROMPT = """Your task is to review a series of multiple-choice questions and evaluate their ability to
be answered without the provided answer choices. Each question is shown together with an
image; base your evaluation on the question together with the image, not on the question
text alone. For questions that begin with an incomplete sentence (e.g., "During swallowing, ..."),
use your knowledge to attempt to complete the sentence accurately. For direct questions that ask
for specific information or identification (e.g., "Which of the following structures is part of
the small intestine?"), assess whether the question is formulated clearly enough that an informed
answer can be given without seeing the multiple-choice options. For mathematical or analytical
questions (e.g., "Find all cosets of the subgroup 4Z of 2Z"), determine if the question provides
enough context and information for a solution to be formulated without additional options.
Please follow this format for your evaluation:
QUESTION: [Insert the question here] REASON: Briefly explain why the choices can or cannot be
removed, given the question and the image. VERDICT: Respond with "YES" if the question is clear
and can be directly answered based on its content alone, or "NO" if it relies on the answer
choices to be understood or answered. Your response should include both the reason and the
verdict, in that order."""


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def judge(item):
    img_b64 = encode_image(item["image_path"])
    user_text = f"QUESTION: {item['question']}"  # CHOICES intentionally omitted
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            },
        ],
        max_completion_tokens=300,
    )
    text = (resp.choices[0].message.content or "").strip()
    upper = text.upper()
    if "VERDICT: YES" in upper:
        v = "YES"
    elif "VERDICT: NO" in upper:
        v = "NO"
    elif upper.rstrip().endswith("YES"):
        v = "YES"
    elif upper.rstrip().endswith("NO"):
        v = "NO"
    else:
        v = "UNPARSED"
    return v, text


def process(item):
    for attempt in range(3):
        try:
            v, reasoning = judge(item)
            break
        except Exception as e:
            v, reasoning = f"ERROR:{e}", ""
            time.sleep(1)
    item["gpt_verdict_v5"] = v
    item["gpt_reasoning_v5"] = reasoning
    return item


def main():
    # gpt_audit_survived_653.json(Step1 생존자만 따로 저장한 파일)은 실제로는 별도
    # 스크립트 없이 blind_test_results_full.json에서 survived==True만 골라낸 것이었다.
    # 그 중간 파일 자체는 안 남아있어서, 여기서 그 필터를 그대로 재현한다.
    blind_results = json.loads((Path(__file__).parent / "blind_test_results_full.json").read_text())
    items = [r for r in blind_results if r["survived"]]
    assert len(items) == 653, f"기대한 653건이 아니라 {len(items)}건임 -- 상류 파일이 바뀌었는지 확인"
    print(f"Running v5 (choices hidden) on {len(items)} survivors, {WORKERS} workers...")

    results = [None] * len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(items):
                n_yes = sum(1 for r in results if r and r.get("gpt_verdict_v5") == "YES")
                print(f"  {done}/{len(items)} done, YES so far: {n_yes}")

    out = Path(__file__).parent / "gpt_audit_v5_survived_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    n_yes = sum(1 for x in results if x["gpt_verdict_v5"] == "YES")
    print(f"\nTOTAL: {len(results)}, YES(final candidates) = {n_yes}, NO = {len(results)-n_yes}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
