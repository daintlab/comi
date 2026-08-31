"""VQAv2 open-ended blind test -- scienceqa_filtering/blind_test_full.py와 동일한 구조/모델/
반복횟수/판정기준(3회 중 과반 정답 = language bias = 탈락)을 쓰되, 선지가 없는 open-ended라
아래만 다르게 처리한다:

  - 모델에게 선지 없이 질문만 주고, "이미지 없이도 추측해서 답하라"고 강제한다
    (거부/모른다 답변을 막기 위해 프롬프트에 명시).
  - 정답 판정은 공식 VQAv2 채점 정규화(vqaEval.py의 processPunctuation +
    processDigitArticle을 그대로 이식: 소문자화, 관사 제거, 구두점 제거, 숫자단어<->숫자
    변환, 축약형 정규화)를 거친 뒤, 10명 어노테이터 답 중 3명 이상이 동의한 답과 일치하면
    정답으로 판정한다(공식 VQA soft-accuracy 대신, 사용자가 요청한 단순 이진 판정).

temperature: 원본 scienceqa_filtering/blind_test_full.py에도 temperature가 전혀 지정되어
있지 않음(default 값 사용, 0이 아님) -- 여기서도 동일하게 지정하지 않는다. gpt-5.2는 추론모델
계열이라 애초에 temperature 파라미터를 지원하지 않을 수도 있어서(넣으면 에러), 원본과 똑같이
아예 생략하는 게 맞다.

max_completion_tokens: 원본은 선지 letter 하나만 답하면 되니 5였음. 여기는 자유 서술 답이라
너무 짧으면 잘릴 수 있어 20으로 늘림(원본에 없는 유일한 하이퍼파라미터 차이, 명시적으로 남김).
"""
import json
import re
import time
from collections import Counter
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
MAX_COMPLETION_TOKENS = 20  # 원본(5)과의 유일한 차이 -- 자유서술 답이라 여유를 줌

SYSTEM_PROMPT = """Answer the question. You are not given the image it refers to.
Guess the most likely answer anyway. Reply with the answer only."""


# =====================================================================
# 공식 VQAv2 채점 정규화 (vqaEval.py의 processPunctuation + processDigitArticle을
# 그대로 이식 -- /home/ysk1207/comi_experiments/vqav2_official/vqaEval.py 참고)
# =====================================================================
_CONTRACTIONS = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                  "couldnt": "couldn't", "dont": "don't", "doesnt": "doesn't", "didnt": "didn't",
                  "hasnt": "hasn't", "havent": "haven't", "hes": "he's", "im": "i'm", "ive": "i've",
                  "isnt": "isn't", "itll": "it'll", "shes": "she's", "shouldve": "should've",
                  "shouldnt": "shouldn't", "thats": "that's", "theres": "there's", "theyre": "they're",
                  "theyve": "they've", "wasnt": "wasn't", "werent": "weren't", "whats": "what's",
                  "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "youre": "you're",
                  "youve": "you've", "youll": "you'll"}
_MANUAL_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
               "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_PUNCT = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
          '>', '<', '@', '`', ',', '?', '!']


def _process_punctuation(text: str) -> str:
    out = text
    for p in _PUNCT:
        if (p + ' ' in text) or (' ' + p in text) or (_COMMA_STRIP.search(text) is not None):
            out = out.replace(p, '')
        else:
            out = out.replace(p, ' ')
    out = _PERIOD_STRIP.sub("", out)
    return out


def normalize_vqa_answer(text: str) -> str:
    text = (text or "").replace("\n", " ").replace("\t", " ").strip()
    text = _process_punctuation(text)
    words = text.lower().split()
    out = []
    for w in words:
        w = _MANUAL_MAP.get(w, w)
        if w not in _ARTICLES:
            out.append(w)
    out = [_CONTRACTIONS.get(w, w) for w in out]
    return " ".join(out)


# =====================================================================
# Blind test
# =====================================================================
def ask_blind(item):
    user_text = f"QUESTION: {item['question']}"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()


def process_item(item):
    picks_raw = []
    for _ in range(N_RUNS):
        for attempt in range(3):
            try:
                picks_raw.append(ask_blind(item))
                break
            except Exception:
                if attempt == 2:
                    picks_raw.append("ERROR")
                time.sleep(1)

    gt_norm = [normalize_vqa_answer(a["answer"]) for a in item["answers"]]
    gt_counts = Counter(gt_norm)
    accepted = {a for a, c in gt_counts.items() if c >= 3}  # 3명 이상 동의한 답만 정답으로 인정

    picks_norm = [normalize_vqa_answer(p) if p != "ERROR" else "ERROR" for p in picks_raw]
    correct_flags = [p in accepted for p in picks_norm]
    n_correct = sum(correct_flags)
    blind_majority_correct = n_correct > N_RUNS / 2

    item["blind_picks_raw"] = picks_raw
    item["blind_picks_normalized"] = picks_norm
    item["blind_correct_flags"] = correct_flags
    item["blind_correct_count"] = n_correct
    item["blind_majority_correct"] = blind_majority_correct
    item["survived"] = not blind_majority_correct
    item["accepted_answers"] = sorted(accepted)
    return item


def run(input_path: str, output_path: str):
    items = json.loads(Path(input_path).read_text())
    print(f"Running BLIND test ({N_RUNS} runs/item, {WORKERS} workers) on {len(items)} items...")

    results = [None] * len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process_item, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(items):
                n_surv = sum(1 for r in results if r and r.get("survived"))
                print(f"  {done}/{len(items)} done, survived so far: {n_surv}")

    Path(output_path).write_text(json.dumps(results, indent=2, ensure_ascii=False))

    n_survived = sum(1 for x in results if x["survived"])
    print(f"\nTOTAL: {len(results)}, survived = {n_survived}, filtered out = {len(results) - n_survived}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    run(args.input, args.output)
