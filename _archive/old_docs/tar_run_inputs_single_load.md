# Tar-7B `run_inputs_single_load.py` 설계 문서

## 개요

Tar-7B 모델의 **단일 로드 배치 멀티모달 추론 스크립트**.
사용자가 외부에서 출력 모달리티(text/image)를 지정하지 않고, **모델이 프롬프트만 보고 자율적으로 결정**한다.

- 위치: `/home/ysk1207/2026/cams/Tar/run_inputs_single_load.py`

## Anole vs Tar 모달리티 결정 방식 비교

| 항목 | Anole (Chameleon) | Tar-7B |
|------|-------------------|--------|
| 이미지 시작 토큰 | `boi` (vocab에 내장) | `<im_start>` (special token으로 추가) |
| 이미지 코드 형식 | 연속 정수 토큰 (1024개) | `<I0>` ~ `<I65535>` 텍스트 토큰 |
| 이미지 디코더 | 내장 VQ decoder | TA-Tok → AR-DTok → VQVAE 파이프라인 |
| 모달리티 결정 | `boi` 토큰 생성 여부 | `<im_start>` 토큰 생성 여부 |
| 모달리티 확률 | `boi_prob` (첫 토큰의 BOI 확률) | `im_start_prob` (첫 토큰의 `<im_start>` 확률) |

## 아키텍처

```
입력 JSON
  ↓
모델 1회 로드 (LLM + TA-Tok + AR-DTok + VQVAE)
  ↓
아이템별 처리:
  1. 입력 이미지 → TA-Tok → <I{n}> 토큰으로 인코딩
  2. 프롬프트 구성 (Qwen2 chat template)
  3. im_start_prob 계산 (첫 토큰에서 <im_start> 확률)
  4. 자유 생성 (<im_start> 강제 추가 없음)
  5. 출력 파싱:
     - <im_start> 발견 → <I{n}> 추출 → AR-DTok → VQVAE → PNG
     - <im_start> 미발견 → 텍스트 출력
  ↓
출력 디렉토리
```

## 핵심 함수

### `encode_image_to_tokens(image_path, visual_tokenizer, device)`
입력 이미지를 TA-Tok으로 인코딩하여 `<I0><I123>...` 형태의 문자열로 변환.

### `build_prompt(input_segs, tokenizer, visual_tokenizer, device)`
입력 세그먼트를 Qwen2 chat template으로 변환.
- text → 그대로 삽입
- image → TA-Tok 인코딩 후 `<I{n}>` 토큰 문자열로 삽입
- **`<im_start>` 미추가** — 모델이 자율 결정

### `compute_im_start_prob(model, tokenizer, input_ids, device)`
Anole의 `boi_prob`에 대응. 첫 생성 토큰 위치에서 `<im_start>` 토큰의 확률을 계산.

### `parse_output(generated_text, image_decoder, ...)`
생성된 텍스트에서:
1. `<im_start>` 존재 여부 확인
2. 있으면 `<I{n}>` 패턴 추출 → `seq_len`에 맞춰 pad/truncate → AR-DTok → VQVAE → PNG 저장
3. 없으면 텍스트 세그먼트로 반환

### `run_one(model, tokenizer, visual_tokenizer, image_decoder, ...)`
단일 아이템 추론. 프롬프트 구성 → im_start_prob 계산 → 생성 → 파싱 → JSON 저장.

## 출력 형식

```json
[
  {"type": "meta", "im_start_prob": 0.000123},
  {"type": "text", "content": "The answer is 42."}
]
```

또는 이미지 생성 시:

```json
[
  {"type": "meta", "im_start_prob": 0.987654},
  {"type": "image", "path": "/path/to/r00_gen.png", "num_codes": 729}
]
```

## CLI 사용법

```bash
cd /home/ysk1207/2026/cams/Tar

# 기본 실행 (Tar-7B, scale=1, 512px)
python run_inputs_single_load.py \
  --input_json /path/to/items.json \
  --out_root /path/to/output \
  --model_path csuhan/Tar-7B

# 커스텀 설정
python run_inputs_single_load.py \
  --input_json /path/to/items.json \
  --out_root /path/to/output \
  --model_path csuhan/Tar-7B \
  --scale 2 --seq_len 729 \
  --cfg_scale 4.0 \
  --repeat 3 \
  --flat \
  --print-raw
```

## 주요 CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model_path` | `csuhan/Tar-7B` | LLM 모델 경로 |
| `--scale` | `1` | 이미지 스케일 (0=small, 1=medium, 2=large) |
| `--seq_len` | `729` | 이미지 코드 토큰 수 |
| `--cfg_scale` | `4.0` | Classifier-free guidance scale |
| `--temperature` | `1.0` | 샘플링 temperature |
| `--top_p` | `0.95` | Top-p 샘플링 |
| `--top_k` | `1200` | Top-k 샘플링 |
| `--repeat` | `1` | 아이템당 반복 생성 횟수 |
| `--flat` | off | 배치 계층 없이 플랫 저장 |
| `--print-raw` | off | 디버깅용 raw 생성 텍스트 출력 |

## 주의사항

1. **모달리티 자율 결정의 한계**: Tar-7B이 학습 시 `<im_start>`를 항상 assistant 응답의 일부로 생성하도록 훈련되었는지에 따라, 자유 생성 시 이미지를 생성하지 않을 수 있음. 이 경우 `--print-raw` 플래그로 raw output을 확인하여 디버깅 필요.

2. **입력 이미지 인코딩**: 입력에 이미지가 포함된 경우(ti2t, ti2i 태스크), TA-Tok으로 인코딩하여 `<I{n}>` 토큰으로 변환. 이는 Anole에서 이미지를 Chameleon의 image tokenizer로 인코딩하는 것에 대응.

3. **모델 웨이트 자동 다운로드**: `--ar_path`, `--encoder_path`, `--decoder_path`를 생략하면 HuggingFace Hub에서 자동 다운로드.
