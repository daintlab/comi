# Anole `run_inputs_single_load.py` 분석

## 개요

Chameleon(Anole) 멀티모달 모델의 **단일 프로세스 배치 추론 스크립트**.
기존 `run_merged_all_in_batches.sh`가 아이템마다 새 Python 프로세스를 생성해 모델을 매번 재로딩하던 비효율을 해결한다.

- 위치: `/nas/home/ysk1207/cams-benchmark/anole/anole/run_inputs_single_load.py`

## 아키텍처

```
[입력 JSON] → 모델 1회 로드 → 배치 단위 반복 → 아이템별 run_one() → [출력 디렉토리]
```

## 입력 형식

`convert_merged_all_to_inputs.py`가 생성하는 JSON:

```json
{
  "items": [
    {
      "task_id": "gsm8k_test_00000",
      "source": "gsm8k",
      "benchmark": "gsm8k",
      "expectation": "text",
      "input": [
        {"type": "text", "content": "질문 텍스트"},
        {"type": "image", "content": "/path/to/image.jpg"},
        {"type": "sentinel", "content": "<END-OF-TURN>"}
      ]
    }
  ]
}
```

## 핵심 함수

### `build_prompt_ui(input_segs, no_eot)`
입력 세그먼트(text/image/sentinel)를 Chameleon 모델의 `batch_prompt_ui` 형식으로 변환.
`--no-eot`가 아니면 `<END-OF-TURN>` 토큰 자동 추가.

### `run_one(model, options, item_dir, input_segs, ...)`
1. `model.stream()`으로 토큰 스트리밍 생성
2. 첫 생성 토큰의 logits에서 **BOI(Begin-of-Image) 확률** 계산 → `boi_prob`
3. `split_interleaved()`로 출력 토큰을 텍스트/이미지 구간 분리
4. 텍스트는 디코딩, 이미지(1024 토큰)는 PNG 저장
5. 결과를 `output_rXX.json`에 저장

### `split_interleaved(tokens, boi, eoi)`
생성된 토큰 시퀀스를 `boi`/`eoi` 토큰 기준으로 텍스트 구간과 이미지 구간으로 분할.

## 출력 구조

```
out_root/
  batch_0000/
    item_0000/
      input.json          # 원본 입력 세그먼트
      meta.json           # task_id, source, benchmark, expectation
      output_r00.json     # [{"type":"meta","boi_prob":0.xx}, {"type":"text","content":"..."}, ...]
      r00_1.png           # 생성된 이미지 (있을 경우)
    item_0001/
      ...
  batch_0001/
    ...
```

`--flat` 옵션 시 `batch_XXXX/` 계층 없이 `item_XXXX/`로 직접 저장.

## 주요 CLI 옵션

| 옵션 | 설명 |
|------|------|
| `--input_json` | 입력 JSON 파일 경로 (필수) |
| `--out_root` | 출력 디렉토리 (필수) |
| `--batch_size` | 배치 크기 (기본 10) |
| `--text-only` | 텍스트만 생성 |
| `--image-only` | 이미지만 생성 |
| `--no-eot` | `<END-OF-TURN>` 토큰 미추가 |
| `--repeat N` | 동일 입력에 대해 N회 반복 생성 |
| `--flat` | 배치 계층 없이 플랫하게 저장 |

## 핵심 특징

1. **BOI 확률 기록**: 첫 생성 토큰에서 BOI 확률을 계산해 모델이 이미지 생성 의도가 있었는지 분석 가능
2. **모달리티 자율 결정**: Chameleon은 `boi`/`eoi` 토큰을 자연스럽게 생성할 수 있어, `--text-only`/`--image-only` 없이도 모델이 스스로 출력 모달리티를 결정
3. **부분 이미지 처리**: 이미지 토큰이 1024개 미만이면 디코딩 건너뜀
4. **에러 격리**: 개별 아이템 에러 시 `error.json` 기록 후 다음 아이템 계속 처리
