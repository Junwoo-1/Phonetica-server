# voice-pron

한국어 음성 입력을 받아 **자모(초성/중성/종성) 단위 발음 정확도**를 측정하는 Python API.

별도의 정답 텍스트(reference) 없이 동작한다. ASR이 인식한 텍스트의 표준 발음 자모 시퀀스(G2P 결과)와, 같은 음성에서 phoneme 모델이 추출한 실제 발음 자모 시퀀스를 가중 Needleman-Wunsch로 정렬해 클라이언트(UI)에서 음절 단위로 조립할 수 있도록 음절(Syllable) 및 자모별/위치별 개별 점수와 종합 점수를 산출한다.

## 아키텍처

```
audio file
   │
   ├─► faster-whisper (large-v3, ko)        ─► 인식 텍스트 ─► g2pkk + jamo ─► ref_jamo[]
   │                                                                          │
   └─► Kkonjeong/wav2vec2-base-korean (CTC) ─► hyp_jamo[]                     │
                                                  │                           │
                                                  └─► Needleman-Wunsch ◄──────┘
                                                            │
                                                            ▼
                                                음절 단위 자모 맵핑 및 개별 점수 산출
                                                            │
                                                            ▼
                                                    SSE 이벤트 스트림
```

## 요구 환경

- Linux + CUDA 13.x (RTX 5090 등 Blackwell 권장)
- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/) 0.6+

## 설치

```bash
uv sync
```

PyTorch 2.11(cu130 stable) 휠을 자동으로 받아온다. `pyproject.toml`의 `[tool.uv.sources]`에 cu130 인덱스가 설정되어 있다.

## 모델 사전 워밍업

첫 요청 지연을 줄이기 위해 모델을 미리 다운로드한다 (Whisper-large-v3 약 3GB).

```bash
uv run python scripts/warmup_models.py
```

## 서버 실행

```bash
uv run uvicorn voice_pron.main:app --host 0.0.0.0 --port 8000
```

엔드포인트:
- `POST /pronounce` — 멀티파트 `file` 필드로 wav/mp3 업로드, SSE 스트림 응답
- `GET /health` — 헬스 체크

## SSE 이벤트 순서

```
accepted → audio_loaded → asr_progress* → asr_completed
        → g2p_completed → phoneme_completed
        → alignment_completed → score → done
```

이상 발생 시 `error` 이벤트로 종료. 각 이벤트 envelope:
```json
{ "event": "score", "data": { ... }, "ts": "...", "request_id": "uuid" }
```

`score` 페이로드 예:
```json
{
  "recognized_word": "닭볶이",
  "whisper_text": "닥뽀끼",
  "overall_score": 95.77,
  "detailed_jamos": [
    { "syl": 0, "pos": "onset", "char": "ㄷ", "score": 90.0 },
    { "syl": 0, "pos": "nucleus", "char": "ㅏ", "score": 100.0 },
    { "syl": 0, "pos": "coda", "char": "ㄱ", "score": 85.0 },
    { "syl": 1, "pos": "onset", "char": "ㅃ", "score": 95.0 },
    { "syl": 1, "pos": "nucleus", "char": "ㅗ", "score": 100.0 },
    { "syl": 2, "pos": "onset", "char": "ㄲ", "score": 80.0 },
    { "syl": 2, "pos": "nucleus", "char": "ㅣ", "score": 100.0 }
  ]
}
```
- syl: 음절 인덱스 (클라이언트에서 한 글자 단위 블록을 생성하는 기준)
- pos: 자모 위치 (onset 초성, nucleus 중성, coda 종성)
- char: 매핑된 실제 자모 텍스트

## CLI로 단일 파일 검증

```bash
uv run python scripts/manual_check.py path/to/sample.wav
```

## 테스트

```bash
uv run pytest                         # 전체
uv run pytest tests/unit -v          # 단위 테스트만
uv run pytest tests/integration -v   # mock 기반 통합 테스트
```

## 점수 계산

- **자모 단위 가중 NW 정렬**
  - 동일 평/경/격음 그룹 sub: 0.3 (예: ㄱ↔ㄲ↔ㅋ)
  - 동일 조음위치 sub: 0.6
  - 인접 모음 sub: 0.3, 단모음↔이중모음: 0.6
  - 자음↔모음 sub: 1.5
  - coda del: 0.5, onset del: 1.0, nucleus del: 1.2, ins: 1.0
- **종합 점수**: `100 * max(0, 1 - total_cost / Σref_weight)`
- **PER**: `(sub + ins + del) / N_ref`
- 음절 기반 매핑: 정렬된 자모 데이터를 유니티 등 클라이언트 UI에서 조립할 수 있도록 원래의 음절 인덱스(syl) 및 위치(pos) 기준으로 재그룹화하여 detailed_jamos 리스트로 반환한다.

## 한계와 엣지 케이스

- ASR이 단어를 인식하지 못하면 G2P 기준이 없어 평가 불가 — `error("NO_SPEECH" | "NO_HANGUL")` 반환.
- 무음 비율 ≥ 95% 입력은 즉시 거부.
- 60초 초과 음성은 `413 Payload Too Large`.
- 영어/숫자가 섞인 발화는 비한글 부분이 평가에서 제외된다.
- 사투리/외래어는 g2pkk의 표준 발음과 차이가 있어 실제보다 낮게 채점될 수 있다.
- 단일 GPU 동시 추론은 `Semaphore(VOICE_CONCURRENCY)`로 직렬화 — 기본 1.

## 환경 변수

`.env` 또는 셸 환경에서 설정 가능 (`VOICE_` prefix):

| 변수 | 기본 | 설명 |
|---|---|---|
| `VOICE_WHISPER_MODEL` | `large-v3` | faster-whisper 모델 크기 |
| `VOICE_WHISPER_DEVICE` | `cuda` | `cuda` 또는 `cpu` |
| `VOICE_WHISPER_COMPUTE_TYPE` | `float16` | `float16` / `int8` 등 |
| `VOICE_PHONEME_MODEL` | `Kkonjeong/wav2vec2-base-korean` | HF 모델 ID |
| `VOICE_PHONEME_DEVICE` | `cuda` | |
| `VOICE_MAX_DURATION_SEC` | `60` | 최대 입력 길이 |
| `VOICE_MAX_FILE_BYTES` | `20971520` | 최대 파일 크기 (20MB) |
| `VOICE_SILENCE_THRESHOLD` | `0.95` | 무음 거부 임계값 |
| `VOICE_LOW_LANGUAGE_PROB` | `0.7` | 한국어 확률 경고 임계값 |
| `VOICE_CONCURRENCY` | `1` | GPU 동시 처리 수 |

## 디렉토리

```
src/voice_pron/
├── main.py              # FastAPI + lifespan
├── config.py            # pydantic-settings
├── api/
│   ├── routes.py        # /pronounce SSE
│   └── sse.py           # 이벤트 envelope
├── audio/loader.py      # 16kHz mono 변환, silence_ratio
├── asr/whisper_asr.py   # faster-whisper async wrapper
├── g2p/
│   ├── pronouncer.py    # g2pkk + 정규화
│   └── jamo_utils.py    # 자모 분해/위치 태깅
├── phoneme/wav2vec_jamo.py  # Kkonjeong CTC + 자모 토큰화
├── align/
│   ├── needleman.py     # 가중 NW
│   └── cost.py          # 자모 거리 행렬
├── scoring/score.py     # 점수 산정 및 detailed_jamos 맵핑
└── pipeline/orchestrator.py  # SSE 이벤트 yield
```
