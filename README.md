# voice-pron

한국어 음성 입력과 **후보 단어 리스트**를 받아 **자모(초성/중성/종성) 단위 발음 정확도**를 측정하는 Python API.

클라이언트가 보낸 후보 단어들 중 phoneme 모델 출력과 가장 잘 맞는 단어를 정답으로 선택하고, 그 단어의 표준 발음 자모 시퀀스와 실제 발음 자모 시퀀스를 가중 Needleman-Wunsch로 정렬해 음절/위치별 개별 점수와 종합 점수를 산출한다. ref 좌표계의 채점 결과(`detailed_jamos`)와 hyp 좌표계의 실제 발음(`heard_jamos`)을 함께 반환해, 클라이언트가 음절 박스 단위로 정답/실제 발음을 시각화할 수 있다.

## 아키텍처

```
audio file + candidates (후보 단어 리스트)
   │
   └─► Kkonjeong/wav2vec2-base-korean (CTC) ─► hyp_jamo[]
                                                   │
   ┌──────────── 후보 단어들 ─────────────────────┐  │
   │ candidate_1 ─► g2pkk + jamo ─► ref_jamo[1]   │  │
   │ candidate_2 ─► g2pkk + jamo ─► ref_jamo[2]   │  │
   │ ...                                          │  │
   └──────────────────────────────────────────────┘  │
                  │                                  │
                  └─► 각 후보별 Needleman-Wunsch ◄───┘
                                  │
                                  ▼
                  weighted error rate 최저 후보 선택
                                  │
                                  ▼
              음절 단위 자모 매핑 + heard_jamos
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

첫 요청 지연을 줄이기 위해 phoneme 모델을 미리 다운로드한다.

```bash
uv run python scripts/download_models.py
```

## WordBank.json

후보 단어들의 자모 시퀀스를 서버 시작 시 미리 계산해두는 캐시 파일. 리포지토리 루트에 위치하며, 형식:

```json
{
  "wordList": [
    { "word": "사과", "pronunciation": "사과" },
    { "word": "닭볶이", "pronunciation": "닥뽀끼" }
  ]
}
```

`pronunciation`이 제공된 경우 g2pkk를 건너뛰고 해당 발음형을 직접 자모 분해한다. g2pkk가 잘못 처리하는 단어("닭볶이" 등)를 수동 발음형으로 우회하는 escape hatch.

`WordBank.json`이 없으면 캐시는 비어있는 상태로 시작하며, 모든 후보는 요청 시 G2P를 거친다.

## 서버 실행

```bash
uv run uvicorn voice_pron.main:app --host 0.0.0.0 --port 8000
```

엔드포인트:
- `POST /pronounce` — 멀티파트 폼 데이터, SSE 스트림 응답
  - `file` (File): wav/mp3 오디오 파일
  - `candidates` (String): 콤마로 구분된 후보 단어 (예: `"닭볶이,포도,오렌지"`, 최대 20개)
- `GET /health` — 헬스 체크

## SSE 이벤트 순서

```
accepted → audio_loaded → phoneme_completed
        → g2p_completed → alignment_completed → score → done
```

이상 발생 시 `error` 이벤트로 종료. 각 이벤트 envelope:
```json
{ "event": "score", "data": { ... }, "ts": "...", "request_id": "uuid" }
```

`score` 페이로드 예:
```json
{
  "recognized_word": "닭볶이",
  "overall_score": 95.77,
  "detailed_jamos": [
    { "syl": 0, "pos": "onset",   "char": "ㄷ", "score": 100.0 },
    { "syl": 0, "pos": "nucleus", "char": "ㅏ", "score": 100.0 },
    { "syl": 0, "pos": "coda",    "char": "ㄱ", "score": 85.0  },
    { "syl": 1, "pos": "onset",   "char": "ㅃ", "score": 95.0  },
    { "syl": 1, "pos": "nucleus", "char": "ㅗ", "score": 100.0 },
    { "syl": 2, "pos": "onset",   "char": "ㄲ", "score": 80.0  },
    { "syl": 2, "pos": "nucleus", "char": "ㅣ", "score": 100.0 }
  ],
  "heard_jamos": [
    { "syl": 0, "pos": "onset",   "char": "ㄷ" },
    { "syl": 0, "pos": "nucleus", "char": "ㅏ" },
    { "syl": 0, "pos": "coda",    "char": "ㄴ" },
    { "syl": 1, "pos": "onset",   "char": "ㅃ" },
    { "syl": 1, "pos": "nucleus", "char": "ㅗ" },
    { "syl": 2, "pos": "onset",   "char": "ㄲ" },
    { "syl": 2, "pos": "nucleus", "char": "ㅣ" }
  ],
  "per": 0.14,
  "weighted_per": 0.042,
  "counts": {"match": 6, "sub": 1, "ins": 0, "del": 0},
  "per_position": {
    "onset":   {"matched": 3, "total": 3, "accuracy": 1.0},
    "nucleus": {"matched": 3, "total": 3, "accuracy": 1.0},
    "coda":    {"matched": 0, "total": 1, "accuracy": 0.0}
  },
  "per_jamo": { "ㄱ@coda": {"ref_count": 1, "correct": 0, "errors": {"sub_to_ㄴ": 1}} },
  "problem_jamos": [],
  "ref_jamo": [ ... ],
  "low_confidence": false
}
```

### `detailed_jamos` vs `heard_jamos`

- **`detailed_jamos`**: 정답 단어(ref) 좌표계. 음절 인덱스 `syl`은 정답 단어 기준 (예: 닭볶이 → 0=닭, 1=볶, 2=이). 채점 결과(`score`: 0~100)와 자모(`char`: 정답 자모) 포함.
- **`heard_jamos`**: 사용자 실제 발음(hyp) 좌표계. `syl`은 wav2vec2 phoneme 모델이 추론한 음절 인덱스로, 사용자가 음절을 빠뜨리거나 추가하면 `detailed_jamos.syl`과 어긋날 수 있음. 점수 정보 없음.

클라이언트는 두 배열을 **독립적으로** 표시하는 것이 권장됨 — `detailed_jamos`로 음절 박스에 점수 표시, `heard_jamos`로 "실제 무슨 발음을 했는지" 별도 영역에 표시.

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
- **후보 선택**: 후보별 `total_cost / Σref_weight`(weighted error rate) 최저값
- **종합 점수**: `100 * max(0, 1 - total_cost / Σref_weight)`
- **PER**: `(sub + ins + del) / N_ref`
- **음절 기반 매핑**: 정렬된 자모 데이터를 음절 인덱스(`syl`) 및 위치(`pos`) 기준으로 재그룹화하여 `detailed_jamos` 리스트로 반환.

## 한계와 엣지 케이스

- 후보 단어가 비어있거나 20개 초과면 `400 Bad Request`.
- 무음 비율 ≥ 95% 입력은 `error("NO_SPEECH")`.
- 60초 초과 음성은 `413 Payload Too Large`.
- 사투리/외래어는 g2pkk의 표준 발음과 차이가 있어 실제보다 낮게 채점될 수 있음 (WordBank `pronunciation` 필드로 우회 가능).
- 단일 GPU 동시 추론은 `Semaphore(VOICE_CONCURRENCY)`로 직렬화 — 기본 1.
- `heard_jamos`는 wav2vec2의 원시 출력을 그대로 노출하므로, CTC 노이즈로 인해 음절 구조가 깨진 형태가 나올 수 있음.

## 환경 변수

`.env` 또는 셸 환경에서 설정 가능 (`VOICE_` prefix):

| 변수 | 기본 | 설명 |
|---|---|---|
| `VOICE_PHONEME_MODEL` | `Kkonjeong/wav2vec2-base-korean` | HF 모델 ID |
| `VOICE_PHONEME_DEVICE` | `cuda` | |
| `VOICE_MAX_DURATION_SEC` | `60` | 최대 입력 길이 |
| `VOICE_MAX_FILE_BYTES` | `20971520` | 최대 파일 크기 (20MB) |
| `VOICE_SILENCE_THRESHOLD` | `0.95` | 무음 거부 임계값 |
| `VOICE_MAX_CANDIDATES` | `20` | 후보 단어 최대 개수 |
| `VOICE_CONCURRENCY` | `1` | GPU 동시 처리 수 |

## 디렉토리

```
src/voice_pron/
├── main.py              # FastAPI + lifespan + WordBank 캐싱
├── config.py            # pydantic-settings
├── api/
│   ├── routes.py        # /pronounce SSE (file + candidates)
│   └── sse.py           # 이벤트 envelope
├── audio/loader.py      # 16kHz mono 변환, silence_ratio
├── g2p/
│   ├── pronouncer.py    # g2pkk + 정규화
│   └── jamo_utils.py    # 자모 분해/위치 태깅
├── phoneme/wav2vec_jamo.py  # Kkonjeong CTC + 자모 토큰화
├── align/
│   ├── needleman.py     # 가중 NW
│   └── cost.py          # 자모 거리 행렬
├── scoring/score.py     # 점수 산정 + detailed_jamos
└── pipeline/orchestrator.py  # 후보 alignment loop + SSE
```
