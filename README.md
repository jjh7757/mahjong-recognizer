# mahjong-recognizer

리치마작 손패 사진에서 패를 읽어 돌려주는 서버. [mahjong_calc](../mahjong_calc) 앱의 2단계(사진 인식) 백엔드.
인식은 Gemini 비전 모델(기본 `gemini-3.8-flash`)이 하고, 이 서버는 키 보관·전처리·요청 제한을 맡는다.
점수 계산은 하지 않는다 — 앱이 한다.

- 배포: [jjh7757/TIL — AWS배포환경](https://github.com/jjh7757/TIL/tree/master/AWS배포환경)
  (GitHub Actions → ECR → SSM → EC2 t4g.small, Caddy 뒤에서 `mahjong-recognizer:8000`).
- 주소: https://mahjong.ai-agent-develop.cloud

## API

모든 엔드포인트는 multipart `image` (JPEG/PNG/WebP, 4MB·4천만 픽셀 이하), 헤더 `X-Api-Key`, 응답에 `notes`·`model`·`latency_ms`·`prep_ms`·`usage` 공통.
좌표는 `[ymin, xmin, ymax, xmax]` 0~1000 정규화, `side` 는 **이미지 기준** `bottom`/`right`/`top`/`left`.
자리(동남서북) 변환과 점수 계산은 앱이 한다. 설계: [mahjong_calc/docs/table_reading_design.md](../mahjong_calc/docs/table_reading_design.md)

```
GET  /               → {"status":"ok","backend":"gemini","model":"gemini-3.8-flash"|null,"git_sha":"...","endpoints":[...],
                        "source":"https://github.com/jjh7757/mahjong-recognizer","license":"AGPL-3.0-only"}

POST /v1/recognize   눕힌 손패 크롭 (울기·화료패 포함 가능)
  → {"tiles":[{"tile":"5m","conf":0.93},...], "count":13,
     "melds":[{"type":"chi|pon|kan_open|kan_closed","tiles":["4p","5p","6p"],"conf":0.8}],
     "win_tile":{"tile":"7s","conf":0.7} | null, "notes":"", ...}

POST /v1/locate      테이블 전경 (앱이 긴 변 1024 정도로 줄여서 보냄)
  → {"hands":[{"side":"bottom","box":[757,321,917,814],"face_up":true,"conf":0.98},...],
     "rivers":[{"side":"bottom","box":[...],"conf":..},...],
     "dead_wall":{"box":[...],"conf":..} | null, "center":{"box":[...]} | null,
     "face_up_sides":["bottom"], "notes":"", ...}

POST /v1/table       테이블 중앙 크롭 (강 4개 + 왕패)
  → {"dora_indicators":[{"tile":"3p","conf":0.9}], "ura_dora_indicators":[] (리치 화료 뒤 공개했을 때만),
     "riichi":{"bottom":true,"right":false,"top":false,"left":false}, "riichi_conf":{...},
     "last_discards":{"bottom":"7s","right":null,...}, "last_discards_conf":{...},
     "rivers":{"bottom":["1z","9m",...,"7s"],...} (버린 순서), "notes":"", ...}
```

`tiles` 는 왼쪽부터 순서. 적도라는 `0m` `0p` `0s`. 자패는 `1z`(동)~`7z`(중).
응답은 5~15초, 읽을 게 없거나 흐리면 1분까지 늘어난다 — 앱은 경과 시간을 보여줘야 한다.

| 코드 | 뜻 | 앱이 할 일 |
| --- | --- | --- |
| 400 | 디코딩 실패, 또는 픽셀 수 초과(`MAX_PIXELS`, 기본 4천만) | 사진을 고쳐 보낸다 |
| 401 | X-Api-Key 틀림(비ASCII 도 401) | — |
| 413 | 이미지 4MB 초과(`MAX_UPLOAD_BYTES`). Content-Length 로 본문을 받기 전에 끊거나(`body too large`), 파일 크기로(`image too large`) | 줄여서 보낸다 |
| 429 | 이 서버의 요청 제한(`too many requests`), 또는 Gemini 가 429 를 돌려줘 재시도 뒤에도 실패(`upstream rate limited`). 둘 다 `Retry-After` 초 | 그 초만큼 쉬고 다시 |
| 503 | 서버에 GEMINI_API_KEY 없음(`recognizer not configured`), 또는 Gemini 5xx·시간 초과·호출 예산 초과(`upstream unavailable`, `Retry-After`) | 쉬고 다시 |
| 502 | 그 밖의 Gemini 호출 실패(빈 응답 등) | 다시 찍는다 |

### 요청 제한 (요청당 1, 사진 수와 무관)

- **토큰 버킷**: 한 번에 `RATE_BURST`(기본 20)개까지 몰아 보낼 수 있고, 분당 `RATE_PER_MIN`(기본 40)개씩 다시 찬다.
  화료 한 번(locate 1 + 판독 최대 10)이나 추적 중 자리·강 끝·도라 산이 겹쳐 오는 묶음을 한 번에 받고, 평균은 분당 40 으로 묶는다.
  `/v2/last3` 는 사진 3~4장이라도 요청 1 로 센다(전엔 사진 수만큼 깎여 10-29 측정에서 429 가 났다).
- **비용 보호**: 시간당 `RATE_PER_HOUR`(기본 600)·하루 `RATE_PER_DAY`(기본 3000) 고정 창 상한. 넘으면 429 와 창이 끝날 때까지의 `Retry-After`.
- **Gemini 429**: 서버가 1~3초 쉬고 한 번 더 보내고, 그래도 429 면 앱에 429 + `Retry-After: 10`(`UPSTREAM_429_RETRY_S`).
- **호출 예산**: Gemini 호출 하나(헤지 포함)는 60초(`budget_s`) 안에 끝나야 한다. 넘으면 늦은 요청을 기다리지 않고 503 + `Retry-After: 5`.

### 시간 (모든 응답, 밀리초)

- `latency_ms`: 판독 함수에 들어와서 나갈 때까지 — 틀 잡기(YOLO11)·토막 내기·Gemini 호출(헤지·여러 번 읽기)·맞추기 모두. 어느 경로든 같은 뜻.
- `prep_ms`: 그중 첫 Gemini 요청을 보내기 전까지(틀 잡기·토막 내기). 단일 호출 경로는 0.
- 요청 받기·디코딩·크기 줄이기는 둘 다에 들어가지 않는다(앱이 잰 왕복 시간 − `latency_ms` ≈ 네트워크 + 디코딩).

코드 구성: `app/schemas.py`(구조화 출력 스키마 3종) · `app/prompts.py`(프롬프트) · `app/normalize.py`(응답 정리, 순수 함수) · `app/recognizer.py`(Gemini 호출) · `app/main.py`(엔드포인트).

## 입력 조건이 정확도를 결정한다

실험([mahjong_calc/experiments/vlm/RESULTS.md](../mahjong_calc/experiments/vlm/RESULTS.md)) 기준:

- 눕힌 손패를 위에서 찍어 손패만 보이게 → 실제 사진 3장 전부 정답
- 테이블 전경(버림패·벽패 섞임, 서 있는 손패) → 패는 거의 못 읽지만 **영역 위치(`/v1/locate`)는 정확히 찾는다**
- 패 한 장이 모델에 들어가는 이미지에서 100px 이상이어야 한다. 서버가 긴 변을 1568px 로 줄이므로 전경은 앱이 구역별로 원본에서 잘라 보내야 한다

앱은 locate 로 구역을 찾고 손패·중앙 크롭을 따로 보내며, 인식 결과는 사용자가 확인·수정하게 한다.

## 환경 변수 (서버 `.env`)

| 이름 | 값 |
| --- | --- |
| `RECOGNIZER_API_KEY` | 앱이 보내는 키. **비우면 기동을 거부한다** — 로컬 실험에서 인증 없이 띄우려면 `ALLOW_NO_AUTH=1` |
| `SOURCE_URL` | `/` 응답의 `source`(AGPL §13 소스 안내). 기본 이 저장소 |
| `GEMINI_API_KEY` | Google AI Studio 키 |
| `GEMINI_MODEL` | 기본 `gemini-3.8-flash` |
| `GEMINI_THINKING` | `default` / `low` / `off` |
| `LOCATE_THINKING` | 구역 나누기(/v1/locate)만 따로. 기본 `default` — `low` 면 모서리 울기·왕패를 자주 놓친다 |
| `RATE_PER_MIN`, `RATE_BURST`, `RATE_PER_HOUR`, `RATE_PER_DAY` | 요청 제한(위 "요청 제한"). 기본 40 / 20 / 600 / 3000 |
| `UPSTREAM_429_RETRY_S`, `UPSTREAM_DOWN_RETRY_S` | Gemini 429 / 5xx·시간 초과 때 앱에 주는 `Retry-After`. 기본 10 / 5 |
| `MAX_CONCURRENT` | 동시에 도는 Gemini 호출(기본 16). 호출 스레드 풀은 이 값 + 4, 헤지 풀은 × SEAT_READS × 2 |
| `MAX_UPLOAD_BYTES`, `MAX_PIXELS` | 사진 하나의 크기(기본 4MB)·픽셀 수(기본 4천만) |
| `SEAT_READS` | 자리 판독을 몇 번 동시에 읽어 맞출지(기본 3) |
| `RIVER_FRAME` | 기본 `1` — 강 끝(/v2/last3)의 강 틀을 YOLO11 검출로 잡아 마지막 패만 잘라 읽음. `0` 이면 강 크롭을 통째로 읽음 |
| `RIVER_CONF` | 강 틀 검출 기준값, 기본 `0.04`(손패 사진으로 학습한 모델이라 강 패는 확신도가 낮다) |

## 로컬 실행

```
pip install -r requirements-dev.txt   # 운영 의존성 + pytest
set GEMINI_API_KEY=...
set RECOGNIZER_API_KEY=...            # 또는 set ALLOW_NO_AUTH=1 (인증 없이, 로컬만)
uvicorn app.main:app --reload
curl -F image=@hand.jpg -H "X-Api-Key: ..." http://localhost:8000/v1/recognize
python scripts/try_endpoint.py --mode hand|locate|table [--url ...] [--key-file ...] photo.jpg
pytest   # master push 시 GitHub Actions 가 먼저 돌리고, 통과해야 배포(.github/workflows/deploy.yml)
```

## 라이선스

이 서버는 **GNU AGPL-3.0**(`LICENSE`)이고 비상업 프로젝트다. 강 끝(`app/river.py`)과 자리 틀(`app/seat.py`)의 패 위치 검출에
쓰는 모델 `app/assets/mahjong-yolo11s.onnx` 는 nikmomo/Mahjong-YOLO(MIT, Copyright (c) 2024 Shin Zhang)의 YOLO11s 이고,
학습 데이터는 Kaggle shinz114514 데이터셋(MIT), 학습 도구는 Ultralytics(학습한 모델 기본 AGPL-3.0)다.
자세한 출처·Docker 이미지의 Python 의존성 라이선스 표·라이선스 전문은 `THIRD_PARTY_NOTICES.md`. 서버는 ultralytics 라이브러리
없이 onnxruntime 으로만 돌린다. 네트워크로 쓰는 사람에게는 `/` 응답의 `source` 로 소스 받을 곳을 알린다(AGPL §13).

Docker 이미지는 기반 이미지를 digest 로 고정하고 비루트 사용자 `app` 으로 돈다(`Dockerfile`).
