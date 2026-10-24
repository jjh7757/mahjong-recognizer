# mahjong-recognizer

리치마작 손패 사진에서 패를 읽어 돌려주는 서버. [mahjong_calc](../mahjong_calc) 앱의 2단계(사진 인식) 백엔드.
인식은 Gemini 비전 모델(기본 `gemini-3.8-flash`)이 하고, 이 서버는 키 보관·전처리·요청 제한을 맡는다.
점수 계산은 하지 않는다 — 앱이 한다.

- 배포: [jjh7757/TIL — AWS배포환경](https://github.com/jjh7757/TIL/tree/master/AWS배포환경)
  (GitHub Actions → ECR → SSM → EC2 t4g.small, Caddy 뒤에서 `mahjong-recognizer:8000`).
- 주소: https://mahjong.ai-agent-develop.cloud

## API

모든 `/v1/*` 는 multipart `image` (JPEG/PNG/WebP, 4MB 이하), 헤더 `X-Api-Key`, 응답에 `notes`·`model`·`latency_ms` 공통.
좌표는 `[ymin, xmin, ymax, xmax]` 0~1000 정규화, `side` 는 **이미지 기준** `bottom`/`right`/`top`/`left`.
자리(동남서북) 변환과 점수 계산은 앱이 한다. 설계: [mahjong_calc/docs/table_reading_design.md](../mahjong_calc/docs/table_reading_design.md)

```
GET  /               → {"status":"ok","backend":"gemini","model":"gemini-3.8-flash"|null,"endpoints":[...]}

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

| 코드 | 뜻 |
| --- | --- |
| 401 | X-Api-Key 틀림 |
| 413 | 이미지 4MB 초과 |
| 429 | 분당 요청 한도 초과 (`RATE_PER_MIN`, 기본 40) |
| 502 | Gemini 호출 실패 |
| 503 | 서버에 GEMINI_API_KEY 없음 |

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
| `RECOGNIZER_API_KEY` | 앱이 보내는 키. 비우면 인증 없음 |
| `GEMINI_API_KEY` | Google AI Studio 키 |
| `GEMINI_MODEL` | 기본 `gemini-3.8-flash` |
| `GEMINI_THINKING` | `default` / `low` / `off` |
| `LOCATE_THINKING` | 구역 나누기(/v1/locate)만 따로. 기본 `default` — `low` 면 모서리 울기·왕패를 자주 놓친다 |
| `RATE_PER_MIN`, `MAX_CONCURRENT`, `MAX_UPLOAD_BYTES` | 요청 제한 |

## 로컬 실행

```
pip install -r requirements.txt
set GEMINI_API_KEY=...
uvicorn app.main:app --reload
curl -F image=@hand.jpg -H "X-Api-Key: ..." http://localhost:8000/v1/recognize
python scripts/try_endpoint.py --mode hand|locate|table [--url ...] [--key-file ...] photo.jpg
pytest
```
