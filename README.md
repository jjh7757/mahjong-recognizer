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
| `WIDE_READS` | `/v2/seatwide` 조각·모서리를 각각 몇 번 읽을지(기본 1 = 호출 2번, `agreement.unanimous` 는 거짓). 0 이면 `SEAT_READS` |
| `LOCAL_SHADOW` | 기본 `0`(끔). `1` 이면 `/v2/seat`·`/v2/last3` 사진을 자체 모델로도 읽어 응답 `local`·서버 로그에 남김(`app/local_read.py`, 답에는 안 씀). 켜면 운영에서 넓은 판독이 느려지고 메모리가 늘어 평소엔 끈다 |
| `RIVER_FRAME` | 기본 `1` — 강 끝(/v2/last3)의 강 틀을 YOLO11 검출로 잡아 마지막 패만 잘라 읽음. `0` 이면 강 크롭을 통째로 읽음 |
| `RIVER_CONF` | 강 틀 검출 기준값, 기본 `0.04`(손패 사진으로 학습한 모델이라 강 패는 확신도가 낮다) |
| `WALL_FRAME` | **기본 `hybrid`(8b-4a, 변수가 없을 때).** 끄기는 `off`(또는 `0`·`none`·`false`·빈 값 — Windows `set WALL_FRAME=` 는 변수를 지워 기본값이 되므로 `off` 로). 실험 값 `axis`(또는 `1`) / `rot` / `rot2`(8b-3b 미채택) / `hybrid`(8b-3c 채택) — 자체 모델 판독(`/v2/localroles`·`/v2/localdora`·`/v2/locallast3`)을 탁자 기준점으로 자른다(아래 절) |
| `WALL_FRAME_GRID_SRC` | 측정용. `WALL_FRAME=hybrid` 에서 요청에 `grid_src` 가 없을 때 쓸 격자 출처(`saved|manual|felt|none|default`) |

## 탁자 기준점 실험 옵션 `WALL_FRAME` (개선 계획 8b "할 일 3", 2026-11-06, 브랜치 `feat/table-frame`)

앱 격자(사진 좌표 선 4개 x1·x2·y1·y2)는 펠트 색·기기 크기·카메라 각도에 기대 탁자가 바뀌면 깨진다. 8b 는 기준점을 마작에 둔다:
**가운데 기기 가운데·방향 θ·패 한 장 짧은 변 u·기기 반폭 D**(`app/table_frame.TableFrame`)와 **패 단위 구역표**(`app/assets/table_frame_zones.json`).
꺼져 있으면(기본) 응답·답이 지금과 같고, 요청의 `frame=` 도 무시한다.

- **구역표·로더** `app/table_frame.py` = mahjong_calc `experiments/zone_points/table_frame.py`(8b-2)를 로직 그대로 옮긴 것(표준 라이브러리만).
  구역표는 앱 `assets/table_frame/zones.json` 복사본이고 앱 `lib/table/table_frame.dart` 와 같은 예제(`tests/fixtures/table_frame.json`)로 같은
  상자를 낸다 — `tests/test_table_frame.py` 가 예제 통과 + (앱 저장소가 있으면, `MAHJONG_CALC_REPO`) 두 파일이 앱 것과 같은지 본다.
  **구역표를 고치면 앱 저장소에서 고치고 여기로 복사한다.**
- **기준점 잡기** `app/table_locate.py` = mahjong_calc `experiments/zone_points/wall_frame.py`(8b-1, metrics 표 1-101)의 기기·패산 찾기를 그대로.
  화료 사진 = 가운데 기기(앞면 패 무리 안 빈 정사각형), 못 찾으면 패산(눕힌 뒷면 사각형, 배패 때 첫 프레임용). 색은 안 쓴다.
  입력은 사진 전체를 자리 판독 검출기(회전 상자 8회차)로 **한 번** 검출한 앞면·뒷면 — `LocalReader.full_raw` 가 같은 사진 바이트면 한 요청에서
  계산한 것을 10초 동안 다른 요청(localroles 맥락 검출·localdora·locallast3 기준점)에 나눠 준다(8단계 P1, 판독기마다 따로).
  요청에 `frame=`(`TableFrame.to_json` JSON, 다른 크기 사진에서 잰 것이면 가로 비로 옮겨 씀)이 오면 잡지 않고 그대로 쓴다(앱이 배패 때 잡은
  패산 기준점을 보내는 길). 응답에 `wall_frame`(모드)·`frame`·`frame_info`(source·기기 점수·패산 줄 수)·`cut`(rot | axis | grid)·`ms.frame`.
  못 잡으면(상자 부족·기기·패산 없음·가운데가 사진 밖·구역이 사진 밖) `frame_error` 를 넣고 **지금 격자 경로로**(`cut: grid`, 답이 비지 않게).
- **`axis`** — 기준점에서 지금 격자와 같은 뜻의 선을 만든다: x1·x2·y1·y2 = 가운데 칸(반폭 R = D + 6u, θ 로 돈 정사각형)의 축 평행 외접 상자,
  sticks = 리치봉 상자(기기 + 0.5u)의 외접 상자(응답 `grid`). 그다음은 지금 코드 그대로(`crop_seat`·`hand_frac`·`tokens`) — 역할 모델·분류기
  입력 분포는 안 바뀐다. localdora 는 후보·짝을 거르기만, locallast3(사진 전체 길)은 그 격자의 `lastZone`.
- **`rot`** — 구역표의 회전 사각형을 그 자리 사람이 아래에 오게 θ 만큼 세워 자른다(`table_locate.warp_zone`, 픽셀 크기 그대로, 사진이 안 닿는 끝은
  잘라 냄). 자리 = `seat` 칸(가운데 쪽으로 seatReadGrow 40/1000 과 같은 픽셀만큼 넓힘), 손패 칸 폭 = a ∈ [−R, R], 강 끝 = `river` 칸,
  표시패 = 가운데에서 max(왕패 칸 13u, 표시패 탐색 칸) 반폭 정사각형을 세운 그림에서 검출·패산 옆 검사·읽기. 크롭 상자는 원본 좌표로 되돌려
  응답한다. 역할 모델 토큰은 상자 네 점을 탁자 좌표(u)로 돌린 뒤 같은 식으로(`roles.tokens_frame` — θ = 0 이면 `tokens` 와 같은 값,
  테스트로 확인). 역할 모델은 거의 축 정렬된 사진으로 배웠으므로 **재학습 없이** 되는지 보는 것이 이 모드의 목적이다.
- **`rot2`**(8b-3b, rot 의 크롭 정의 손실 세 가지를 고친 후보 — mahjong_calc metrics 표 1-105 '다음 후보'): rot 과 같되
  ① 자리 읽기 크롭은 구역표 `seat_read`(= `seat` 를 그 사람 왼쪽·이웃 자리 쪽으로 3u 넓힘, a ≥ −R − 3 — 바람개비 경계로 뻗은 손패 끝 장이
  잘리지 않게, 어느 자리 패인지는 역할 모델이 가름; 채점·`zone_of` 의 구역은 `seat` 그대로), ② 크롭 안 `loose` 역할 상자도 틀에 안 쓰였고
  홀로 놓였으면(표시패 짝과 같은 간격 tw×0.5 로 묶어 혼자) 크롭 밖 것과 같은 조건(손패에서 패 높이 3배 안, 합쳐 딱 하나)으로 떼어 둔 화료패
  후보(`separate_tile.from = loose_inside`), ③ 표시패는 세워 자르지 않고 지금과 같은 사진 전체 검출(`detect_other`)에서 **패산 표시패 후보만**
  구역으로 거르고 짝은 지금처럼 사진 전체에서 묶는다(`cut: whole`, 짝마다 `zone`). 강 끝(`sides=`)은 rot 과 같다.
  **판정(mahjong_calc metrics 표 1-107, 2026-11-06): 미채택** — 개발 32회 앱 흐름 패 433 → 429/436(기준 ≥ 431). 겨냥한 손실은 다 사라졌으나
  나눔 역할 모델에서 171245 모서리 白 울기 채우기 실패·171505 오독이 새로 생김. 회귀 prod 장면 553 → 599·unsafe 11 → 2, 시간 +0.065초.
- **`hybrid`**(8b-3c, mahjong_calc metrics 표 1-108·1-109): 요청마다 **앱이 이 탁자 격자를 가졌는지**로 고른다 — 격자 출처 `grid_src` 가
  `saved|manual|felt`(저장·손으로 맞춤·펠트 자동 맞추기 성공)면 **지금 경로**(격자 그대로, WALL_FRAME 없음과 같은 답, 기준점은 잡아 응답에만,
  `wall_frame: saved`), `none|default`(기본값 격자·자동 맞추기 실패·갤러리 사진, 앱이 명시적으로 보냄)면 **rot2**(`wall_frame: rot2`), 기준점을 못 잡으면
  지금 경로(`wall_frame: none`). 문턱·색 없음. `grid_src` 는 `/v2/localroles`·`/v2/localdora`(격자 `x1·x2·y1·y2` 선택 인자도 받음)·`/v2/locallast3`
  의 질의 인자이고, 없으면 `WALL_FRAME_GRID_SRC`, 그것도 없으면 **saved**(8b-4a: 격자 질의가 없어도 — 옛 앱 답 무변경). 응답 `wall_frame_src` = [출처, query|env|grid|no_grid],
  `wall_frame_check` = 격자 가운데 칸 중심 ↔ 기준점 거리·θ(진단만). 서버 로그 `localroles …`·`localdora …` 줄 앞에 `wall_frame=`.
  격자 ↔ 기준점을 1u·3° 로 대조해 고르는 첫 설계는 측정 전에 버렸다(저장 격자 가운데 칸 중심이 기기 중심에서 거치마다 1~3u 어긋나고 사진 한 장
  θ 가 0.3~4.2° 흔들려 개발 saved 6/31).
  **판정(2026-11-06): 채택 — 기본 켬 후보.** 개발 32회 앱 흐름 장면 32/32 답 같음(경로 saved 32/32), 회귀 prod 세 변형 같음, 테스트 펴기 없이
  158 → 191·색 바꾼 145 → 179(none → rot2)·펴기 189 = 189, 표시패 50장 같음, 시간 +0.020초. **8b-4a(2026-11-07):** `grid_src` 가 없으면 격자 질의가 없어도 지금 경로로 바꾸고(옛 앱의 localdora 가 rot2 로 가던 위험 제거)
  **기본값을 hybrid 로** — 개발 서버 단독(`grid_src` 없이) 끔 / 기본: 회귀 prod `orig` 26장 답 같음(패 421 = 421), 표시패 개발 20장 같음(전부 `cut: grid`).
  앱 4a(`grid_src` 보내기, `none` 이면 강 끝 `sides=` 사진 전체)와의 앱 흐름 판정은 mahjong_calc metrics 표 1-111·1-112: 저장 격자 답 그대로,
  펴기 없이 155 → 201/211·색 바꾼 145 → 182, 다만 경고 없는 틀린 자동 0 → 3(F1400, 펴기 경로와 같은 장면)으로 미리 적은 기준상 미채택 — 해석은
  계획 세션. 서버 기본값(이 절)은 grid_src 없는 옛 앱에 답 무변경이라 그 판정과 별개다. 운영 배포는 사용자 결정.
- **표시패**(`/v2/localdora`, axis·rot): 기준점이 있으면 앞면 후보·짝을 구역표 표시패 탐색 칸(`dora_search`) ∪ 왕패 칸(`dead_wall`, 네 변)에 가운데가
  드는 것만 남기고(`kept` = [남은 수, 전체]), 짝 패마다 바람개비 칸 `zone`(bottom/right/top/left/center). 손패와 표시패를 크롭으로 가르지는
  않는다(왕패 칸이 손패 띠와 겹친다 — 구분은 지금처럼 패산 옆 검사·짝 규칙·앱 몫).
- **`/v2/locallast3?sides=right,top,left`**: `images` 에 사진 전체 한 장을 보내면 서버가 그 자리들의 강 끝을 자른다(`rot` = `river` 칸,
  `axis` = 기준점 격자, 끔·못 잡음 = 요청의 `x1·x2·y1·y2` 로 앱 `GridZones.lastZone` 과 같게). `sides` 가 없으면 지금처럼 앱이 자른 크롭들.
- 확인: `python -X utf8 scripts/wall_frame_check.py [--modes ,rot2] [키 ...]` — mahjong_calc 의 사진 몇 장을 서버 코드로(HTTP 없이) 모드마다 읽어 손패
  맞은 수·표시패·기준점(8b-1 결과와의 차이)·시간을 찍는다. 운영 배포는 판정(mahjong_calc docs/improvement_plan_1105.md 8b 합격 기준) 뒤.

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
