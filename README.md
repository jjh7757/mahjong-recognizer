# mahjong-recognizer

리치마작 손패 사진에서 패를 읽어 돌려주는 서버. [mahjong_calc](../mahjong_calc) 앱의 2단계(사진 인식) 백엔드.
인식은 Gemini 비전 모델(기본 `gemini-3.8-flash`)이 하고, 이 서버는 키 보관·전처리·요청 제한을 맡는다.
점수 계산은 하지 않는다 — 앱이 한다.

- 배포: [jjh7757/TIL — AWS배포환경](https://github.com/jjh7757/TIL/tree/master/AWS배포환경)
  (GitHub Actions → ECR → SSM → EC2 t4g.small, Caddy 뒤에서 `mahjong-recognizer:8000`).
- 주소: https://mahjong.ai-agent-develop.cloud

## API

```
GET  /                      → {"status":"ok","backend":"gemini","model":"gemini-3.8-flash"|null}
POST /v1/recognize          multipart `image` (JPEG/PNG, 4MB 이하), 헤더 X-Api-Key
  → {"tiles":[{"tile":"5m","conf":0.93}, ...], "count":13, "notes":"", "model":"gemini-3.8-flash", "latency_ms":15000}
```

`tiles` 는 왼쪽부터 순서. 적도라는 `0m` `0p` `0s`. 자패는 `1z`(동)~`7z`(중).
응답까지 10~20초 걸린다 — 앱은 진행 표시를 띄워야 한다.

| 코드 | 뜻 |
| --- | --- |
| 401 | X-Api-Key 틀림 |
| 413 | 이미지 4MB 초과 |
| 429 | 분당 요청 한도 초과 (`RATE_PER_MIN`, 기본 20) |
| 502 | Gemini 호출 실패 |
| 503 | 서버에 GEMINI_API_KEY 없음 |

## 입력 조건이 정확도를 결정한다

실험([mahjong_calc/experiments/vlm/RESULTS.md](../mahjong_calc/experiments/vlm/RESULTS.md)) 기준:

- 눕힌 손패를 위에서 찍어 손패만 보이게 → 실제 사진 13/13 정답
- 테이블 전경(버림패·벽패 섞임, 서 있는 손패) → 거의 못 읽음

앱은 가이드 프레임으로 손패 영역만 찍게 하고, 인식 결과는 사용자가 확인·수정하게 한다.

## 환경 변수 (서버 `.env`)

| 이름 | 값 |
| --- | --- |
| `RECOGNIZER_API_KEY` | 앱이 보내는 키. 비우면 인증 없음 |
| `GEMINI_API_KEY` | Google AI Studio 키 |
| `GEMINI_MODEL` | 기본 `gemini-3.8-flash` |
| `GEMINI_THINKING` | `default` / `low` / `off` |
| `RATE_PER_MIN`, `MAX_CONCURRENT`, `MAX_UPLOAD_BYTES` | 요청 제한 |

## 로컬 실행

```
pip install -r requirements.txt
set GEMINI_API_KEY=...
uvicorn app.main:app --reload
curl -F image=@hand.jpg -H "X-Api-Key: ..." http://localhost:8000/v1/recognize
pytest
```
