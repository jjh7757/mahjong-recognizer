# mahjong-recognizer

리치마작 손패 사진에서 패를 검출해 돌려주는 서버. [mahjong_calc](../mahjong_calc) 앱의 2단계(사진 인식) 백엔드다.
점수 계산은 하지 않는다 — 앱이 한다.

- FastAPI + ONNX Runtime(CPU). 모델은 YOLO11n 을 ONNX 로 내보낸 것.
- 배포: [jjh7757/TIL — AWS배포환경](https://github.com/jjh7757/TIL/tree/master/AWS배포환경)
  (GitHub Actions → ECR → SSM → EC2 t4g.small, Caddy 뒤에서 `mahjong-recognizer:8000`).

## API

```
GET  /                      → {"status":"ok","model":"model"|null}
POST /v1/recognize          multipart `image` (JPEG/PNG, 2MB 이하), 헤더 X-Api-Key
  → {"tiles":[{"tile":"5m","conf":0.93,"box":[x,y,w,h],"row":0,"group":0}, ...],
     "count":14, "model_version":"model", "image_size":[w,h], "infer_ms":420}
```

`tiles` 는 줄(row) → 왼쪽부터 순서. 옆 패와 패 폭의 60% 이상 떨어지면 `group` 이 바뀐다
(손패와 부른 묶음을 띄워 놓으면 자동으로 나뉜다). 뒷면은 `back`.

환경 변수: `RECOGNIZER_API_KEY`(비우면 인증 없음), `CONF_THRESHOLD`(기본 0.5), `MAX_UPLOAD_BYTES`.

## 로컬 실행

```
pip install -r requirements.txt
uvicorn app.main:app --reload
curl -F image=@hand.jpg -H "X-Api-Key: ..." http://localhost:8000/v1/recognize
pytest
```

## 모델 학습 (training/)

```
pip install ultralytics
python training/gen_dataset.py training/dataset --train 600 --val 100
python training/train.py training/dataset --epochs 20
```

`train.py` 가 `models/model.onnx` 와 `models/classes.json` 을 갱신한다. 패 이미지는
[FluffyStuff/riichi-mahjong-tiles](https://github.com/FluffyStuff/riichi-mahjong-tiles)(CC0).
실제 사진으로 재학습할 때는 같은 38클래스 라벨 체계를 쓴다.

## 서버 측 설정 (한 번)

`/srv/mahjong-recognizer/docker-compose.yml` 은 TIL 템플릿에 포트 8000, 메모리 상한 512M.
`.env` 에 `RECOGNIZER_API_KEY` 하나.
