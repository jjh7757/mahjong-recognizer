FROM python:3.12-slim

WORKDIR /srv/app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# wget: 서버 compose 의 healthcheck 가 쓴다
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# AGPL-3.0(서버) + 제3자 모델·데이터 출처(app/assets/mahjong-yolo11s.onnx)
COPY LICENSE THIRD_PARTY_NOTICES.md ./

ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
