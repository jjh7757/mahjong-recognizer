# 기반 이미지는 digest 로 고정(multi-arch 목록 digest, linux/arm64 빌드에 그대로 쓴다). 2026-10-30 python:3.12-slim.
# 올릴 때: docker buildx imagetools inspect python:3.12-slim 의 목록 digest 로 바꾸고 테스트·배포.
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

WORKDIR /srv/app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# wget: 서버 compose 의 healthcheck 가 쓴다. app: 서버를 돌리는 비루트 사용자(쓰는 파일 없음, 업로드 임시 파일은 /tmp).
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --no-create-home --shell /usr/sbin/nologin app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# AGPL-3.0(서버) + 제3자 모델·데이터·Python 의존성 출처(app/assets/mahjong-yolo11s.onnx 등)
COPY LICENSE THIRD_PARTY_NOTICES.md ./

ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
