"""복기 질문 라우트(POST /v2/review/ask)를 노트북에서 띄우는 개발용 서버 — 운영에 쓰지 않는다.

11-24: app/main.py 에서 /v2/review/ask 라우트를 뗐다(사용자 결정) — 지금 띄우면 그 경로는 404. 쓰려면 라우트부터 복구(커밋 9fe2d5e).

python scripts/review_dev_server.py [--port 8321] [--host 127.0.0.1] [--api-key dev-key] [--fake] [--env-file PATH]

진짜 `app.main:app` 을 그대로 띄운다 — 요청 검사 · 요청 제한 · 오류 코드가 운영과 같아서, 앱이 만든 요청이 서버 규격에
맞는지를 그대로 본다(mahjong_calc docs/paifu_review_agent_app_design_1122.md 5절). 바꾸는 것은 셋뿐:
  - 키: app.main 을 import 하기 전에 RECOGNIZER_API_KEY 를 --api-key 값으로 넣는다(main 이 import 때 읽음).
  - --fake: GEMINI_API_KEY 를 환경에서 지워 진짜 생성기 · 판독기가 안 만들어지게 하고, 시작 뒤 review_generator 를
    scripts/review_ask.py 의 fake_generate()(대본 가짜 LLM)로 바꿔 끼운다. Gemini 호출 0 · 돈 0.
    사진 판독 라우트(/v1 · /v2 판독)는 판독기가 없어 503 이다.
  - CORS: 웹 빌드(flutter run -d chrome)에서 부를 수 있게 이 스크립트 안에서만 연다(모든 origin · 헤더 · 메서드,
    Retry-After 노출). app/main.py 는 건드리지 않는다 — 운영 서버에는 CORS 가 없다.
--fake 가 없으면 실제 Gemini 로 답한다(돈이 든다 — 사용자 허락 뒤에만). GEMINI_API_KEY 가 환경에 있거나 --env-file 에
`GEMINI_API_KEY=…` 줄로 있어야 하고(값은 찍지 않는다), 없으면 까닭을 찍고 종료 코드 2.
--env-file 에서는 GEMINI_API_KEY 한 줄만 읽고 다른 줄은 모두 무시한다(실행 스크립트 agent_run_*.py 가 자식에 그 키 하나만
넘기는 것과 같게). 환경에 GEMINI_API_KEY 가 이미 있으면 그 값을 쓴다. 모델(REVIEW_MODEL · GEMINI_MODEL) · 생각 수준
(REVIEW_THINKING) · 요청 제한(REVIEW_RATE_PER_MIN 6 · BURST 3 … 운영 기본)은 이 프로세스의 환경 변수로만 바꾼다.
scripts/ 는 배포 워크플로를 일으키지 않는다(.github/workflows/deploy.yml 의 paths-ignore).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 같은 폴더의 review_ask.py(fake_generate)

log = logging.getLogger("review_dev_server")


def _read_gemini_key(path: Path) -> str | None:
    """`GEMINI_API_KEY=VALUE` 줄의 값(앞의 `export ` 허용, 값 양끝 따옴표 벗김). 다른 줄은 읽어 넘기지 않는다.
    없으면 None. 값은 어디에도 찍지 않는다."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip().removeprefix("export ").strip() != "GEMINI_API_KEY":
            continue
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        return v
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="개발용 복기 서버(운영에 쓰지 않음)")
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--api-key", default="dev-key", help="이 서버가 받을 X-Api-Key")
    ap.add_argument("--fake", action="store_true", help="Gemini 대신 대본 가짜 LLM(scripts/review_ask.py fake_generate)")
    ap.add_argument("--env-file", type=Path, help="GEMINI_API_KEY 한 줄만 읽을 KEY=VALUE 파일(--fake 가 아닐 때, 다른 줄은 무시)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # app.main 은 import 때 환경을 읽는다 — 그 전에 맞춘다.
    os.environ["RECOGNIZER_API_KEY"] = args.api_key
    os.environ.pop("ALLOW_NO_AUTH", None)
    if args.fake:
        os.environ.pop("GEMINI_API_KEY", None)
    else:
        if args.env_file:
            if not args.env_file.is_file():
                print(f"--env-file 이 없다: {args.env_file}", file=sys.stderr)
                return 2
            # GEMINI_API_KEY 한 줄만 — 받을 키(RECOGNIZER_API_KEY) · 모델 · 제한 같은 다른 줄은 무시한다.
            gemini_key = _read_gemini_key(args.env_file)
            if gemini_key:
                os.environ.setdefault("GEMINI_API_KEY", gemini_key)
        if not os.environ.get("GEMINI_API_KEY"):
            print("GEMINI_API_KEY 가 없다 — 환경 변수나 --env-file 로 주거나, 돈 없이 흐름만 보려면 --fake", file=sys.stderr)
            return 2

    import uvicorn
    from fastapi.middleware.cors import CORSMiddleware

    from app import main as server

    server.app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Retry-After"],
    )

    if args.fake:
        from review_ask import fake_generate

        gen = fake_generate()
        gen.model = "fake"  # 응답의 model(라우트가 getattr(review_generator, "model") 로 읽음)

        def use_fake() -> None:
            # app.main.startup() 뒤에 돈다(등록 순서). GEMINI_API_KEY 가 없어 startup 은 review_generator 를 안 만든다.
            server.review_generator = gen
            log.warning("FAKE LLM — Gemini 호출 없음 (review_generator = scripts/review_ask.py fake_generate)")

        # app.main 과 같은 on_event(이 FastAPI 판에는 add_event_handler 가 없다).
        server.app.on_event("startup")(use_fake)

    log.warning("개발용 복기 서버 — 운영에 쓰지 않는다: http://%s:%d/v2/review/ask (X-Api-Key 는 --api-key 값, %s)",
                args.host, args.port, "가짜 LLM" if args.fake else "실제 Gemini")
    uvicorn.run(server.app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
