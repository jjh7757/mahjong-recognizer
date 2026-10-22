"""YOLO 학습 → ONNX 내보내기 → ../models/ 에 복사.

    python train.py <dataset 폴더> [--epochs 20] [--imgsz 640] [--model yolo11n.pt] [--device cpu]

GPU 가 없으면 CPU 로 돈다(합성 600장·20 epoch 기준 1시간 안팎).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default="synthetic")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.dataset / "dataset.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        project=str(ROOT / "training" / "runs"),
        name=args.name,
        exist_ok=True,
        patience=8,
        # 색상 증강은 끈다: 적도라(붉은 5)와 일반 5의 구분이 색에 달려 있다
        hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,
        fliplr=0.0,  # 좌우 반전하면 글자가 뒤집힌다
        plots=False,
    )

    best = ROOT / "training" / "runs" / args.name / "weights" / "best.pt"
    onnx_path = YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, opset=17, simplify=True, dynamic=False)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    shutil.copy(onnx_path, models_dir / "model.onnx")
    shutil.copy(args.dataset / "classes.json", models_dir / "classes.json")
    print("exported:", models_dir / "model.onnx")


if __name__ == "__main__":
    main()
