# Third-party notices

This server (mahjong-recognizer) is licensed under the GNU Affero General Public License v3.0 (see `LICENSE`).
It is a non-commercial project. The components below are included or used as described.

## Tile detection model — `app/assets/mahjong-yolo11s.onnx`

Used only to find where tiles lie in a discard-pile (river) crop (`app/river.py`); tile kinds are read by Gemini.
Run with onnxruntime; the Ultralytics library itself is not used by this server.

- Source: nikmomo/Mahjong-YOLO — https://github.com/nikmomo/Mahjong-YOLO (`mahjong-yolos-best.onnx`, unmodified, renamed)
- Model: Ultralytics YOLO11s, trained by the author above (ONNX metadata: Ultralytics 8.3.162, 2025-07-16)
- Ultralytics states that models trained with Ultralytics YOLO are licensed under AGPL-3.0 by default
  (https://www.ultralytics.com/license). This server is distributed under AGPL-3.0 accordingly.

The repository is released under the MIT License:

```
MIT License

Copyright (c) 2024 Shin Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Training data of the model above

- Kaggle dataset "Mahjong hand photos taken with mobile camera" by shinz114514 —
  https://www.kaggle.com/datasets/shinz114514/mahjong-hand-photos-taken-with-mobile-camera
- License: MIT (Kaggle dataset metadata, checked 2026-10-26)

## Reference sheet — `app/assets/ref_sheet_ps.jpg`

Photographed and cut by the author of this project (own tile set). No third-party content.
