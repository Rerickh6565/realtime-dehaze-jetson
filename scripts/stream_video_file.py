"""
stream_video_file.py — Dehaze a video file and stream it via Flask MJPEG.

Reads a video file frame-by-frame, runs TensorRT FP16 dehazing inference,
and serves a side-by-side (hazy | dehazed) MJPEG stream at http://0.0.0.0:5000.
The video loops continuously.

Requirements:
    - TRT model weights at weights/ldnet_trt.pth  (run scripts/convert_trt.py first)

Usage:
    python scripts/stream_video_file.py
    # Then open http://<jetson-ip>:5000 in a browser
"""

import time

import cv2
import numpy as np
import torch
from flask import Flask, Response
from torch2trt import TRTModule

# ── Config ──────────────────────────────────────────────────────────────────
TRT_WEIGHTS  = "weights/ldnet_trt.pth"
VIDEO_PATH   = "assets/bikers.mp4"
INPUT_SIZE   = (256, 256)   # Must match the size used during TRT conversion
OUTPUT_SIZE  = (640, 480)
FLASK_HOST   = "0.0.0.0"
FLASK_PORT   = 5000

# ── Load TRT model ──────────────────────────────────────────────────────────
print("Loading TensorRT model...")
model = TRTModule()
model.load_state_dict(torch.load(TRT_WEIGHTS))
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params / 1e6:.3f}M")

# ── Open video ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video file: {VIDEO_PATH}")

app = Flask(__name__)
frame_count = 0


def process_frame(frame: np.ndarray):
    """Preprocess, infer, and postprocess a single BGR frame."""
    t0 = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, INPUT_SIZE)
    tensor = torch.from_numpy(small).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).cuda()
    t1 = time.time()

    with torch.no_grad():
        output = model(tensor)
    t2 = time.time()

    out = output.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.resize(out, OUTPUT_SIZE), cv2.COLOR_RGB2BGR)
    t3 = time.time()

    return out, (t1 - t0), (t2 - t1), (t3 - t2), (t3 - t0)


def generate():
    """Generator yielding MJPEG frames, looping the input video."""
    global frame_count
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video back to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, OUTPUT_SIZE)
        frame_count += 1

        out, tpre, tinf, tpost, ttot = process_frame(frame)
        stitched = np.hstack((frame, out))
        fps = 1.0 / ttot if ttot > 0 else 0

        print(
            f"Frame {frame_count:05d} | "
            f"pre:{tpre*1000:.1f}ms  inf:{tinf*1000:.1f}ms  "
            f"post:{tpost*1000:.1f}ms  total:{ttot*1000:.1f}ms  FPS:{fps:.2f}"
        )

        _, buf = cv2.imencode(".jpg", stitched)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buf.tobytes() + b"\r\n"
        )


@app.route("/")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT)
