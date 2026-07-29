"""
camera_raw_stream.py — Raw camera MJPEG passthrough for testing.

Captures raw frames from /dev/video0 and streams them as an MJPEG feed
at http://0.0.0.0:5000. No inference — use this to verify that the camera
is working correctly inside the container before running the dehazing stream.

Usage:
    python scripts/camera_raw_stream.py
    # Open http://<jetson-ip>:5000 in a browser to see the raw feed
"""

import cv2
from flask import Flask, Response

# ── Config ──────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
FLASK_HOST   = "0.0.0.0"
FLASK_PORT   = 5000

# ── Open camera ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}.")

app = Flask(__name__)


def generate():
    """Generator yielding raw MJPEG frames from the camera."""
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)
