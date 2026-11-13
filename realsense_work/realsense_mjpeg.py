#!/usr/bin/env python3
"""리얼센스(D455) 컬러 스트림을 MJPEG 으로 낸다 — data_stream 이 로봇에서 받는 모양 그대로.

왜 이 모양인가: `server_stream_multi_source.py` 의 `robot_video_loop` 가
`GET /stream/<번호>` 의 multipart MJPEG 을 읽는다. 같은 모양으로 내면 **데이터
스트림 쪽에 새 풀러를 만들지 않아도** 된다(주소 한 줄과 소스 한 줄만 는다).

왜 pyrealsense2 가 아닌가: 이 파이(aarch64)에 librealsense 가 없고 바퀴도 없다.
D455 는 UVC 라 컬러 스트림이 /dev/video* 로 그냥 보인다 — 깊이가 필요해지면
그때 SDK 를 올린다(깊이는 UVC 로 안 나온다. 그것이 이 선택의 한계다).

  python3 realsense_mjpeg.py [포트] [장치번호] [너비] [높이] [fps]
  기본   8090      4        1280  720   15
  GET /            브라우저에서 보기
  GET /stream/<n>  MJPEG (번호는 무시한다 — 장치가 하나다)
  GET /health      상태 JSON
"""
import json
import socket
import sys
import threading
import time
from http import server

import cv2

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
DEV = int(sys.argv[2]) if len(sys.argv) > 2 else 4      # D455: 0/1 깊이 · 2/3 적외선 · 4/5 컬러
WIDTH = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
HEIGHT = int(sys.argv[4]) if len(sys.argv) > 4 else 720
FPS = int(sys.argv[5]) if len(sys.argv) > 5 else 15
QUALITY = 85
CAMERA_ID = "%s-realsense-d455-color" % socket.gethostname()   # 기기 이름에서 딴다 — pi6 → ds5 로 옮겨도 그대로 맞다
HFOV_DEG = 90.0            # D455 컬러의 **제원값**이다(관측이 아니다 — 보정 전에는 근거로 쓰지 않는다)
BOUNDARY = "frame"

_lock = threading.Lock()
_latest = {"jpeg": None, "seq": 0, "ts": None}
_state = {"opened": False, "reads": 0, "fails": 0, "since": time.time()}


def grab_loop():
    gap = 1.0 / max(1, FPS)
    cap = None
    while True:
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, FPS)
            _state["opened"] = cap.isOpened()
            if not cap.isOpened():
                _state["fails"] += 1
                time.sleep(2.0)
                continue
            print("열림 /dev/video%d  %dx%d @%d" % (DEV, WIDTH, HEIGHT, FPS), flush=True)
        ok, frame = cap.read()
        if not ok or frame is None:
            _state["fails"] += 1
            _state["opened"] = False
            cap.release()
            cap = None
            time.sleep(0.5)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        if not ok:
            _state["fails"] += 1
            continue
        with _lock:
            _latest["jpeg"] = buf.tobytes()
            _latest["seq"] += 1
            _latest["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + \
                ".%03dZ" % int((time.time() % 1) * 1000)
        _state["reads"] += 1
        time.sleep(gap)


PAGE = ("<!doctype html><meta charset=utf-8><title>%s realsense</title>"
        "<body style='margin:0;background:#111'>"
        "<img src='/stream/1' style='width:100%%;height:auto'>" % socket.gethostname()).encode()


class Handler(server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                 # 접속마다 찍지 않는다
        pass

    def do_GET(self):
        if self.path.startswith("/stream"):
            return self.stream()
        if self.path.startswith("/health"):
            with _lock:
                seq, ts = _latest["seq"], _latest["ts"]
            body = json.dumps({"camera_id": CAMERA_ID, "device": "/dev/video%d" % DEV,
                               "width": WIDTH, "height": HEIGHT, "fps": FPS,
                               "opened": _state["opened"], "frames": seq,
                               "read_fails": _state["fails"], "last_frame": ts,
                               "uptime_s": round(time.time() - _state["since"], 1)},
                              ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE)

    def stream(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=%s" % BOUNDARY)
        self.end_headers()
        sent = 0
        try:
            while True:
                with _lock:
                    jpeg, seq, ts = _latest["jpeg"], _latest["seq"], _latest["ts"]
                if jpeg is None or seq == sent:
                    time.sleep(0.01)
                    continue
                sent = seq
                # v2 프레임 머리를 한 줄로 싣는다 (DS-03). 받는 쪽은 이것을 sender 로 저장하고
                # 비전은 frame_ref{camera_id, sequence_id, timestamp} 로 되짚는다.
                meta = json.dumps({"camera_id": CAMERA_ID, "sequence_id": seq,
                                   "timestamp": ts,
                                   "camera": {"hfov_deg": HFOV_DEG, "source": "datasheet"}},
                                  ensure_ascii=False)
                self.wfile.write(b"--" + BOUNDARY.encode() + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n")
                self.wfile.write(b"Frame-Meta: " + meta.encode() + b"\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    threading.Thread(target=grab_loop, daemon=True).start()
    for _ in range(100):                       # 첫 장이 올 때까지 잠깐 기다렸다 알린다
        with _lock:
            if _latest["jpeg"] is not None:
                break
        time.sleep(0.1)
    print("MJPEG  http://0.0.0.0:%d/stream/1   상태 /health" % PORT, flush=True)
    Server(("0.0.0.0", PORT), Handler).serve_forever()


main()
