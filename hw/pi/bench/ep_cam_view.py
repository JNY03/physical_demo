# -*- coding: utf-8 -*-
"""
피지컬팀 mk2 — RoboMaster EP 카메라 미리보기 (검증용)
========================================================
EP 카메라 프레임을 브라우저에서 바로 보기 위한 **검증 도구**다. 운영 경로가 아니다
— 운영은 `media.py` 의 JPEG/RTP 로 엣지에 쏜다(HW-R-07). 이건 "카메라가 실제로
무엇을 보고 있나"를 사람이 눈으로 확인하는 용도다(`go1_cam_view.py` 와 같은 자리).

  파이에서:  python3 -m bench.ep_cam_view
  PC 에서 :  http://pi1.local:8080

MJPEG(multipart/x-mixed-replace)라 브라우저만 있으면 되고 플러그인이 필요 없다.
cv2 를 쓰지 않는다 — SDK 가 numpy 배열을 주므로 Pillow 로 바로 JPEG 로 굽는다.
opencv 를 이미지에 넣으면 90MB 가 는다.
"""
import argparse
import io
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from robomaster import robot

_latest = {"jpeg": None, "at": 0.0, "n": 0}
_lock = threading.Lock()

PAGE = """<!doctype html><meta charset=utf-8><title>EP camera</title>
<style>body{margin:0;background:#111;color:#ccc;font:14px system-ui}
img{display:block;max-width:100%;margin:0 auto}
p{padding:8px 12px;margin:0}</style>
<p>RoboMaster EP \u2014 /stream (MJPEG)</p><img src="/stream">""".encode("utf-8")


def grab(ep, fps, quality, max_width, swap_rb):
    period = 1.0 / fps
    while True:
        t0 = time.time()
        try:
            frame = ep.camera.read_video_frame(strategy="newest", timeout=5)
        except Exception as e:
            print(f"[EP캠] 프레임 실패: {type(e).__name__}: {e}", flush=True)
            time.sleep(0.5)
            continue
        arr = np.asarray(frame)
        if swap_rb and arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]
        img = Image.fromarray(arr)
        if max_width and img.width > max_width:
            h = round(img.height * max_width / img.width)
            img = img.resize((max_width, h))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        with _lock:
            _latest["jpeg"] = buf.getvalue()
            _latest["at"] = time.time()
            _latest["n"] += 1
        time.sleep(max(0.0, period - (time.time() - t0)))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                                  # 접속마다 찍히면 프레임 로그가 묻힌다

    def do_GET(self):
        if self.path == "/stream":
            return self._stream()
        body = PAGE
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        try:
            while True:
                with _lock:
                    jpeg, n = _latest["jpeg"], _latest["n"]
                if jpeg is None or n == last:
                    time.sleep(0.02)
                    continue
                last = n
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpeg)).encode()
                                 + b"\r\n\r\n" + jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass                              # 브라우저 탭을 닫은 것뿐이다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default=os.environ.get("HW_EP_CONN_TYPE", "rndis"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--width", type=int, default=960, help="0 이면 원본(1280)")
    ap.add_argument("--swap-rb", action="store_true",
                    help="색이 뒤집혀 보이면(파랑↔빨강) 이 옵션을 준다")
    a = ap.parse_args()

    ep = robot.Robot()
    ep.initialize(conn_type=a.conn)
    print(f"[EP캠] 접속 — sn={ep.get_sn()}", flush=True)
    ep.camera.start_video_stream(display=False)

    threading.Thread(target=grab, daemon=True,
                     args=(ep, a.fps, a.quality, a.width, a.swap_rb)).start()

    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"[EP캠] http://<파이주소>:{a.port}  (Ctrl-C 로 종료)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            ep.camera.stop_video_stream()
        except Exception:
            pass
        # SDK close() 가 내부 스레드 join 에서 멈추는 것을 실측했다(2026-09-14).
        # 검증 도구이므로 프로세스를 그냥 끝낸다.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
