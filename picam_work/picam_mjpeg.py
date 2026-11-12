#!/usr/bin/env python3
"""라즈베리파이 모듈 카메라(CSI)를 MJPEG 으로 낸다 — data_stream 이 로봇에서 받는 모양 그대로.

왜 이 모양인가: `server_stream_multi_source.py` 의 `robot_video_loop` 가
`GET /stream/<번호>` 의 multipart MJPEG 을 읽는다. 같은 모양으로 내면 데이터
스트림 쪽에 **새 풀러를 만들지 않아도** 된다 — pi6 의 realsense_mjpeg.py 와 같은 선택이다.

왜 cv2.VideoCapture 가 아닌가: pi6 의 D455 는 UVC 라 /dev/video* 가 바로 컬러를 낸다.
**모듈 카메라(imx708)는 CSI 라 그 자리가 베이어 RAW 다** — 같은 코드를 쓰면 열리기는
하고 쓸 수 없는 그림이 나온다. 그래서 libcamera 쪽(rpicam-vid)에서 받는다.
이 파이에는 picamera2 도 cv2 도 없어 **표준 라이브러리만** 쓴다(설치가 필요 없다).

  python3 picam_mjpeg.py [포트] [카메라번호] [너비] [높이] [fps]
  기본   8090      0        1280  720   10
  GET /            브라우저에서 보기
  GET /stream/<n>  MJPEG (번호는 무시한다 — 이 파이의 카메라는 하나다)
  GET /health      상태 JSON
"""
import json
import shutil
import socket
import subprocess
import sys
import threading
import time
from http import server

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
CAM = int(sys.argv[2]) if len(sys.argv) > 2 else 0
WIDTH = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
HEIGHT = int(sys.argv[4]) if len(sys.argv) > 4 else 720
FPS = int(sys.argv[5]) if len(sys.argv) > 5 else 10
QUALITY = 80
CAMERA_ID = "%s-imx708" % socket.gethostname()   # 기기 이름에서 딴다 — pi11 · pi12 가 같은 파일을 쓴다
HFOV_DEG = 66.0            # Camera Module 3 (표준) 의 **제원값**이다(관측이 아니다 — 보정 전에는 근거로 쓰지 않는다)
BOUNDARY = "frame"

SOI, EOI = b"\xff\xd8", b"\xff\xd9"
_lock = threading.Lock()
_latest = {"jpeg": None, "seq": 0, "ts": None}
_state = {"opened": False, "frames": 0, "restarts": 0, "since": time.time(), "last_err": None}


def _cmd():
    exe = shutil.which("rpicam-vid") or shutil.which("libcamera-vid")
    if not exe:
        raise SystemExit("rpicam-vid 가 없습니다 (apt install rpicam-apps)")
    return [exe, "--camera", str(CAM), "-t", "0", "--nopreview",
            "--codec", "mjpeg", "--quality", str(QUALITY),
            "--width", str(WIDTH), "--height", str(HEIGHT),
            "--framerate", str(FPS), "-o", "-"]


def grab_loop():
    """rpicam-vid 의 stdout 에서 JPEG 한 장씩 잘라낸다. 죽으면 다시 띄운다."""
    while True:
        proc = None
        try:
            proc = subprocess.Popen(_cmd(), stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, bufsize=0)
            _state["opened"] = True
            print("열림 camera%d  %dx%d @%d" % (CAM, WIDTH, HEIGHT, FPS), flush=True)
            buf = b""
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    raise RuntimeError("rpicam-vid stdout 이 닫혔습니다")
                buf += chunk
                while True:
                    i = buf.find(SOI)
                    if i < 0:
                        if len(buf) > 4 << 20:
                            buf = b""           # 쓰레기만 쌓이면 버린다
                        break
                    j = buf.find(EOI, i + 2)
                    if j < 0:
                        if i:
                            buf = buf[i:]       # 시작 앞은 버린다
                        break
                    jpeg = buf[i:j + 2]
                    buf = buf[j + 2:]
                    with _lock:
                        _latest["jpeg"] = jpeg
                        _latest["seq"] += 1
                        _latest["ts"] = time.time()
                    _state["frames"] += 1
        except Exception as e:                   # noqa: BLE001 — 어떤 실패든 다시 띄운다
            _state["opened"] = False
            _state["restarts"] += 1
            _state["last_err"] = str(e)
            print("다시 띄웁니다: %s" % e, flush=True)
        finally:
            if proc is not None:
                try:
                    proc.kill()
                    proc.wait(timeout=3)
                except Exception:                # noqa: BLE001
                    pass
        time.sleep(2.0)


PAGE = ("<!doctype html><meta charset=utf-8><title>%s</title>"
        "<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui}"
        "img{max-width:100%%;display:block}p{padding:8px 12px;margin:0}</style>"
        "<p>%s &middot; %dx%d @%dfps &middot; <a style=color:#8bf href=/health>/health</a></p>"
        "<img src=/stream/1>") % (CAMERA_ID, CAMERA_ID, WIDTH, HEIGHT, FPS)


class Handler(server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                   # 접속마다 찍지 않는다
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif path == "/health":
            with _lock:
                seq, ts = _latest["seq"], _latest["ts"]
            body = json.dumps({"camera_id": CAMERA_ID, "opened": _state["opened"],
                               "frames": _state["frames"], "seq": seq,
                               "last_frame_at": ts, "age_s": None if ts is None else round(time.time() - ts, 3),
                               "restarts": _state["restarts"], "last_error": _state["last_err"],
                               "width": WIDTH, "height": HEIGHT, "fps": FPS,
                               "hfov_deg_spec": HFOV_DEG,
                               "uptime_s": round(time.time() - _state["since"], 1)},
                              ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
        elif path.startswith("/stream"):
            self.stream()
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def stream(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + BOUNDARY)
        self.end_headers()
        last, gap = -1, 1.0 / max(1, FPS)
        try:
            while True:
                with _lock:
                    jpeg, seq, ts = _latest["jpeg"], _latest["seq"], _latest["ts"]
                if jpeg is None or seq == last:
                    time.sleep(gap / 4)
                    continue
                last = seq
                head = ("--%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n"
                        "X-Frame-Seq: %d\r\nX-Timestamp: %.6f\r\n\r\n"
                        % (BOUNDARY, len(jpeg), seq, ts)).encode("latin-1")
                self.wfile.write(head)
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass                                  # 보는 쪽이 끊은 것은 정상이다


def main():
    threading.Thread(target=grab_loop, daemon=True).start()
    httpd = server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    httpd.daemon_threads = True
    print("http://0.0.0.0:%d/  ·  /stream/1  ·  /health" % PORT, flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
