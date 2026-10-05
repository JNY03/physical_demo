# -*- coding: utf-8 -*-
"""
피지컬팀 mk2 — Go1 정면 영상 업로더 (HW-R-07)
=================================================
Go1 정면 카메라를 인지 서버에 프레임 단위로 올린다.

    카메라 ──ws H.264──► pi1 ──ffmpeg──► JPEG ──HTTP POST──► 서버 /upload

서버(FastAPI)가 받아서 `/stream/<camera_id>` 로 되돌려 주고, `/control/<camera_id>` 로
탐지 결과를 낸다. Unity 의 5009 수신기가 먹는 그 형식이다.

## 왜 JPEG 인가 — 서버가 그것만 받는다

`robot/media.py` 의 운영 경로는 RTP over UDP 다(v8 §5-10). 이 서버는 그 경로가 아니라
multipart/form-data 단건 업로드만 받으므로(`POST /upload: file, camera_id`) 여기서는
JPEG 로 변환해 올린다. 엣지노드가 서면 media.py 경로로 돌아간다.

## ffmpeg 플래그는 bench/go1_cam_view.py 의 실측값을 그대로 쓴다

- `-fflags nobuffer` 금지: 출력이 파이프면 두어 프레임만 내보내고 나머지를 조용히 버린다.
- `-threads 1`(입출력 양쪽): 기본 프레임 단위 멀티스레딩이 프레임을 쥐고 있어 지연이
  301ms 였다. 1스레드로 묶어 37ms.

## 프레임은 쌓지 않고 버린다

무선 홉이라 업로드가 밀릴 수 있다. 큐를 두면 지연이 누적돼 화면이 과거를 보여 준다.
최신 프레임만 들고 있다가 올리고, 올리는 동안 들어온 것은 버린다.

사용:
    cd ~/hw/pi && python3 -m robot.go1_front_upload
    cd ~/hw/pi && python3 -m robot.go1_front_upload --fps 15 --quality 4
"""
import argparse
import os
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.go1_camera import Go1CameraSource      # noqa: E402

DEFAULT_URL = "http://210.110.250.33:7864/upload"


def _multipart(jpeg, camera_id):
    """requests 없이 multipart/form-data 를 만든다. 말단에 의존성을 늘리지 않는다."""
    b = uuid.uuid4().hex
    pre = (
        "--" + b + "\r\n"
        'Content-Disposition: form-data; name="camera_id"\r\n\r\n' + camera_id + "\r\n"
        "--" + b + "\r\n"
        'Content-Disposition: form-data; name="file"; filename="frame.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    post = ("\r\n--" + b + "--\r\n").encode()
    return pre + jpeg + post, "multipart/form-data; boundary=" + b


class Uploader(threading.Thread):
    """최신 프레임 하나만 들고 올린다. 올리는 중에 온 프레임은 버린다."""

    daemon = True

    def __init__(self, url, camera_id, min_interval):
        super().__init__()
        self.url = url
        self.camera_id = camera_id
        self.min_interval = min_interval
        self.cond = threading.Condition()
        self.pending = None
        self.stop_flag = False
        self.sent = 0
        self.dropped = 0
        self.failed = 0
        self.last_error = ""
        self.last_ms = 0.0

    def offer(self, jpeg):
        with self.cond:
            if self.pending is not None:
                self.dropped += 1
            self.pending = jpeg
            self.cond.notify()

    def run(self):
        last = 0.0
        while not self.stop_flag:
            with self.cond:
                while self.pending is None and not self.stop_flag:
                    self.cond.wait(0.5)
                jpeg, self.pending = self.pending, None
            if jpeg is None:
                continue
            wait = self.min_interval - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            body, ctype = _multipart(jpeg, self.camera_id)
            req = urllib.request.Request(self.url, data=body, method="POST")
            req.add_header("Content-Type", ctype)
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    r.read()
                self.sent += 1
                self.last_ms = (time.time() - t0) * 1000.0
            except Exception as e:                      # 서버가 잠깐 죽어도 계속 간다
                self.failed += 1
                self.last_error = type(e).__name__ + ": " + str(e)[:80]


def run(cam_id, url, camera_id, fps, quality, verbose):
    up = Uploader(url, camera_id, 1.0 / max(fps, 0.1))
    up.start()

    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-threads", "1",
         "-probesize", "32k", "-analyzeduration", "0",
         "-f", "h264", "-i", "pipe:0",
         "-f", "mjpeg", "-q:v", str(quality), "-threads", "1", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def feed():
        while True:
            try:
                for chunk in Go1CameraSource(cam_id):
                    proc.stdin.write(chunk)
                    proc.stdin.flush()   # 빠뜨리면 파이썬 버퍼에 갇혀 ffmpeg 이 굶는다
            except Exception as e:
                print("[cam] 재연결: " + str(e), flush=True)
            time.sleep(1.0)

    threading.Thread(target=feed, daemon=True).start()

    buf = b""
    frames = 0
    last_log = time.time()
    while True:
        d = os.read(proc.stdout.fileno(), 65536)
        if not d:
            print("[ffmpeg] 출력 종료", flush=True)
            break
        buf += d
        while True:
            s = buf.find(b"\xff\xd8")
            e = buf.find(b"\xff\xd9", s + 2) if s >= 0 else -1
            if s < 0 or e < 0:
                break
            up.offer(buf[s:e + 2])
            frames += 1
            buf = buf[e + 2:]
        if verbose and time.time() - last_log >= 5.0:
            last_log = time.time()
            print("[up] 디코드 %df  전송 %d  드롭 %d  실패 %d  왕복 %.0fms  %s"
                  % (frames, up.sent, up.dropped, up.failed, up.last_ms, up.last_error),
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description="Go1 정면 영상 -> 인지 서버 업로드")
    ap.add_argument("--cam", type=int, default=1, help="Go1 카메라 번호 (1=정면)")
    ap.add_argument("--url", default=os.environ.get("HW_UPLOAD_URL", DEFAULT_URL))
    ap.add_argument("--camera-id", default=os.environ.get("HW_UPLOAD_CAMERA_ID", "go1_front"))
    ap.add_argument("--fps", type=float, default=float(os.environ.get("HW_UPLOAD_FPS", "10")))
    ap.add_argument("--quality", type=int, default=int(os.environ.get("HW_UPLOAD_Q", "5")),
                    help="JPEG q:v. 낮을수록 고화질·고대역폭")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    print("[init] cam=%d -> %s camera_id=%s fps=%s q=%d"
          % (a.cam, a.url, a.camera_id, a.fps, a.quality), flush=True)
    run(a.cam, a.url, a.camera_id, a.fps, a.quality, not a.quiet)


if __name__ == "__main__":
    main()
