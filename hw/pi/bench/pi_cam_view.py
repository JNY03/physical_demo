"""
피지컬팀 mk2 — 라즈베리파이 카메라 모듈 웹 뷰어
==================================================
이 파이의 **CSI 카메라 모듈**(리본 케이블로 직결된 것)이 지금 무엇을 보고 있는지
브라우저에서 확인하기 위한 도구다. 운영 구성요소가 아니다.

`bench/go1_cam_view.py` 와 헷갈리지 말 것 — 그쪽은 **Go1 로봇 안의 나노 5대**가
ws 로 보내 주는 H.264 를 중계한다. 이 파일은 그 경로를 전혀 건드리지 않고,
파이에 직접 물린 카메라만 본다. 포트도 8090 이 아닌 8091 로 따로 쓴다.

## 경로

    카메라 모듈 ──CSI──► rpicam-vid (--codec mjpeg) ──파이프──► JPEG 프레임 분해
                                                            └─► HTTP multipart ──► <img>

**파이썬에서 인코딩하지 않는다.** `rpicam-vid` 가 ISP 하드웨어로 JPEG 를 만들어
주므로 이 프로세스가 하는 일은 파이프에서 `FFD8…FFD9` 경계를 찾아 나누고 뿌리는
것뿐이다. picamera2·flask 같은 추가 의존성이 없다 — 이 파이에는 둘 다 깔려 있지
않고, 표준 라이브러리와 이미 설치된 rpicam-apps 만으로 충분하다.

상류 프로세스는 **보는 사람이 몇이든 하나**이고, 아무도 보지 않으면 20초 뒤
종료해 카메라를 놓는다. 카메라는 한 번에 한 프로세스만 잡을 수 있으므로,
뷰어를 띄워 둔 채로 다른 캡처 도구를 돌리려다 막히는 일이 없도록 한 것이다.

사용:
    cd ~/hw/pi && python3 -m bench.pi_cam_view 8091
    # 브라우저에서 http://<이 호스트>:8091/

환경 변수로 조절한다(기본값은 괄호 안):
    PI_CAM_WIDTH(1280) PI_CAM_HEIGHT(720) PI_CAM_FPS(15) PI_CAM_QUALITY(70)
    PI_CAM_HFLIP(0) PI_CAM_VFLIP(0) PI_CAM_ROTATION(0)   # 0 또는 180

끝점:
    /                페이지
    /stream.mjpg     multipart MJPEG 스트림
    /snapshot.jpg    가장 최근 프레임 한 장
    /status          JSON — 카메라 인식 여부·fps·오류 문자열
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _i(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


WIDTH = _i("PI_CAM_WIDTH", 1280)
HEIGHT = _i("PI_CAM_HEIGHT", 720)
FPS = _i("PI_CAM_FPS", 15)
QUALITY = _i("PI_CAM_QUALITY", 70)
HFLIP = _i("PI_CAM_HFLIP", 0)
VFLIP = _i("PI_CAM_VFLIP", 0)
ROTATION = _i("PI_CAM_ROTATION", 0)

IDLE_STOP_S = 20.0          # 보는 사람이 없으면 이만큼 뒤 카메라를 놓는다
FRAME_WAIT_S = 8.0          # 첫 프레임을 기다리는 한도 (카메라 기동 ~1초)
BUF_LIMIT = 8 << 20         # 경계를 못 찾고 버퍼만 부풀면 버린다


# ---------------------------------------------------------------- 카메라 인식
_LIST_CACHE = {"t": 0.0, "v": None}


def list_cameras(max_age=5.0):
    """`rpicam-hello --list-cameras` 로 붙어 있는 카메라 목록을 읽는다.

    스트림이 돌아가는 중에는 **묻지 않는다.** 카메라는 한 프로세스만 잡을 수
    있어서, 굳이 확인하려다 돌아가는 스트림을 방해할 이유가 없다. 이미 프레임이
    나오고 있다는 것보다 확실한 인식 증거도 없다.
    """
    now = time.time()
    if _LIST_CACHE["v"] is not None and now - _LIST_CACHE["t"] < max_age:
        return _LIST_CACHE["v"]
    out = []
    try:
        r = subprocess.run(["rpicam-hello", "--list-cameras"],
                           capture_output=True, text=True, timeout=15)
        for line in (r.stdout + r.stderr).splitlines():
            line = line.strip()
            if re.match(r"^\d+\s*:", line):
                out.append(line)
    except Exception:
        pass
    _LIST_CACHE.update(t=now, v=out)
    return out


# ---------------------------------------------------------------- 상류 하나
class Camera:
    """rpicam-vid 하나를 띄워 MJPEG 프레임을 최신 한 장만 들고 있는다.

    최신 한 장만 두는 것은 의도다. 큐에 쌓아 두면 느린 뷰어가 몇 초 전 장면을
    보게 된다 — 늦은 프레임은 버리는 편이 맞다.
    """

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg = None
        self.seq = 0
        self.fps = 0.0
        self.err = None
        self.running = False
        self.last_want = 0.0
        self.proc = None

    def argv(self):
        a = ["rpicam-vid", "-n", "-t", "0", "--codec", "mjpeg",
             "--width", str(WIDTH), "--height", str(HEIGHT),
             "-q", str(QUALITY),
             # --flush 가 없으면 rpicam-vid 가 출력 버퍼를 채울 때까지 들고 있다가
             # 뭉텅이로 내보낸다. 파이프로 받는 쪽에서는 그대로 지연이 된다.
             "--flush", "-o", "-"]
        if FPS > 0:
            a += ["--framerate", str(FPS)]
        if HFLIP:
            a += ["--hflip"]
        if VFLIP:
            a += ["--vflip"]
        if ROTATION:
            a += ["--rotation", str(ROTATION)]
        return a

    def want(self):
        """뷰어가 있다는 표시. 필요하면 상류를 띄운다."""
        self.last_want = time.time()
        with self.cond:
            if self.running:
                return
            self.running = True
            self.err = None
        threading.Thread(target=self._run, daemon=True).start()

    def _fail(self, msg):
        with self.cond:
            self.err = msg
            self.running = False
            self.cond.notify_all()

    def _run(self):
        tail = deque(maxlen=12)
        try:
            self.proc = subprocess.Popen(self.argv(),
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
        except FileNotFoundError:
            self._fail("rpicam-vid 를 찾을 수 없다 — sudo apt install rpicam-apps")
            return
        except Exception as e:
            self._fail(f"{type(e).__name__}: {e}")
            return
        threading.Thread(target=self._drain, args=(tail,), daemon=True).start()
        try:
            self._collect()
        except Exception as e:
            with self.cond:
                self.err = f"{type(e).__name__}: {e}"
        finally:
            self._stop(tail)

    def _drain(self, tail):
        """rpicam-vid 의 stderr 를 모아 둔다.

        카메라가 안 붙어 있으면 이 프로세스는 `no cameras available` 한 줄만 남기고
        곧바로 끝난다. 그 줄을 버리면 화면에는 '왜인지 모르게 영상이 안 나옴'만
        남는다 — 그대로 페이지에 띄워 주려고 붙잡아 둔다.
        """
        try:
            for raw in self.proc.stderr:
                line = raw.decode("utf-8", "replace").strip()
                if line:
                    tail.append(line)
        except Exception:
            pass

    def _collect(self):
        buf = b""
        n, t0 = 0, time.time()
        fd = self.proc.stdout.fileno()
        while True:
            # read(n) 은 n 바이트가 찰 때까지 막는다. os.read 로 오는 대로 받는다.
            d = os.read(fd, 65536)
            if not d:
                break
            buf += d
            while True:
                s = buf.find(b"\xff\xd8")
                e = buf.find(b"\xff\xd9", s + 2) if s >= 0 else -1
                if s < 0 or e < 0:
                    break
                with self.cond:
                    self.jpeg, self.seq = buf[s:e + 2], self.seq + 1
                    self.cond.notify_all()
                buf = buf[e + 2:]
                n += 1
            if len(buf) > BUF_LIMIT:
                buf = b""               # 프레임 경계가 아닌 무언가다. 다시 맞춘다
            dt = time.time() - t0
            if dt >= 2.0:
                self.fps, n, t0 = round(n / dt, 1), 0, time.time()
            if time.time() - self.last_want > IDLE_STOP_S:
                break                   # 아무도 안 본다 — 카메라를 놓는다

    def _stop(self, tail):
        p, self.proc = self.proc, None
        if p:
            for kill in (p.terminate, p.kill):
                try:
                    kill()
                    p.wait(timeout=3)
                    break
                except Exception:
                    pass
        with self.cond:
            self.running = False
            self.fps = 0.0
            if self.seq == 0 and self.err is None:
                # 한 장도 못 받고 끝났다. 이유는 stderr 에 있다.
                hint = " / ".join(t for t in tail if t) or "출력 없이 종료"
                self.err = hint
            self.cond.notify_all()

    def wait_first(self, timeout=FRAME_WAIT_S):
        """첫 프레임이 나오거나, 상류가 이유를 남기고 죽거나, 시간이 다 될 때까지.

        `wait_frame(-1)` 로 대신하면 안 된다 — seq 가 0 이라 조건이 곧바로 어긋나
        기다리지 않고 빈손으로 돌아온다. 그러면 카메라가 없을 때 rpicam-vid 가
        남긴 진짜 이유 대신 '시간 초과' 만 화면에 뜬다.
        """
        deadline = time.time() + timeout
        with self.cond:
            while self.jpeg is None and self.err is None and self.running:
                left = deadline - time.time()
                if left <= 0:
                    break
                self.cond.wait(left)
            return self.seq, self.jpeg, self.err

    def wait_frame(self, last_seq, timeout=FRAME_WAIT_S):
        with self.cond:
            if self.seq == last_seq and self.running:
                self.cond.wait(timeout)
            return self.seq, self.jpeg, self.err

    def status(self):
        with self.cond:
            running, seq, fps, err = self.running, self.seq, self.fps, self.err
        cams = ["(스트리밍 중 — 확인 생략)"] if running and seq else list_cameras()
        return {
            "running": running, "frames": seq, "fps": fps, "error": err,
            "cameras": cams,
            "config": {"width": WIDTH, "height": HEIGHT, "fps": FPS,
                       "quality": QUALITY, "hflip": HFLIP, "vflip": VFLIP,
                       "rotation": ROTATION},
        }


CAM = Camera()


# ---------------------------------------------------------------- 페이지
PAGE = r"""<!doctype html><meta charset="utf-8"><title>파이 카메라</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{background:#111;color:#ddd;font:14px/1.6 system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:16px;margin:0 0 10px;font-weight:600}
figure{margin:0;background:#000;border:1px solid #333;border-radius:6px;overflow:hidden;
       max-width:1280px}
figcaption{padding:6px 10px;font-size:12px;color:#9ad;border-top:1px solid #222;
           display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
img{width:100%;display:block;background:#000;max-height:78vh;object-fit:contain}
a{color:#7bf}
.stat{color:#7a8}
#err{display:none;background:#3a1400;border:1px solid #a40;color:#fb8;
     padding:10px 14px;border-radius:6px;margin-bottom:12px;font-size:13px;
     max-width:1280px;white-space:pre-wrap}
.note{color:#777;font-size:12px;margin-top:14px;max-width:1280px}
</style>
<h1>라즈베리파이 카메라 모듈 — 실시간</h1>
<div id="err"></div>
<figure>
  <img id="v" alt="카메라 영상">
  <figcaption><span class="stat" id="s">연결 중…</span>
    <span><a href="/snapshot.jpg" target="_blank">스냅샷</a> ·
          <a href="/status" target="_blank">상태</a></span></figcaption>
</figure>
<p class="note">해상도·프레임률은 환경 변수로 바꾼다
(<code>PI_CAM_WIDTH/HEIGHT/FPS/QUALITY</code>). 보는 사람이 없으면 20초 뒤 카메라를
놓으므로, 다른 캡처 도구와 번갈아 써도 서로 막지 않는다. 이 뷰어는 파이에 직접 물린
CSI 카메라만 본다 — Go1 로봇 카메라는 8090 쪽이다.</p>
<script>
const img = document.getElementById('v'), st = document.getElementById('s'),
      er = document.getElementById('err');

// 끊기면 스스로 다시 붙는다. 상류가 잠깐 없거나 서버를 재기동해도 사람이
// 새로고침할 필요가 없다 — 얼어붙은 화면을 방치하지 않는 것이 목적이다.
function connect(){
  img.src = '/stream.mjpg?t=' + Date.now();
}
img.addEventListener('error', () => { st.textContent = '재연결 중…'; setTimeout(connect, 1500); });
connect();

async function poll(){
  try{
    const s = await (await fetch('/status', {cache:'no-store'})).json();
    if (s.error){
      er.style.display = 'block';
      er.textContent = '카메라를 열지 못했다:\n' + s.error
        + '\n\n케이블이 제대로 꽂혀 있는지, /boot/firmware/config.txt 의 camera_auto_detect=1'
        + ' 인지 확인한 뒤 재부팅해 보라. `rpicam-hello --list-cameras` 가 비어 있으면'
        + ' 파이가 아직 모듈을 못 본 상태다.';
    } else {
      er.style.display = 'none';
    }
    const c = s.config;
    st.textContent = (s.running ? '스트리밍' : '대기')
      + ` · ${c.width}x${c.height} · ${s.fps || 0} fps · ${s.frames} 프레임`;
  }catch(e){ /* 서버 재기동 중일 뿐이다 */ }
  setTimeout(poll, 2000);
}
poll();
</script>
"""


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                                  # 스트림 한 줄마다 로그를 남길 이유가 없다

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif path == "/status":
            self._send(200, "application/json; charset=utf-8",
                       json.dumps(CAM.status(), ensure_ascii=False).encode("utf-8"))
        elif path == "/snapshot.jpg":
            self._snapshot()
        elif path == "/stream.mjpg":
            self._stream()
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def _snapshot(self):
        CAM.want()
        seq, jpg, err = CAM.wait_first()
        if jpg is None:
            msg = f"프레임 없음: {err or '카메라 기동 중'}"
            self._send(503, "text/plain; charset=utf-8", msg.encode("utf-8"))
            return
        self._send(200, "image/jpeg", jpg)

    def _stream(self):
        CAM.want()
        last, jpg, err = CAM.wait_first()
        if jpg is None:
            # 여기서 200 을 주고 멈춰 있으면 <img> 는 영원히 로딩 중이 된다.
            # 오류로 끝내야 페이지의 error 처리가 재시도를 돈다.
            msg = f"카메라를 열지 못했다: {err or '시간 초과'}"
            self._send(503, "text/plain; charset=utf-8", msg.encode("utf-8"))
            return
        self.send_response(200)
        self.send_header("Cache-Control", "no-store, private")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=fr")
        self.end_headers()
        try:
            while True:
                if jpg is not None:
                    self.wfile.write(b"--fr\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    self.wfile.flush()
                CAM.want()
                last, jpg, err = CAM.wait_frame(last)
                if jpg is None:
                    break
                if not CAM.running and CAM.err:
                    break
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                              # 브라우저가 닫은 것뿐이다


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8091
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    cams = list_cameras()
    print(f"카메라: {', '.join(cams) if cams else '인식된 카메라 없음 — 케이블/설정 확인'}",
          flush=True)
    print(f"대기 중 — http://<이 호스트>:{port}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
