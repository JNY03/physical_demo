#!/usr/bin/env python3
"""replay_source.py — 녹화해 둔 장으로 **살아 있는 소스를 흉내 낸다** (파이 없이 끝까지 시험, 2026-11-26).

왜: 파이들이 꺼져 있으면 데이터 서버 → 비전 → 오케스트레이터를 끝까지 시험할 수가 없다. 저장된
stream_data/<소스>/<실행>/ (frames/*.jpg + frames.jsonl + states.jsonl) 을 지금 시각 · 늘어나는 번호로 다시 흘린다.

    venv/bin/python tools/replay_source.py --source pi12 [--run latest|<실행>|<폴더>] [--fps 5] [--loop]
        [--mode redis|mjpeg|upload] [--seconds N] [--as 이름] [--trim] [--resize 640x360|0.5]

흘리는 길 (--mode)
  redis   (기본) Redis camera_stream:<이름> 에 데이터 서버(Source.add_frame)와 **같은 칸**으로 직접 XADD 한다.
          image = JPEG 바이트 · header = {"stream","n","frame_file","via","received_by","received_at","sender",
          "replay"} JSON · 그 밖에 sender 의 칸 한 칸씩 (frame_fields 와 같다). 최근 --maxlen(기본 300)개만 남긴다.
          states.jsonl 이 있으면(robot1 · a2 …) state_stream:<이름> 에도 원래 순서대로 {topic, payload, header}.
          --as replay-test-pi12 처럼 시험 이름을 주면 진짜 소스 키를 건드리지 않는다. --trim 이면 끝날 때
          그 시험 키(camera_stream:<이름> · state_stream:<이름>)를 지운다 (진짜 소스 이름이면 지우지 않는다).
          데이터 서버는 띄우지 않아도 된다 — vision_infer live · device_stage(--camera-url none) 가 Redis 에서 읽는다.
          source_probe 는 Redis 의 최근 장 나이로 live 를 낸다. --health 면 그 소스의 서버 포트(pi12 = 10009)에
          /health · /frame · /stream 도 열어 「데이터 서버가 받는 중」처럼 보이게 한다(진짜 서버가 꺼져 있을 때만).
  mjpeg   카메라 파이(picam_mjpeg · realsense_mjpeg · go1-camview · a2-bridge)와 같은 모양의 HTTP 를 연다:
          GET /stream/<n> (multipart/x-mixed-replace, 칸마다 Content-Length + 녹화 때 칸 머리 그대로 — picam 은
          X-Frame-Seq · X-Timestamp, v2 프레임 헤더는 Frame-Meta JSON) · /snapshot/1 · /health · /.
          --bind 127.0.0.1 --port 18090 (소스가 여럿이면 18090, 18091, … 차례로). 진짜 데이터 서버를 여기 붙이려면
          서버의 영상 포트(8090 고정)를 바꿔야 하므로 tools/server_port_override.py 로 서버를 띄운다 (아래 「끝까지」).
          device_stage 는 --camera-url http://127.0.0.1:18090/stream/1 로 바로 읽는다.
  upload  데이터 서버의 소스 포트에 POST /upload (multipart: 이미지 + sender JSON) · POST /state 로 밀어 넣는다.
          서버를 고치지 않고 서버 → Redis → 저장까지 **진짜 서버 길**을 지난다 (서버는 --no-pull 로 띄운다).
          보내는 곳은 --to URL (기본 http://127.0.0.1:<그 소스의 서버 포트>). via 는 「upload:/upload」가 된다.

시각과 번호 (live 처럼)
  번호는 1 부터 늘어난다(--loop 로 돌아도 이어 늘어난다). sender 칸 중 X-Frame-Seq · sequence_id · frame_seq · seq 는
  새 번호, X-Timestamp(초 · 밀리초 꼴 유지) · X-Timestamp-Iso · timestamp 는 **지금** 시각, Content-Length ·
  X-Image-Bytes · X-Sha1 · X-Image-Width/Height · width/height 는 실제 보낸 장으로 고친다. 그 밖의 칸은 녹화 그대로.
  해상도는 그대로 두고 --resize 를 줄 때만 줄인다 (cv2 가 있어야 한다). 상태 payload 의 timestamp 도 지금으로.
  frames.jsonl 이 없는 옛 폴더는 frames/*.jpg (또는 폴더의 *.jpg) 만 읽고 sender 를 picam 꼴로 만든다.

여럿 · 끝내기
  --source pi12 --source robot1   소스마다 스레드 하나. --run · --as 는 한 번 주면 전부에, 여러 번이면 차례로 짝짓는다.
  Ctrl+C(SIGINT) · SIGTERM 이면 깨끗이 멈춘다. --seconds N 이면 N초 뒤에 끝낸다. --loop 가 없으면 끝까지 한 번만.
  끝에 소스마다 보낸 장 · 상태 수와 (redis 모드) 스트림 길이를 찍는다. --json 이면 그 요약을 JSON 한 줄로.

끝까지 (파이 없이) — 자세한 명령은 data_stream_가이드.md 「파이 없이 끝까지 시험」
  A) Redis 로만:   replay_source.py --source pi12 --source robot1 --loop
                   → vision/vision_infer.py -d pi12 robot1 (live) · device_stage.py --source pi12 --camera-url none
  B) 진짜 서버로:  server_stream_multi_source.py --on pi12 robot1 --no-pull --save /tmp/ds_replay
                   + replay_source.py --mode upload --source pi12 --source robot1 --loop
  C) MJPEG 풀:     replay_source.py --mode mjpeg --source pi12 --port 18090 --loop
                   + tools/server_port_override.py --on pi12 --pi12 127.0.0.1:18090 --save /tmp/ds_replay
표준 라이브러리 + redis(redis 모드) + cv2(--resize 때만). 서버 파일을 import 하지 않는다(cv2 · torch 를 끌고 오지 않게).
"""
import argparse
import base64
import hashlib
import json
import os
import signal
import socket
import sys
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from http import server as httpserver
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

KST = timezone(timedelta(hours=9))
RESERVED = ("image", "header")              # server_stream_multi_source.RESERVED 와 같다
BOUNDARY = "frame"                          # picam_mjpeg 와 같은 경계
SEQ_KEYS = ("X-Frame-Seq", "sequence_id", "frame_seq", "seq")
ISO_KEYS = ("X-Timestamp-Iso", "timestamp_iso")
LEN_KEYS = ("Content-Length", "X-Image-Bytes")
TEST_PREFIX = "replay-test-"


def iso_now(t=None):
    """서버 received_now 와 같은 꼴 — ISO 밀리초 +09:00"""
    return datetime.fromtimestamp(time.time() if t is None else t, KST).isoformat(timespec="milliseconds")


def parse_iso(s):
    try:
        return datetime.fromisoformat(str(s)).timestamp()
    except (TypeError, ValueError):
        return None


def log(msg):
    print("[replay] " + msg, flush=True)


# ---------------------------------------------------------------- 녹화 읽기
def data_root():
    try:
        from ds_resources import settings
        return settings.ds_home() / "stream_data"
    except Exception:  # noqa: BLE001
        return REPO / "stream_data"


def _count_frames(folder):
    d = folder / "frames"
    try:
        return sum(1 for _ in (d if d.is_dir() else folder).glob("*.jpg"))
    except OSError:
        return 0


def find_run(source, run="latest", root=None):
    """--run → 실행 폴더 Path. 'latest' = 장이 있는 가장 최근 실행. 폴더 경로나 실행 이름도 받는다."""
    root = Path(root) if root else data_root()
    if run and run != "latest":
        p = Path(run).expanduser()
        for cand in (p, root / source / run, root / run):
            if cand.is_dir():
                return cand.resolve()
        raise SystemExit("실행 폴더를 못 찾음: %s (찾은 곳 %s)" % (run, root / source))
    base = root / source
    if not base.is_dir():
        raise SystemExit("녹화가 없음: %s" % base)
    # 서버가 만든 실행 폴더(frames.jsonl 이 있는 것)를 먼저 — 손으로 모은 sample 같은 폴더가 이름순으로 앞서지 않게
    dirs = [x for x in base.iterdir() if x.is_dir()]
    for d in sorted(dirs, key=lambda x: ((x / "frames.jsonl").is_file(), x.name), reverse=True):
        if _count_frames(d) > 0:
            return d.resolve()
    raise SystemExit("장이 있는 실행 폴더가 없음: %s" % base)


class Recording:
    """한 실행 폴더 — frames: [(jpg 경로, sender 사전, 받은 시각 epoch 또는 None, frame_file)],
    states: [(받은 시각, 기록 사전)]. 이미지는 보낼 때 읽는다 (수만 장도 메모리를 안 먹게)."""

    def __init__(self, folder, root=None):
        self.dir = Path(folder)
        self.root = Path(root) if root else self.dir.parent.parent
        self.frames, self.states = [], []
        jl = self.dir / "frames.jsonl"
        if jl.is_file():
            with open(jl, "rb") as f:
                for line in f:
                    try:
                        h = json.loads(line)
                    except ValueError:
                        continue
                    ff = h.get("frame_file") or h.get("file")
                    p = (self.root / ff) if ff else None
                    if p is None or not p.is_file():
                        cand = self.dir / "frames" / ("%09d.jpg" % int(h.get("n") or 0))
                        p = cand if cand.is_file() else None
                    if p is None:
                        continue
                    self.frames.append((p, dict(h.get("sender") or {}), parse_iso(h.get("received_at")), ff))
        if not self.frames:                       # 옛 폴더 · 손으로 모은 장
            d = self.dir / "frames" if (self.dir / "frames").is_dir() else self.dir
            for p in sorted(d.glob("*.jpg")):
                self.frames.append((p, {}, None, None))
        sl = self.dir / "states.jsonl"
        if sl.is_file():
            with open(sl, "rb") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if rec.get("topic") is not None:
                        self.states.append((parse_iso(rec.get("received_at")), rec))
        # 상태를 장 사이에 끼울 자리: states_before[i] = i 번째 장 앞에 보낼 상태들 (받은 시각 순서)
        self.states_before = [[] for _ in range(len(self.frames) + 1)]
        times = [t for _, _, t, _ in self.frames]
        j = 0
        for t, rec in sorted(self.states, key=lambda x: (x[0] is None, x[0] or 0)):
            if t is None:
                self.states_before[0].append(rec)
                continue
            while j < len(times) and (times[j] is None or times[j] < t):
                j += 1
            self.states_before[j].append(rec)


# ---------------------------------------------------------------- sender · 상태 고치기
def _like_ms(v):
    try:
        return float(v) > 1e11
    except (TypeError, ValueError):
        return False


def _same_type(orig, value):
    return str(value) if isinstance(orig, str) else value


def refresh_sender(sender, seq, now, image, size=None):
    """녹화 sender → 지금 보낸 것처럼. 새 사전을 돌려준다 (원본은 건드리지 않는다)."""
    s = dict(sender or {})
    if not s:                                     # frames.jsonl 이 없는 옛 장 — picam 꼴로
        s = {"Content-Type": "image/jpeg", "Content-Length": "0", "X-Frame-Seq": "0", "X-Timestamp": "0"}
    for k in SEQ_KEYS:
        if k in s:
            s[k] = _same_type(s[k], seq)
    if "X-Timestamp" in s:
        s["X-Timestamp"] = ("%d" % int(now * 1000)) if _like_ms(s["X-Timestamp"]) else ("%.6f" % now)
    for k in ISO_KEYS:
        if k in s:
            s[k] = iso_now(now)
    if "timestamp" in s:                          # v2 프레임 헤더 (ISO 글자) · 드론 meta (epoch 숫자)
        v = s["timestamp"]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            s["timestamp"] = int(now * 1000) if _like_ms(v) else round(now, 6)
        elif isinstance(v, str) and parse_iso(v) is not None:
            s["timestamp"] = iso_now(now)
        elif isinstance(v, str):
            s["timestamp"] = ("%d" % int(now * 1000)) if _like_ms(v) else ("%.6f" % now)
    for k in LEN_KEYS:
        if k in s:
            s[k] = _same_type(s[k], len(image))
    if "X-Sha1" in s:
        s["X-Sha1"] = hashlib.sha1(image).hexdigest()[:len(str(s["X-Sha1"])) or 16]
    if size:
        w, h = size
        for kw, kh in (("X-Image-Width", "X-Image-Height"), ("width", "height")):
            if kw in s:
                s[kw] = _same_type(s[kw], w)
            if kh in s:
                s[kh] = _same_type(s[kh], h)
    return s


def is_part_headers(sender):
    """녹화 sender 가 MJPEG 칸 머리 사전인가 (picam · 옛 pi7) — 아니면 Frame-Meta JSON 이었다 (v2 · realsense)."""
    return not sender or any(str(k).lower() == "content-type" for k in sender)


def state_payload(rec, now):
    """states.jsonl 한 줄 → 보낼 payload 바이트 (timestamp 는 지금으로)"""
    raw, enc = rec.get("payload"), rec.get("enc")
    if enc == "base64":
        return base64.b64decode(raw or "")
    if enc == "json" or isinstance(raw, (dict, list)):
        if isinstance(raw, dict) and isinstance(raw.get("timestamp"), str) and parse_iso(raw["timestamp"]) is not None:
            raw = dict(raw, timestamp=iso_now(now))
        return json.dumps(raw, ensure_ascii=False).encode("utf-8")
    return str(raw if raw is not None else "").encode("utf-8")


def frame_fields(image, header):
    """server_stream_multi_source.frame_fields 와 같다 (그 파일을 import 하지 않으려고 옮겨 둠 — 시험이 둘을 맞춰 본다)"""
    fields = {"image": image, "header": json.dumps(header, ensure_ascii=False)}
    for k, v in (header.get("sender") or {}).items():
        k = str(k)
        if not k or k in RESERVED:
            continue
        fields[k] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return fields


def parse_resize(text):
    """'640x360' → (640, 360) · '0.5' → 0.5 · None"""
    if not text:
        return None
    t = str(text).lower().strip()
    if "x" in t:
        w, h = t.split("x", 1)
        return int(w), int(h)
    f = float(t)
    if not 0 < f <= 4:
        raise ValueError("배율은 0 ~ 4")
    return f


def resize_jpeg(image, spec, quality=85):
    """(JPEG, (w, h)). spec 이 없으면 그대로."""
    if not spec:
        return image, None
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return image, None
    if isinstance(spec, tuple):
        w, h = spec
    else:
        h0, w0 = img.shape[:2]
        w, h = max(1, int(round(w0 * spec))), max(1, int(round(h0 * spec)))
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return (buf.tobytes() if ok else image), (w, h)


# ---------------------------------------------------------------- 내보내는 쪽
class Latest:
    """최신 한 장 (HTTP 창구가 기다렸다 가져간다)"""

    def __init__(self):
        self.cv = threading.Condition()
        self.item = None                          # (seq, jpeg, sender, now)
        self.frames = 0
        self.since = time.time()

    def put(self, seq, jpeg, sender, now):
        with self.cv:
            self.item = (seq, jpeg, sender, now)
            self.frames += 1
            self.cv.notify_all()

    def wait_new(self, last_seq, timeout):
        with self.cv:
            if self.item is None or self.item[0] == last_seq:
                self.cv.wait(timeout)
            return self.item


def mjpeg_part(jpeg, sender):
    """MJPEG 한 칸 (경계 줄 ~ 이미지 뒤 CRLF). 녹화 sender 가 칸 머리였으면 그 머리 그대로, 아니면 Frame-Meta."""
    lines = ["--" + BOUNDARY, "Content-Type: image/jpeg", "Content-Length: %d" % len(jpeg)]
    if is_part_headers(sender):
        for k, v in sender.items():
            if str(k).lower() in ("content-type", "content-length"):
                continue
            v = str(v).replace("\r", " ").replace("\n", " ")
            lines.append("%s: %s" % (k, v))
    else:
        lines.append("Frame-Meta: " + json.dumps(sender, ensure_ascii=True, separators=(",", ":")))
    head = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1", "replace")
    return head + jpeg + b"\r\n"


def make_handler(name, latest, stop, extra_health=None):
    class Handler(httpserver.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/") or "/"
            if path == "/health":
                item = latest.item
                h = {"source": name, "replay": True, "frames": latest.frames,
                     "seq": item[0] if item else None,
                     "age_s": round(time.time() - item[3], 3) if item else None,
                     "uptime_s": round(time.time() - latest.since, 1),
                     "status": {"video": "replay 중 (%d장)" % latest.frames}}
                h.update(extra_health or {})
                return self._send(200, "application/json; charset=utf-8", json.dumps(h, ensure_ascii=False).encode())
            if path in ("/snapshot/1", "/snapshot", "/frame"):
                item = latest.item or latest.wait_new(None, 3.0)
                if item is None:
                    return self._send(204, "text/plain", b"")
                return self._send(200, "image/jpeg", item[1], {
                    "X-Frame-N": str(item[0]),
                    "X-Frame-Meta": json.dumps(item[2], ensure_ascii=True, separators=(",", ":"))})
            if path == "/stream" or path.startswith("/stream/"):
                return self._stream()
            if path == "/":
                page = ("<!doctype html><meta charset=utf-8><title>replay %s</title>"
                        "<body style='margin:0;background:#111;color:#ddd;font:14px sans-serif'>"
                        "<p style='padding:8px'>replay %s · <a style='color:#8bf' href=/health>/health</a></p>"
                        "<img src=/stream/1 style='max-width:100%%'>" % (name, name))
                return self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))
            self._send(404, "text/plain; charset=utf-8", b"not found")

        def _stream(self):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + BOUNDARY)
            self.end_headers()
            last = None
            try:
                while not stop.is_set():
                    item = latest.wait_new(last, 0.5)
                    if item is None or item[0] == last:
                        continue
                    last = item[0]
                    self.wfile.write(mjpeg_part(item[1], item[2]))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
    return Handler


def start_http(bind, port, handler):
    httpd = httpserver.ThreadingHTTPServer((bind, port), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.25}, daemon=True).start()
    return httpd


def _multipart(fields):
    """[(이름, 파일이름 또는 None, content-type, 바이트)] → (본문, content-type)"""
    b = "----replay" + uuid.uuid4().hex
    out = []
    for name, fname, ctype, data in fields:
        disp = 'form-data; name="%s"' % name + ('; filename="%s"' % fname if fname else "")
        out.append(("--%s\r\nContent-Disposition: %s\r\nContent-Type: %s\r\n\r\n" % (b, disp, ctype)).encode())
        out.append(data)
        out.append(b"\r\n")
    out.append(("--%s--\r\n" % b).encode())
    return b"".join(out), "multipart/form-data; boundary=" + b


def _post(url, body, ctype, timeout=5.0):
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


# ---------------------------------------------------------------- 소스 하나
class Player:
    """녹화 하나를 한 소스로 흘린다 (스레드 하나). out 은 redis · mjpeg · upload 중 하나."""

    def __init__(self, src, name, rec, mode, fps=5.0, loop=False, resize=None, redis=None, maxlen=300,
                 latest=None, upload_url=None, stop=None, run_label=""):
        self.src, self.name, self.rec, self.mode = src, name, rec, mode
        self.fps, self.loop, self.resize = max(0.1, float(fps)), loop, resize
        self.r, self.maxlen = redis, maxlen
        self.latest = latest
        self.upload_url = (upload_url or "").rstrip("/")
        self.stop = stop or threading.Event()
        self.run_label = run_label
        self.host = os.environ.get("DS_NODE_NAME") or socket.gethostname()
        self.seq = 0
        self.n_state = 0
        self.sent = self.states = self.errors = 0
        self.err = None
        self.done = threading.Event()
        self.t = None

    # 한 장 · 한 상태
    def _emit_state(self, rec, now):
        payload = state_payload(rec, now)
        if self.mode == "redis":
            self.n_state += 1
            header = {"stream": self.name, "n": self.n_state, "topic": rec.get("topic"), "qos": rec.get("qos"),
                      "retain": rec.get("retain"), "via": "replay:" + self.run_label,
                      "received_by": self.host, "received_at": iso_now(now)}
            self.r.xadd("state_stream:" + self.name,
                        {"topic": str(rec.get("topic")), "payload": payload,
                         "header": json.dumps(header, ensure_ascii=False)},
                        maxlen=self.maxlen, approximate=True)
        elif self.mode == "upload":
            obj = {"topic": rec.get("topic"), "payload": payload.decode("utf-8", "replace"),
                   "qos": rec.get("qos"), "retain": rec.get("retain")}
            if rec.get("enc") == "json":
                obj["payload"] = json.loads(payload)
            _post(self.upload_url + "/state", json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                  "application/json")
        else:
            return                                 # mjpeg: 카메라 창구에는 상태가 없다 (MQTT 는 흉내 내지 않는다)
        self.states += 1

    def emit(self, idx):
        path, sender0, _t, frame_file = self.rec.frames[idx]
        now = time.time()
        for rec in self.rec.states_before[idx]:
            self._emit_state(rec, now)
        image = path.read_bytes()
        image, size = resize_jpeg(image, self.resize)
        self.seq += 1
        sender = refresh_sender(sender0, self.seq, now, image, size)
        if self.mode == "redis":
            header = {"stream": self.name, "n": self.seq, "frame_file": None if size else frame_file,
                      "via": "replay:" + self.run_label, "received_by": self.host, "received_at": iso_now(now),
                      "sender": sender,
                      "replay": {"source": self.src, "run": self.run_label, "index": idx + 1}}
            self.r.xadd("camera_stream:" + self.name, frame_fields(image, header),
                        maxlen=self.maxlen, approximate=True)
        elif self.mode == "upload":
            body, ctype = _multipart([("image", "frame.jpg", "image/jpeg", image),
                                      ("meta", None, "application/json",
                                       json.dumps(sender, ensure_ascii=False).encode("utf-8"))])
            _post(self.upload_url + "/upload", body, ctype)
        if self.latest is not None:
            self.latest.put(self.seq, image, sender, now)
        self.sent += 1

    def run(self):
        n = len(self.rec.frames)
        gap = 1.0 / self.fps
        nxt = time.time()
        idx = 0
        try:
            while not self.stop.is_set():
                if idx >= n:
                    if not self.loop:
                        break
                    idx = 0
                try:
                    self.emit(idx)
                except Exception as e:  # noqa: BLE001 — 한 장 실패로 멈추지 않는다 (서버가 잠깐 꺼진 것 등)
                    self.errors += 1
                    if self.err != str(e)[:200]:
                        self.err = str(e)[:200]
                        log("%s 보내기 실패 (%s) — 계속" % (self.name, self.err))
                idx += 1
                nxt += gap
                delay = nxt - time.time()
                if delay < -1.0:                  # 많이 밀렸으면 따라잡으려 몰아 보내지 않는다
                    nxt = time.time()
                elif delay > 0:
                    self.stop.wait(delay)
        finally:
            self.done.set()

    def start(self):
        self.t = threading.Thread(target=self.run, name="replay-" + self.name, daemon=True)
        self.t.start()
        return self


# ---------------------------------------------------------------- 명령
def server_ports():
    """소스 → 데이터 서버 포트 (source_probe 와 같은 표를 글로 읽는다)"""
    try:
        import source_probe
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import source_probe
    return source_probe._tables()[0]


def _pair(values, n, what, default):
    values = list(values or [])
    if not values:
        return [default] * n
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise SystemExit("%s 는 한 번 또는 --source 수(%d)만큼 주세요 (%d개)" % (what, n, len(values)))
    return values


def trim(r, names, real):
    """시험 키만 지운다 — 진짜 소스 이름(real)은 지우지 않는다."""
    gone = []
    for nm in names:
        if nm in real:
            log("%s 는 진짜 소스 이름이라 지우지 않음 (--as 로 시험 이름을 주세요)" % nm)
            continue
        for k in ("camera_stream:" + nm, "state_stream:" + nm):
            if r.delete(k):
                gone.append(k)
    return gone


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="녹화 장으로 살아 있는 소스를 흉내 낸다 (파이 없이 끝까지 시험)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="자세한 설명은 파일 맨 위 주석 · data_stream_가이드.md 「파이 없이 끝까지 시험」")
    ap.add_argument("--source", action="append", required=True, metavar="소스",
                    help="녹화 소스 (stream_data/<소스>). 여러 번 주면 함께 흘린다")
    ap.add_argument("--run", action="append", metavar="실행",
                    help="latest(기본) · 실행 이름(20261126_200152) · 폴더 경로")
    ap.add_argument("--as", dest="as_name", action="append", metavar="이름",
                    help="흘릴 이름 (기본 = 소스 이름). 시험이면 replay-test-pi12 처럼")
    ap.add_argument("--mode", choices=("redis", "mjpeg", "upload"), default="redis")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--loop", action="store_true", help="끝나면 처음부터 (번호는 이어 늘어난다)")
    ap.add_argument("--seconds", type=float, default=0, help="이만큼 흘리고 끝낸다 (0 = 끝없이 / 한 바퀴)")
    ap.add_argument("--resize", metavar="WxH|배율", help="예: 640x360 또는 0.5 (기본: 원본 해상도)")
    ap.add_argument("--data", metavar="폴더", help="stream_data 폴더 (기본 DS_HOME/stream_data)")
    ap.add_argument("--redis", default=os.environ.get("REDIS_URL") or "redis://127.0.0.1:6380/0")
    ap.add_argument("--maxlen", type=int, default=300, help="Redis 스트림에 남길 개수 (서버와 같은 300)")
    ap.add_argument("--trim", action="store_true", help="끝날 때 시험 키(camera_stream:<이름> · state_stream:<이름>)를 지운다")
    ap.add_argument("--health", action="store_true",
                    help="[redis] 그 소스의 데이터 서버 포트에 /health · /frame · /stream 을 연다 (서버가 꺼져 있을 때)")
    ap.add_argument("--bind", default="127.0.0.1", help="[mjpeg] 열 주소 (기본 127.0.0.1)")
    ap.add_argument("--port", type=int, default=18090, help="[mjpeg] 첫 포트 (소스마다 +1, 기본 18090)")
    ap.add_argument("--to", action="append", metavar="URL",
                    help="[upload] 보낼 데이터 서버 (기본 http://127.0.0.1:<그 소스의 서버 포트>)")
    ap.add_argument("--json", action="store_true", help="끝 요약을 JSON 한 줄로")
    a = ap.parse_args(argv)
    try:
        a.resize = parse_resize(a.resize)
    except ValueError as e:
        ap.error("--resize: %s" % e)
    return a


def main(argv=None):
    a = parse_args(argv)
    n = len(a.source)
    runs = _pair(a.run, n, "--run", "latest")
    if a.as_name and len(a.as_name) != n:
        raise SystemExit("--as 는 --source 수(%d)만큼 차례로 주세요 (%d개)" % (n, len(a.as_name)))
    names = list(a.as_name or a.source)
    if len(set(names)) != len(names):
        raise SystemExit("흘릴 이름이 겹칩니다: %s" % names)
    root = Path(a.data).expanduser().resolve() if a.data else data_root()

    stop = threading.Event()
    r = None
    if a.mode == "redis" or a.trim:
        import redis as redis_mod
        r = redis_mod.Redis.from_url(a.redis, socket_connect_timeout=2, socket_timeout=5)
        r.ping()
    ports = server_ports() if (a.mode == "upload" or a.health) else {}
    tos = _pair(a.to, n, "--to", None)

    players, servers = [], []
    try:
        for i, (src, run, nm) in enumerate(zip(a.source, runs, names)):
            folder = find_run(src, run, root)
            rec = Recording(folder, root)
            if not rec.frames:
                raise SystemExit("%s 에 장이 없음" % folder)
            label = "%s/%s" % (src, folder.name)
            latest = Latest() if (a.mode == "mjpeg" or a.health) else None
            upload = None
            if a.mode == "upload":
                upload = tos[i] or ("http://127.0.0.1:%d" % ports[nm] if nm in ports else None)
                if not upload:
                    raise SystemExit("%s 의 데이터 서버 포트를 모름 — --to http://주소:포트 로 주세요" % nm)
            if a.mode == "mjpeg":
                port = a.port + i
                try:
                    servers.append(start_http(a.bind, port, make_handler(nm, latest, stop)))
                except OSError as e:
                    raise SystemExit("%s:%d 를 못 엶 (%s) — 다른 replay 가 떠 있나? --port 로 바꾸세요" % (a.bind, port, e))
                where = "http://%s:%d/stream/1" % (a.bind, port)
            elif a.mode == "upload":
                where = upload + "/upload"
            else:
                where = "redis camera_stream:%s" % nm
                if a.health:
                    if nm in ports:
                        try:
                            servers.append(start_http("127.0.0.1", ports[nm], make_handler(nm, latest, stop)))
                            where += " · /health :%d" % ports[nm]
                        except OSError as e:
                            log("%s :%d 를 못 엶 (%s) — 데이터 서버가 떠 있나? /health 없이 계속" % (nm, ports[nm], e))
                    else:
                        log("%s 는 서버 포트가 없는 이름 — --health 생략 (source_probe 는 Redis 나이로 live)" % nm)
            players.append(Player(src, nm, rec, a.mode, a.fps, a.loop, a.resize, r, a.maxlen,
                                  latest, upload, stop, label))
            log("%s ← %s (%d장 · 상태 %d건) → %s  %.1ffps%s" % (
                nm, folder, len(rec.frames), len(rec.states), where, a.fps, " loop" if a.loop else ""))
    except BaseException:
        for s in servers:
            s.shutdown()
            s.server_close()
        raise

    def _stop(*_):
        stop.set()
    try:
        signal.signal(signal.SIGTERM, _stop)
    except (ValueError, OSError):
        pass

    len0 = {}
    if r is not None and a.mode == "redis":
        len0 = {p.name: r.xlen("camera_stream:" + p.name) for p in players}
    t0 = time.time()
    for p in players:
        p.start()
    try:
        while not stop.is_set():
            if all(p.done.is_set() for p in players):
                break
            if a.seconds and time.time() - t0 >= a.seconds:
                break
            stop.wait(0.2)
    except KeyboardInterrupt:
        pass
    stop.set()
    for p in players:
        if p.t:
            p.t.join(3.0)
    for s in servers:
        s.shutdown()
        s.server_close()

    out = {"mode": a.mode, "seconds": round(time.time() - t0, 1), "sources": []}
    for p in players:
        row = {"source": p.src, "name": p.name, "run": p.run_label, "sent": p.sent, "states": p.states,
               "errors": p.errors, "last_seq": p.seq}
        if r is not None and a.mode == "redis":
            row["xlen_before"] = len0.get(p.name)
            row["xlen_after"] = r.xlen("camera_stream:" + p.name)
        out["sources"].append(row)
    if a.trim and r is not None:
        out["trimmed"] = trim(r, names, set(ports or server_ports()) | set(a.source))
    if a.json:
        print(json.dumps(out, ensure_ascii=False), flush=True)
    else:
        for row in out["sources"]:
            log("끝 %s: 보낸 장 %d · 상태 %d · 오류 %d%s" % (
                row["name"], row["sent"], row["states"], row["errors"],
                " · XLEN %s → %s" % (row["xlen_before"], row["xlen_after"]) if "xlen_after" in row else ""))
        if "trimmed" in out:
            log("지움: %s" % (" · ".join(out["trimmed"]) or "없음"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
