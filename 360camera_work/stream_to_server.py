"""
stream_to_server.py — pro2_auto_bg.py 의 영상판 (사진 촬영 → 실시간 프레임)

[무엇을 대체하나]
  기존 : 촬영(20s) → 다운로드 → ProStitcher stitch(50s) → 업로드     ← stitch 가 병목
  이것 : 인카메라 실시간 스티칭 RTMP → 프레임 뽑기 → 업로드           ← stitch 없음

  서버(pano_receiver, anchor360_live)와 유니티는 한 줄도 바꾸지 않는다.
  같은 /upload_pano 에 같은 equirect.jpg 를 올리므로 live.py 감시 루프가 그대로 집어간다.
  (2026-10-01 부터 같은 사진을 10001 데이터 스트림에도 보낸다 — 아래 [2026-10-01 수정])

[2026-10-14 수정 — 통합데이터규약 v2 (CAM360-01)]
  장마다 보내는 폼을 바꿨다 (첫 번째 7866 · 두 번째부터 10001 모두 같은 폼).
       capture_id  그대로 (pano_receiver 의 저장 폴더 이름)
       camera_id   값이 cam360-001_main 으로 바뀐다 (--camera-id 기본값). meta 의 camera_id 와 같은 값
       meta        새 칸, application/json — 프레임 헤더 (규약 §3-5):
                   schema_version "2.0" · source_id "cam360-001" · node_id(이 노트북 이름) · zone_id "zoneA" ·
                   camera_id · sequence_id(정수) · session_id(기동마다 12자리) · timestamp(ISO 밀리초 +09:00) ·
                   encoding "jpeg" · width · height
       seq · timestamp 글자 칸은 지웠다 (meta 가 대신한다)
  sequence_id 는 예전 seq 와 같은 값 — 목표 fps 로 고른 장의 순번이다. 고르는 순간 매기고 업로드에 실패해도
  번호가 오르므로 받는 쪽은 실패를 불연속으로 본다. 솎아 버린 장은 번호를 매기지 않는다.
  timestamp 는 그 장을 카메라 스트림에서 받은 시각(예전 timestamp 칸과 같은 순간)이다.
  pano_receiver 는 meta 를 같은 폴더의 equirect.json 에 남기고, anchor360_live 가 그것으로 원본 프레임을 가리킨다.
  ※ 아래 [2026-10-01 수정] 2) 의 seq · timestamp 칸 설명은 이력이다.

[2026-10-01 수정 — 데이터 스트림(server_stream_multi_source.py) 연결]
  1) 기본으로 서버 두 곳에 보낸다
       7866  pano_receiver → anchor360_live → 유니티 (예전 그대로. 응답을 기다리고 시간을 잰다)
       10001 데이터 스트림 파일의 360 입구 (같은 /upload_pano 모양. 따로 보내서 7866 을 막지 않는다)
     --upload 를 주면 기본 두 곳 대신 준 곳들로만 보낸다.
  2) 장마다 두 칸을 더 보낸다 (나머지 칸·파일은 그대로)
       seq       = 보낸 순번 n_up (1부터, 화면의 #번호와 같다)
       timestamp = 그 장을 카메라 스트림에서 받은 시각 t_frame (이 노트북 시계, epoch 초)
     pano_receiver(7866)는 모르는 칸을 무시하므로 그쪽에는 영향이 없다.
  3) --upload 를 여러 번 주면 같은 사진을 모두에 보낸다.
     첫 번째는 지금처럼 응답까지 기다린다 (업로드·PC구간 측정이 예전과 같다).
     두 번째부터는 따로 보낸다 — 느리거나 꺼진 곳이 첫 번째와 카메라 읽기를 막지 않는다.
  4) 영상이 끊기면 스스로 다시 붙는다. 새 프레임이 RECONNECT_S(5초) 넘게 없으면
     같은 주소로 다시 열기 → 카메라 프리뷰 다시 켜기 → 제어 세션부터 다시 잡기 순으로 될 때까지 시도한다.

[지연 누적 방지 — 이 스크립트의 핵심]
  cv2.VideoCapture 는 RTMP 프레임을 내부 버퍼에 쌓는다. 메인 루프가 5fps 로
  read() 를 부르면 15fps 로 들어오는 프레임이 계속 밀려 '오래된 프레임'이 나온다.
  사진 방식에서 stitch 큐가 적체된 것과 같은 현상이 버퍼 안에서 재현된다.
  그래서 리더 스레드가 쉬지 않고 read() 해서 최신 한 장만 들고 있고,
  메인 루프는 그 최신 장만 집어간다. 오래된 프레임은 즉시 버린다.

[디스크]
  서버에 capture_id 폴더 하나를 계속 덮어쓴다. 디스크 증가 0.  (7866 pano_receiver 기준.
  10001 데이터 스트림은 받은 장을 전부 파일로 남긴다)
  PC 에는 아무것도 저장하지 않는다(메모리에서 바로 업로드).

[서버 쪽 주의]
  anchor360_live.py 의 --poll 기본값이 1.0초다. 5fps 로 올려도 초당 1장만 집는다.
  반드시 --poll 0.1 로 낮춰서 띄울 것.

[사용]
  python stream_to_server.py                      # 서버 7866 + 10001 로 5fps 무한
  python stream_to_server.py --fps 3              # 3fps
  python stream_to_server.py --count 30           # 30장만 올리고 종료(측정용)
  python stream_to_server.py --dry-run --count 20 # 업로드 없이 스트림 성능만
  python stream_to_server.py --upload http://210.110.250.33:7866/upload_pano
                                                  # 예전처럼 7866 한 곳만
  python stream_to_server.py --upload http://210.110.250.33:7866/upload_pano --upload http://127.0.0.1:10001/upload_pano
                                                  # 7866 + 이 노트북에서 도는 데이터 스트림 파일

[의존]
  requests, opencv-python

[종료]
  Ctrl+C  → _stopPreview 후 통계 출력
"""

import sys
import json
import time
import uuid
import socket
import signal
import argparse
import threading
import statistics
from datetime import datetime, timedelta, timezone

import cv2
import requests

DEF_IP         = "192.168.100.124"
DEF_UPLOAD     = "http://210.110.250.33:7866/upload_pano"
DEF_UPLOAD_STREAM = "http://210.110.250.33:10001/upload_pano"  # 데이터 스트림 360 입구 (2026-10-01 추가)
DEF_CAPTURE_ID = "LIVE_STREAM"
DEF_CAMERA_ID  = "cam360-001_main"     # 2026-10-14 통합데이터규약 v2 (예전 "pro2_anchor")
SOURCE_ID      = "cam360-001"          # 카메라가 달린 장비 (규약 §2-3)
ZONE_ID        = "zoneA"
SESSION_ID     = uuid.uuid4().hex[:12] # 기동할 때 한 번 — sequence_id 가 다시 시작되는 경계
NODE_ID        = socket.gethostname()  # 이 노트북
KST            = timezone(timedelta(hours=9))


def iso_ms(t=None):
    """epoch 초 → '2026-10-12T14:10:31.086+09:00'. 밀리초 아래는 버린다 (규약 §2-2)."""
    return datetime.fromtimestamp(time.time() if t is None else t, KST).isoformat(timespec="milliseconds")

RECONNECT_S     = 5.0     # 새 프레임이 이만큼 없으면 끊긴 것으로 보고 다시 붙는다
READ_TIMEOUT_MS = 5000    # 프레임 읽기 제한시간 (OpenCV 4.6 이상). 기본 30초 대신 5초 안에 멈춘 연결을 알아챈다
                          # 열기 제한시간은 기본(30초) 그대로 둔다 — 실측 콜드스타트 1.9~5초

fingerprint = None
hb_stop = threading.Event()
hb_thread = None
CMD_URL = None
STATE_URL = None
_running = True


# ── 카메라 제어 (preview_probe.py 와 동일) ───────────────
def headers():
    h = {"Content-Type": "application/json"}
    if fingerprint:
        h["Fingerprint"] = fingerprint
    return h


def heartbeat():
    while not hb_stop.is_set():
        try:
            requests.post(STATE_URL, json={}, headers=headers(), timeout=3)
        except Exception:
            pass
        time.sleep(1)


def connect(max_retry=3):
    global fingerprint
    for attempt in range(1, max_retry + 1):
        try:
            body = requests.post(CMD_URL,
                                 json={"name": "camera._connect", "parameters": {}},
                                 timeout=10).json()
            fp = body.get("results", {}).get("Fingerprint")
            if fp:
                fingerprint = fp
                print("[연결] Fingerprint=%s" % fp)
                return True
            err = body.get("error", {})
            print("[연결] 시도 %d: %s — %s"
                  % (attempt, err.get("code"), err.get("description")))
            if err.get("description") == "already connected by another":
                print("        → 다른 창(preview_probe / pro2_auto_bg)이 세션을 쥐고 있습니다.")
                return False
        except Exception as e:
            print("[연결] 시도 %d 실패: %s" % (attempt, e))
        time.sleep(1.5)
    return False


def send(name, params=None, timeout=20):
    body = {"name": name}
    if params is not None:
        body["parameters"] = params
    try:
        return requests.post(CMD_URL, json=body, headers=headers(), timeout=timeout).json()
    except Exception as e:
        return {"state": "exception", "error": {"description": str(e)}}


def start_preview(args):
    params = {
        "origin": {"mime": "h264", "width": 1920, "height": 1440,
                   "framerate": 30, "bitrate": 20480},
        "stiching": {"mode": "pano", "mime": "h264",
                     "width": args.width, "height": args.height,
                     "framerate": 30, "bitrate": args.bitrate},
        "stabilization": True,
    }
    send("camera._stopPreview", {})          # 남아 있을 수 있으니 먼저 끈다
    time.sleep(1.0)
    res = send("camera._startPreview", params)
    if res.get("state") != "done":
        err = res.get("error", {})
        print("[프리뷰] 실패: %s — %s" % (err.get("code"), err.get("description")))
        return None
    url = res.get("results", {}).get("_previewUrl", "")
    # 카메라가 자기 자신 기준(127.0.0.1)으로 주므로 카메라 IP 로 바꾼다
    url = url.replace("//127.0.0.1", "//" + args.ip).replace("//localhost", "//" + args.ip)
    if "://" in url and ":" not in url.split("://", 1)[1].split("/", 1)[0]:
        scheme, rest = url.split("://", 1)
        host, path = rest.split("/", 1)
        url = "%s://%s:1935/%s" % (scheme, host, path)
    print("[프리뷰] %s  (%dx%d, %d kbps)" % (url, args.width, args.height, args.bitrate))
    return url


# ── 최신 프레임 한 장만 들고 있는 리더 ───────────────────
class Latest:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.t = 0.0
        self.seq = 0
        self.dropped = 0

    def put(self, f, t):
        with self.lock:
            if self.frame is not None:
                self.dropped += 1      # 메인이 안 집어간 것 = 버린 프레임
            self.frame, self.t, self.seq = f, t, self.seq + 1

    def take(self):
        with self.lock:
            f, t, s = self.frame, self.t, self.seq
            self.frame = None
            return f, t, s


def open_capture(url):
    """스트림을 연다. 열리면 cap, 못 열면 None.
    OpenCV 4.6 이상이면 읽기 제한시간을 건다 — 멈춘 연결에서 read() 가 30초씩 묶이지 않게."""
    cap = None
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS])
        except Exception:                      # 제한시간 인자를 못 받는 OpenCV
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(url)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def reader(cap, latest, stop, stat, reopen):
    """
    cap.read() 가 진행 중일 때 다른 스레드가 cap.release() 를 부르면
    libavcodec 이 'Assertion fctx->async_lock failed' 로 프로세스를 죽인다.
    그래서 cap 의 수명을 이 스레드가 전적으로 소유하고, 여기서만 해제한다.
    끊긴 스트림을 풀고 다시 여는 것도 같은 이유로 이 스레드가 한다 (2026-10-01).
    """
    last_ok = time.time()
    try:
        while not stop.is_set():
            try:
                ok, f = cap.read() if cap is not None else (False, None)
            except Exception:                  # 읽기 중 예외도 끊김과 같이 다룬다
                ok, f = False, None
            if ok:
                stat["read_ok"] += 1
                last_ok = time.time()
                latest.put(f, last_ok)
                continue
            stat["read_fail"] += 1
            if stop.is_set():
                break
            if time.time() - last_ok < RECONNECT_S:
                time.sleep(0.05)
                continue
            # ── 끊김: 쥐고 있던 cap 을 풀고 다시 연다 ──
            print("\n[스트림] %.0f초 동안 새 프레임이 없습니다 — 다시 붙습니다" % (time.time() - last_ok))
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
            cap = reopen()                     # 될 때까지 시도한다. 멈추라고 하면 None
            if cap is not None:
                stat["reconnect"] += 1
            last_ok = time.time()
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


# ── 두 번째부터의 보낼 곳 (2026-10-01) ───────────────────
def short_name(url):
    """'http://210.110.250.33:7866/upload_pano' → '210.110.250.33:7866'"""
    return url.split("://", 1)[-1].split("/", 1)[0]


class ExtraUploader:
    """
    --upload 두 번째부터. 따로 도는 스레드가 보낸다 — 느리거나 꺼진 곳이
    첫 번째 업로드와 카메라 읽기를 막지 않게. 보내는 중에 새 장이 오면
    못 보낸 장은 버리고 최신 장으로 바꾼다 (Latest 와 같은 원리).
    """

    def __init__(self, url):
        self.url = url
        self.name = short_name(url)
        self.cv = threading.Condition()
        self.item = None
        self.closing = False
        self.sent = self.failed = self.skipped = 0
        self.status = "-"
        self.ups = []
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def put(self, form, data, meta=None):
        """meta = 프레임 헤더 JSON 글자 (첫 번째 업로드와 같은 것)"""
        with self.cv:
            if self.item is not None:
                self.skipped += 1              # 아직 못 보낸 장 = 최신 장으로 바꿔치기
            self.item = (form, data, meta)
            self.cv.notify()

    def close(self, timeout=3.0):
        with self.cv:
            self.closing = True
            self.cv.notify()
        self.th.join(timeout)

    def _run(self):
        sess = requests.Session()
        while True:
            with self.cv:
                while self.item is None and not self.closing:
                    self.cv.wait(0.5)
                if self.item is None:
                    return
                form, data, meta = self.item
                self.item = None
            t1 = time.perf_counter()
            files = {"file": ("equirect.jpg", data, "image/jpeg")}
            if meta is not None:
                files["meta"] = (None, meta, "application/json")
            try:
                r = sess.post(self.url, data=form, files=files, timeout=15)
                self.status = r.json().get("status", "?")
                self.sent += 1
            except Exception as e:
                self.status = "ERR:%s" % type(e).__name__     # 한 줄에 여러 곳이 찍히므로 짧게
                self.failed += 1
            self.ups.append(time.perf_counter() - t1)


# ── main ─────────────────────────────────────────────────
def main():
    global CMD_URL, STATE_URL, hb_thread, _running

    ap = argparse.ArgumentParser(description="Pro2 실시간 스트림 → 서버 업로드")
    ap.add_argument("--ip", default=DEF_IP)
    ap.add_argument("--upload", action="append", default=None, metavar="URL",
                    help="보낼 곳. 여러 번 주면 같은 사진을 모두에 보낸다. "
                         "안 주면 %s 와 %s 두 곳" % (DEF_UPLOAD, DEF_UPLOAD_STREAM))
    ap.add_argument("--capture-id", default=DEF_CAPTURE_ID,
                    help="서버에 덮어쓸 폴더명. 고정이라 디스크가 늘지 않는다")
    ap.add_argument("--camera-id", default=DEF_CAMERA_ID,
                    help="카메라 키 (기본 %s). meta 의 camera_id 와 pano_receiver 저장 폴더 이름이 된다" % DEF_CAMERA_ID)
    ap.add_argument("--rotate", type=int, default=0,
                    help="capture_id 를 N개 폴더로 돌려쓴다. 0=고정 1개. "
                         "서버 live.py 의 wait_stable 이 '쓰이는 중'으로 오판해 "
                         "1.5초를 버리는 것을 피하려면 8 이상 권장")
    ap.add_argument("--fps", type=float, default=5.0, help="서버로 올릴 목표 fps")
    ap.add_argument("--quality", type=int, default=92, help="JPEG 품질 (1~100)")
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--bitrate", type=int, default=10240)
    ap.add_argument("--count", type=int, default=0, help="N장 올리고 종료 (0=무한)")
    ap.add_argument("--dry-run", action="store_true", help="업로드 없이 스트림만 측정")
    ap.add_argument("--url", default=None, help="스트림 주소 직접 지정")
    args = ap.parse_args()

    # 첫 번째(7866)는 예전처럼 응답을 기다리고, 두 번째(10001)부터는 따로 보낸다.
    # --upload 를 주면 기본 두 곳 대신 준 곳들로만 보낸다.
    uploads = args.upload or [DEF_UPLOAD, DEF_UPLOAD_STREAM]

    CMD_URL   = "http://%s:20000/osc/commands/execute" % args.ip
    STATE_URL = "http://%s:20000/osc/state" % args.ip

    print("=" * 70)
    print("stream_to_server — 카메라 %s → %s" % (args.ip, uploads[0]))
    for u in uploads[1:]:
        print("  + 같은 사진을 따로 보낼 곳 → %s" % u)
    print("  목표 %.1f fps | JPEG q%d | capture_id=%s%s%s"
          % (args.fps, args.quality, args.capture_id,
             ("  (%d개 회전)" % args.rotate) if args.rotate > 0 else "  (고정)",
             "  [DRY-RUN]" if args.dry_run else ""))
    print("=" * 70)

    if not connect():
        sys.exit(1)
    hb_stop.clear()
    hb_thread = threading.Thread(target=heartbeat, daemon=True)
    hb_thread.start()
    time.sleep(0.5)

    url = args.url or start_preview(args)
    if not url:
        hb_stop.set()
        sys.exit(2)

    t_open = time.time()
    cap = open_capture(url)
    if cap is None:
        print("[스트림] 열기 실패: %s" % url)
        send("camera._stopPreview", {})
        hb_stop.set()
        sys.exit(3)
    print("[스트림] 열림 %.2fs" % (time.time() - t_open))

    latest = Latest()
    stop = threading.Event()
    rstat = {"read_ok": 0, "read_fail": 0, "reconnect": 0}
    url_box = [url]

    def reopen():
        """끊긴 스트림을 다시 연다 (리더 스레드가 부른다). 될 때까지 시도하고, 멈추라고 하면 None.
        1번째: 같은 주소로 다시 열기 → 2번째부터: 카메라 프리뷰를 다시 켜고 열기
        (프리뷰가 안 켜지면 제어 세션부터 다시 잡는다 — 카메라가 재부팅됐거나 세션이 끊긴 경우)"""
        attempt = 0
        while not stop.is_set():
            attempt += 1
            if attempt >= 2 and not args.url:
                new_url = start_preview(args)
                if not new_url and not stop.is_set():
                    print("[재연결] 프리뷰가 안 켜집니다 — 카메라 제어 세션부터 다시 잡습니다")
                    if connect():
                        new_url = start_preview(args)
                    else:
                        print("[재연결] 세션을 새로 못 잡음 — 이 스크립트의 세션이 아직 살아 있어도"
                              " 이렇게 나온다. 다음 시도에서 프리뷰부터 다시 켠다")
                if new_url:
                    url_box[0] = new_url
            if stop.is_set():
                return None
            t0 = time.time()
            c = open_capture(url_box[0])
            if c is not None:
                print("[재연결] 스트림 다시 열림 (%d번째 시도, 여는 데 %.1fs)" % (attempt, time.time() - t0))
                return c
            wait = min(10.0, 2.0 * attempt)
            print("[재연결] %d번째 시도 실패 — %.0f초 뒤 다시" % (attempt, wait))
            stop.wait(wait)
        return None

    th = threading.Thread(target=reader, args=(cap, latest, stop, rstat, reopen), daemon=True)
    th.start()

    sess = requests.Session()
    extras = [] if args.dry_run else [ExtraUploader(u) for u in uploads[1:]]
    ages, encs, ups, totals = [], [], [], []
    n_up = 0
    t_start = time.time()
    period = 1.0 / args.fps
    next_t = time.time()

    def bye(sig, frm):
        global _running
        _running = False
    signal.signal(signal.SIGINT, bye)

    print("\n프레임나이 = 카메라에서 받은 뒤 집을 때까지 기다린 시간 (작을수록 실시간)")
    print("PC구간     = 나이 + JPEG인코딩 + 업로드\n")

    while _running and (args.count == 0 or n_up < args.count):
        now = time.time()
        if now < next_t:
            time.sleep(min(0.005, next_t - now))
            continue
        next_t += period
        if next_t < now:                      # 밀렸으면 따라잡지 말고 현재로 리셋
            next_t = now + period

        frame, t_frame, seq = latest.take()
        if frame is None:
            continue

        age = now - t_frame
        t0 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        t_enc = time.perf_counter() - t0
        if not ok:
            continue
        data = buf.tobytes()

        cap_id = (args.capture_id if args.rotate <= 0
                  else "%s_%02d" % (args.capture_id, n_up % args.rotate))
        meta = {"schema_version": "2.0", "source_id": SOURCE_ID, "node_id": NODE_ID, "zone_id": ZONE_ID,
                "camera_id": args.camera_id,
                "sequence_id": n_up + 1,          # 고른 장의 순번 = 예전 seq (화면의 #번호와 같다)
                "session_id": SESSION_ID,
                "timestamp": iso_ms(t_frame),     # 카메라 스트림에서 이 장을 받은 시각
                "encoding": "jpeg",
                "width": int(frame.shape[1]), "height": int(frame.shape[0])}
        meta_json = json.dumps(meta, ensure_ascii=False)
        form = {"capture_id": cap_id, "camera_id": args.camera_id}
        files = {"file": ("equirect.jpg", data, "image/jpeg"),
                 "meta": (None, meta_json, "application/json")}

        if not args.dry_run:
            for x in extras:                      # 두 번째부터는 따로 보낸다 — 기다리지 않는다
                x.put(form, data, meta_json)
        t1 = time.perf_counter()
        if args.dry_run:
            t_up = 0.0
            status = "dry"
        else:
            try:
                r = sess.post(uploads[0],
                              data=form,
                              files=files,
                              timeout=15)
                status = r.json().get("status", "?")
            except Exception as e:
                status = "ERR:%s" % str(e)[:40]
            t_up = time.perf_counter() - t1

        n_up += 1
        tot = age + t_enc + t_up
        ages.append(age); encs.append(t_enc); ups.append(t_up); totals.append(tot)
        print("  #%-4d 나이 %5.0fms  인코딩 %4.0fms  업로드 %5.0fms  | PC구간 %5.0fms"
              "  %4dKB  %s%s" % (n_up, age*1000, t_enc*1000, t_up*1000, tot*1000,
                                 len(data)//1024, status,
                                 "".join("  | %s %s" % (x.name, x.status) for x in extras)))

    # ── 정리 ──
    stop.set()
    send("camera._stopPreview", {})   # 스트림을 끊어야 read() 가 즉시 반환된다
    th.join(timeout=3.0)              # 리더가 스스로 cap.release() 하고 빠져나감
    for x in extras:
        x.close()                     # 보내던 장까지만 마치고 끝낸다
    hb_stop.set()

    el = time.time() - t_start
    print("\n" + "=" * 70)
    print("종료 — %.1f초 동안 %d장 업로드 (%.2f fps 실측)" % (el, n_up, n_up/el if el else 0))
    print("  리더 스레드 수신 %d프레임 (%.1f fps), read 실패 %d"
          % (rstat["read_ok"], rstat["read_ok"]/el if el else 0, rstat["read_fail"]))
    print("  버린 프레임 %d장  ← 목표 fps 로 솎아낸 결과 (정상)" % latest.dropped)
    if rstat["reconnect"]:
        print("  스트림이 끊겨 다시 붙은 횟수 %d" % rstat["reconnect"])
    for x in extras:
        print("  + %s  보냄 %d · 실패 %d · 밀려서 최신 장으로 바꿈 %d · 업로드 중앙 %.0fms"
              % (x.name, x.sent, x.failed, x.skipped,
                 statistics.median(x.ups) * 1000 if x.ups else 0.0))
    if totals:
        def s(v):
            v = sorted(v)
            return (statistics.median(v)*1000, v[int(len(v)*0.9)]*1000, v[-1]*1000)
        for name, v in [("프레임 나이", ages), ("JPEG 인코딩", encs),
                        ("업로드", ups), ("PC 구간 합계", totals)]:
            m, p90, mx = s(v)
            print("  %-12s 중앙 %6.0fms   p90 %6.0fms   최대 %6.0fms" % (name, m, p90, mx))
        print("\n  ※ 종단 지연 = 위 'PC 구간 합계' + 서버 live.py 감시주기 + live 총처리시간")
    print("=" * 70)


if __name__ == "__main__":
    main()
