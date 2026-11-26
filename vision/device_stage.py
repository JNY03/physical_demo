#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""device_stage.py — 기기 단계 실행체: 파이(또는 아무 CPU 기기)에서 고른 작은 모델 하나를 돌린다 (2026-11-26).

기능 계약의 **기기 단계**(예: object_detect.device, 역할 슬롯 detector)를 실제로 돌리는 프로그램이다.
k3s 가 docker/device-runner 이미지로 파드를 띄우고 이것을 부른다(orchestrator/k3s_executor.py 가 명령을 채운다).
혼자 도는 파일이다 — 표준 라이브러리 + numpy · cv2 · redis (+ 레지스트리를 읽을 때 PyYAML) + 고른 실행기만 쓴다.
torch · ultralytics · ds_resources 를 import 하지 않는다.

입력 (프레임 — 가장 최근 것 하나만, 밀린 것은 건너뛴다)
  1) 카메라 HTTP (MJPEG multipart, picam_mjpeg · realsense_mjpeg · a2-bridge 의 GET /stream/1)
     --camera-url URL 이면 그것. auto(기본)면 **이 파드가 그 카메라의 파이에 있을 때만**
     http://$HOST_IP:8090/stream/1 (HOST_IP 는 k3s 가 status.hostIP 로 준다, 없으면 127.0.0.1).
     「그 카메라의 파이」= NODE_NAME 이 소스 이름과 같거나 DS_SOURCE_NODE 와 같을 때.
  2) Redis camera_stream:<소스> (서버가 받아 넣는 것 — image 칸 = JPEG 바이트, header 칸 = JSON)
     HTTP 가 안 되면 여기로 내려가고 HTTP_RETRY_S 마다 다시 HTTP 를 본다. 너무 오래된 프레임(--max-age)은 버린다.

출력 (프레임은 내지 않는다 — 작은 결과만)
  Redis 스트림 device_stage:<소스>:<기능>  (최근 --maxlen 개)
    result  JSON {schema "dstage/1", ts(ISO ms), ts_unix, frame_seq, frame_ts, stream_entry_id, source, function,
                  stage "device", model, kind, unit_id, node, via(http|redis), width, height,
                  n(검출 총수), dets [{cls, name, conf, xyxy}] (상위 --top-k, 분할이면 polygon 을 24점 이하로),
                  depth(깊이 모델만: {min, p5, median, p95, max, valid_ratio, grid 6x8}),
                  proc_ms(디코드+추론), infer_ms(실행기가 잰 것), cpu_pct(이 프로세스, 직전 창), rss_mb, fps_target}
    header  JSON {stream, model, n, stream_entry_id, lag_ms} — vision_stream 과 같은 머리
  하트비트  cpu_telemetry:<UNIT_ID> (TTL 60초) = {"ts": epoch 초, schema "dstel/1", frames, fps, proc_ms_p50/p90,
                  cpu_pct, rss_mb, via, model, source, function, node, error}
            컨트롤러(ds_resources/controller.py heartbeat_age)가 이 ts 로 running 을 판정한다(30초 안).
            UNIT_ID 가 없으면(손으로 띄움) <기능>.device@<소스>.

모델 고르기
  --model <키>  vision/model_registry.yaml models.<키> 의 runner(「모듈:클래스」)로 vision/runners 에서 만든다.
                매개변수는 vision_config.yaml models.<키> → 없으면 레지스트리 models.<키>.config → 빈 사전.
                레지스트리에 없는 키면 vision/runners/<키>.py 의 Runner 클래스를 찾는다(이름만으로 더할 수 있게).
  --runner 모듈:클래스 · --cfg JSON  레지스트리 없이 직접 (시험용). cfg 안의 models/… · external/… 는 DS_HOME 기준.
  --imgsz · --threads 는 cfg 에 그 칸이 없을 때만 넣는다(모델 파일이 입력 크기를 고정한 경우 그쪽이 이긴다).

자원 계약 (기능 계약의 기기 단계: fps 1~5 · cpu_m · mem_mb)
  --fps 는 1~5 로 자른다. 처리 시간이 주기보다 길면 그만큼 늦게 돈다(쌓지 않는다).
  스레드는 --threads(기본: DS_CPU_M 또는 cpu_m 를 1000 으로 나눠 올림, 1~4). cv2 는 1 스레드.
  메모리 상한은 k3s limits 가 건다 — 여기서는 rss_mb 를 하트비트에 남기고 DS_MEM_MB 를 넘으면 한 번 알린다.

  python vision/device_stage.py --model yolo26n_ncnn --source pi13 --function object_detect --fps 2 --imgsz 320
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import sys
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
KST = timezone(timedelta(hours=9))
SCHEMA = "dstage/1"
TEL_SCHEMA = "dstel/1"
FPS_MIN, FPS_MAX = 1.0, 5.0              # 기기 단계 계약 (vision/model_registry.yaml functions.*.stages.device.fps)
HEARTBEAT_S = 5.0                         # 컨트롤러의 CPU_HEARTBEAT_FRESH_S(30초)보다 넉넉히 자주
TELEMETRY_TTL_S = 60
HTTP_RETRY_S = 30.0                       # Redis 로 내려간 뒤 HTTP 를 다시 볼 간격
HTTP_STALL_S = 5.0                        # HTTP 에서 이만큼 새 프레임이 없으면 Redis 로 내려간다
PATH_ROOTS = ("models", "external")       # cfg 안 상대 경로 중 DS_HOME 기준으로 푸는 첫 폴더 (ds_resources/settings.py 와 같다)
POLY_MAX = 24


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def iso_ms(t=None):
    return datetime.fromtimestamp(time.time() if t is None else t, KST).isoformat(timespec="milliseconds")


def ds_home():
    v = os.environ.get("DS_HOME")
    return Path(v).expanduser() if v else HERE.parent


def resolve_paths(v, home=None):
    """cfg 안의 models/… · external/… 상대 경로를 DS_HOME 기준 절대 경로로 (사전 · 목록은 안까지)"""
    home = ds_home() if home is None else Path(home)
    if isinstance(v, dict):
        return {k: resolve_paths(x, home) for k, x in v.items()}
    if isinstance(v, list):
        return [resolve_paths(x, home) for x in v]
    if isinstance(v, str) and v and not v.startswith(("/", "~")):
        head = v.replace("\\", "/").split("/", 1)[0]
        if head in PATH_ROOTS and "/" in v:
            return str(home / v)
    return v


def _num(v, kind):
    """머리에서 온 글자 숫자 → 숫자 (ISO 시각처럼 숫자가 아니면 그대로)"""
    if isinstance(v, (bytes, bytearray)):
        v = v.decode(errors="replace")
    if isinstance(v, str):
        try:
            return kind(float(v)) if kind is int else kind(v)
        except ValueError:
            return v
    return v


def clamp_fps(fps):
    try:
        f = float(fps)
    except (TypeError, ValueError):
        f = 2.0
    return min(FPS_MAX, max(FPS_MIN, f))


def threads_for(cpu_m):
    """cpu_m(밀리코어) → 추론 스레드 수 (1~4). 모르면 2"""
    try:
        return max(1, min(4, int(math.ceil(float(cpu_m) / 1000.0))))
    except (TypeError, ValueError):
        return 2


# ---------------------------------------------------------------- 모델
def _yaml(path):
    try:
        import yaml
    except ImportError:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except OSError:
        return {}


def model_spec(key, registry=None, config=None):
    """키 → (레지스트리 항목, cfg). 항목이 없으면 vision/runners/<키>.py:Runner 로 본다"""
    reg = _yaml(registry or HERE / "model_registry.yaml")
    spec = dict(((reg.get("models") or {}).get(key)) or {})
    if not spec.get("runner"):
        if (HERE / "runners" / ("%s.py" % key)).is_file():
            spec["runner"] = "%s:Runner" % key
        else:
            raise SystemExit("모델 %r 을 모른다 — 레지스트리 models.%s 도, vision/runners/%s.py 도 없다" % (key, key, key))
    vc = _yaml(config or HERE / "vision_config.yaml")
    cfg = (vc.get("models") or {}).get(key)
    if cfg is None:
        cfg = spec.get("config") or {}
    return spec, dict(cfg)


def make_runner(key, spec, cfg):
    sys.path.insert(0, str(HERE))
    from runners import load_runner
    return load_runner(key, spec, "cpu", cfg)


def compact_dets(dets, top_k, kind):
    """실행기 검출 → 결과에 실을 것 (상위 top_k, 밑줄 칸 · 마스크 버림, 분할 외곽은 POLY_MAX 점 이하로)"""
    out = []
    for d in sorted(dets or [], key=lambda d: -float(d.get("conf") or 0))[:max(0, int(top_k))]:
        x = {"cls": int(d.get("cls", -1)), "name": str(d.get("name", "")), "conf": round(float(d.get("conf") or 0), 3),
             "xyxy": [round(float(v), 1) for v in (d.get("xyxy") or [])[:4]]}
        if d.get("track_id") is not None:
            x["track_id"] = int(d["track_id"])
        poly = d.get("polygon")
        if poly is None and d.get("_poly") is not None:
            poly = d["_poly"]
        if kind == "segment" and poly is not None and len(poly):
            pts = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in poly]
            step = max(1, int(math.ceil(len(pts) / float(POLY_MAX))))
            x["polygon"] = pts[::step]
        out.append(x)
    return out


def depth_summary(depth):
    """깊이 지도 → 작은 요약 (상대 깊이 — 거리값으로 쓰지 않는다): 분위수 + 6x8 칸 중앙값"""
    import numpy as np
    d = np.asarray(depth, np.float32)
    ok = np.isfinite(d) & (d > 0)
    if not ok.any():
        return {"valid_ratio": 0.0}
    v = d[ok]
    q = np.percentile(v, [0, 5, 50, 95, 100])
    h, w = d.shape[:2]
    grid = []
    for i in range(6):
        row = []
        for j in range(8):
            c = d[i * h // 6:(i + 1) * h // 6, j * w // 8:(j + 1) * w // 8]
            cm = c[np.isfinite(c) & (c > 0)]
            row.append(round(float(np.median(cm)), 3) if cm.size else None)
        grid.append(row)
    return {"min": round(float(q[0]), 3), "p5": round(float(q[1]), 3), "median": round(float(q[2]), 3),
            "p95": round(float(q[3]), 3), "max": round(float(q[4]), 3), "valid_ratio": round(float(ok.mean()), 3),
            "grid": grid}


# ---------------------------------------------------------------- 입력
class MjpegReader:
    """MJPEG multipart 를 따로 읽어 **가장 최근 한 장**만 들고 있는다 (처리가 느려도 쌓이지 않게).
    머리: Content-Length (있으면), X-Frame-Seq · X-Timestamp (picam) 또는 Frame-Meta JSON (realsense)."""

    def __init__(self, url, timeout=5.0):
        self.url, self.timeout = url, timeout
        self.lock = threading.Lock()
        self.latest = None                      # (jpeg, seq, ts, 받은 시각)
        self.error = None
        self.stop = False
        self.got = 0
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        try:
            resp = urllib.request.urlopen(self.url, timeout=self.timeout)
            while not self.stop:
                head = {}
                line = resp.readline()
                if not line:
                    raise IOError("스트림이 끝남")
                if not line.startswith(b"--"):
                    continue
                while True:
                    line = resp.readline()
                    if not line:
                        raise IOError("스트림이 끝남")
                    line = line.strip()
                    if not line:
                        break
                    k, _, v = line.decode("latin-1").partition(":")
                    head[k.strip().lower()] = v.strip()
                n = int(head.get("content-length") or 0)
                if n <= 0:
                    raise IOError("Content-Length 없는 MJPEG 은 받지 않는다")
                jpeg = resp.read(n)
                seq, ts = head.get("x-frame-seq"), head.get("x-timestamp")
                if head.get("frame-meta"):
                    try:
                        meta = json.loads(head["frame-meta"])
                        seq = meta.get("sequence_id", seq)
                        ts = meta.get("timestamp", ts)
                    except ValueError:
                        pass
                with self.lock:
                    self.got += 1
                    self.latest = (jpeg, seq if seq is not None else self.got, ts, time.time())
        except Exception as e:  # noqa: BLE001 — 끊기면 부르는 쪽이 Redis 로 내려간다
            self.error = str(e)[:200]

    def alive(self):
        return self.t.is_alive()

    def take(self):
        with self.lock:
            x, self.latest = self.latest, None
        return x

    def close(self):
        self.stop = True


class FrameInput:
    """HTTP 를 먼저, 안 되면 Redis camera_stream:<소스>. next() → dict 또는 None (새 프레임 없음)"""

    def __init__(self, r, source, camera_url=None, max_age=5.0):
        self.r, self.source, self.url, self.max_age = r, source, camera_url, max_age
        self.key = "camera_stream:" + source
        self.http = None
        self.http_last = 0.0
        self.http_down_at = None
        self.last_id = None
        self.via = None
        if self.url:
            self._open_http()

    def _open_http(self):
        self.http = MjpegReader(self.url)
        self.http_last = time.time()
        self.http_down_at = None

    def _http_dead(self, why):
        log("카메라 HTTP %s 안 됨 (%s) — Redis %s 로" % (self.url, why, self.key))
        if self.http:
            self.http.close()
        self.http = None
        self.http_down_at = time.time()

    def next(self):
        now = time.time()
        if self.url and self.http is None and self.http_down_at and now - self.http_down_at >= HTTP_RETRY_S:
            self._open_http()
        if self.http is not None:
            x = self.http.take()
            if x is not None:
                self.http_last = now
                self.via = "http"
                jpeg, seq, ts, _at = x
                return {"jpeg": jpeg, "seq": seq, "ts": ts, "entry_id": None, "via": "http"}
            if not self.http.alive():
                self._http_dead(self.http.error or "끊김")
            elif now - self.http_last > HTTP_STALL_S:
                self._http_dead("%.0f초 새 프레임 없음" % (now - self.http_last))
            else:
                return None
        return self._redis()

    def _redis(self, block_ms=1000):
        # 「$ 다음에 오는 한 장」을 기다린다 — 최신 프레임을 고르려고 xrevrange 를 되풀이하면 같은 JPEG 을
        # 몇 번이고 받아 테일넷 대역을 먹는다. 처리하는 사이 들어온 것은 건너뛴다(실시간 우선, vision_infer 와 같다)
        got = self.r.xread({self.key: "$"}, count=1, block=int(block_ms))
        if not got:
            return None
        eid, f = got[0][1][0]
        eid = eid.decode() if isinstance(eid, bytes) else str(eid)
        if eid == self.last_id:
            return None
        self.last_id = eid
        ms = int(eid.split("-")[0])
        if self.max_age and time.time() - ms / 1000.0 > self.max_age:
            return None                          # 옛 프레임 — 카메라가 멈춘 것. 결과를 내지 않는다
        g = {(k.decode() if isinstance(k, bytes) else k): v for k, v in f.items()}
        seq = None
        try:
            h = json.loads(g.get("header") or b"{}")
            seq = h.get("n")
            snd = h.get("sender") or {}
            seq = snd.get("sequence_id", seq)
        except ValueError:
            h = {}
        ts = g.get("timestamp")
        if isinstance(ts, bytes):
            ts = ts.decode(errors="replace")
        self.via = "redis"
        return {"jpeg": g.get("image") or b"", "seq": seq if seq is not None else eid, "ts": ts,
                "entry_id": eid, "via": "redis", "put_ms": ms}

    def close(self):
        if self.http:
            self.http.close()


def pick_camera_url(arg, source, env=None):
    """--camera-url → 실제 URL 또는 None(Redis 만). auto 는 그 카메라의 파이에서 돌 때만 로컬 카메라"""
    env = os.environ if env is None else env
    if arg in (None, "", "none", "redis"):
        return None
    if arg != "auto":
        return arg
    if env.get("DS_CAMERA_URL"):
        return env["DS_CAMERA_URL"]
    node = env.get("NODE_NAME") or ""
    if node and node in (source, env.get("DS_SOURCE_NODE") or ""):
        return "http://%s:%s/stream/1" % (env.get("HOST_IP") or "127.0.0.1", env.get("DS_CAMERA_PORT") or "8090")
    return None


# ---------------------------------------------------------------- 자원
class ProcStat:
    """이 프로세스의 CPU % (직전 창, 코어 하나 = 100) · RSS MB"""

    def __init__(self):
        self.t, self.c = time.time(), time.process_time()

    def cpu_pct(self):
        t, c = time.time(), time.process_time()
        dt = t - self.t
        pct = 100.0 * (c - self.c) / dt if dt > 0 else 0.0
        self.t, self.c = t, c
        return round(pct, 1)

    @staticmethod
    def rss_mb():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return round(int(line.split()[1]) / 1024.0, 1)
        except OSError:
            pass
        try:
            import resource
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        except Exception:  # noqa: BLE001
            return None


def pctl(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 1)


# ---------------------------------------------------------------- 돌리기
class DeviceStage:
    def __init__(self, r, runner, a, key, kind, camera_url):
        self.r, self.runner, self.a, self.key, self.kind = r, runner, a, key, kind
        self.fps = clamp_fps(a.fps)
        self.out_key = "device_stage:%s:%s" % (a.source, a.function)
        self.unit_id = a.unit_id or os.environ.get("UNIT_ID") or "%s.device@%s" % (a.function, a.source)
        self.tel_key = "cpu_telemetry:" + self.unit_id
        self.node = os.environ.get("NODE_NAME") or socket.gethostname()
        self.inp = FrameInput(r, a.source, camera_url, a.max_age)
        self.stat = ProcStat()
        self.win_ms, self.win_n, self.win_t = [], 0, time.time()
        self.frames = 0
        self.last_cpu = 0.0
        self.error = None
        self.mem_cap = float(os.environ.get("DS_MEM_MB") or 0) or None
        self.mem_warned = False
        self.stopping = False

    def heartbeat(self, force=False):
        now = time.time()
        if not force and now - self.win_t < HEARTBEAT_S:
            return
        dt = now - self.win_t
        self.last_cpu = self.stat.cpu_pct()
        rss = ProcStat.rss_mb()
        if self.mem_cap and rss and rss > self.mem_cap * 0.9 and not self.mem_warned:
            log("메모리 %.0f MB — 계약 mem_mb %.0f 의 90%% 를 넘음 (k3s 가 상한에서 죽인다)" % (rss, self.mem_cap))
            self.mem_warned = True
        doc = {"ts": now, "schema": TEL_SCHEMA, "unit_id": self.unit_id, "source": self.a.source,
               "function": self.a.function, "stage": "device", "model": self.key, "node": self.node,
               "via": self.inp.via, "frames": self.frames, "window_s": round(dt, 2),
               "fps": round(self.win_n / dt, 2) if dt > 0 else None, "fps_target": self.fps,
               "proc_ms_p50": pctl(self.win_ms, 0.5), "proc_ms_p90": pctl(self.win_ms, 0.9),
               "cpu_pct": self.last_cpu, "rss_mb": rss, "error": self.error}
        try:
            self.r.set(self.tel_key, json.dumps(doc, ensure_ascii=False), ex=TELEMETRY_TTL_S)
        except Exception as e:  # noqa: BLE001
            log("하트비트를 못 씀 (%s)" % e)
        self.win_ms, self.win_n, self.win_t = [], 0, now

    def process(self, fr):
        import cv2
        import numpy as np
        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(fr["jpeg"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.error = "JPEG 을 못 풀었다"
            return None
        res = {}
        if self.kind == "depth":
            info, depth, _K = self.runner(img)
            res["depth"] = depth_summary(depth)
            dets = []
        else:
            info, dets = self.runner(img)
        proc_ms = (time.perf_counter() - t0) * 1000.0
        now = time.time()
        h, w = img.shape[:2]
        rec = {"schema": SCHEMA, "ts": iso_ms(now), "ts_unix": round(now, 3), "frame_seq": _num(fr.get("seq"), int),
               "frame_ts": _num(fr.get("ts"), float), "stream_entry_id": fr.get("entry_id"), "source": self.a.source,
               "function": self.a.function, "stage": "device", "model": self.key, "kind": self.kind,
               "unit_id": self.unit_id, "node": self.node, "via": fr.get("via"), "width": w, "height": h,
               "n": len(dets or []), "dets": compact_dets(dets, self.a.top_k, self.kind),
               "proc_ms": round(proc_ms, 1), "infer_ms": (info or {}).get("infer_ms"),
               "cpu_pct": self.last_cpu, "rss_mb": ProcStat.rss_mb(), "fps_target": self.fps}
        rec.update(res)
        lag = int(now * 1000 - fr["put_ms"]) if fr.get("put_ms") else None
        header = {"stream": self.a.source, "model": self.key, "n": self.frames + 1,
                  "stream_entry_id": fr.get("entry_id"), "lag_ms": lag}
        self.r.xadd(self.out_key, {"result": json.dumps(rec, ensure_ascii=False),
                                   "header": json.dumps(header, ensure_ascii=False)},
                    maxlen=self.a.maxlen, approximate=True)
        self.frames += 1
        self.win_n += 1
        self.win_ms.append(proc_ms)
        self.error = None
        return rec

    def run(self):
        period = 1.0 / self.fps
        log("기기 단계 %s · 소스 %s · 모델 %s (%s) · %.1f fps · 입력 %s → %s · 하트비트 %s" % (
            self.a.function, self.a.source, self.key, self.kind, self.fps, self.inp.url or "redis", self.out_key,
            self.tel_key))
        self.heartbeat(force=True)
        redis_err = None
        while not self.stopping:
            t0 = time.time()
            fr = None
            try:
                fr = self.inp.next()
                if fr is not None:
                    self.process(fr)
                    if self.a.max_frames and self.frames >= self.a.max_frames:
                        break
                if redis_err is not None:
                    log("Redis 다시 됨")
                    redis_err = None
            except Exception as e:  # noqa: BLE001 — Redis · 실행기 오류로 죽지 않는다(하트비트에 남긴다)
                self.error = str(e)[:200]
                if redis_err is None:
                    log("처리 실패: %s" % self.error)
                redis_err = e
                time.sleep(1.0)
            try:
                self.heartbeat()
            except Exception:  # noqa: BLE001
                pass
            wait = period - (time.time() - t0) if fr is not None else min(0.05, period)
            if wait > 0:
                time.sleep(wait)
        self.heartbeat(force=True)
        self.inp.close()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="기기 단계 실행체 — 고른 작은 모델을 CPU 로 돌려 결과만 Redis 에 낸다")
    env = os.environ.get
    ap.add_argument("--model", default=env("DS_MODEL"), help="레지스트리 모델 키 (DS_MODEL)")
    ap.add_argument("--source", default=env("DS_SOURCE"), help="소스 이름 (DS_SOURCE)")
    ap.add_argument("--function", default=env("DS_FUNCTION") or "object_detect", help="기능 이름 (DS_FUNCTION)")
    ap.add_argument("--fps", type=float, default=float(env("DS_FPS") or 2), help="목표 fps (1~5 로 자른다)")
    ap.add_argument("--imgsz", type=int, default=None, help="입력 크기 — cfg 에 imgsz 가 없을 때만 넣는다")
    ap.add_argument("--threads", type=int, default=None, help="추론 스레드 (기본: DS_CPU_M 으로)")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--redis", default=env("REDIS_URL") or "redis://127.0.0.1:6380/0")
    ap.add_argument("--camera-url", default=env("DS_CAMERA") or "auto",
                    help="auto (그 카메라의 파이면 로컬 MJPEG) | none (Redis 만) | URL")
    ap.add_argument("--max-age", type=float, default=5.0, help="Redis 프레임이 이보다 오래되면 버린다 (초, 0=안 봄)")
    ap.add_argument("--maxlen", type=int, default=300)
    ap.add_argument("--unit-id", default=None)
    ap.add_argument("--runner", default=None, help="모듈:클래스 — 레지스트리 대신 (시험용)")
    ap.add_argument("--cfg", default=env("DS_MODEL_CFG"), help="실행기 cfg JSON — 레지스트리 · vision_config 대신")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--vision-config", default=None)
    ap.add_argument("--max-frames", type=int, default=0, help="이만큼 처리하고 끝낸다 (시험용, 0=끝없이)")
    a = ap.parse_args(argv)
    if not a.model and not a.runner:
        ap.error("--model (또는 DS_MODEL) 이 필요하다 — 기기 슬롯에 고른 모델이 없으면 띄우지 않는다")
    if not a.source:
        ap.error("--source (또는 DS_SOURCE) 가 필요하다")
    return a


def main(argv=None):
    a = parse_args(argv)
    cv_threads = 1
    try:
        import cv2
        cv2.setNumThreads(cv_threads)
    except ImportError:
        pass
    key = a.model or a.runner.split(":")[-1]
    if a.runner:
        spec = {"runner": a.runner}
        cfg = {}
    else:
        spec, cfg = model_spec(a.model, a.registry, a.vision_config)
    if a.cfg:
        cfg = json.loads(a.cfg)
    cfg = resolve_paths(cfg)
    if a.imgsz:
        cfg.setdefault("imgsz", a.imgsz)
    cfg.setdefault("threads", a.threads or threads_for(os.environ.get("DS_CPU_M")))
    import redis
    r = redis.Redis.from_url(a.redis, socket_connect_timeout=3, socket_timeout=10)
    t0 = time.perf_counter()
    runner = make_runner(key, spec, cfg)
    kind = getattr(runner, "kind", None) or spec.get("kind") or "detect"
    log("실행기 %s 준비 %.0f ms (threads %s)" % (key, (time.perf_counter() - t0) * 1000, cfg.get("threads")))
    st = DeviceStage(r, runner, a, key, kind, pick_camera_url(a.camera_url, a.source))

    def _stop(*_):
        st.stopping = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    st.run()
    close = getattr(runner, "close", None)
    if callable(close):
        close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
