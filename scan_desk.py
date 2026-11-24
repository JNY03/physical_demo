"""스캔 탐지 창구 — /scan/* (2026-11-24). 스캔탐지_창구_계획_261124.md · 스캔탐지창구_요청_VZ_261123.md.

가시화가 장치를 45° 씩 돌리며 각도마다 POST /scan/mark 를 보내면, 이미 돌고 있는 targeted 결과에서
**그 표시 뒤의 한 장**을 골라 붙들고, /scan/finish 에서 그 장의 값으로 판정 · 방위 · 거리를 모은다(vision/scan_assess.py).
새 추론은 하지 않고 장치도 움직이지 않는다.

Redis (서버 · 비전이 함께 쓰는 키)
  vision_scan                {scan_id, devices, targets}  판이 사는 동안 TTL 20초를 늘린다. 비전(LiveRunner.check_scan)이
                             읽어 targeted 대상을 바꾸고, 키가 사라지면(끝 · 취소 · 이 서버가 죽음) 원래 대상으로 되돌린다
  vision_scan_ack:<scan_id>  비전의 반영 응답 {ok, error, version}
  vision_scan_mark:<소스>     각도 표시 번호. 비전이 보고 그 기기의 트랙 · 거리 기억을 비우고 결과 profile.mark 에 싣는다
읽기: camera_stream:<소스> (원본) · vision_stream:<소스>:targeted (결과 · 오버레이)

판은 **서버 전체에 하나**다(대상 바꾸기가 비전 프로세스 단위라서 — 가시화 동의 2026-11-24).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("scan_assess", HERE / "vision" / "scan_assess.py")
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)

#: 가시화가 보내는 장치 ID → 소스 (2026-11-24 확인). go1-002(pi1)는 EP 로봇으로 바뀔 수 있다
DEVICE_IDS = {"go1-001": "robot1", "go1-002": "robot2", "a2-001": "a2"}

KEY_TTL_S = 20            # vision_scan 수명 — 이 서버가 죽으면 이 안에 비전이 원래 대상으로 돌아간다
LOADING_S = 60            # prepare → ready 한도
IDLE_S = 120              # 판 도중 요청이 이만큼 없으면 닫는다
SETTLE_MS = 500           # mark 뒤 이만큼 지난 원본부터 (회전 직후 흔들림)
CAPTURE_S = 1.85          # mark 가 기다리는 한도 (응답은 2초 안). 최악 = 0.5초 + 장 2개 처리 — demo_door 는 장당 0.5~0.7초(4050 노트북)
STREAM_FRESH_S = 5.0      # 소스 · 비전이 살아 있다고 보는 마지막 항목 나이
KEEP_DONE = 3             # 메모리에 남기는 끝난 판
TERMINAL = ("done", "failed", "cancelled")


class ScanError(Exception):
    def __init__(self, code, error, detail=None, **extra):
        super().__init__(detail or error)
        self.code, self.error, self.detail, self.extra = code, error, detail, extra


def _iso(ms):
    return datetime.fromtimestamp(ms / 1000.0, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _js(raw):
    try:
        v = json.loads(raw or b"{}")
    except (TypeError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


def _id_ms(eid):
    if isinstance(eid, bytes):
        eid = eid.decode()
    return int(str(eid).split("-")[0])


class Scan:
    def __init__(self, req, source, now):
        self.scan_id = req["scan_id"]
        self.device, self.source = req["device"], source
        self.target, self.query, self.query_ko = req["target"], req.get("query"), req.get("query_ko")
        self.steps, self.step_deg = req.get("steps"), req.get("step_deg")
        self.targets = [sa.target_name(self.target, self.query)]
        self.state, self.error = "loading", None
        self.profile_version = None
        self.created = self.touched = now
        self.frames = {}          # 각도 → {rec, header, overlay, original, frame_n, captured_at, order}
        self.order = 0
        self.waiter = None
        self.result = None
        self.computed_at = None


class ScanDesk:
    def __init__(self, redis_url, save_root=None, aliases=None, log=print, redis_client=None, clock=time.time):
        self.url, self.save_root, self.log, self.clock = redis_url, save_root, log, clock
        self.aliases = dict(aliases or {})
        self.aliases.update(DEVICE_IDS)
        self.lock = threading.RLock()
        self.scans = {}               # scan_id → Scan (진행 중 하나 + 끝난 것 몇 개)
        self.active = None
        self._r = redis_client
        self._stop = threading.Event()

    # -- Redis
    @property
    def r(self):
        if self._r is None:
            import redis
            self._r = redis.Redis.from_url(self.url, socket_connect_timeout=2, socket_timeout=5)
        return self._r

    def now_ms(self):
        s, us = self.r.time()                  # 원본 항목 ID 와 같은 시계(Redis 서버)
        return int(s) * 1000 + int(us) // 1000

    def start(self):
        threading.Thread(target=self._beat, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()

    def _beat(self):
        while not self._stop.wait(2.0):
            try:
                self.tick()
            except Exception as e:             # Redis 가 잠깐 안 돼도 다음 박자에
                self.log("스캔 창구 박자 실패 (%s: %s)" % (type(e).__name__, e))

    def tick(self):
        """2초마다: 키 수명 연장 · loading → ready 확인 · 시간 초과."""
        with self.lock:
            sc = self.scans.get(self.active) if self.active else None
            if sc is None:
                return
            now = self.clock()
            if sc.state == "loading":
                self._check_ready(sc)
                if sc.state == "loading" and now - sc.created > LOADING_S:
                    return self._close(sc, "failed", "timeout")
            if sc.state in ("ready", "collecting") and now - sc.touched > IDLE_S:
                return self._close(sc, "failed", "timeout")
            if sc.state not in TERMINAL:
                self.r.expire("vision_scan", KEY_TTL_S)

    # -- 판 열기 · 닫기
    def prepare(self, port_source, body):
        req = self._check_prepare(body)
        with self.lock:
            old = self.scans.get(req["scan_id"])
            if old is not None:                # 같은 scan_id 면 같은 판(멱등)
                old.touched = self.clock()
                return self._state_body(old, prepare=True)
            if self.active and self.scans[self.active].state not in TERMINAL:
                raise ScanError(409, "busy", "another scan is running", busy_scan_id=self.active)
            src = self.aliases.get(req["device"].strip().lower())
            if src is None:
                raise ScanError(400, "unknown_device", "device %r is not in the device table" % req["device"])
            if src != port_source:
                raise ScanError(400, "wrong_port", "device %s is source %s, not %s" % (req["device"], src, port_source),
                                stream=src)
            sc = Scan(req, src, self.clock())
            self._remember(sc)
            err = self._check_streams(src)
            if err:
                sc.state, sc.error = "failed", err
                self.log("스캔 %s 준비 실패 — %s" % (sc.scan_id, err))
                return self._state_body(sc, prepare=True)
            self.active = sc.scan_id
            self.r.set("vision_scan", json.dumps({"scan_id": sc.scan_id, "devices": [src],
                                                  "targets": [{"name": t} for t in sc.targets]}), ex=KEY_TTL_S)
            self.log("스캔 %s 준비 — %s(%s) · %s → %s" % (sc.scan_id, req["device"], src, sc.target, sc.targets[0]))
            self._check_ready(sc)
            return self._state_body(sc, prepare=True)

    def _check_prepare(self, body):
        if not isinstance(body, dict):
            raise ScanError(400, "bad_request", "JSON object expected")
        sid, dev, tgt = body.get("scan_id"), body.get("device"), body.get("target")
        if not isinstance(sid, str) or not sid.strip() or len(sid) > 120:
            raise ScanError(400, "bad_request", "scan_id is required (string, at most 120)")
        if not isinstance(dev, str) or not dev.strip():
            raise ScanError(400, "bad_request", "device is required")
        if tgt not in ("door", "box", "object"):
            raise ScanError(400, "bad_request", "target must be door | box | object")
        if tgt == "object" and not (isinstance(body.get("query"), str) and body["query"].strip()):
            raise ScanError(400, "bad_request", "query (English name) is required for target object")
        for k in ("steps", "step_deg"):
            if body.get(k) is not None and not isinstance(body[k], (int, float)):
                raise ScanError(400, "bad_request", "%s must be a number" % k)
        return dict(body, scan_id=sid.strip())

    def _check_streams(self, src):
        now = self.now_ms()
        cam = self.r.xrevrange("camera_stream:" + src, "+", "-", count=1)
        if not cam or now - _id_ms(cam[0][0]) > STREAM_FRESH_S * 1000:
            return "no_stream"
        vis = self.r.xrevrange("vision_stream:%s:targeted" % src, "+", "-", count=1)
        if not vis or now - _id_ms(vis[0][0]) > STREAM_FRESH_S * 1000:
            return "no_vision"
        return None

    def _check_ready(self, sc):
        """비전 응답이 ok 이고, 그 판 대상으로 낸 결과가 한 장이라도 나왔으면 ready."""
        ack = self.r.get("vision_scan_ack:" + sc.scan_id)
        if ack:
            a = _js(ack)
            if not a.get("ok"):
                err = str(a.get("error") or "rejected")
                return self._close(sc, "failed", ("gpu: " + err) if "memory" in err.lower() else err)
        else:
            return
        for _eid, f in self.r.xrevrange("vision_stream:%s:targeted" % sc.source, "+", "-", count=5):
            prof = _js(f.get(b"result")).get("profile") or {}
            if prof.get("scan") == sc.scan_id:
                sc.state = "ready"
                sc.profile_version = prof.get("version")
                self.log("스캔 %s ready (프로필 %s, %.1fs)" % (sc.scan_id, sc.profile_version, self.clock() - sc.created))
                return

    def _close(self, sc, state, error=None):
        sc.state, sc.error = state, error
        if self.active == sc.scan_id:
            self.active = None
            raw = self.r.get("vision_scan")
            if raw and _js(raw).get("scan_id") == sc.scan_id:
                self.r.delete("vision_scan")
        if sc.waiter is not None:
            sc.waiter["cancel"] = True
        self.log("스캔 %s %s%s" % (sc.scan_id, state, (" — %s" % error) if error else ""))
        self._save(sc)

    def _remember(self, sc):
        self.scans[sc.scan_id] = sc
        done = [s for s in self.scans.values() if s.state in TERMINAL and s.scan_id != sc.scan_id]
        for s in sorted(done, key=lambda s: s.created)[:-KEEP_DONE or None]:
            self.scans.pop(s.scan_id, None)

    def _get(self, scan_id):
        sc = self.scans.get(scan_id or "")
        if sc is None:
            raise ScanError(404, "unknown_scan", "no scan %r" % scan_id)
        return sc

    # -- 상태
    def status(self, scan_id):
        with self.lock:
            sc = self._get(scan_id)
            if sc.state == "loading":
                self._check_ready(sc)
            return self._state_body(sc)

    def _state_body(self, sc, prepare=False):
        out = {"scan_id": sc.scan_id, "state": sc.state, "profile_version": sc.profile_version, "error": sc.error}
        if prepare:
            out.update(stream=sc.source, model="targeted", targets=list(sc.targets))
        else:
            out["angles_captured"] = sorted(sc.frames)
        return out

    # -- 각도 표시
    def mark(self, body):
        if not isinstance(body, dict) or not isinstance(body.get("rotation_deg"), (int, float)):
            raise ScanError(400, "bad_request", "rotation_deg (number) is required")
        rot = body["rotation_deg"]
        rot = int(rot) if float(rot).is_integer() else float(rot)
        with self.lock:
            sc = self._get(body.get("scan_id"))
            if sc.state not in ("ready", "collecting"):
                raise ScanError(409, "bad_state", "scan is %s" % sc.state, state=sc.state)
            sc.touched = self.clock()
            sc.state = "collecting"
            t0 = self.now_ms()
            no = int(self.r.incr("vision_scan_mark:" + sc.source))
            w = sc.waiter = {"mark": no, "cut_ms": t0 + SETTLE_MS, "cancel": False}
            src, sid = sc.source, sc.scan_id
        got = self._wait_result(src, sid, w, t0)
        with self.lock:
            if sc.waiter is w:
                sc.waiter = None
            if sc.state not in ("collecting",) or w["cancel"]:
                return {"scan_id": sid, "rotation_deg": rot, "frame_captured": False, "frame_n": None,
                        "captured_at": None, "error": sc.error or sc.state}
            if got is None:
                self.log("스캔 %s %s° — %.1f초 안에 결과 없음" % (sid, rot, CAPTURE_S))
                return {"scan_id": sid, "rotation_deg": rot, "frame_captured": False, "frame_n": None,
                        "captured_at": None}
            res, header, overlay = got
            ver = (res.get("profile") or {}).get("version")
            if sc.profile_version is not None and ver != sc.profile_version:
                self._close(sc, "failed", "profile_changed")
                return {"scan_id": sid, "rotation_deg": rot, "frame_captured": False, "frame_n": None,
                        "captured_at": None, "error": "profile_changed"}
            eid = header.get("stream_entry_id")
            original = None
            if eid:
                cam = self.r.xrange("camera_stream:" + src, eid, eid)
                original = cam[0][1].get(b"image") if cam else None
            sc.order += 1
            sc.frames[rot] = {"rec": res, "header": header, "overlay": overlay, "original": original,
                              "frame_n": res.get("n"), "order": sc.order,
                              "captured_at": _iso(_id_ms(eid)) if eid else None}
            sc.touched = self.clock()
            f = sc.frames[rot]
            return {"scan_id": sid, "rotation_deg": rot, "frame_captured": True, "frame_n": f["frame_n"],
                    "captured_at": f["captured_at"]}

    def _wait_result(self, src, scan_id, w, t0):
        """원본 시각 ≥ mark + 0.5초 · 그 판 · 그 표시 번호인 첫 targeted 결과 → (결과, header, 오버레이) 또는 None."""
        key = "vision_stream:%s:targeted" % src
        last = "%d-0" % t0
        deadline = self.clock() + CAPTURE_S
        while not w["cancel"]:
            left = deadline - self.clock()
            if left <= 0:
                return None
            got = self.r.xread({key: last}, count=20, block=max(1, int(left * 1000)))
            for _k, entries in got or []:
                for eid, f in entries:
                    last = eid
                    res, h = _js(f.get(b"result")), _js(f.get(b"header"))
                    prof = res.get("profile") or {}
                    sid = h.get("stream_entry_id")
                    if prof.get("scan") != scan_id or prof.get("mark") != w["mark"] or not sid:
                        continue
                    if _id_ms(sid) < w["cut_ms"]:
                        continue
                    return res, h, f.get(b"overlay")
        return None

    # -- 산출
    def finish(self, body):
        with self.lock:
            sc = self._get((body or {}).get("scan_id"))
            if sc.state in ("running", "done"):
                return {"scan_id": sc.scan_id, "state": sc.state}
            if sc.state not in ("ready", "collecting"):
                raise ScanError(409, "bad_state", "scan is %s" % sc.state, state=sc.state)
            sc.state = "running"
            sc.touched = self.clock()
        threading.Thread(target=self._compute, args=(sc,), daemon=True).start()
        return {"scan_id": sc.scan_id, "state": "running"}

    def _compute(self, sc):
        try:
            with self.lock:
                if sc.state != "running":
                    return
                if not sc.frames:
                    return self._close(sc, "failed", "no_frames")
                cap, prev = [], None
                for rot, f in sorted(sc.frames.items(), key=lambda kv: kv[1]["order"]):
                    h = hashlib.sha1(f["original"]).hexdigest() if f["original"] else None
                    frozen = prev[0] if (prev and h and prev[1] == h) else None
                    cap.append({"rotation_deg": rot, "frame_n": f["frame_n"], "rec": f["rec"], "frozen_of": frozen})
                    prev = (rot, h)
                cap.sort(key=lambda c: c["rotation_deg"])
                out = sa.assemble({"scan_id": sc.scan_id, "target": sc.target, "query": sc.query,
                                   "targets": sc.targets, "profile_version": sc.profile_version}, cap)
                sc.computed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
                out["computed_at"] = sc.computed_at
                sc.result = out
                self._close(sc, "done")
        except Exception as e:
            with self.lock:
                self._close(sc, "failed", "%s: %s" % (type(e).__name__, e))

    def result(self, scan_id):
        with self.lock:
            sc = self._get(scan_id)
            if sc.state != "done" or sc.result is None:
                return {"scan_id": sc.scan_id, "state": sc.state, "error": sc.error}
            return dict(sc.result, state="done", error=None)

    def cancel(self, body):
        with self.lock:
            sc = self._get((body or {}).get("scan_id"))
            if sc.state not in TERMINAL:
                self._close(sc, "cancelled")
            return {"scan_id": sc.scan_id, "state": sc.state}

    # -- 그림
    def frame(self, scan_id, rotation_deg, kind):
        with self.lock:
            sc = self._get(scan_id)
            try:
                rot = float(rotation_deg)
            except (TypeError, ValueError):
                raise ScanError(400, "bad_request", "rotation_deg (number) is required")
            f = next((v for k, v in sc.frames.items() if float(k) == rot), None)
            if f is None:
                raise ScanError(404, "no_frame", "no frame at %s°" % rotation_deg)
            if kind in ("original", None, ""):
                img = f["original"]
            elif kind == "overlay":
                img = f["overlay"]
            elif kind == "crop":
                img = self._crop(sc, rot, f)
            else:
                raise ScanError(400, "bad_request", "kind must be original | overlay | crop")
            if not img:
                raise ScanError(404, "no_frame", "%s image is gone" % kind)
            return img

    def _crop(self, sc, rot, f):
        e = next((e for e in ((sc.result or {}).get("frames") or []) if float(e["rotation_deg"]) == rot), None)
        if e is None:                                    # 아직 산출 전 — 그 장으로 바로 셈한다
            e, _ = sa.frame_entry(rot, f["frame_n"], f["rec"], sc.targets[0])
        if not e.get("found") or not e.get("box_xyxy") or not f["original"]:
            raise ScanError(404, "no_frame", "target not found at %s°" % rot)
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise ScanError(501, "no_cv2", "crop needs opencv on the server")
        img = cv2.imdecode(np.frombuffer(f["original"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ScanError(404, "no_frame", "original image cannot be decoded")
        H, W = img.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in e["box_xyxy"]]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            raise ScanError(404, "no_frame", "empty box")
        ok, buf = cv2.imencode(".jpg", img[y1:y2, x1:x2], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return buf.tobytes()

    # -- 기록
    def _save(self, sc):
        if self.save_root is None:
            return
        try:
            d = Path(self.save_root) / "scans" / re.sub(r"[^A-Za-z0-9_\-]", "_", sc.scan_id)[:120]
            d.mkdir(parents=True, exist_ok=True)
            body = sc.result or {"scan_id": sc.scan_id, "state": sc.state, "error": sc.error}
            body = dict(body, state=sc.state, error=sc.error, device=sc.device, stream=sc.source,
                        query_ko=sc.query_ko, steps=sc.steps, step_deg=sc.step_deg)
            (d / "result.json").write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
            for rot, f in sc.frames.items():
                for kind in ("original", "overlay"):
                    if f.get(kind):
                        (d / ("%s_%s.jpg" % (rot, kind))).write_bytes(f[kind])
        except OSError as e:
            self.log("스캔 %s 기록을 못 남김 (%s)" % (sc.scan_id, e))

    def health(self):
        with self.lock:
            return {"busy_scan_id": self.active, "devices": dict(DEVICE_IDS)}

    # -- HTTP 이음 (server_stream_multi_source.py 의 Handler 가 부른다)
    def handle(self, method, path, port_source, query, body):
        """→ (코드, JSON 사전) 또는 (200, (JPEG 바이트, "image/jpeg"))."""
        from urllib.parse import parse_qs
        q = {k: v[0] for k, v in parse_qs(query or "").items()}
        try:
            if method == "POST":
                try:
                    obj = json.loads(body.decode("utf-8")) if body else {}
                except (UnicodeDecodeError, ValueError):
                    raise ScanError(400, "bad_request", "body must be JSON")
                if path == "/scan/prepare":
                    return 200, self.prepare(port_source, obj)
                if path == "/scan/mark":
                    return 200, self.mark(obj)
                if path == "/scan/finish":
                    return 200, self.finish(obj)
                if path == "/scan/cancel":
                    return 200, self.cancel(obj)
            elif method == "GET":
                if path == "/scan/status":
                    return 200, self.status(q.get("scan_id"))
                if path == "/scan/result":
                    return 200, self.result(q.get("scan_id"))
                if path == "/scan/frame":
                    return 200, (self.frame(q.get("scan_id"), q.get("rotation_deg"), q.get("kind")), "image/jpeg")
            raise ScanError(404, "not_found", "no such path")
        except ScanError as e:
            return e.code, dict({"error": e.error, "detail": e.detail}, **e.extra)
        except Exception as e:                           # Redis 안 됨 등
            return 503, {"error": "unavailable", "detail": "%s: %s" % (type(e).__name__, e)}
