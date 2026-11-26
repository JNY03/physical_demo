#!/usr/bin/env python3
"""지금 실행의 상태 한 장 — 런처의 「실행 상태」 칸이 3초마다 읽는다 (2026-11-27).

    venv/bin/python tools/run_status.py [--json] [--out 파일] [--redis URL] [--timeout 0.4]

왜: 무엇이 도는지 보려고 가시화 웹을 열어야 했다. 런처 안에서 바로 보이게 한다.

보는 것 (읽기만 한다 — 아무것도 켜거나 끄지 않는다. stdlib + redis 만, torch · cv2 · yaml 없이)
  프로세스      /proc 의 명령 줄 — 데이터 서버 · 오케스트레이터 · 비전 (python 으로 도는 것만, ds_run.sh 껍데기는 뺀다)
  데이터 서버   소스 포트마다 /health (포트는 server_stream_multi_source 의 SOURCES 표를 글로 읽는다 — source_probe 와 같다)
                입력 fps 는 지난 호출과의 받은 장 수 차이로 셈한다(장을 Redis 에서 끌어오지 않게 — 그림이 무겁다)
  오케스트레이터 state/orchestrator_state.json (StateFile 읽기 전용) 의 살아 있는 단위 + orchestration_stream 마지막 기록들
  비전          Redis vision_telemetry:<호스트>:<pid> (TTL 60초) — 처리 fps · 모델
  GPU           nvidia-smi (없으면 빈 칸)
아무것도 안 돌면 빈 상태({"active": false, "sources": []…})를 낸다.
"""
import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ds_resources.streams import ORCH_STREAM, REASON_CODES  # noqa: E402  (stdlib 만 쓴다)

STATE = REPO / "state" / "orchestrator_state.json"
PREV = Path("/tmp/ds_launcher/run_status_prev.json")   # 입력 fps 셈용 — 지난 호출의 받은 장 수
PREV_MAX_AGE_S = 30.0                                  # 이보다 오래된 지난 값으로는 fps 를 셈하지 않는다
TELEMETRY_PREFIX = "vision_telemetry:"
ORCH_TAIL = 200                                        # 읽을 오케스트레이션 기록 수 (그림이 없어 가볍다)

#: 프로세스 역할 — 명령 줄에 이것이 있으면 (python 으로 도는 것만)
ROLES = (("data_server", ("server_stream_multi_source.py",)),
         ("orchestrator", ("ds_resources.cli", "run")),
         ("vision", ("vision_infer.py",)),
         ("slice", ("slice360_live.py",)))
ROLE_KO = {"data_server": "데이터 서버", "orchestrator": "오케스트레이터", "vision": "비전", "slice": "슬라이스"}
STATE_KO = {"live": "수신 중", "ready": "연결됨 · 새 장 없음", "down": "안 들어옴"}
KIND_KO = {"refused": "거절", "waiting": "대기", "degraded": "낮춤", "failed": "실패", "stopped": "멈춤", "ok": "처리 중"}


# ------------------------------------------------------------------ 읽는 곳 (시험이 가짜로 바꾼다)
def _procs():
    """[(pid, argv)] — /proc 이 없으면 빈 목록"""
    out = []
    proc = Path("/proc")
    if not proc.is_dir():
        return out
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = [a.decode("utf-8", "replace") for a in (d / "cmdline").read_bytes().split(b"\0") if a]
        except OSError:
            continue
        if argv:
            out.append((int(d.name), argv))
    return out


def _health(port, timeout):
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001
        return None


def _ports():
    """소스 → 데이터 서버 포트 (source_probe 와 같은 표 — 서버 파일을 import 하지 않고 글로 읽는다)"""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import source_probe
        return dict(source_probe._tables()[0])
    except Exception:  # noqa: BLE001
        txt = (REPO / "server_stream_multi_source.py").read_text(encoding="utf-8")
        m = re.search(r"^SOURCES\s*=\s*\{(.*?)\}", txt, re.S | re.M)
        return {k: int(v) for k, v in re.findall(r'"([\w\-]+)"\s*:\s*(\d+)', m.group(1))} if m else {}


def _nvidia_smi():
    exe = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi") else None)
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2).stdout
    except Exception:  # noqa: BLE001
        return None
    return out


def _state_units():
    """오케스트레이터 상태 파일의 단위들 (읽기 전용 — 잠금을 잡지 않는다: 잡으면 오케스트레이터가 못 뜬다)"""
    try:
        from ds_resources.state import StateFile
        return StateFile(STATE, writer=False).units()
    except Exception:  # noqa: BLE001
        return {}


def _redis(url):
    try:
        import redis
        r = redis.Redis.from_url(url, socket_connect_timeout=0.5, socket_timeout=1, decode_responses=True)
        r.ping()
        return r
    except Exception:  # noqa: BLE001
        return None


def _is_wsl():
    try:
        return "microsoft" in open("/proc/version", encoding="utf-8").read().lower()
    except OSError:
        return False


def _local_node():
    return os.environ.get("DS_NODE_NAME") or socket.gethostname().lower()


# ------------------------------------------------------------------ 조각
def processes(procs):
    """역할별 프로세스 [{role, role_ko, pid, cmd}] — python 으로 도는 것만 (ds_run.sh · bash -c 껍데기는 같은 글을 품고 있다)"""
    me = os.getpid()
    out = []
    for pid, argv in procs:
        if pid == me or not argv or "python" not in os.path.basename(argv[0]):
            continue
        line = " ".join(argv)
        for role, keys in ROLES:
            if all(any(a == k or a.endswith("/" + k) for a in argv) for k in keys):
                out.append({"role": role, "role_ko": ROLE_KO[role], "pid": pid, "cmd": line[:200]})
                break
    return sorted(out, key=lambda p: (p["role"], p["pid"]))


def gpu_info(raw):
    """nvidia-smi 글 → [{name, used_mb, total_mb, util_pct}]"""
    out = []
    for line in (raw or "").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 3:
            continue
        try:
            out.append({"name": p[0], "used_mb": int(float(p[1])), "total_mb": int(float(p[2])),
                        "util_pct": int(float(p[3])) if len(p) > 3 and p[3] not in ("", "[N/A]") else None})
        except ValueError:
            continue
    return out


def gpu_text(gpus):
    if not gpus:
        return "GPU 정보 없음 (nvidia-smi 를 못 읽음)"
    return " · ".join("%s %.1f / %.1f GB%s" % (g["name"].replace("NVIDIA GeForce ", ""), g["used_mb"] / 1024.0,
                                              g["total_mb"] / 1024.0,
                                              (" · 사용률 %d%%" % g["util_pct"]) if g.get("util_pct") is not None else "")
                      for g in gpus)


def gpu_where(node, local, gpu_name, wsl):
    """controller.gpu_where 와 같은 글 — 「book-… GPU (RTX 4050) — 이 기기 · WSL」 (controller 는 yaml 을 끌어와 여기서 다시 쓴다)"""
    if (node or "").lower() != (local or "").lower():
        return "%s GPU" % node
    here = "이 기기 · WSL" if wsl else "이 기기"
    return "%s GPU%s — %s" % (node, (" (%s)" % gpu_name) if gpu_name else "", here)


def telemetry(r):
    """[{host, pid, models, streams}] — 비전 프로세스마다 (TTL 60초라 남은 것은 살아 있는 것이다)"""
    out = []
    if r is None:
        return out
    try:
        for k in r.scan_iter(TELEMETRY_PREFIX + "*", count=100):
            raw = r.get(k)
            if not raw:
                continue
            try:
                doc = json.loads(raw)
            except ValueError:
                continue
            k = k.decode() if isinstance(k, bytes) else k
            host, _, pid = k[len(TELEMETRY_PREFIX):].rpartition(":")
            out.append({"host": host, "pid": int(pid) if pid.isdigit() else None, "models": sorted(doc.get("models") or []),
                        "streams": doc.get("streams") or {}, "ts": doc.get("ts")})
    except Exception:  # noqa: BLE001
        return out
    return out


def orch_tail(r, n=ORCH_TAIL):
    """orchestration_stream 마지막 n 건 (오래된 것 → 새것), 마지막 「requested」부터만"""
    if r is None:
        return []
    try:
        rows = r.xrevrange(ORCH_STREAM, count=n)
    except Exception:  # noqa: BLE001
        return []
    recs = []
    for eid, f in rows:
        f = {(k.decode() if isinstance(k, bytes) else k): (v.decode("utf-8", "replace") if isinstance(v, bytes) else v)
             for k, v in f.items()}
        f["_id"] = eid.decode() if isinstance(eid, bytes) else eid
        recs.append(f)
        if f.get("event") == "requested":
            break
    return list(reversed(recs))


def _j(txt):
    try:
        v = json.loads(txt) if txt else None
        return v if isinstance(v, (dict, list)) else None
    except ValueError:
        return None


def item_source(item):
    """배치표 항목 이름 → 소스 — 「input:a2」 · 「gpu:targeted:robot1」 · 「targeted_detect.device@pi12」"""
    item = str(item or "")
    if "@" in item:
        return item.rsplit("@", 1)[1]
    return item.rsplit(":", 1)[-1] if ":" in item else ""


def unit_sources(u):
    if not u:
        return []
    if u.get("devices"):
        return list(u["devices"])
    s = item_source(u.get("task") or "")
    return [s] if s else []


def _why(rec_or_item):
    """사람이 읽는 까닭 — 문구가 있으면 문구, 없으면 이유 코드의 뜻"""
    txt = str(rec_or_item.get("reason") or "").strip()
    code = rec_or_item.get("code")
    return txt or REASON_CODES.get(code or "", "") or (code or "")


def last_reasons(recs, units):
    """소스 → {kind, code, text} — 그 소스에 마지막으로 일어난 일 (거절 · 대기 · 낮춤 · 실패 · 멈춤 · 처리 중)"""
    out = {}

    def put(src, kind, why, code=None):
        if src:
            out[src] = {"kind": kind, "code": code or "", "text": "%s: %s" % (KIND_KO[kind], why) if why else KIND_KO[kind]}

    for rec in recs:
        ev = rec.get("event")
        det = _j(rec.get("detail")) or {}
        if ev == "planned":
            plan = _j(rec.get("plan")) or {}
            for key, kind in (("refused", "refused"), ("waiting", "waiting"), ("degraded", "degraded")):
                for it in plan.get(key) or []:
                    if isinstance(it, dict):
                        put(item_source(it.get("item")), kind, _why(it), it.get("code"))
        elif ev == "running":
            for s in list(det.get("sources_up") or []) + ([det["source_up"]] if det.get("source_up") else []):
                put(s, "ok", "")
            for s in det.get("waiting") or []:
                put(s, "waiting", REASON_CODES["input_wait"], "input_wait")
        elif ev in ("degraded", "failed", "stopped"):
            code = rec.get("code") or det.get("code")
            kind = {"failed": "failed", "stopped": "stopped"}.get(ev) or (
                "waiting" if (code or "").startswith("input_wait") else "degraded")
            srcs = []
            if det.get("item"):
                srcs.append(item_source(det["item"]))
            if det.get("source_down"):
                srcs.append(det["source_down"])
            if isinstance(det.get("fps"), dict) and ev == "degraded":
                srcs += list(det["fps"])
            if not srcs and det.get("unit_id"):
                srcs = unit_sources(units.get(det["unit_id"]))
            for s in srcs:
                put(s, kind, _why(rec), code)
    return out


def orch_summary(recs, procs_):
    alive = [p for p in procs_ if p["role"] == "orchestrator"]
    if not recs:
        return {"running": bool(alive), "pid": alive[0]["pid"] if alive else None, "request_id": "", "last_event": "",
                "last_reason": "", "last_ts": "", "text": "오케스트레이터 실행 중 · 기록 없음" if alive else "오케스트레이터 꺼짐"}
    last = recs[-1]
    rid = next((r.get("request_id") for r in recs if r.get("event") == "requested"), last.get("request_id", ""))
    why = _why(last)
    return {"running": bool(alive), "pid": alive[0]["pid"] if alive else None, "request_id": rid or "",
            "last_event": last.get("event", ""), "last_reason": why, "last_ts": last.get("ts", ""),
            "text": "%s · 마지막 %s%s" % ("오케스트레이터 실행 중" if alive else "오케스트레이터 꺼짐 (지난 기록)",
                                        last.get("event", ""), (" — " + why) if why else "")}


def _fps_txt(v):
    return "-" if v is None else "%.1f" % v


def load_prev(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_prev(path, data):
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass


# ------------------------------------------------------------------ 한 장
def snapshot(redis_url="redis://127.0.0.1:6380/0", timeout=0.4, prev_path=None, now=None):
    prev_path = PREV if prev_path is None else prev_path
    now = time.time() if now is None else now
    procs_ = processes(_procs())
    gpus = gpu_info(_nvidia_smi())
    local, wsl = _local_node(), _is_wsl()
    gname = gpus[0]["name"] if gpus else ""
    server_on = any(p["role"] == "data_server" for p in procs_)
    orch_on = any(p["role"] == "orchestrator" for p in procs_)

    # 데이터 서버 — 켜져 있을 때만 포트를 두드린다 (꺼져 있으면 0.4초 × 포트 수를 아낀다)
    ports = _ports() if server_on else {}
    health = {}
    if ports:
        with ThreadPoolExecutor(max_workers=min(12, len(ports))) as ex:
            got = list(ex.map(lambda kv: (kv[0], _health(kv[1], timeout)), sorted(ports.items())))
        health = {s: h for s, h in got if h}

    r = _redis(redis_url)
    tel = telemetry(r)
    recs = orch_tail(r)
    units = _state_units()
    # 오케스트레이터가 꺼져 있으면 파일에 남은 「도는 중」 단위는 지난 실행의 것이다 — 보이지 않는다
    active = {uid: u for uid, u in units.items()
              if orch_on and u.get("status") in ("starting", "loaded", "running", "local")}
    reasons = last_reasons(recs, units) if (orch_on or server_on or tel) else {}

    # 어느 소스를 보일까 — 서버가 받는 것 · 비전이 처리하는 것 · 오케스트레이터가 맡은 것
    names = set(health)
    for t in tel:
        names |= set(t["streams"])
    for u in active.values():
        names |= set(unit_sources(u))
    if orch_on:
        req = next((_j(x.get("request")) for x in recs if x.get("event") == "requested"), None) or {}
        names |= set(req.get("sources") or [])

    prev = load_prev(prev_path)
    keep = {}
    rows = []
    for s in sorted(names):
        h = health.get(s)
        frames = int(h.get("frames") or 0) if h else None
        in_fps = None
        if frames is not None:
            keep[s] = [frames, now]
            p = prev.get(s)
            if p and 0 < now - p[1] <= PREV_MAX_AGE_S and frames >= p[0]:
                in_fps = round((frames - p[0]) / (now - p[1]), 2)
        if h is None:
            state = "down"
        elif frames and (in_fps is None or in_fps > 0):
            state = "live"
        else:
            state = "ready"
        procs_fps = []
        models = set()
        where = []
        for t in tel:
            st = t["streams"].get(s)
            if st is None:
                continue
            f = st.get("fps")
            if f is None and not st.get("frames"):
                f = 0.0
            procs_fps.append(("+".join(t["models"]), f))
            models |= set(t["models"])
            where.append(gpu_where(t["host"], local, gname, wsl))
        for u in active.values():
            if s not in unit_sources(u):
                continue
            models |= set(u.get("models") or [])
            if u.get("kind") == "gpu":
                where.append(gpu_where(u.get("node") or local, local, gname, wsl))
            elif u.get("kind") == "cpu":
                where.append("%s CPU (%s)" % (u.get("node") or "?", u.get("task") or u.get("function") or ""))
        if h:
            for m, v in (h.get("vision") or {}).items():
                if isinstance(v, dict) and v.get("age_s") is not None and v["age_s"] < 30:
                    models.add(m)
        vals = [f for _, f in procs_fps if f is not None]
        proc_fps = min(vals) if vals else None
        proc_txt = _fps_txt(proc_fps) if len(procs_fps) <= 1 else " · ".join(
            "%s %s" % (m, _fps_txt(f)) for m, f in procs_fps)
        why = reasons.get(s) or {}
        rows.append({"source": s, "state": state, "state_ko": STATE_KO[state], "port": ports.get(s),
                     "frames": frames, "in_fps": in_fps, "in_fps_text": _fps_txt(in_fps),
                     "proc_fps": proc_fps, "proc_fps_text": proc_txt, "models": sorted(models),
                     "where": " · ".join(dict.fromkeys(where)) or "-",
                     "reason_kind": why.get("kind", ""), "reason_code": why.get("code", ""),
                     "reason": why.get("text", "")})
    if keep:
        save_prev(prev_path, keep)
    is_active = bool(procs_)
    return {"schema": "run_status/1", "ts": now, "active": is_active, "redis": r is not None,
            "processes": procs_, "orchestrator": orch_summary(recs if is_active else [], procs_),
            "sources": rows, "gpu": gpus, "gpu_text": gpu_text(gpus),
            "summary": ("실행 중: " + ", ".join(dict.fromkeys(p["role_ko"] for p in procs_))) if is_active else "실행 중인 것 없음"}


def write_atomic(path, text):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".run_status-", dir=str(p.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, p)


def main(argv=None):
    ap = argparse.ArgumentParser(description="지금 실행의 상태 한 장 (읽기만)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default="", help="JSON 을 이 파일에도 쓴다 (원자적 — 런처가 기다리지 않고 읽게)")
    ap.add_argument("--redis", default="redis://127.0.0.1:6380/0")
    ap.add_argument("--timeout", type=float, default=0.4)
    a = ap.parse_args(argv)
    snap = snapshot(a.redis, a.timeout)
    txt = json.dumps(snap, ensure_ascii=False)
    if a.out:
        write_atomic(a.out, txt)
    if a.json:
        print(txt)
    elif not a.out:
        print(snap["summary"])
        print(snap["orchestrator"]["text"])
        for x in snap["sources"]:
            print("%-12s %-10s 입력 %-5s 처리 %-12s %-18s %-30s %s" % (
                x["source"], x["state_ko"], x["in_fps_text"], x["proc_fps_text"], "+".join(x["models"]) or "-",
                x["where"], x["reason"]))
        print(snap["gpu_text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
