#!/usr/bin/env python3
"""소스(카메라 · 로봇)가 **지금 닿는지** 본다 — 런처가 고를 수 있는 것만 켜게 (2026-11-26).

    venv/bin/python tools/source_probe.py [--sources pi12 a2 ...] [--json] [--timeout 1.5]

왜: 꺼진 기기를 골라 놓고 「실행」을 누르면 한참 뒤에야 영상이 없다는 것을 안다. 고르기 전에 알려 준다.

보는 것 (순서대로, 처음 닿는 것으로 끝)
  1. 데이터 서버가 이미 그 소스를 받고 있나  — 로컬 포트 /health (가장 확실하다: 지금 장이 들어온다)
  2. Redis 에 최근 장이 있나                — camera_stream:<소스> 의 마지막 장 나이
  3. 그 기기의 영상 창구가 열려 있나        — server_stream_multi_source 가 쓰는 주소의 TCP 연결
상태: live(장이 들어오는 중) · ready(기기는 닿는다) · down(안 닿는다) · unknown(주소를 모른다)
읽기만 한다 — 아무것도 켜거나 끄지 않는다.
"""
import argparse
import json
import re
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FRESH_S = 5.0                      # 이 안의 장이 있으면 live


#: 소스마다 영상이 나오는 창구 (server_stream_multi_source 머리말과 같다)
CAM_PORT = {"robot1": 8090, "realsense": 8090, "a2": 8090, "pi11": 8090, "pi12": 8090, "pi13": 8090,
            "ep-001": 8091, "drone": 8890, "drone2": 8890}


def _tables():
    """(소스 → 서버 포트, 소스 → 기기 주소). 서버 포트는 server_stream_multi_source 의 SOURCES 표를 **글로** 읽는다
    (그 파일을 import 하면 cv2 · torch 까지 끌려 온다). 주소는 ds_resources.settings 가 읽는 hosts 파일에서."""
    txt = (REPO / "server_stream_multi_source.py").read_text(encoding="utf-8")
    ports = {}
    m = re.search(r"^SOURCES\s*=\s*\{(.*?)\}", txt, re.S | re.M)
    if m:
        for k, v in re.findall(r'"([\w\-]+)"\s*:\s*(\d+)', m.group(1)):
            ports[k] = int(v)
    hosts = {}
    try:
        from ds_resources import settings
        hosts = {k: v for k, v in (settings.load_hosts() or {}).items() if v}
        for n, v in (settings.load_user_sources() or {}).items():   # 사람이 더한 소스도 함께 본다
            ports.setdefault(n, int(v["port"]))
            hosts[n] = v["host"]
            CAM_PORT.setdefault(n, int(v.get("cam_port") or 8090))
    except Exception:  # noqa: BLE001
        pass
    return ports, hosts


def _tcp(host, port, timeout):
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def _health(port, timeout):
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001
        return None


def _redis_age(src, r):
    if r is None:
        return None
    try:
        rows = r.xrevrange("camera_stream:" + src, count=1)
        if not rows:
            return None
        eid = rows[0][0]
        ms = int((eid.decode() if isinstance(eid, bytes) else eid).split("-")[0])
        return time.time() - ms / 1000.0
    except Exception:  # noqa: BLE001
        return None


def probe_one(src, ports, hosts, r, timeout):
    out = {"source": src, "state": "unknown", "why": "", "port": ports.get(src)}
    h = _health(ports[src], timeout) if src in ports else None
    if h:
        out["frames"] = h.get("frames")
        out["state"] = "live" if (h.get("frames") or 0) > 0 else "ready"
        out["why"] = str((h.get("status") or {}).get("video") or "데이터 서버가 받는 중")
        return out
    age = _redis_age(src, r)
    if age is not None and age <= FRESH_S:
        out["state"] = "live"
        out["why"] = "최근 장 %.1f초 전" % age
        return out
    host = hosts.get(src)
    cam = CAM_PORT.get(src)
    if host and cam:
        if _tcp(host, cam, timeout):
            out["state"] = "ready"
            out["why"] = "%s:%d 열려 있음" % (host, cam)
        else:
            out["state"] = "down"
            out["why"] = "%s:%d 에 닿지 않음" % (host, cam)
        return out
    if age is not None:
        out["state"] = "down"
        out["why"] = "마지막 장이 %.0f초 전 (지금은 안 들어옴)" % age
        return out
    out["why"] = "이 소스의 주소를 모름 (그 기기가 서버로 밀어 넣는 방식일 수 있음)"
    return out


def probe(sources=None, timeout=1.5, redis_url="redis://127.0.0.1:6380/0"):
    ports, hosts = _tables()
    srcs = sorted(sources or ports)
    r = None
    try:
        import redis
        r = redis.Redis.from_url(redis_url, socket_connect_timeout=1, socket_timeout=1)
    except Exception:  # noqa: BLE001
        r = None
    with ThreadPoolExecutor(max_workers=min(12, max(1, len(srcs)))) as ex:
        rows = list(ex.map(lambda s: probe_one(s, ports, hosts, r, timeout), srcs))
    return {"sources": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description="소스가 지금 닿는지 본다 (읽기만)")
    ap.add_argument("--sources", nargs="*", default=[])
    ap.add_argument("--timeout", type=float, default=1.5)
    ap.add_argument("--redis", default="redis://127.0.0.1:6380/0")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = probe(a.sources or None, a.timeout, a.redis)
    if a.json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        for x in res["sources"]:
            print("%-12s %-7s %s" % (x["source"], x["state"], x["why"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
