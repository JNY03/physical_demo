#!/usr/bin/env python3
"""스캔 탐지 창구를 한 판 돌려 본다(2026-11-24) — 가시화 없이, 장치를 돌리지 않고.

  python tools/scan_smoke.py --url http://127.0.0.1:10003 --device go1-001 --target door
  python tools/scan_smoke.py --target object --query "fire extinguisher" --interval 1

prepare → ready 기다림 → mark 를 steps 번(각도 0, 45, …, 간격 --interval 초) → finish → result 를 찍고
가시화가 바라는 시간 한도(prepare→ready 60초 · mark 2초 · finish→done 30초)와 견준다. --frames 폴더를 주면
각도 그림(original · overlay · crop)을 받아 둔다.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

LIMITS = {"ready": 60.0, "mark": 2.0, "done": 30.0}


def call(url, method="GET", body=None, raw=False):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            out = r.read()
            return r.status, (out if raw else json.loads(out))
    except urllib.error.HTTPError as e:
        out = e.read()
        try:
            return e.code, json.loads(out)
        except ValueError:
            return e.code, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:10003", help="장치 소스의 서버 포트 (기본 robot1 :10003)")
    ap.add_argument("--device", default="go1-001")
    ap.add_argument("--target", default="door", choices=("door", "box", "object"))
    ap.add_argument("--query", help="target object 의 영어 이름")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--step-deg", type=float, default=45.0)
    ap.add_argument("--interval", type=float, default=1.0, help="mark 사이 쉬는 초 (회전 대신)")
    ap.add_argument("--frames", metavar="폴더", help="각도 그림을 받아 둘 폴더")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    sid = "SMOKE-%s" % datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    over = {}

    t = time.time()
    code, out = call(base + "/scan/prepare", "POST", {"scan_id": sid, "device": a.device, "target": a.target,
                                                       "query": a.query, "steps": a.steps, "step_deg": a.step_deg})
    print("prepare %d %s" % (code, json.dumps(out, ensure_ascii=False)))
    if code != 200 or out.get("state") == "failed":
        return 1
    while out.get("state") == "loading":
        time.sleep(1.0)
        code, out = call(base + "/scan/status?scan_id=" + sid)
    t_ready = time.time() - t
    print("status  %s  (%.1fs)" % (json.dumps(out, ensure_ascii=False), t_ready))
    over["ready"] = t_ready
    if out.get("state") != "ready":
        return 1

    worst = 0.0
    for i in range(a.steps):
        rot = round(i * a.step_deg, 1)
        rot = int(rot) if float(rot).is_integer() else rot
        t = time.time()
        code, m = call(base + "/scan/mark", "POST", {"scan_id": sid, "rotation_deg": rot})
        dt = time.time() - t
        worst = max(worst, dt)
        print("mark %5s° %d  %.2fs  %s" % (rot, code, dt, json.dumps(m, ensure_ascii=False)))
        if i + 1 < a.steps:
            time.sleep(a.interval)
    over["mark"] = worst

    t = time.time()
    call(base + "/scan/finish", "POST", {"scan_id": sid})
    while True:
        code, res = call(base + "/scan/result?scan_id=" + sid)
        if res.get("state") not in ("running", "collecting"):
            break
        time.sleep(0.2)
    over["done"] = time.time() - t
    print("result  %s" % json.dumps(res, ensure_ascii=False, indent=1))

    if a.frames:
        d = Path(a.frames) / sid
        d.mkdir(parents=True, exist_ok=True)
        for f in res.get("frames") or []:
            for kind in ("original", "overlay", "crop"):
                code, img = call("%s/scan/frame?scan_id=%s&rotation_deg=%s&kind=%s" % (base, sid, f["rotation_deg"], kind),
                                 raw=True)
                if code == 200:
                    (d / ("%s_%s.jpg" % (f["rotation_deg"], kind))).write_bytes(img)
        print("그림  %s" % d)

    print("시간  " + " · ".join("%s %.2fs (한도 %.0fs)%s" % (k, v, LIMITS[k], "" if v <= LIMITS[k] else " **넘음**")
                               for k, v in over.items()))
    return 0 if res.get("state") == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
