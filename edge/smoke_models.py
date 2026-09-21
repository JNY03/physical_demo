"""모델이 실제로 올라와 돌아가는지 한 번에 확인한다.

실행: ./venv312/bin/python smoke_models.py

**왜 별도 스크립트인가**: 서버를 띄워 화면을 보는 것으로는 "왜 비어 있는지"를
알 수 없다. 모델이 안 올라온 것인지, 올라왔는데 결과가 0건인지, 결과는 있는데
그리지 못한 것인지 — 셋이 화면에서는 똑같이 검은 화면이다. 여기서는 단계마다
숫자를 찍어 어디서 끊기는지 드러낸다(AI-O-02).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from server import load_config, build_state, stage1, stage_depth, STATE  # noqa: E402
from models import class_prior_range  # noqa: E402


def pull_frame(host: str, port: int, cam: int):
    r = requests.get(f"http://{host}:{port}/api/frame?cam={cam}&wait=5", timeout=20)
    if r.status_code != 200:
        return None, None
    parts = r.content.split(b"--drone")
    meta = jpeg = None
    for p in parts:
        if b'name="meta"' in p:
            meta = json.loads(p.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n"))
        elif b'name="frame"' in p:
            jpeg = p.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if jpeg else None
    return meta, bgr


def main() -> None:
    cfg = load_config()
    print("=" * 62)
    build_state(cfg)
    print("=" * 62)

    print("\n[1] 모델 적재")
    for name in ("ovd", "ovd2", "clip", "depth"):
        o = STATE.get(name)
        ok = o is not None and (not hasattr(o, "available") or o.available())
        err = STATE.get(f"{name}_error") or getattr(o, "error", None)
        print(f"  {name:7s} {'OK' if ok else '없음'}"
              + (f"  — {str(err)[:90]}" if err and not ok else ""))

    node = (cfg.get("terminals") or [{}])[0]
    host, port = node.get("host", "CHANGE-ME"), int(node.get("port", 8890))
    print(f"\n[2] 말단에서 프레임 당기기 — {host}:{port}")
    meta, bgr = pull_frame(host, port, int((node.get("cameras") or [0])[0]))
    if meta is None or bgr is None:
        print("  실패 — 말단이 떠 있는지 확인한다")
        return
    print(f"  frame_seq {meta['frame_seq']}  {bgr.shape[1]}x{bgr.shape[0]}  "
          f"후보 {len(meta.get('candidates', []))}  트랙 {len(meta.get('tracks', []))}")
    print(f"  카메라 기하 {meta.get('camera')}")

    print("\n[3] stage1 — YOLOE 검출")
    t = time.perf_counter()
    rec = stage1(cfg, meta, bgr)
    ms = (time.perf_counter() - t) * 1000
    dets = rec.get("ovd", [])
    print(f"  {ms:.0f}ms  검출 {len(dets)}개  per_track {len(rec.get('per_track', []))}개")
    labs: dict[str, int] = {}
    for d in dets[:200]:
        labs[d.get("label", "?")] = labs.get(d.get("label", "?"), 0) + 1
    if labs:
        top = sorted(labs.items(), key=lambda x: -x[1])[:8]
        print("  라벨: " + ", ".join(f"{k}×{v}" for k, v in top))
    else:
        print("  ⚠ 검출 0개")

    print("\n[4] depth — MoGe2-Aerial")
    t = time.perf_counter()
    ranged = stage_depth(cfg, rec, meta, bgr)
    ms = (time.perf_counter() - t) * 1000
    dp = rec.get("depth", {})
    print(f"  {ms:.0f}ms  ran={dp.get('ran')}  거리 산출 {len(ranged)}건"
          + (f"  err={str(dp.get('error'))[:80]}" if dp.get("error") else ""))
    for r in ranged[:6]:
        rr = r["range"]
        alt = rr.get("alternative")
        print(f"    track {r['track_id']:>4}  {rr['meters']:7.2f}m  [{rr['source']}]"
              + (f"  alt {alt['meters']:.2f}m [{alt['source']}]" if alt else ""))

    print("\n[5] 오버레이 렌더")
    from server import reflected_overlay, colorize_depth
    # 말단이 반영한 것처럼 meta.tracks에 근거를 얹어 본다
    by = {r["track_id"]: r["range"] for r in ranged}
    lab_by = {p["track_id"]: p.get("ovd") for p in rec.get("per_track", [])}
    for t_ in meta["tracks"]:
        tid = t_["track_id"]
        if tid in lab_by and lab_by[tid]:
            t_["ovd"] = {"age_s": 0.3, "payload": lab_by[tid]}
        if tid in by:
            t_["range"] = {**by[tid], "stale_warning": True,
                           "meters_now": by[tid]["meters"], "scale_ratio": 1.0}
    img = reflected_overlay(bgr, meta)
    cv2.imwrite("/tmp/smoke_obstacle.png", img)
    print(f"  obstacle information → /tmp/smoke_obstacle.png  {img.shape}")

    if STATE.get("depth") is not None and STATE["depth"].available():
        dmap = STATE["depth"].depth_map(bgr, hfov_deg=(meta.get("camera") or {}).get("hfov_deg"))
        if dmap is not None:
            v = dmap[np.isfinite(dmap) & (dmap > 0)]
            print(f"  depth 맵 {dmap.shape}  유효 {v.size}px  "
                  f"범위 {v.min():.1f}~{v.max():.1f}m  중앙 {np.median(v):.1f}m")
            cv2.imwrite("/tmp/smoke_depth.png", colorize_depth(dmap))
            print("  depth → /tmp/smoke_depth.png")


if __name__ == "__main__":
    main()
