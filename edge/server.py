"""엣지 노드 — 말단에서 프레임을 **당겨 와** 의미·거리를 붙이고 돌려준다.

implements: AI-E-04, AI-S-03, AI-S-04, AI-S-06, AI-C-05, AI-C-06, AI-B-08, AI-O-02

── 흐름 ─────────────────────────────────────────────────────────────────────
  1. Puller가 말단의 `GET /api/frame?cam=N`을 친다. 응답 본문에 meta와 frame이
     함께 온다 — **요청한 IP를 말단이 되찾아 되쏘지 않는다.** 같은 연결로 답하면
     NAT·LTE·재접속에서 안 깨지고 왕복도 절반이다.
  2. YOLOE로 검출하고 MoGe2-Aerial로 거리를 낸 뒤, 둘을 **한 판정에 함께 실어**
     `POST /api/verdict`로 말단에 밀어 넣는다.
  3. 말단이 KLT 플로우로 좌표를 현재 시점에 맞추고, 그 결과가 **다음 프레임의
     meta["obstacles"]** 에 실려 돌아온다. 화면은 그것을 그린다.

정체성은 **엣지가 소유한다**(`ObjectTracks`의 obj_id). 말단에는 추적기가 없다 —
정체성 공간이 둘이면 매칭이 틀렸을 때 근거가 엉뚱한 객체에 붙는다.

── 왜 pull인가 ──────────────────────────────────────────────────────────────
예전에는 말단이 카메라 속도로 밀어넣었고 CLIP이 프레임당 800ms~1s라 비동기 큐가
6:1로 밀렸다. 지금은 **이쪽이 여유 있을 때만 당긴다** — backpressure가 구조에서
나온다. 당기지 않은 프레임은 버려지는 게 아니라 말단이 최신 것만 들고 있다가
다음에 준다. 링크가 끊기면 말단이 스풀에 쌓고, 돌아오면 `/api/spool`로 훑어 간다.

── 스트리밍 ─────────────────────────────────────────────────────────────────
오버레이 MJPEG은 이 프로세스의 로컬 포트에 뜬다(config `server.port`). 기기를
옮겨도 같은 포트에 같은 화면이 뜬다 — 주소가 설정 한 줄이기 때문이다.
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import signal
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import requests

from edge_rpn import Rpn as EdgeRpn
from missed import (MissedTracker, crop_of, find_missed,
                    geometric_verdict, suspect_classes)
from models import (AerialDepth, ClipScorer, TorchClipScorer, WorldOvd,
                    YoloeOvd, ZipDepth, class_prior_range, resolve)
from semantics import (FeatureDictionary, LlmFeatureGenerator, Ollama,
                       ObjectTracks, VlmDescriber, absorb, concept_key,
                       expand_box, iou, merge_by_consensus, merge_overlapping)
from store import CropGate, Rotation


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else Path(__file__).resolve().parent / "config.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["_config_path"], cfg["_root"] = str(p), str(p.parent)
    return cfg


STATE: dict = {}
# 종료 신호. **MJPEG 스트리밍 스레드가 이걸 봐야 Ctrl-C가 먹는다.**
#
# socketserver.ThreadingMixIn은 block_on_close가 기본 True라 server_close()가
# **모든 요청 스레드를 join**한다. 스트리밍 핸들러는 `while True`로 프레임을
# 밀고 있으므로 끝나지 않고, 그래서 Ctrl-C를 눌러도 프로세스가 포트를 쥔 채
# 영원히 매달린다(2026-09-21 실측: SIGINT 후 15초가 지나도 LISTEN 유지).
# daemon_threads=True로는 안 풀린다 — 그건 인터프리터 종료 시점 이야기이고
# join은 그 전에 걸린다.
SHUTDOWN = threading.Event()
# 스트림 세 갈래. 각각 카메라별 **최신 한 장만** 들고 있는다 — 실시간 감시라
# 밀린 프레임은 가치가 없다.
ORIG: dict[int, dict] = {}      # 원본
DEPTH: dict[int, dict] = {}     # MoGe2-Aerial depth
LIVE: dict[int, dict] = {}      # 반영 결과(박스+클래스+미터)
# 말단이 마지막으로 보내 온 meta. 정렬 검증(/api/obstacles)에 쓴다 —
# moved_px와 quality의 분포를 봐야 플로우가 실제로 듣는지 알 수 있다.
LAST_META: dict[int, dict] = {}
# 미검출 후보 패널. 위 세 패널과 갱신 주기가 달라(비동기) 따로 둔다.
MISSED: dict[int, dict] = {}
LIVE_LOCK = threading.Lock()


def reflected_overlay(bgr, meta: dict):
    """**말단이 정렬한 장애물**을 그린다 — 박스 + 클래스명 + 미터 거리.

    그리는 것은 `meta["obstacles"]`이지 엣지가 방금 본 검출이 아니다. 이 구분이
    이 파이프라인의 핵심이다:

        박스   말단이 KLT 플로우로 **현재 프레임에 맞춘** 좌표
        클래스 몇 프레임 전 관측에서 온 YOLOE 근거 (괄호 안이 그 나이)
        거리   MoGe2-Aerial이 낸 미터 값, 누적 배율로 보정

    엣지가 방금 본 것을 그리면 왕복 지연이 화면에서 사라져, 검증하려는 대상
    자체가 없어진다. 여기 보이는 "낡은 라벨이 정확한 자리를 따라가는 것"이
    정렬이 실제로 동작한다는 증거다.

    **나이와 품질을 숨기지 않는다**(AI-S-03). 거리는 pose 보정이 안 돼 있으므로
    `stale_warning`이 붙은 값에는 `~`를 찍고, 플로우 드리프트가 쌓이면
    `quality`가 떨어지며 박스가 흐려진다.
    """
    img = bgr.copy()
    GREEN, YELLOW = (0, 220, 0), (0, 200, 255)
    n_lab = n_rng = 0
    taken: list[tuple[int, int, int]] = []

    obstacles = meta.get("obstacles") or []
    # **화면에 그릴 수를 제한한다.** 실측에서 22개가 한 번에 떠 서로를 덮었다
    # (2026-09-21 사무실 장면). 저장과 전송은 그대로 두고 그리기만 줄인다 —
    # 데이터를 버리는 것이 아니라 볼 수 있게 하는 것이다.
    scfg = (STATE.get("cfg") or {}).get("stream", {})
    min_conf = float(scfg.get("draw_min_conf", 0.35))
    max_draw = int(scfg.get("draw_max_obstacles", 8))
    hidden = len(obstacles)
    # 거리 > conf > 최신 순. 정보가 많고 확실한 것부터 자리를 가져간다.
    obstacles = sorted(
        (o for o in obstacles
         if (o.get("conf") or 0) >= min_conf or o.get("range")),
        key=lambda o: (-(bool(o.get("range"))), -(o.get("conf") or 0)))[:max_draw]
    hidden -= len(obstacles)

    for o in obstacles:
        box = o.get("box")
        if not box:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        lab, rng = o.get("label"), (o.get("range") or {})
        if not lab and rng.get("meters") is None:
            continue

        # 정렬 품질이 낮으면 선을 얇게 — 박스가 어디 있는지 덜 믿는다는 표시다.
        q = float(o.get("quality", 1.0))
        color = GREEN if q >= 0.6 else YELLOW
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if q >= 0.6 else 1)
        if lab:
            n_lab += 1

        bits = []
        if lab:
            conf = o.get("conf")
            bits.append(f"{lab}" + (f" {conf:.2f}" if isinstance(conf, (int, float)) else ""))
            age = o.get("age_s")
            if isinstance(age, (int, float)) and age >= 0.1:
                bits.append(f"({age:.1f}s)")
        if rng.get("meters") is not None:
            n_rng += 1
            m = rng.get("meters_now", rng["meters"])
            ratio = rng.get("scale_ratio")
            mark = "~" if rng.get("stale_warning") else ""
            bits.append(f"{mark}{m:.1f}m")
            if ratio and abs(ratio - 1.0) > 0.08:
                bits.append(f"({rng['meters']:.0f}m/{ratio:.2f})")
            if rng.get("source") == "class_prior":
                bits.append("[size prior]")
        txt = " ".join(bits)

        lx, ly = x1, max(20, y1 - 6)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        for _ in range(4):
            if not any(abs(ly - py) < th + 6 and lx < px + pw and px < lx + tw
                       for px, py, pw in taken):
                break
            ly += th + 8
        taken.append((lx, ly, tw))
        cv2.rectangle(img, (lx - 2, ly - th - 4), (lx + tw + 2, ly + 4), (0, 0, 0), -1)
        cv2.putText(img, txt, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    fl = meta.get("flow") or {}
    moved = max((o.get("moved_px", 0) for o in obstacles), default=0)
    head = (f'{meta.get("frame_id","?")}  cam{meta.get("camera_id",0)}  '
            f'drawn {len(obstacles)}(+{hidden} hidden)  range {n_rng}  '
            f'| flow {fl.get("flow_ms","?")}ms/{fl.get("flow_points","?")}pt  '
            f'moved {moved:.0f}px')
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, head, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def colorize_depth(dmap, lo_pct: float = 2.0, hi_pct: float = 98.0,
                   *, title: str = "depth", unit: str = "m"):
    """미터 depth를 눈으로 볼 수 있게. 가까울수록 밝다.

    **고정 범위가 아니라 백분위수로 정규화한다.** 하늘이나 먼 지형이 들어오면
    최대값이 수백 미터로 튀어 관심 영역이 전부 같은 색으로 뭉갠다. 2~98%로 자르면
    장면의 실제 분포에 맞춰진다 — 대신 **색은 절대 거리를 뜻하지 않는다.** 숫자는
    반영 화면의 미터 값으로 읽고, 이 화면은 기하가 말이 되는지 보는 용도다.
    """
    valid = np.isfinite(dmap) & (dmap > 0)
    if not valid.any():
        return None
    lo, hi = np.percentile(dmap[valid], [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-3
    norm = np.clip((dmap - lo) / (hi - lo), 0, 1)
    u8 = (255 * (1.0 - norm)).astype(np.uint8)      # 가까울수록 밝게
    img = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    img[~valid] = (0, 0, 0)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, f"{title}  {lo:.3g}{unit} ~ {hi:.3g}{unit} (2~98%)",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def _publish(bucket: dict, cfg: dict, cam: int, img, info: dict | None = None) -> None:
    """공통 발행 — JPEG로 굽고 최신 한 장만 들고 있는다."""
    q = int(cfg.get("stream", {}).get("jpeg_quality", 70))
    scale = float(cfg.get("stream", {}).get("scale", 1.0))
    if scale and scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return
    with LIVE_LOCK:
        bucket[cam] = {"jpeg": enc.tobytes(), "at": time.time(), "info": info or {}}


def publish_original(cfg: dict, meta: dict, bgr) -> None:
    """원본 — 아무것도 그리지 않은 말단 프레임."""
    _publish(ORIG, cfg, int(meta.get("camera_id", 0)), bgr,
             {"frame_id": meta.get("frame_id"), "seq": meta.get("frame_seq")})


def publish_depth(cfg: dict, meta: dict, dmap) -> None:
    # **단위를 속이지 않는다.** provider가 상대값이면 m를 붙이면 안 된다 —
    # 화면에 "0.0m ~ 0.2m"라고 떠서 실내 20cm 장면처럼 보였다(2026-09-21).
    d = STATE.get("depth")
    metric = bool(getattr(d, "metric", True)) if d is not None else True
    prov = cfg.get("depth", {}).get("provider", "depth")
    img = colorize_depth(dmap, title=prov if metric else f"{prov} (relative, not metric)",
                         unit="m" if metric else "")
    if img is not None:
        _publish(DEPTH, cfg, int(meta.get("camera_id", 0)), img,
                 {"frame_id": meta.get("frame_id")})


def missed_panel(crops: list, infos: list, *, width: int = 1280,
                 rows: int = 3, cols: int = 4, stats: dict | None = None):
    """미검출 후보를 **crop 이미지 + 클래스 후보 + 근거**로 나란히 그린다.

    위 패널이 "무엇을 잡았나"라면 이것은 "무엇을 놓쳤나"다. 두 질문은 서로
    보완이라 같은 화면에 있어야 한다 — 검출 결과만 보면 빠진 것이 보이지 않고,
    빠진 것만 보면 그게 정말 빠진 건지 이미 잡힌 건지 알 수 없다.

    **확정 라벨을 쓰지 않는다.** CLIP은 닫힌 사전의 argmax라 '해당 없음'이 없고,
    정답이 사전에 없으면 가장 덜 틀린 것을 고른다. 그래서 1등만 크게 쓰지 않고
    상위 몇 개를 점수와 함께 적는다 — 그게 근거이고, 판단은 보는 사람 몫이다
    (AI-S-03: 신뢰도와 근거 충분도는 별개).

    `gap`(1등−2등)을 같이 적는다. 작으면 사전이 그 crop을 갈라내지 못했다는
    뜻이고, 그 사실이 1등 이름보다 더 쓸모 있을 때가 많다.
    """
    cell_w = width // cols
    thumb = int(cell_w * 0.42)
    cell_h = max(thumb + 12, 108)
    H = 26 + rows * cell_h
    canvas = np.full((H, width, 3), 22, np.uint8)

    n = min(len(crops), rows * cols)
    for i in range(n):
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w, 26 + r * cell_h
        cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x0 + cell_w - 3, y0 + cell_h - 3),
                      (48, 48, 48), 1)
        cr = crops[i]
        if cr is not None and cr.size:
            ch, cw = cr.shape[:2]
            sc = min(thumb / max(1, cw), (cell_h - 12) / max(1, ch))
            tw, th = max(1, int(cw * sc)), max(1, int(ch * sc))
            canvas[y0 + 6:y0 + 6 + th, x0 + 6:x0 + 6 + tw] = cv2.resize(cr, (tw, th))

        info = infos[i] if i < len(infos) else {}
        tx, ty = x0 + thumb + 12, y0 + 18

        # 1줄: 슬롯 · 안정성 · objectness
        obj = info.get("objectness")
        head = f"s{info.get('slot', i+1)}"
        if info.get("seen"):
            head += f"  seen {info['seen']}x"
        if isinstance(obj, (int, float)):
            head += f"  obj {obj:.2f}"
        cv2.putText(canvas, head, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (190, 190, 190), 1)
        ty += 17

        # 2줄: **거리와 실제 크기** — 판단에 가장 직접 쓰이는 값이다.
        # "5m 앞 0.3m짜리"와 "5m 앞 1.8m짜리"는 후보 클래스가 전혀 다르다.
        z, w_m, h_m = info.get("dist_m"), info.get("w_m"), info.get("h_m")
        if z is not None:
            # 돌출은 "면이 아니라 물체다"의 근거다. 같은 줄에 이어 쓴다 —
            # 오른쪽에 따로 두었더니 셀 폭을 넘어 옆 썸네일을 덮었다.
            pr = info.get("protrusion_m")
            line = f"{z:.1f}m | {w_m*100:.0f}x{h_m*100:.0f}cm"
            if pr is not None:
                line += f" | +{pr*100:.0f}cm"
            cv2.putText(canvas, line, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, (120, 220, 160), 1)
        else:
            bx = info.get("box") or [0, 0, 0, 0]
            cv2.putText(canvas, f"거리 미측정  ({int(bx[2]-bx[0])}x{int(bx[3]-bx[1])} px)",
                        (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1)
        ty += 17

        # 3줄~: **의심 클래스** — 확정 문턱을 못 넘은 OVD 검출. 추가 연산 0.
        sus = info.get("suspects") or []
        if sus:
            cv2.putText(canvas, "의심:", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (140, 140, 140), 1)
            ty += 14
            for k, (name, cf, iv) in enumerate(sus[:3]):
                col = (0, 200, 255) if k == 0 else (150, 150, 150)
                cv2.putText(canvas, f"  {name[:20]} {cf:.2f} (iou {iv})",
                            (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)
                ty += 14
        else:
            # 겹치는 검출이 하나도 없다 = OVD가 이 자리를 아예 안 봤다.
            # 그게 가장 강한 "미검출" 신호다.
            cv2.putText(canvas, "OVD 반응 없음", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 160, 255), 1)

    st = stats or {}
    head = (f"missed candidates — RPN {st.get('rpn', '?')} "
            f"-> explained {st.get('known', '?')} "
            f"/ merged {st.get('merged_away', 0)} "
            f"/ background {st.get('background', 0)} {st.get('dropped_by', {})} "
            f"=> slots {st.get('slots', '?')} "
            f"(ripe {st.get('ripe', '?')}, shown {st.get('shown', n)})")
    cv2.rectangle(canvas, (0, 0), (width, 26), (0, 0, 0), -1)
    cv2.putText(canvas, head, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)
    return canvas


def compose_panels(cam: int):
    """원본·depth·반영을 가로로 이어 붙인 한 장.

    **왜 합치는가 — 브라우저 연결 한도.** MJPEG는 끝나지 않는 응답이라 연결을
    영구 점유한다. HTTP/1.1 동일 출처 동시 연결은 6개가 한도인데, 카메라 2대 ×
    패널 3개면 정확히 6개라 슬롯이 꽉 찬다. 그러면 뒤늦게 붙는 패널은 연결을
    못 얻어 영영 빈 화면이 되고, `fetch("/api/command")`마저 슬롯이 없어
    매달린다 — 버튼이 disabled로 굳는 원인이 이것이었다(2026-09-21).

    카메라당 한 스트림으로 합치면 연결이 2개로 줄어 네 슬롯이 남는다.

    덤이 하나 더 있다. 세 버킷은 서로 다른 시각에 갱신되므로 따로 스트리밍하면
    **나란히 놓인 세 화면이 서로 다른 프레임**이었다. 합치면 한 장 안에서
    같은 프레임끼리 비교된다.
    """
    with LIVE_LOCK:
        parts = [(name, (bucket.get(cam) or {}).get("jpeg"))
                 for name, bucket in (("original", ORIG), ("depth", DEPTH),
                                      ("obstacle information", LIVE))]
    imgs, labels = [], []
    for name, jpg in parts:
        if jpg is None:
            continue
        im = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            imgs.append(im)
            labels.append(name)
    if not imgs:
        return None
    # 높이를 맞춰 가로로 잇는다. 기준은 첫 장 — 원본이 있으면 그 해상도다.
    h = imgs[0].shape[0]
    resized = [im if im.shape[0] == h else
               cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
               for im in imgs]
    # 아직 안 들어온 패널은 검은 칸으로 자리를 지킨다 — 칸이 사라지면 옆 패널이
    # 밀려서 "어느 것이 무엇인지"가 매 프레임 바뀐다.
    canvas = np.hstack(resized)
    x = 0
    for im, name in zip(resized, labels):
        cv2.putText(canvas, name, (x + 10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4)
        cv2.putText(canvas, name, (x + 10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1)
        x += im.shape[1]
        if x < canvas.shape[1]:
            cv2.line(canvas, (x, 0), (x, h), (60, 60, 60), 2)
    return canvas


def publish_live(cfg: dict, meta: dict, bgr, rec: dict | None = None,
                 unknown=None) -> None:
    """반영 결과 — 박스 + 클래스명 + 미터 거리.

    rec/unknown은 받기만 하고 그리지 않는다. 화면이 보여줘야 하는 것은 엣지의
    중간 결과가 아니라 **말단이 지금 들고 있는 상태**이기 때문이다. 인자를 남긴
    이유는 호출부를 건드리지 않기 위해서다(추론이 꺼져 있으면 rec이 None으로 온다).
    """
    _publish(LIVE, cfg, int(meta.get("camera_id", 0)),
             reflected_overlay(bgr, meta),
             {"frame_id": meta.get("frame_id"), "seq": meta.get("frame_seq"),
              "n_obstacles": len(meta.get("obstacles") or [])})


def apply_domain(cfg: dict, domain: str) -> dict:
    """도메인 사전과 배경 클래스를 갈아 끼운다.

    사전과 배경 클래스는 **한 쌍**이다. 하천의 water_surface와 도시의 road_surface는
    같은 역할(배경 흡수)을 하지만 이름이 다르므로 따로 옮기면 배경 판정이 깨진다.
    그래서 domains[<key>]가 둘을 같이 들고 있다.

    텍스트 임베딩 캐시를 비우는 것이 중요하다 — 사전이 바뀌면 프롬프트 집합이
    바뀌는데 캐시를 그대로 두면 이전 도메인의 앵커로 계속 점수를 매긴다.
    """
    spec = cfg["clip"].get("domains", {}).get(domain)
    if not spec:
        raise KeyError(f"모르는 도메인: {domain} "
                       f"(가능: {list(cfg['clip'].get('domains', {}))})")
    fd = FeatureDictionary(resolve(cfg, spec["dictionary"]),
                           int(cfg["clip"].get("dictionary_top_k", 3)))
    STATE["dict"] = fd
    # 배경 목록은 **사전이 선언한다**. config에 두면 사전을 고칠 때 어긋난다
    # (2026-09-21에 실제로 어긋났다 — semantic.FeatureDictionary.background_classes 참고).
    STATE["background_classes"] = fd.background_classes()
    STATE["active_domain"] = domain
    cfg["clip"]["active_domain"] = domain
    # OVD 어휘도 도메인을 따라간다. 사전과 어휘는 같은 도메인의 두 면이고, 따로
    # 옮기면 CLIP은 하천을 보는데 OVD는 도시를 보는 상태가 된다(AI-C-15).
    if STATE.get("ovd2") is not None:
        words = (cfg.get("ovd2", {}).get("vocabularies") or {}).get(domain) or []
        if list(words) != list(STATE["ovd2"].vocab):
            STATE["ovd2"].set_vocabulary(words)
    if STATE.get("clip") is not None:
        STATE["clip"].prompts = []          # 앵커 재생성 강제
        STATE["clip"].text_emb = None
    STATE.pop("clip_cache", None)            # 이전 도메인 판정 재사용 금지

    # VLM 장면 프롬프트도 같은 도메인으로 옮긴다. 두 키(clip.active_domain /
    # vlm.active_domain)가 따로 놀면 사전은 도시인데 장면 서술은 하천을 묻는
    # 상태가 된다 — 2026-09-21에 실제로 그렇게 떠 있었다.
    cfg.setdefault("vlm", {})["active_domain"] = domain
    if STATE.get("vlm") is not None:
        sp = cfg["vlm"].get("scene_prompts", {}).get(domain)
        if sp:
            STATE["vlm"].scene_prompt = sp

    return {"domain": domain, "dictionary": spec["dictionary"],
            "classes": fd.classes(),
            "background_classes": sorted(STATE["background_classes"])}


# --------------------------------------------------------------------------
def stage1(cfg: dict, meta: dict, bgr) -> dict:
    t0 = time.perf_counter()
    h, w = bgr.shape[:2]
    # 말단은 더 이상 RPN 후보를 보내지 않는다 — 검출은 여기서만 한다.
    skipped: list[str] = []

    ovd_dets, ovd_ms = [], 0.0
    if STATE.get("ovd") is not None:
        t = time.perf_counter()
        ovd_dets = STATE["ovd"].detect(bgr)
        ovd_ms = (time.perf_counter() - t) * 1000.0
    elif cfg["ovd"].get("enabled", True):
        skipped.append(f"ovd_unavailable:{STATE.get('ovd_error','?')}")

    # 두 번째 눈 — 현장 어휘로 좁혀 본다. 없으면 이 단계만 빠진다(AI-C-05).
    ovd2_dets, ovd2_ms = [], 0.0
    if STATE.get("ovd2") is not None:
        t = time.perf_counter()
        ovd2_dets = STATE["ovd2"].detect(bgr)
        ovd2_ms = (time.perf_counter() - t) * 1000.0

    # **확정은 합의로 정한다**(consensus.py). 단일 모델의 conf는 그 모델의 확신일
    # 뿐 근거 충분도가 아니다(AI-S-03). 실측에서 YOLOE는 0.94/0.87처럼 **높은
    # conf로** 틀렸으므로 임계를 올려서는 그 오답을 거를 수 없었다.
    # verdict를 **흡수보다 먼저** 매긴다 — absorb가 확신 검출만 흡수하기 때문이다.

    # 합치기 **전** 목록을 남긴다 — 근거는 모델별로 따로 트랙에 붙여야 하고,
    # 합친 목록에서는 어느 모델이 말했는지가 한 항목에 뭉개진다.
    pf_dets = list(ovd_dets)
    if ovd2_dets:
        ovd_dets, consensus_stats = merge_by_consensus(ovd_dets, ovd2_dets,
                                                       {**cfg.get("ovd2", {})})
    else:
        # 두 번째 눈이 없으면 예전 규칙(단일 conf 임계)으로 물러난다. 축소지
        # 실패가 아니다 — 다만 확정의 근거가 약해졌다는 사실은 남긴다.
        cconf = float(cfg["ovd"].get("confident_conf", 0.35))
        for det in ovd_dets:
            det["verdict"] = "confident" if det["conf"] >= cconf else "tentative"
            det["sources"] = ["prompt_free"]
        consensus_stats = {"degraded": "single_model"}
    ovd_counts = {"confident": sum(1 for d in ovd_dets if d["verdict"] == "confident"),
                  "tentative": sum(1 for d in ovd_dets if d["verdict"] == "tentative")}

    # ── 정체성은 **엣지가 소유한다** ─────────────────────────────────────
    # 예전에는 말단이 RPN+ByteTrack으로 track_id를 만들고, 여기서 YOLOE 검출을
    # 그 id에 IoU로 배정했다(`_assign`). 그 층을 통째로 걷어냈다:
    #   * 말단에 RPN이 없다 — KLT 플로우 정렬로 바꿨다(프레임당 350ms → 6ms).
    #   * 정체성 공간이 둘이면 매칭이 틀렸을 때 근거가 엉뚱한 객체에 붙는다.
    #     실측으로 "computer chair 0.76" 하나가 118개 트랙에 동시에 붙어 화면을
    #     덮은 적이 있다(2026-09-21).
    # 이제 `ObjectTracks`가 검출을 직접 추적해 `obj_id`를 만들고, 말단은 그것을
    # 그대로 받아 좌표만 현재 시점으로 옮긴다. 공간이 하나뿐이라 어긋날 곳이 없다.
    for d in ovd_dets:
        d.setdefault("concept", concept_key(d["label"],
                                            cfg.get("ovd2", {}).get("synonyms") or {}))
    cam_key = int(meta.get("camera_id", 0))
    ot = STATE.setdefault("obj_tracks", {}).get(cam_key)
    if ot is None:
        ot = STATE["obj_tracks"][cam_key] = ObjectTracks(cfg)
    objects = [o.to_dict() for o in ot.update(ovd_dets, time.perf_counter())]

    # 말단에 보낼 장애물. **확정만 보낸다** — 잠정 검출은 "배경이 아닌데 모르는
    # 것"이라 말단 상태를 바꿀 근거가 못 된다(AI-S-04). 잠정도 저장소와 오버레이
    # 에는 남으므로 가시화가 판단할 수 있다.
    send_tentative = bool(cfg["ovd"].get("send_tentative", False))
    obstacles = [{"obs_id": o["obj_id"], "box": o["box"],
                  "ovd": {"label": o["label"], "conf": o["conf"],
                          "concept": o["concept"], "confirmed": o["confirmed"]}}
                 for o in objects
                 if o.get("label") and (o["confirmed"] or send_tentative)]

    return {
        "stage": 1,
        "frame_id": meta["frame_id"], "frame_seq": meta.get("frame_seq"),
        "node_id": meta.get("node_id"), "camera_id": meta.get("camera_id", 0),
        "observed_at": meta.get("observed_at") or meta.get("captured_at"),
        "received_at": datetime.now(timezone.utc).isoformat(),
        "frame_wh": [w, h],
        "ovd": ovd_dets, "ovd_count": len(ovd_dets),
        "ovd2_count": len(ovd2_dets), "consensus": consensus_stats,
        "objects": objects,
        "objects_confirmed": sum(1 for o in objects if o["confirmed"]),
        "object_stats": dict(getattr(ot, "last_stats", {})),
        "ovd_verdicts": ovd_counts,
        "obstacles": obstacles,
        # 말단이 보내 온 **정렬 상태**. 플로우가 죽으면 여기서 먼저 보인다.
        "pi_flow": meta.get("flow"),
        "pi_obstacles": len(meta.get("obstacles") or []),
        "timing_ms": {"ovd": round(ovd_ms, 1), "ovd2": round(ovd2_ms, 1),
                      "stage1_total": round((time.perf_counter() - t0) * 1000.0, 1)},
        "skipped": skipped,
    }


def stage2(cfg: dict, record: dict, bgr) -> dict:
    """미지 후보를 특징 사전에 맞춰 근거를 만든다. **현재 구성에서는 돌지 않는다.**

    두 전제가 다 사라졌다:
      * `clip.enabled = false` (사용자 2026-09-21: YOLOE만 쓴다)
      * 미지 후보의 출처였던 **말단 RPN이 없다** — KLT 플로우 정렬로 바꿨다.
        후보가 없으니 "OVD가 설명 못 한 것"을 골라낼 대상 자체가 없다.

    코드를 남겨 둔 이유는 되돌릴 수 있게 하기 위해서다. 다시 켜려면 미지 후보를
    어디서 낼지부터 정해야 한다 — 엣지에서 class-agnostic 제안기를 돌리거나,
    YOLOE의 낮은 conf 검출을 후보로 쓰는 길이 있다.
    """
    cc = cfg["clip"]
    unknown = record.pop("_unknown", [])
    root = resolve(cfg, cfg["store"]["root"])
    st = cfg["store"]
    h, w = bgr.shape[:2]

    if STATE.get("clip") is None or not unknown:
        record["stage2"] = {"skipped": "clip_unavailable" if unknown else "no_unknown"}
        return record

    fd = STATE["dict"]
    dict_prompts, dict_index = fd.prompt_plan()
    free_prompts = list(cc.get("prompts", []))
    STATE["clip"].set_prompts(dict_prompts + free_prompts)
    base_free = len(dict_prompts)

    ctx = float(cfg["unknown"].get("crop_context", 1.5))
    take = unknown[: int(cc.get("max_crops_per_frame", 60))]

    # 같은 track_id를 매 프레임 다시 분류하지 않는다. 추적이 정체성을 유지하므로
    # 한 번 본 객체는 recheck 창 안에서는 이전 판정을 재사용한다. crop을 **줄이는**
    # 것이 아니라 **중복을 줄이는** 것이라, RPN이 후보를 넓게 내는 목적과 충돌하지 않는다.
    dedup = bool(cc.get("dedup_by_track", True))
    recheck = float(cc.get("dedup_recheck_s", 3.0))
    cache: dict = STATE.setdefault("clip_cache", {})
    now = time.time()
    for tid in [k for k, v in cache.items() if now - v["at"] > recheck * 4]:
        cache.pop(tid, None)          # 오래된 항목은 버린다(무한 증가 방지)

    crops, kept, reused = [], [], 0
    for d in take:
        tid = d.get("track_id")
        if dedup and tid is not None:
            hit = cache.get(tid)
            if hit and (now - hit["at"]) <= recheck:
                d["clip"] = {**hit["clip"], "reused_from_frame": hit["frame_id"],
                             "reused_age_s": round(now - hit["at"], 2)}
                reused += 1
                continue
        x1, y1, x2, y2 = expand_box(d["box"], ctx, w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(bgr[y1:y2, x1:x2])
        kept.append(d)

    t = time.perf_counter()
    sims = (STATE["clip"].score(crops, int(cc.get("batch_size", 16)))
            if crops else [])
    clip_ms = (time.perf_counter() - t) * 1000.0

    bg_set = STATE.get("background_classes", set())
    bg_margin = float(cc.get("background_margin", 0.010))
    cf_margin = float(cc.get("confident_margin", 0.030))
    use_abs = bool(cc.get("use_target_sim_threshold", True))
    default_target_sim = float(cc.get("default_target_sim", 0.22))
    top_k = int(cc.get("top_k", 5))

    counts = {"background": 0, "unidentified": 0}
    per_track, saved = [], []
    crop_dir = root / "unknown_crops"
    untracked_saved = 0            # 이 프레임에서 실제로 저장한 수(상한 적용)

    for d, crop, row in zip(kept, crops, sims):
        per_class = fd.score_row(row, dict_index)
        ranked = sorted(per_class.items(), key=lambda kv: -kv[1])
        gap = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 1.0
        top1 = ranked[0][0]

        # **두 갈래.** CLIP이 설명한 것은 확신이 아니라 추론이다(사용자 2026-09-21).
        # 실제 오추론 사례: 차량 일부나 여러 차량이 합쳐진 박스를 damaged_vehicle로
        # 찍었다. 사전은 닫힌 집합이고 argmax에는 "해당 없음"이 없으므로, 점수가
        # 아무리 갈려도 그것이 "맞다"는 근거가 되지 못한다.
        # 그래서 CLIP은 말단 레코드에 올리지 않고 **가시화로 관리자에게 전달할
        # 근거**로만 쓴다 — 연초록(OVD 잠정)과 같은 성격이다(AI-S-04, AI-L-03).
        #
        # 판정 축이 둘이다(2026-09-21 재보정으로 추가):
        #   상대(gap)  1등과 2등의 차이. 사전 안에서 얼마나 갈렸나.
        #   절대(top1) 1등 점수 자체. 사전 어느 항목과도 별로 안 닮았으면 높은 gap도
        #              의미가 없다 — 8개 중 덜 안 닮은 것을 고른 것뿐이다.
        # 상대만 쓰던 이전 판은 crop 512개 실측에서 배경/비배경 gap 분포가 거의 겹쳤다
        # (비배경 gap이 배경보다 큰 비율 0.683, 0.5면 구분 불가). 절대 축은 사전이
        # 이미 항목마다 들고 있던 `target_sim_threshold`를 실제로 쓰는 것이다 —
        # 지금까지 선언만 되어 있고 판정에 반영되지 않았다.
        need = fd.entries.get(top1, {}).get("target_sim_threshold")
        if need is None:
            need = default_target_sim
        meets_abs = (not use_abs) or (ranked[0][1] >= float(need))

        if top1 in bg_set and gap >= bg_margin:
            verdict = "background"
        else:
            verdict = "unidentified"
        # 얼마나 갈렸는지는 버리지 않고 근거로 남긴다 — 관리자가 정렬·필터할 축이다.
        # 판정을 바꾸지는 않는다(둘 다 unidentified다).
        strength = ("strong" if (top1 not in bg_set and gap >= cf_margin and meets_abs)
                    else "weak")
        counts[verdict] += 1

        free_row = row[base_free:]
        clip_ev = {
            "verdict": verdict, "status": "inference",  # 확신이 아니다
            "strength": strength,
            "ranking": [{"class": k, "score": round(v, 4),
                         "verified": bool(fd.entries.get(k, {}).get("_verified", True))}
                        for k, v in ranked[:top_k]],
            "top1_gap": round(float(gap), 4),
            "top1_score": round(float(ranked[0][1]), 4),
            "target_sim_threshold": round(float(need), 4),
            "meets_absolute": bool(meets_abs),
            "free_prompt_best": (free_prompts[int(np.argmax(free_row))]
                                 if free_row else None),
            "free_prompt_score": (round(float(np.max(free_row)), 4)
                                  if free_row else None),
        }
        d["clip"] = clip_ev
        if dedup and d.get("track_id") is not None:
            cache[d["track_id"]] = {"clip": clip_ev, "at": now,
                                    "frame_id": record["frame_id"]}

        # 배경은 **저장하지 않는다**(저장소가 오탐으로 차는 것을 막는 유일한 경로.
        # RPN precision@50 0.19~0.25). 다만 오버레이에는 **회색으로 그린다** —
        # 도로·건물을 실제로 지우고 있는지 눈으로 확인할 수 있어야 한다
        # (사용자 2026-09-21).
        if verdict == "background" and not st.get("save_background_crops", False):
            continue

        # **같은 물체를 프레임마다 다시 저장하지 않는다.** 1000번째 crop은 새 이유를
        # 담고 있지 않으므로 학습 후보가 아니다(AI-L-01). 정체성 기준으로 거르고,
        # track_id가 없는 후보는 프레임당 몇 장까지만 받는다.
        gate = STATE.get("crop_gate")
        cam_i = int(record.get("camera_id", 0))
        if gate is None:
            allow = True
        elif untracked_saved >= gate.max_per_frame:
            allow = False          # 한 프레임이 저장을 독차지하지 않게 한다
        else:
            allow = gate.allow(cam_i, d["box"], clip_ev.get("verdict", ""))
            if allow:
                untracked_saved += 1

        if st.get("save_unknown_crops", True) and allow:
            crop_dir.mkdir(parents=True, exist_ok=True)
            base = f"{record['frame_id']}_u{len(saved):03d}"
            cv2.imwrite(str(crop_dir / f"{base}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            # 추론 결과와 **근거를 함께** 남긴다 — 가시화가 "왜 이렇게 봤는지"를
            # 보여줄 수 있어야 한다(AI-L-03). 이것은 정답이 아니라 후보다(AI-L-02).
            (crop_dir / f"{base}.json").write_text(json.dumps({
                "frame_id": record["frame_id"], "frame_seq": record.get("frame_seq"),
                "observed_at": record.get("observed_at"),
                "track_id": d.get("track_id"), "box": d["box"],
                "objectness": d.get("objectness"), "best_iou_with_ovd": d.get("best_iou"),
                "clip": clip_ev, "verified": False,
                "for_visualization": True,   # 저장된 것은 전부 가시화 대상이다
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            d["crop"] = f"unknown_crops/{base}.jpg"
            saved.append(base)

        # **말단에 보내지 않는다.** CLIP은 추론이지 확신이 아니므로 말단의 지속
        # 레코드를 바꿀 근거가 못 된다(사용자 2026-09-21). 말단이 반영하는 것은
        # OVD 확신뿐이고, CLIP 결과는 store에 남아 가시화가 소비한다.
        if cfg["clip"].get("send_to_terminal", False) and d.get("track_id") is not None:
            # 관측 당시 박스를 함께 보낸다 — 말단이 "이 근거가 관측된 위치"와
            # "지금 위치"를 비교해 시점 반영이 실제로 일어났는지 검증한다.
            per_track.append({"track_id": d["track_id"],
                              "clip": {**clip_ev, "box": d["box"]},
                              "crop": d.get("crop")})

    # 재사용분도 카운트에 넣어 "이 프레임에서 무엇이 무엇으로 보였나"가 온전히 남게 한다.
    for d in take:
        if "reused_from_frame" in d.get("clip", {}):
            counts[d["clip"]["verdict"]] = counts.get(d["clip"]["verdict"], 0) + 1

    record["unknown"] = take
    record["stage2"] = {"counts": counts, "saved_crops": len(saved),
                        "clip_ms": round(clip_ms, 1), "scored": len(crops),
                        "reused_from_track": reused,
                        "sent_to_terminal": len(per_track)}
    record["timing_ms"]["clip"] = round(clip_ms, 1)
    record["_stage2_per_track"] = per_track
    return record


def detect_domain(cfg: dict, switch: bool = True) -> dict:
    """VLM이 현재 장면을 보고 어느 도메인인지 고르면 CLIP 사전을 그에 맞게 바꾼다.

    사용자 제안(2026-09-21). 고리가 이렇게 닫힌다:

        VLM(장면이 무엇인가) → 도메인 선택 → CLIP 사전 교체 → 근거 품질 향상

    왜 필요한가: 사전은 닫힌 집합이고 argmax에는 "해당 없음"이 없다. 그래서 **틀린
    도메인의 사전을 쓰면 조용히 그럴듯한 오답이 나온다** — 실측으로 하천 사전을 도시
    영상에 쓰자 미지 crop의 71.7%가 tire/plastic_bag으로 몰렸다. 사람이 도메인을 미리
    지정하는 대신 장면에서 읽어내면 이 실패 방식 자체가 사라진다.

    판정은 VLM 한 번으로 끝내지 않는다. VLM은 자유 서술을 하고, 그 서술을 설정에 있는
    도메인 **후보 목록**과 맞춘다. 후보 밖이면 바꾸지 않고 사유를 돌려준다 —
    모르는 도메인으로 멋대로 갈아타는 것이 틀린 사전을 쓰는 것보다 낫지 않다.
    """
    if STATE.get("vlm") is None:
        return {"error": f"vlm_unavailable:{STATE.get('vlm_error','?')}"}
    last = STATE.get("last_frame")
    if not last:
        return {"error": "no frame received yet"}

    domains = cfg["clip"].get("domains", {})
    menu = "\n".join(f'- "{k}": {v.get("hint", k)}' for k, v in domains.items())
    prompt = (
        "You are looking at one frame from a drone camera. Decide which operating "
        "domain this scene belongs to.\n\nCandidate domains:\n" + menu +
        '\n\nAnswer strictly as JSON:\n'
        '{"domain": "<one of the keys above, or \'unknown\'>", '
        '"scene": "<one short sentence describing what you see>", '
        '"confidence": 0.0-1.0}\n'
        "Use 'unknown' if the scene clearly matches none of them."
    )
    t0 = time.perf_counter()
    try:
        raw = STATE["vlm"].ollama.generate(STATE["vlm"].model, prompt,
                                           images_b64=[_b64(last["jpeg"])])
    except Exception as exc:
        return {"error": repr(exc)[:200]}
    from semantic import _extract_json

    parsed = _extract_json(raw) or {}
    picked = str(parsed.get("domain", "unknown")).strip()
    out = {"frame_id": last["frame_id"], "detected": picked,
           "scene": str(parsed.get("scene", ""))[:300],
           "confidence": parsed.get("confidence"),
           "previous_domain": STATE.get("active_domain"),
           "vlm_ms": round((time.perf_counter() - t0) * 1000.0, 1),
           "model": STATE["vlm"].model, "switched": False,
           "status": "hypothesis"}     # 장면 판단도 가설이다(AI-S-04)

    if picked in domains and switch and picked != STATE.get("active_domain"):
        out.update(apply_domain(cfg, picked))
        out["switched"] = True
    elif picked not in domains:
        out["reason"] = ("후보 밖이라 바꾸지 않았다 — 모르는 도메인으로 갈아타는 것이 "
                         "틀린 사전을 쓰는 것보다 낫지 않다")
    elif picked == STATE.get("active_domain"):
        out["reason"] = "이미 그 도메인이다"

    root = resolve(cfg, cfg["store"]["root"]) / "scene"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{last['frame_id']}.domain.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def _b64(raw: bytes) -> str:
    import base64

    return base64.b64encode(raw).decode()


# --------------------------------------------------------------------------
# VLM 환경 서술 — **명령이 올 때만** 돈다(사용자 2026-09-21).
# --------------------------------------------------------------------------
def describe_scene(cfg: dict) -> dict:
    """가장 최근 프레임의 환경 서술. LLM과 마찬가지로 프레임 처리 경로에 없다.

    주기 실행을 뺀 이유: 이것은 객체 판정이 아니라 **현재 상황 서술**이라 매 프레임
    필요하지 않고, ollama 호출이 초 단위라 상시로 돌면 비동기 큐만 잡아먹는다.
    가시화나 운영자가 "지금 무슨 상황인가"를 물을 때 답하면 된다.
    """
    if STATE.get("vlm") is None:
        return {"error": f"vlm_unavailable:{STATE.get('vlm_error','?')}"}
    last = STATE.get("last_frame")
    if not last:
        return {"error": "no frame received yet"}
    t = time.perf_counter()
    desc = STATE["vlm"].describe_scene(last["jpeg"])
    out = {"frame_id": last["frame_id"], "frame_seq": last.get("frame_seq"),
           "observed_at": last.get("observed_at"),
           "domain": cfg["vlm"].get("active_domain"),
           "age_s": round(time.time() - (last.get("at") or time.time()), 2),
           "scene": desc, "vlm_ms": round((time.perf_counter() - t) * 1000.0, 1)}
    root = resolve(cfg, cfg["store"]["root"]) / "scene"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{last['frame_id']}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# 비동기 워커
# --------------------------------------------------------------------------
def worker(cfg: dict) -> None:
    """비동기 단계 — CLIP 근거와 MoGe2-Aerial 거리. 끝나는 대로 말단에 민다.

    stage1과 나눈 이유는 예산이다. CLIP이 프레임당 800ms~1s, depth가 그 위에
    얹히므로 이걸 동기로 두면 당기는 주기가 그만큼 늘어난다. 나눠 두면 stage1은
    프레임 속도를 지키고 느린 근거는 준비되는 대로 따라붙는다.

    **거리 근거는 여기서 나오므로 태생적으로 늦다.** 말단이 `ego_compensated=False`로
    표시해 그 사실을 숨기지 않는다(drone_rpi/records.py Range).
    """
    q: queue.Queue = STATE["async_q"]
    n = 0
    every = int(cfg.get("depth", {}).get("every_n_frames", 1))
    while True:
        item = q.get()
        if item is None:
            return
        record, bgr, meta, puller = item
        n += 1

        # **주기적으로 오래된 것부터 지운다.** 비동기에서 도는 이유는 당기는 주기를
        # 건드리지 않기 위해서다 — 디렉터리 훑기는 파일 수에 비례한다.
        rot = STATE.get("rotation")
        if rot is not None:
            gone = rot.maybe_prune()
            if gone:
                g = STATE.get("crop_gate")
                if g is not None:
                    g.sweep()
                print(f"  [보존] 정리 {gone}  누적 {rot.removed_total}개 / "
                      f"{rot.freed_total/1048576:.0f}MB", flush=True)

        try:
            record = stage2(cfg, record, bgr)
            pt = record.pop("_stage2_per_track", [])
            if pt and puller is not None:
                puller.push_verdict({
                    "stage": 2, "frame_id": record["frame_id"],
                    "frame_seq": record.get("frame_seq"),
                    "camera_id": meta.get("camera_id", 0),
                    "observed_at": record.get("observed_at"),
                    "per_track": pt})

            # 거리 — mode가 "async"일 때만 여기서 돈다. 기본은 동기이고, 그때는
            # 이미 stage1 판정에 실려 나갔다(Puller.run).
            if (cfg.get("depth", {}).get("mode", "sync") == "async"
                    and every > 0 and (n % every == 0)):
                if stage_depth(cfg, record, meta, bgr) and puller is not None:
                    puller.push_verdict({
                        "stage": "depth", "frame_id": record["frame_id"],
                        "frame_seq": record.get("frame_seq"),
                        "camera_id": meta.get("camera_id", 0),
                        "observed_at": record.get("observed_at"),
                        "obstacles": record.get("obstacles") or []})

            # 미검출 분석 — 이 프레임에서 OVD가 놓친 것. 화면 하단 패널용이라
            # 말단 상태를 바꾸지 않는다.
            # **주기를 둔다.** 실측 662~687ms로 depth(493ms)와 같은 워커를
            # 나눠 쓰면 서로 밀어낸다. 미검출 후보는 슬롯으로 누적되므로
            # 매 프레임 볼 필요가 없다 — min_seen(3)에 도달하는 데 몇 프레임이
            # 더 걸릴 뿐이다.
            ue = int(cfg.get("unknown", {}).get("every_n_frames", 4))
            if cfg.get("unknown", {}).get("enabled", True) and ue > 0 and n % ue == 0:
                stage_missed(cfg, record, meta, bgr)

            _persist(cfg, record, bgr)
            # 로그는 **실제로 돈 것만** 말한다. 꺼 둔 단계를 0으로 찍으면 매
            # 프레임 눈이 그걸 훑게 되고, 정작 도는 단계가 묻힌다.
            s2 = record.get("stage2", {})
            bits = []
            if not s2.get("skipped"):
                c = s2.get("counts", {})
                bits.append(f"CLIP {record['timing_ms'].get('clip',0):.0f}ms "
                            f"배경 {c.get('background',0)}/미식별 {c.get('unidentified',0)}")
            ms_ = record.get("missed") or {}
            if ms_ and not ms_.get("skipped"):
                bits.append(f"missed rpn {ms_.get('rpn',0)}"
                            f"→설명 {ms_.get('known',0)}"
                            f"/배경 {ms_.get('background',0)}"
                            f"→{ms_.get('shown',0)}장 "
                            f"({record['timing_ms'].get('missed',0):.0f}ms)")
            dp = record.get("depth")
            if dp and dp.get("ran"):
                bits.append(f"depth {record['timing_ms'].get('depth',0):.0f}ms "
                            f"거리 {dp.get('n_ranged',0)}건")
            if bits:
                print(f"  [async] {record['frame_id']}  " + "  ".join(bits), flush=True)
        except Exception as exc:
            # 조용한 실패가 stage1을 멀쩡해 보이게 만든다. 스택까지 남긴다(AI-O-02).
            import traceback
            print(f"  [async] {record.get('frame_id')} 실패: {exc!r}", flush=True)
            traceback.print_exc()


def _persist(cfg: dict, record: dict, bgr) -> None:
    st = cfg["store"]
    root = resolve(cfg, st["root"])
    fid = record["frame_id"]
    if st.get("save_record", True):
        (root / "records").mkdir(parents=True, exist_ok=True)
        (root / "records" / f"{fid}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    if st.get("save_overlay", True):
        (root / "overlay").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / "overlay" / f"{fid}.jpg"),
                    overlay(bgr, record.get("ovd", []), record.get("unknown", []),
                            bool(cfg["clip"].get("visualize_weak", True))))


def overlay(bgr, ovd_dets, unknown, show_weak: bool = True):
    """색 = **판정 범주**. 세 가지뿐이다(사용자 2026-09-21).

        진초록  확인됨      OVD가 확신을 갖고 이름 붙임  → 말단 레코드에 반영
        노랑    미식별      배경이 아닌데 뭔지 모름      → 근거만, 관리자 판단용
        회색    배경        버림                         → crop 저장 안 함

    **연초록을 없애고 노랑으로 합쳤다.** 잠정 OVD(낮은 conf)는 미식별과 같은
    범주다 — 둘 다 "배경이 아닌데 모르는 것"이고, 다른 것은 어느 근거가 붙어
    있느냐뿐이다. 그래서 라벨에 **두 근거를 나란히** 적는다:

        car 0.22? | vehicle_part 0.24?      ← OVD 근거 + CLIP 근거
        ? | road_debris 0.27?               ← CLIP 근거만
        person 0.19?                        ← OVD 근거만 (RPN 후보가 없던 검출)

    관리자는 이 둘을 같이 보고 판단한다(AI-S-03: 신뢰도와 근거 충분도는 별개,
    AI-L-03: 판단에 쓸 근거를 사용자에게 제공한다).

    배경도 그린다 — 도로·건물을 실제로 지우고 있는지 눈으로 확인할 수 있어야 한다.
    저장은 여전히 안 한다. 그리는 것과 저장하는 것은 다른 결정이다.
    """
    GREEN, YELLOW, GREY = (0, 200, 0), (0, 215, 255), (150, 150, 150)
    out = bgr.copy()
    n = {"confirmed": 0, "unidentified": 0, "background": 0, "weak_hidden": 0}

    def label(txt, x, y, color):
        cv2.putText(out, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    # 1) 확인됨 — OVD 확신
    for d in ovd_dets:
        if d.get("verdict") != "confident":
            continue
        n["confirmed"] += 1
        x1, y1, x2, y2 = [int(t) for t in d["box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, 2)
        label(f"{d['label']} {d['conf']:.2f}", x1, max(12, y1 - 5), GREEN)

    # 2) 미식별 / 배경 — RPN 후보에서 온 것. 두 근거를 나란히 적는다.
    hinted = set()
    for d in unknown:
        v = d.get("clip", {}).get("verdict")
        if v not in ("background", "unidentified"):
            continue
        # weak는 화면에서만 숨긴다 — **저장은 그대로다**. 프레임당 미식별이 53~62개라
        # 전부 그리면 관리자가 볼 수 없다(2026-09-21). 데이터를 버리는 것이 아니라
        # 가시화 기본값을 정하는 것이고, show_weak로 되돌린다.
        if (v == "unidentified" and not show_weak
                and d.get("clip", {}).get("strength") != "strong"):
            n["weak_hidden"] += 1
            continue
        n[v] += 1
        is_bg = v == "background"
        color = GREY if is_bg else YELLOW
        x1, y1, x2, y2 = [int(t) for t in d["box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1 if is_bg else 2)

        parts = []
        h = d.get("ovd_hint")
        if h:
            parts.append(f"{h['label']} {h['conf']:.2f}?")
            hinted.add(h["label"])
        top = d["clip"]["ranking"][0]
        parts.append(f"{top['class']} {top['score']:.2f}?")
        label(" | ".join(parts), x1, min(bgr.shape[0] - 4, y2 + 13), color)

    # 3) RPN 후보가 없던 잠정 OVD 검출도 미식별이다 — 빠뜨리면 근거가 사라진다.
    for d in ovd_dets:
        if d.get("verdict") != "tentative":
            continue
        n["unidentified"] += 1
        x1, y1, x2, y2 = [int(t) for t in d["box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), YELLOW, 1)
        label(f"{d['label']} {d['conf']:.2f}?", x1, max(12, y1 - 5), YELLOW)

    hidden = f"  +{n['weak_hidden']} weak hidden" if n["weak_hidden"] else ""
    lines = [f"confirmed {n['confirmed']} (green)",
             f"unidentified {n['unidentified']} (yellow){hidden}  "
             f"background {n['background']} (grey, dropped)"]
    for k, txt in enumerate(lines):
        y = 26 + k * 26
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(out, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, th)
    return out


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def parse_multipart(body: bytes, ctype: str) -> dict[str, bytes]:
    import email

    msg = email.message_from_bytes(
        b"Content-Type: " + ctype.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    parts: dict[str, bytes] = {}
    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        name = None
        for token in part.get("Content-Disposition", "").split(";"):
            token = token.strip()
            if token.startswith("name="):
                name = token[5:].strip('"')
        if name:
            parts[name] = part.get_payload(decode=True)
    return parts




# ─────────────────────────────────────────────────────────────────────────────
# stage_depth — MoGe2-Aerial 거리 산출
# ─────────────────────────────────────────────────────────────────────────────
def stage_depth(cfg: dict, record: dict, meta: dict, bgr) -> list[dict]:
    """프레임 한 장의 depth 맵을 내고 트랙별 대표 거리를 뽑는다.

    **프레임당 한 번만 추론한다.** 트랙마다 crop을 넣으면 트랙 수만큼 느려지는데,
    depth는 장면 전체의 기하라 한 번 내고 박스로 잘라 쓰면 된다.

    **record["obstacles"] 항목에 range를 직접 얹는다.** 별도 목록으로 돌려주면
    말단에서 obs_id로 다시 조인해야 하고, 그 조인이 한 군데라도 빠지면 거리만
    조용히 사라진다.

    두 경로를 같이 낸다:
      * `moge2_aerial` — 신경망 metric depth. 장면 전체를 덮지만 작은 객체에 약하다.
      * `class_prior`  — 라벨 + 박스 높이 + focal_px. **추가 모델 없이** 나오고,
        100m의 사람처럼 픽셀이 10px뿐인 경우 오히려 이쪽이 낫다.

    둘이 다르면 **합치지 않고 둘 다 보낸다.** 어느 쪽이 맞는지는 이 코드가 알 수
    없고, 평균을 내면 둘 다 아닌 값이 된다. 판단은 소비자 몫이다(AI-S-03).
    """
    dcfg = cfg.get("depth", {})
    depth_model: AerialDepth | None = STATE.get("depth")
    cam_meta = meta.get("camera") or {}
    hfov = cam_meta.get("hfov_deg")
    focal = cam_meta.get("focal_px")
    priors = {k: float(v) for k, v in (dcfg.get("class_height_m") or {}).items()
              if not k.startswith("$")}

    dmap, ms = None, 0.0
    if depth_model is not None and depth_model.available():
        t = time.perf_counter()
        dmap = depth_model.depth_map(bgr, hfov_deg=hfov)
        ms = (time.perf_counter() - t) * 1000.0

    if dmap is not None:
        publish_depth(cfg, meta, dmap)
        # **미검출 분석이 이 맵을 쓴다.** depth는 async라 미검출과 같은 워커에서
        # 번갈아 도는데, 방금 낸 맵을 캐시해 두면 미검출 쪽이 다시 추론하지
        # 않고도 거리와 실제 크기를 낼 수 있다.
        with LIVE_LOCK:
            STATE.setdefault("depth_cache", {})[int(meta.get("camera_id", 0))] = (
                dmap, time.time())

    pct = float(dcfg.get("box_percentile", 25.0))
    lo, hi = float(dcfg.get("min_m", 0.3)), float(dcfg.get("max_m", 400.0))
    out: list[str] = []          # 거리를 붙인 obs_id

    # 대상은 **말단에 보낼 장애물**이다. 모든 검출에 거리를 붙이면 프레임당
    # 수십 개가 되고 대부분 잠정이라 말단 상태를 거리로 채우게 된다.
    for o in record.get("obstacles", []):
        box = o.get("box")
        if not box:
            continue
        lab = (o.get("ovd") or {}).get("label", "")
        conf = float((o.get("ovd") or {}).get("conf") or 0.0)

        # **metric provider일 때만** 깊이 맵에서 미터를 뽑는다. 상대 역depth에서
        # 뽑은 수를 미터라고 부르면 그 순간 조용히 틀린 거리가 파이프라인에 든다.
        nn = (depth_model.range_for_box(dmap, box, percentile=pct)
              if dmap is not None and getattr(depth_model, "metric", True) else None)
        prior = class_prior_range(lab, box, focal, priors)

        cand = None
        if nn is not None and lo <= nn <= hi:
            cand = {"meters": nn, "source": "moge2_aerial", "confidence": conf}
        if prior is not None and lo <= prior <= hi:
            alt = {"meters": prior, "source": "class_prior", "confidence": conf}
            # 사전 쪽을 주값으로 올리는 조건은 하나뿐이다: 박스가 너무 작아
            # depth 맵이 그 자리에서 할 말이 없을 때. 임의로 섞지 않는다.
            small = (float(box[3]) - float(box[1])) < float(dcfg.get("prior_wins_below_px", 24))
            if cand is None or small:
                cand, alt = alt, cand
            if alt is not None:
                cand = {**cand, "alternative": alt}
        if cand is not None:
            # 장애물 항목에 직접 얹는다 — 별도 목록으로 두면 말단에서 다시
            # 맞춰야 하고, obs_id로 두 번 조인하는 코드가 생긴다.
            o["range"] = cand
            out.append(o["obs_id"])

    record.setdefault("timing_ms", {})["depth"] = round(ms, 1)
    record["depth"] = {"ran": dmap is not None, "n_ranged": len(out),
                       "provider": dcfg.get("provider", "moge2_aerial"),
                       "metric": bool(getattr(depth_model, "metric", True))
                       if depth_model is not None else None,
                       "hfov_deg": hfov, "focal_px": focal,
                       "error": getattr(depth_model, "error", None)
                       if depth_model is not None else "not_loaded"}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# stage_missed — OVD가 놓친 것 찾기 (비동기)
# ─────────────────────────────────────────────────────────────────────────────
def stage_missed(cfg: dict, record: dict, meta: dict, bgr) -> dict:
    """RPN → 설명된 것 제거 → 배경 제거 → 남은 것을 crop해 패널에 올린다.

    **비동기다.** RPN이 프레임당 수십~수백 ms이고, 이 결과는 화면 하단 패널용
    이라 말단 상태를 바꾸지 않는다 — 동기 예산에 넣을 이유가 없다.
    """
    rpn = STATE.get("edge_rpn")
    if rpn is None:
        record["missed"] = {"skipped": "edge_rpn_unavailable"}
        return record

    t0 = time.perf_counter()
    boxes, scores, _t = rpn.propose(bgr)
    rpn_ms = (time.perf_counter() - t0) * 1000.0
    items = [{"box": [float(v) for v in b], "objectness": float(sc)}
             for b, sc in zip(boxes, scores)]

    h, w = bgr.shape[:2]
    unknown, stats = find_missed(cfg, items, record.get("ovd", []), (w, h))
    stats["rpn"] = len(items)
    stats["rpn_ms"] = round(rpn_ms, 1)

    ucfg = cfg.get("unknown", {})
    ctx = float(ucfg.get("crop_context", 1.5))
    crops, keep = [], []
    for d in unknown[: int(ucfg.get("max_per_frame", 60))]:
        cr = crop_of(bgr, d["box"], context=ctx)
        if cr is not None:
            crops.append(cr)
            keep.append(d)

    # ── 배경 버리기: **기하 규칙**(CLIP 아님) ────────────────────────────
    # CLIP + 특징 사전이 하던 일이다. 그쪽은 1등-2등 gap이 0.002~0.009라 사실상
    # 구분을 못 했고(닫힌 사전 argmax에 '해당 없음'이 없다) crop 수에 선형이라
    # 프레임당 ~1100ms였다. 기하는 **물리적으로 말이 되는 근거**를 쓰고,
    # 이미 낸 depth 맵 하나로 끝나 사실상 공짜다.
    #
    # 실측 분포(2026-09-21, 미지 후보 24개):
    #   돌출(m)     p10 -0.343  p25 -0.121  p50 0.025  p75 0.424  p90 0.683
    # 음수는 주변보다 뒤에 있다는 뜻 = 면이다. 0 근처를 경계로 깨끗이 갈린다.
    cam = int(meta.get("camera_id", 0))
    with LIVE_LOCK:
        cached = (STATE.get("depth_cache") or {}).get(cam)
    dm_cached = cached[0] if cached and (time.time() - cached[1]) < 3.0 else None
    focal = (meta.get("camera") or {}).get("focal_px")

    t = time.perf_counter()
    verdicts = [geometric_verdict(d["box"], dm_cached, focal, cfg) for d in keep]
    geo_ms = (time.perf_counter() - t) * 1000.0

    # **정체성을 준다.** RPN 박스는 매 프레임 조금씩 달라서, 그대로 그리면 같은
    # 물체가 다른 칸으로 옮겨 다니고 화면이 계속 바뀌어 읽을 수가 없다
    # (사용자 2026-09-21). 슬롯으로 묶으면 칸이 자리를 지키고, 덤으로 프레임을
    # 건너 정보가 쌓인다(seen/votes). 배경 버리기도 여기서 걸린다.
    tr = STATE.setdefault("missed_tracks", {}).get(cam)
    if tr is None:
        tr = STATE["missed_tracks"][cam] = MissedTracker(cfg)
    show = tr.update(keep, crops, verdicts)

    # **쓸모 있는 정보를 붙인다.** 의심 클래스는 이 후보와 겹치는 낮은 conf
    # OVD 검출이다 — 추가 연산 0이고, 겹치는 것이 없으면 그 자체가 가장 강한
    # 미검출 신호다. 거리·실제 크기는 기하 판정이 이미 계산해 두었다.
    for sl in show:
        sl["suspects"] = suspect_classes(sl["box"], record.get("ovd", []))

    stats["background"] = sum(1 for v in verdicts if v["verdict"] == "background")
    # 어느 규칙이 버렸는지 — 한 규칙이 전부 먹고 있으면 임계가 틀린 것이다.
    why: dict[str, int] = {}
    for v in verdicts:
        if v["verdict"] == "background":
            why[v["reason"]] = why.get(v["reason"], 0) + 1
    stats["dropped_by"] = why
    stats["no_depth"] = sum(1 for v in verdicts if v.get("reason") == "no_depth")
    stats.update(tr.stats())
    stats["shown"] = len(show)
    stats["geo_ms"] = round(geo_ms, 1)

    scfg = cfg.get("stream", {})
    img = missed_panel(
        [sl["crop"] for sl in show],
        [{"objectness": sl.get("best_obj"), "seen": sl.get("seen"),
          "slot": sl["id"], "suspects": sl.get("suspects") or [],
          "dist_m": sl.get("dist_m"), "w_m": sl.get("w_m"), "h_m": sl.get("h_m"),
          "protrusion_m": sl.get("protrusion_m"),
          "box": sl["box"]} for sl in show],
        width=int(scfg.get("missed_width", 1280)),
        rows=int(scfg.get("missed_rows", 3)),
        cols=int(scfg.get("missed_cols", 4)),
        stats=stats)
    _publish(MISSED, cfg, cam, img, {"frame_id": meta.get("frame_id"), **stats})
    record["missed"] = stats
    record.setdefault("timing_ms", {})["missed"] = round(rpn_ms + geo_ms, 1)
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Puller — 말단에서 당겨 오고, 끝난 판정을 도로 밀어 넣는다.
# ─────────────────────────────────────────────────────────────────────────────
class Puller:
    """말단 하나를 담당하는 워커. **세 동작이 서로 독립이다.**

    예전에는 당기기·추론·보내기가 한 루프에 묶여 있어서 셋 중 하나만 멈출 수
    없었다. 실제로 그게 불편한 경우가 있다:

      * 말단이 뭘 보고 있는지 **화면만** 확인하고 싶다 → 가져오기만 켠다
      * GPU를 다른 일에 쓰는 동안 **수집은 계속** 하고 싶다 → 추론만 끈다
      * 말단 상태를 건드리지 않고 **엣지에서만** 돌려 보고 싶다 → 보내기만 끈다
      * 비행 중 말단 레코드를 얼리고 싶다 → 보내기만 끈다

    그래서 게이트를 셋으로 나눴다. 각각 따로 켜고 끈다(POST /api/command).

        pull_on    당기기   가져온 프레임은 화면과 last_frame에 남는다
        infer_on   추론     stage1/stage2/depth
        push_on    보내기   말단 POST /api/verdict

    의존은 자연스럽게 걸린다 — 안 당기면 추론할 게 없고, 추론 안 하면 보낼 게 없다.
    끌 때는 역순이 안전하다(보내기 → 추론 → 당기기).

    `fetch_once`는 **연속 당기기와 별개의 한 장짜리 동작**이다. pull_on이 꺼져
    있어도 한 장만 가져온다.
    """

    def __init__(self, cfg: dict, node: dict):
        self.cfg = cfg
        self.name = node.get("name", node["host"])
        self.base = f"http://{node['host']}:{int(node.get('port', 8890))}"
        self.cams = list(node.get("cameras", [0]))
        self.timeout = (float(node.get("connect_timeout_s", 3.0)),
                        float(node.get("read_timeout_s", 20.0)))
        self.idle_sleep = float(node.get("idle_sleep_s", 0.2))
        self.drain_spool = bool(node.get("drain_spool", True))

        # ── 세 게이트. 기동 상태는 설정에서 온다 ──────────────────────────────
        self.pull_on = threading.Event()
        self.infer_on = threading.Event()
        self.push_on = threading.Event()
        auto = node.get("autostart", {})
        if auto.get("pull", True):
            self.pull_on.set()
        if auto.get("inference", True):
            self.infer_on.set()
        if auto.get("push", True):
            self.push_on.set()

        self._fetch_once = threading.Event()
        self._fetch_result: dict | None = None
        self._fetch_done = threading.Event()

        # 카메라별로 마지막에 받은 frame_seq. 이걸 since로 보내면 말단이 새
        # 프레임이 없을 때 204로 끝낸다 — 실측 중복 90%가 여기서 사라진다.
        self.last_seq: dict[int, int] = {}
        self.long_poll_s = float(node.get("long_poll_s", 2.0))
        self.pulled = self.not_modified = self.failed = self.spooled_in = 0
        self.inferred = self.pushed = self.push_suppressed = 0
        self.inference_dropped = 0
        self.last_error: str | None = None
        self.last_frame: dict | None = None      # 마지막으로 가져온 것의 요약
        self._stop = threading.Event()
        self._inference_q: queue.Queue = queue.Queue(maxsize=1)
        self._inference_thread: threading.Thread | None = None
        self._i = 0

    # -- 제어 --------------------------------------------------------------
    def set_gate(self, gate: str, on: bool) -> dict:
        ev = {"pull": self.pull_on, "inference": self.infer_on,
              "push": self.push_on}.get(gate)
        if ev is None:
            raise KeyError(gate)
        ev.set() if on else ev.clear()
        return self.status()

    def fetch(self, timeout_s: float = 15.0) -> dict:
        """한 장만 가져온다. pull_on과 무관하게 동작한다.

        루프 스레드에 시켜서 가져온다 — 여기서 직접 requests를 치면 연속 당기기와
        같은 말단을 동시에 두드리게 되고, 말단은 최신 한 장만 들고 있어서 둘 중
        하나가 빈손으로 돌아온다.
        """
        self._fetch_done.clear()
        self._fetch_result = None
        self._fetch_once.set()
        if not self._fetch_done.wait(timeout_s):
            return {"ok": False, "error": "timeout", "terminal": self.name}
        return self._fetch_result or {"ok": False, "error": "no result"}

    def status(self) -> dict:
        return {"terminal": self.name, "base": self.base, "cameras": self.cams,
                "gates": {"pull": self.pull_on.is_set(),
                          "inference": self.infer_on.is_set(),
                          "push": self.push_on.is_set()},
                "counters": {"pulled": self.pulled,
                             "not_modified": self.not_modified,
                             "failed": self.failed,
                             "from_spool": self.spooled_in,
                             "inferred": self.inferred, "pushed": self.pushed,
                             "push_suppressed": self.push_suppressed,
                             "inference_dropped": self.inference_dropped},
                "last_frame": self.last_frame, "last_error": self.last_error}

    # -- 전송 --------------------------------------------------------------
    def _get_frame(self, path: str):
        # 롱폴이므로 읽기 타임아웃은 long_poll_s보다 넉넉해야 한다.
        r = requests.get(self.base + path,
                         timeout=(self.timeout[0],
                                  max(self.timeout[1], self.long_poll_s + 5)))
        if r.status_code == 204:
            self.not_modified += 1       # "새 것 없음" — 실패가 아니다
            return None
        if r.status_code == 404:
            return None
        r.raise_for_status()
        parts = parse_multipart(r.content, r.headers.get("Content-Type", ""))
        if "meta" not in parts:
            return None
        meta = json.loads(parts["meta"])
        jpeg = parts.get("frame")
        bgr = (cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
               if jpeg else None)
        return meta, bgr, jpeg

    def push_verdict(self, verdict: dict) -> bool:
        """끝난 판정을 말단에 밀어 넣는다. 끝나는 시각을 아는 쪽이 이쪽이다.

        **보내기 게이트가 꺼져 있으면 여기서 막는다.** 호출부마다 확인하지 않고 한
        곳에서 막는 이유는, 판정 경로가 셋(stage1·stage2·depth)이라 어느 하나를
        빠뜨리면 "껐는데 일부가 계속 간다"가 되기 때문이다.
        """
        if not self.push_on.is_set():
            self.push_suppressed += 1
            return False
        try:
            requests.post(self.base + "/api/verdict", json=verdict,
                          timeout=self.timeout)
            self.pushed += 1
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {self.base}"
            return False

    def _ack_spool(self, spool_id: str) -> None:
        try:
            requests.post(self.base + "/api/spool/ack", json={"spool_id": spool_id},
                          timeout=self.timeout)
        except Exception:
            pass          # 못 지워도 다음에 다시 받는다 — 중복이 유실보다 낫다

    # -- 루프 --------------------------------------------------------------
    def stop(self):
        self._stop.set()
        if self._inference_thread is not None:
            try:
                self._inference_q.put_nowait(None)
            except queue.Full:
                pass

    def run(self) -> None:
        cfg = self.cfg
        self._inference_thread = threading.Thread(
            target=self._inference_loop, name=f"infer-{self.name}", daemon=True)
        self._inference_thread.start()
        while not self._stop.is_set():
            one_shot = self._fetch_once.is_set()
            if not (self.pull_on.is_set() or one_shot):
                time.sleep(self.idle_sleep)
                continue

            got, from_spool = None, False
            try:
                cam = self.cams[self._i % len(self.cams)]
                self._i += 1
                # **since + 롱폴.** 같은 프레임을 다시 받지 않고, 새 프레임이
                # 나오는 즉시 깨어난다. 폴링 간격만큼 늦던 지연도 사라진다.
                # 한 장짜리 요청(fetch)은 기다리지 않는다 — "지금 뭐가 보이나"를
                # 묻는 것이라 없으면 없다고 답하는 편이 낫다.
                wait = 0.0 if one_shot else self.long_poll_s
                got = self._get_frame(
                    f"/api/frame?cam={cam}&since={self.last_seq.get(cam, 0)}"
                    f"&wait={wait}")
                # 실시간이 없을 때만 단절 구간을 훑는다 — 현재 프레임이 먼저다.
                # 한 장짜리 요청에는 스풀을 쓰지 않는다: "지금 뭐가 보이나"를
                # 묻는 것이지 과거를 묻는 게 아니다.
                if got is None and self.drain_spool and not one_shot:
                    got = self._get_frame("/api/spool")
                    from_spool = got is not None
            except Exception as exc:
                self.failed += 1
                # 예외 이름 + 대상 주소. repr 전체는 길어서 화면 한 줄에 안 맞고,
                # 정작 필요한 "어디로 가려다 실패했나"가 뒤로 밀린다.
                self.last_error = f"{type(exc).__name__}: {self.base}"
                if one_shot:
                    self._fetch_result = {"ok": False, "error": self.last_error,
                                          "terminal": self.name}
                    self._fetch_once.clear()
                    self._fetch_done.set()
                    continue
                time.sleep(min(5.0, 0.5 * min(self.failed, 10)))
                continue

            if got is None:
                if one_shot:
                    self._fetch_result = {"ok": False, "error": "no frame on terminal",
                                          "terminal": self.name}
                    self._fetch_once.clear()
                    self._fetch_done.set()
                    continue
                # 롱폴이 이미 기다렸으므로 여기서 또 자지 않는다. 자면 그만큼
                # 다음 프레임을 늦게 집는다.
                continue

            meta, bgr, jpeg = got
            if bgr is None:
                if one_shot:
                    self._fetch_result = {"ok": False, "error": "frame decode failed"}
                    self._fetch_once.clear()
                    self._fetch_done.set()
                continue

            self.pulled += 1
            if from_spool:
                self.spooled_in += 1
            else:
                self.last_seq[int(meta.get("camera_id", 0))] = int(
                    meta.get("frame_seq", 0))
            # 원본은 **추론과 무관하게** 항상 발행한다 — 추론을 꺼 둔 채로도
            # 말단이 뭘 보고 있는지는 화면으로 확인할 수 있어야 한다.
            publish_original(cfg, meta, bgr)
            with LIVE_LOCK:
                LAST_META[int(meta.get("camera_id", 0))] = meta
            self.last_frame = {"frame_id": meta.get("frame_id"),
                               "frame_seq": meta.get("frame_seq"),
                               "camera_id": meta.get("camera_id"),
                               "observed_at": meta.get("observed_at"),
                               "age_s": round(time.time() - (meta.get("observed_at") or 0), 2),
                               "wh": meta.get("frame_wh"),
                               "n_obstacles": len(meta.get("obstacles") or []),
                               "flow": meta.get("flow"),
                               "from_spool": from_spool,
                               "at": time.time()}
            if one_shot:
                self._fetch_result = {"ok": True, **self.last_frame,
                                      "terminal": self.name,
                                      "inference": self.infer_on.is_set()}
                self._fetch_once.clear()
                self._fetch_done.set()

            # 원본 발행은 추론과 분리한다. 추론이 늦어도 Pi3의 최신 프레임은
            # 계속 화면에 흐르고, 엣지는 가장 최근 프레임 하나만 따라간다.
            if self.infer_on.is_set():
                item = (meta, bgr, from_spool)
                try:
                    self._inference_q.put_nowait(item)
                except queue.Full:
                    try:
                        self._inference_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._inference_q.put_nowait(item)
                    except queue.Full:
                        pass
                    self.inference_dropped += 1
            elif from_spool and meta.get("spool_id"):
                self._ack_spool(meta["spool_id"])

    def _inference_loop(self) -> None:
        """최신 Pi3 프레임만 추론하고, 늦은 결과는 말단 좌표로 반영한다."""
        while not self._stop.is_set():
            try:
                item = self._inference_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                return
            meta, bgr, from_spool = item
            cfg = self.cfg
            try:
                record = stage1(cfg, meta, bgr)
                self.inferred += 1
                dcfg = cfg.get("depth", {})
                sync_depth = (dcfg.get("mode", "sync") == "sync"
                              and STATE.get("depth") is not None
                              and STATE["depth"].available())
                if sync_depth:
                    stage_depth(cfg, record, meta, bgr)
                obs = record.get("obstacles") or []
                if obs:
                    self.push_verdict({"stage": 1, "frame_id": record["frame_id"],
                                       "frame_seq": record.get("frame_seq"),
                                       "camera_id": meta.get("camera_id", 0),
                                       "observed_at": record.get("observed_at"),
                                       "obstacles": obs})
                publish_live(cfg, meta, bgr, record, record.get("unknown", []))
                if from_spool and meta.get("spool_id"):
                    self._ack_spool(meta["spool_id"])
                want_async = (cfg.get("unknown", {}).get("enabled", True)
                              or STATE.get("clip") is not None
                              or not sync_depth)
                if want_async:
                    try:
                        STATE["async_q"].put_nowait((record, bgr, meta, self))
                    except queue.Full:
                        STATE["async_dropped"] = STATE.get("async_dropped", 0) + 1
            except Exception as exc:
                import traceback
                print(f"  [{self.name}] stage1 실패: {exc!r}", flush=True)
                traceback.print_exc()


def find_puller(name: str | None):
    """이름으로 말단을 찾는다. 하나뿐이면 이름 없이도 잡힌다."""
    pls = STATE.get("pullers", [])
    if not pls:
        return None
    if not name:
        return pls[0] if len(pls) == 1 else None
    for pl in pls:
        if pl.name == name:
            return pl
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HTTP — 오버레이 스트리밍과 운영 창구. **로컬 포트에 뜬다.**
# ─────────────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code: int, obj: dict):
        raw = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    # ------------------------------------------------------------------
    # 실시간 스트림
    # ------------------------------------------------------------------
    def _composed(self, cam: int):
        """원본·depth·반영을 한 장으로 합쳐 내보내는 MJPEG.

        연결 하나로 세 화면을 보낸다 — 이유는 compose_panels의 docstring에 있다.
        """
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        q = int(STATE["cfg"].get("stream", {}).get("jpeg_quality", 70))
        fps = float(STATE["cfg"].get("stream", {}).get("max_fps", 12))
        last = 0.0
        try:
            while not SHUTDOWN.is_set():
                with LIVE_LOCK:
                    at = max([(b.get(cam) or {}).get("at", 0.0)
                              for b in (ORIG, DEPTH, LIVE)] or [0.0])
                if at > last:
                    last = at
                    img = compose_panels(cam)
                    if img is not None:
                        ok, enc = cv2.imencode(".jpg", img,
                                               [cv2.IMWRITE_JPEG_QUALITY, q])
                        if ok:
                            d = enc.tobytes()
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n"
                                + f"Content-Length: {len(d)}\r\n\r\n".encode()
                                + d + b"\r\n")
                            self.wfile.flush()
                SHUTDOWN.wait(1.0 / max(1.0, fps))
        except (BrokenPipeError, ConnectionResetError):
            return

    def _stream(self, cam: int, store: dict | None = None):
        """MJPEG(multipart/x-mixed-replace). 브라우저 <img>가 그대로 받는다.

        WebSocket이나 별도 런타임을 들이지 않은 이유: 이 경로에 필요한 것은
        '최신 프레임을 계속 밀어 넣기'뿐이고 MJPEG는 표준 HTTP만으로 그걸 한다.
        영상 픽셀은 업무·관측 평면이 아니라 이 별도 미디어 경로로만 나간다(AI-C-08).
        """
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = 0.0
        fps = float(STATE["cfg"].get("stream", {}).get("max_fps", 12))
        try:
            # SHUTDOWN.wait()로 잔다 — time.sleep이면 종료 신호를 받고도 최대
            # 한 프레임 간격을 더 자고, 연결이 많으면 그만큼 종료가 늦어진다.
            while not SHUTDOWN.is_set():
                with LIVE_LOCK:
                    item = (LIVE if store is None else store).get(cam)
                if item and item["at"] > last:
                    last = item["at"]
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(item['jpeg'])}\r\n\r\n"
                                     .encode() + item["jpeg"] + b"\r\n")
                    self.wfile.flush()
                SHUTDOWN.wait(1.0 / max(1.0, fps))
        except (BrokenPipeError, ConnectionResetError):
            return

    def _page(self):
        with LIVE_LOCK:
            cams = sorted(LIVE)
        terms = [t.get("name", t.get("host", "?"))
                 for t in STATE["cfg"].get("terminals", []) if t.get("enabled", True)]
        if not cams:
            # 아직 한 장도 안 당겼을 때의 자리표시. terminals[].cameras는 **정수
            # 목록**이다(말단 config의 pi.cameras는 dict 목록이라 모양이 다르다 —
            # 예전에 그 둘을 같은 것으로 보고 여기서 터졌다).
            cams = [c for t in STATE["cfg"].get("terminals", [])
                    for c in t.get("cameras", []) if isinstance(c, int)] or [0]

        # **카메라당 한 스트림이다.** 원본·depth·반영을 서버에서 한 장으로
        # 합쳐 보낸다(compose_panels). 따로 보내면 카메라 2대 × 패널 3개 = 6개가
        # 브라우저 동시 연결 한도를 정확히 채워, 뒤늦게 붙는 패널은 빈 화면이 되고
        # /api/command마저 슬롯이 없어 매달린다(그래서 버튼이 굳었다).
        panes = "".join(
            f'<section><h2>camera {c} &nbsp;·&nbsp; original &nbsp;·&nbsp; depth &nbsp;·&nbsp; obstacle information - box, class, distance(m)</h2>'
            f'<img src="/panels?camera={c}" alt="camera {c}">'
            # 아래: OVD가 놓친 것. 위 패널이 "무엇을 잡았나"라면 이쪽은
            # "무엇을 놓쳤나"다 — 둘은 서로 보완이라 같은 화면에 있어야 한다.
            # 검출 결과만 보면 빠진 것이 안 보이고, 빠진 것만 보면 그게 정말
            # 빠진 건지 이미 잡힌 건지 알 수 없다.
            f'<h2 class="sub">missed candidates &nbsp;·&nbsp; crop '
            f'&nbsp;·&nbsp; class candidates &nbsp;·&nbsp; evidence</h2>'
            f'<img src="/missed?camera={c}" alt="missed {c}"></section>'
            for c in cams)

        # 말단 선택 — 하나뿐이면 굳이 고르게 하지 않는다.
        sel = ("".join(f'<option value="{t}">{t}</option>' for t in terms)
               if len(terms) > 1 else "")
        sel_html = (f'<select id="term" title="대상 말단">{sel}</select>'
                    if sel else f'<span class="mut">{terms[0] if terms else "말단 없음"}</span>')

        html = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>drone edge live</title>
<style>
 :root{{color-scheme:dark light;--bg:#111;--fg:#eee;--mut:#9aa;--line:#333}}
 body{{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}}
 main{{padding:12px}}
 section{{margin-bottom:16px}}
 h2{{font-size:13px;font-weight:600;color:var(--mut);margin:0 0 6px;
     text-transform:uppercase;letter-spacing:.06em}}
 h2.sub{{margin-top:10px;color:#8a8}}
 img{{width:100%;display:block;background:#000;border-radius:4px}}
 figcaption{{color:var(--mut);padding:4px 2px}}
 .mut{{color:var(--mut)}}
 /* 제어 막대 — 화면 위에 고정해 둔다. 스트림을 보면서 누르는 것이 목적이라
    스크롤로 사라지면 안 된다. */
 #ctl{{position:sticky;top:0;z-index:9;display:flex;flex-wrap:wrap;gap:8px;
      align-items:center;padding:10px 16px;background:#161616;
      border-bottom:1px solid var(--line)}}
 #ctl button{{font:inherit;padding:6px 12px;border-radius:6px;cursor:pointer;
      border:1px solid #444;background:#222;color:var(--fg)}}
 #ctl button:hover{{background:#2c2c2c}}
 #ctl button:disabled{{opacity:.45;cursor:progress}}
 /* 켜는 것과 끄는 것을 색으로 가른다 — 비행 중에 잘못 누르면 곤란하다. */
 /* 토글 버튼. 켜져 있으면 채워서 보여준다 — 누른 뒤 색만 사라지면
    무엇이 켜져 있는지 화면만 보고 알 수 없다. */
 #ctl button.tog{{min-width:11em}}
 /* 지금 그 상태인 버튼을 채워서 보여준다. 누른 뒤 색이 사라진 채로 남으면
    무엇이 켜져 있는지 화면만 보고 알 수 없다. */
 #ctl button.active{{background:#14361f;border-color:#2a6;color:#8fd8a8}}
 .sep{{width:1px;height:22px;background:var(--line);margin:0 4px}}
 select{{font:inherit;background:#222;color:var(--fg);border:1px solid #444;
      border-radius:6px;padding:5px 8px}}
 /* 게이트 표시등 — 지금 무엇이 켜져 있는지가 버튼보다 먼저 보여야 한다. */
 .pill{{padding:3px 9px;border-radius:999px;border:1px solid var(--line);
      color:var(--mut);font-size:12px}}
 .pill.live{{border-color:#2a6;color:#6d9}}
 #msg{{margin-left:auto;color:var(--mut);max-width:46ch;overflow:hidden;
      text-overflow:ellipsis;white-space:nowrap}}
 /* 실패는 줄을 따로 준다. 상태 줄에 끼워 넣으면 길이 제한에 잘려서
    정작 원인이 보이지 않는다. */
 #err{{padding:8px 16px;background:#2a1414;border-bottom:1px solid #633;
      color:#f99;font-size:13px;white-space:pre-wrap}}
</style>
<div id="ctl">
  {sel_html}
  <span class="sep"></span>
  <button id="b-pull" class="tog">이미지 가져오기</button>
  <button id="b-infer" class="tog">추론 시작</button>
  <span class="sep"></span>
  <span class="pill" id="p-pull">가져오기</span>
  <span class="pill" id="p-infer">추론</span>
  <span class="pill" id="p-push">보내기</span>
  <span id="msg"></span>
</div>
<div id="err" hidden></div>

<main>{panes}</main>

<script>
// 버튼 → POST /api/command. CLI(`server.py ctl …`)와 **같은 창구**를 친다 —
// 두 벌을 만들면 한쪽만 고치게 된다.
//
// 버튼은 둘뿐이고 **누를 때마다 뒤집힌다.** 라벨과 색이 지금 상태를 말한다.
// "추론"은 추론과 보내기를 함께 다룬다 — 추론만 켜고 결과를 안 보내는 것은
// 디버깅용이고, 그건 CLI(ctl push-off)로 한다. 화면에서 늘 고를 일이 아니다.
const $ = (id) => document.getElementById(id);
const termEl = $("term");
const term = () => (termEl ? termEl.value : null);
let GATES = {{pull: false, inference: false, push: false}};
let busy = false;

// 실패를 화면에 남긴다. 서버 로그를 볼 수 없는 자리에서 버튼만 눌러 보는
// 사람에게는 이게 유일한 단서다(AI-O-02: 조용히 실패하지 않는다).
function showErr(text) {{
  const e = $("err");
  if (!text) {{ e.hidden = true; e.textContent = ""; return; }}
  const ts = new Date().toTimeString().slice(0, 8);
  e.hidden = false;
  e.textContent = `[${{ts}}] ${{text}}`;
}}

function paint(g) {{
  if (!g) return;
  GATES = g;
  // 말단 쪽 실패는 버튼을 안 눌러도 생긴다(당기다 끊김) — 상태에서 끌어온다.
  if (g.__last_error) showErr(g.__last_error);
  for (const [key, id] of [["pull","p-pull"],["inference","p-infer"],["push","p-push"]]) {{
    $(id).classList.toggle("live", !!g[key]);
  }}
  const pull = $("b-pull"), infer = $("b-infer");
  pull.textContent = g.pull ? "이미지 정지" : "이미지 가져오기";
  pull.classList.toggle("active", !!g.pull);
  infer.textContent = g.inference ? "추론 종료" : "추론 시작";
  infer.classList.toggle("active", !!g.inference);
}}

async function post(command, extra) {{
  const body = Object.assign({{command}}, extra || {{}});
  const t = term(); if (t) body.terminal = t;
  const r = await fetch("/api/command", {{
    method: "POST", headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify(body),
    // **타임아웃을 건다.** 이게 없으면 서버가 응답을 못 줄 때 버튼이 disabled로
    // 영원히 굳는다 — 브라우저 연결 슬롯이 MJPEG로 꽉 찼을 때 실제로 그랬다.
    signal: AbortSignal.timeout(20000),
  }});
  return [r.ok, await r.json()];
}}

async function toggle(kind) {{
  if (busy) return;
  busy = true;
  const btns = [$("b-pull"), $("b-infer")];
  btns.forEach(b => b.disabled = true);
  $("msg").style.color = "var(--mut)";
  try {{
    let last = null, ok = true;
    if (kind === "pull") {{
      [ok, last] = await post(GATES.pull ? "stop_pull" : "start_pull");
    }} else {{
      // 추론과 보내기를 같이 뒤집는다. 끌 때는 역순(보내기 먼저) —
      // 추론을 먼저 끄면 이미 큐에 있던 판정이 뒤늦게 말단으로 샌다.
      const on = !GATES.inference;
      const seq = on ? ["start_inference", "start_push"]
                     : ["stop_push", "stop_inference"];
      for (const c of seq) {{
        [ok, last] = await post(c);
        if (!ok) break;
      }}
    }}
    paint(last && last.gates);
    if (ok) {{
      $("msg").textContent = "";
      showErr(null);
    }} else {{
      const why = (last && last.error) || `HTTP error`;
      $("msg").textContent = "failed";
      $("msg").style.color = "#f66";
      showErr(`${{kind === "pull" ? "pull" : "inference"}} toggle failed — ${{why}}`);
    }}
  }} catch (e) {{
    const why = e.name === "TimeoutError" ? "no response from edge (20s)" : String(e);
    $("msg").textContent = "failed";
    $("msg").style.color = "#f66";
    showErr(`${{kind === "pull" ? "pull" : "inference"}} toggle failed — ${{why}}`);
  }} finally {{
    btns.forEach(b => b.disabled = false);
    busy = false;
    refresh();
  }}
}}

$("b-pull").addEventListener("click", () => toggle("pull"));
$("b-infer").addEventListener("click", () => toggle("infer"));

// 표시등은 주기적으로 맞춘다 — CLI로 바꿨거나 다른 탭에서 눌렀을 수 있다.
async function refresh() {{
  if (busy) return;
  const body = {{command: "terminal_status"}};
  const t = term(); if (t) body.terminal = t;
  try {{
    const r = await fetch("/api/command", {{
      method: "POST", headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(body), signal: AbortSignal.timeout(8000)}});
    if (r.ok) {{
      const j = await r.json();
      const g = Object.assign({{}}, j.gates);
      if (j.last_error) g.__last_error = `terminal ${{j.terminal}} — ${{j.last_error}}`;
      paint(g);
      if (!j.last_error && !busy) showErr(null);
    }}
  }} catch (e) {{
    showErr(`edge unreachable — ${{e.name === "TimeoutError" ? "no response (8s)" : e}}`);
  }}
}}
refresh();
setInterval(refresh, 3000);
if (termEl) termEl.addEventListener("change", refresh);
</script>
</html>"""
        raw = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        cfg = STATE["cfg"]
        if self.path == "/" or self.path.startswith("/index"):
            self._page()
            return
        if self.path.startswith("/stream"):
            m = re.search(r"camera=(\d+)", self.path)
            self._stream(int(m.group(1)) if m else 0)
            return
        if self.path.startswith("/api/obstacles"):
            # 정렬 검증용 원본 덤프. **moved_px와 quality의 분포**를 봐야
            # 플로우가 실제로 듣는지 알 수 있다 — 화면만 보면 "안 움직였다"와
            # "정렬이 안 됐다"가 똑같아 보인다.
            m = re.search(r"camera=(\d+)", self.path)
            cam = int(m.group(1)) if m else 0
            with LIVE_LOCK:
                meta = dict(LAST_META.get(cam) or {})
            obs = meta.get("obstacles") or []
            moved = sorted(o.get("moved_px", 0) for o in obs)
            qual = sorted(o.get("quality", 1.0) for o in obs)
            def med(a):
                return round(a[len(a) // 2], 2) if a else None
            self._json(200, {
                "camera": cam, "frame_id": meta.get("frame_id"),
                "frame_seq": meta.get("frame_seq"),
                "flow": meta.get("flow"),
                "n_obstacles": len(obs),
                "n_labeled": sum(1 for o in obs if o.get("label")),
                "n_ranged": sum(1 for o in obs if o.get("range")),
                "moved_px": {"median": med(moved), "max": moved[-1] if moved else None},
                "quality": {"median": med(qual), "min": qual[0] if qual else None},
                "obstacles": obs[:40]})
            return
        # 세 갈래. **이 셋만 있다** — 기준선(/raw)과 비교(/compare)는 걷어냈다.
        # 파이프라인이 pull + depth로 바뀌어 "아무것도 얹지 않은 YOLOE"가 더 이상
        # 지금 구성의 대조군이 아니다(사용자 2026-09-21).
        if self.path.startswith("/panels"):
            m = re.search(r"camera=(\d+)", self.path)
            self._composed(int(m.group(1)) if m else 0)
            return
        if self.path.startswith("/original"):
            m = re.search(r"camera=(\d+)", self.path)
            self._stream(int(m.group(1)) if m else 0, store=ORIG)
            return
        if self.path.startswith("/missed"):
            m = re.search(r"camera=(\d+)", self.path)
            self._stream(int(m.group(1)) if m else 0, store=MISSED)
            return
        if self.path.startswith("/depth"):
            m = re.search(r"camera=(\d+)", self.path)
            self._stream(int(m.group(1)) if m else 0, store=DEPTH)
            return
        if self.path == "/api/live":
            with LIVE_LOCK:
                self._json(200, {str(k): {**v["info"], "age_s": round(time.time() - v["at"], 2)}
                                 for k, v in LIVE.items()})
            return
        # **/api/verdicts는 없다.** 말단이 커서로 회수하던 창구였는데, 지금은
        # 끝나는 시각을 아는 이쪽이 POST /api/verdict로 민다(Puller.push_verdict).
        if self.path == "/api/health":
            self._json(200, {
                "ok": True,
                "ovd": STATE.get("ovd") is not None,
                "clip": STATE.get("clip") is not None,
                "llm": STATE.get("llm") is not None,
                "vlm": STATE.get("vlm") is not None,
                "ovd_error": STATE.get("ovd_error"), "clip_error": STATE.get("clip_error"),
                "llm_error": STATE.get("llm_error"), "vlm_error": STATE.get("vlm_error"),
                "ovd_vocabulary_size": len(STATE["ovd"].vocab) if STATE.get("ovd") else 0,
                "ovd_mode": getattr(STATE.get("ovd"), "mode", None),
                "ovd2": STATE.get("ovd2") is not None,
                "ovd2_error": STATE.get("ovd2_error"),
                "ovd2_vocabulary": list(getattr(STATE.get("ovd2"), "vocab", []) or []),
                "active_domain": STATE.get("active_domain"),
                "domains": list(cfg["clip"].get("domains", {})),
                "dictionary": STATE["dict"].classes() if STATE.get("dict") else [],
                "background_classes": sorted(STATE.get("background_classes", [])),
                "pending_async": STATE["async_q"].qsize(),
                "async_dropped": STATE.get("async_dropped", 0),
                # depth는 "올라갔나"와 "쓸 수 있나"가 다르다 — LoRA 키가 안 붙어도
                # 객체는 생긴다. 둘을 따로 낸다.
                "depth": bool(STATE.get("depth") and STATE["depth"].available()),
                "depth_error": getattr(STATE.get("depth"), "error", None),
                "depth_unmatched_keys": getattr(STATE.get("depth"), "unmatched", None),
                # 당기는 쪽 상태 — 말단이 안 뜨면 여기서 먼저 보인다.
                "terminals": [{"name": pl.name, "base": pl.base,
                               "pulled": pl.pulled, "failed": pl.failed,
                               "from_spool": pl.spooled_in, "pushed": pl.pushed,
                               "last_error": pl.last_error}
                              for pl in STATE.get("pullers", [])],
                # 상한이 실제로 무엇을 막고 있는지. 개수만 보면 5.9GB를 놓친다.
                "store": STATE["rotation"].usage() if STATE.get("rotation") else {},
                "uptime_s": round(time.time() - STATE.get("started_at", time.time())),
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        # **/api/observe는 없다.** 말단이 밀던 시절의 창구였는데 지금은 이쪽이
        # 당긴다(Puller). 남겨 두면 두 경로가 공존해 어느 쪽으로 들어왔는지에 따라
        # backpressure가 달라진다 — 그래서 지웠다.
        if self.path == "/api/command":
            return self._command(STATE["cfg"])
        return self._json(404, {"error": "unknown path"})

    def _command(self, cfg: dict):
        """가시화에서 오는 명령. 지금은 '이 클래스를 찾아라' 하나다.

        **LLM은 여기서만 돈다**(사용자 2026-09-21) — 프레임 처리 경로에는 없다.
        생성된 특징은 사전에 올라가 다음 프레임부터 CLIP 앵커로 쓰인다.
        """
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as exc:
            self._json(400, {"error": repr(exc)[:200]})
            return
        cmd = body.get("command")

        # ── 말단 제어 ─────────────────────────────────────────────────────────
        # 가져오기·추론·보내기를 **따로** 켜고 끈다. 하나로 묶으면 "화면만 보고
        # 싶다"나 "말단 레코드를 얼려 두고 엣지에서만 돌려 보고 싶다" 같은 경우에
        # 전부 멈추는 수밖에 없다. 의존은 자연스럽게 걸리므로(안 당기면 추론할 게
        # 없다) 끌 때는 역순이 안전하다: 보내기 → 추론 → 당기기.
        TERMINAL_CMDS = {
            "fetch_image":     None,                      # 한 장만 가져오기
            "start_inference": ("inference", True),
            "stop_inference":  ("inference", False),
            "start_push":      ("push", True),
            "stop_push":       ("push", False),
            "start_pull":      ("pull", True),
            "stop_pull":       ("pull", False),
        }
        if cmd in TERMINAL_CMDS or cmd == "terminal_status":
            name = body.get("terminal")
            pl = find_puller(name)
            if pl is None:
                pls = [x.name for x in STATE.get("pullers", [])]
                return self._json(404, {
                    "error": ("ambiguous terminal — specify one"
                              if pls else "no terminal configured"),
                    "available": pls})
            if cmd == "terminal_status":
                return self._json(200, pl.status())
            if cmd == "fetch_image":
                # pull_on과 무관하게 한 장. 추론 게이트가 꺼져 있으면 가져오기만 한다.
                res = pl.fetch(float(body.get("timeout_s", 15.0)))
                return self._json(200 if res.get("ok") else 503,
                                  {"command": cmd, **res, "gates": pl.status()["gates"]})
            gate, on = TERMINAL_CMDS[cmd]
            return self._json(200, {"command": cmd, **pl.set_gate(gate, on)})

        if cmd == "describe_scene":
            self._json(200, describe_scene(cfg))
            return
        if cmd == "detect_domain":
            self._json(200, detect_domain(cfg, bool(body.get("switch", True))))
            return
        if cmd == "set_domain":
            try:
                self._json(200, apply_domain(cfg, str(body.get("domain", ""))))
            except KeyError as exc:
                self._json(400, {"error": str(exc)})
            return
        if cmd != "find_class":
            self._json(400, {"error": "unknown command",
                             "supported": ["find_class", "describe_scene",
                                           "detect_domain", "set_domain"]})
            return
        if STATE.get("llm") is None:
            self._json(503, {"error": f"llm_unavailable:{STATE.get('llm_error','?')}"})
            return
        query = str(body.get("query", "")).strip()
        if not query:
            self._json(400, {"error": "query 없음"})
            return
        key = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_") or "query_class"
        fd = STATE["dict"]
        if key in fd.entries:
            self._json(200, {"class": key, "already_known": True,
                             "features": fd.entries[key].get("features", [])})
            return
        try:
            gen = STATE["llm"].generate(query, key)
            ok, why = validate_entry(gen["entry"])
            if ok:
                fd.add(key, gen["entry"], provenance={
                    "source": "llm", "model": gen["model"], "query": query,
                    "at": time.time(), "trigger": "visualization_command"})
                STATE["clip"].prompts = []   # 사전이 바뀌었으니 텍스트 임베딩 재생성
            self._json(200, {"class": key, "accepted": ok, "reason": why,
                             "features": (gen["entry"] or {}).get("features"),
                             "rationale": (gen["entry"] or {}).get("rationale"),
                             "latency_ms": gen["latency_ms"], "model": gen["model"],
                             "verified": False})
        except Exception as exc:
            self._json(500, {"error": repr(exc)[:300]})




# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def build_state(cfg: dict) -> None:
    """무거운 것들을 올린다. **하나가 실패해도 나머지로 간다**(AI-C-05).

    없는 단계는 실패가 아니라 축소다. 무엇이 빠졌는지는 /api/health에 드러난다 —
    조용히 빠지면 며칠을 모르고 돈다(2026-09-21에 실제로 그랬다: transformers 버전
    차이로 stage2가 매 프레임 죽었는데 비동기 안이라 stage1은 멀쩡해 보였다).
    """
    STATE["cfg"] = cfg
    STATE["async_q"] = queue.Queue(
        maxsize=int(cfg.get("server", {}).get("async_queue", 4)))
    STATE["started_at"] = time.time()

    if cfg.get("ovd", {}).get("enabled", True):
        try:
            STATE["ovd"] = YoloeOvd(cfg)
            print(f"OVD 로드됨 — YOLOE prompt-free, 내장 어휘 {len(STATE['ovd'].vocab)}종")
        except Exception as exc:
            STATE["ovd"], STATE["ovd_error"] = None, repr(exc)[:200]
            print(f"OVD 없음 → 해당 단계만 비활성: {STATE['ovd_error']}")

    st_cfg = cfg.get("store", {})
    STATE["crop_gate"] = CropGate(st_cfg)
    STATE["rotation"] = Rotation(resolve(cfg, st_cfg.get("root", "store")), st_cfg)
    rot, g = STATE["rotation"], STATE["crop_gate"]
    print(f"보존 정책 — crop 게이트 IoU {g.match_iou:.2f} 슬롯당 {g.max_per_slot}장/"
          f"{g.min_interval_s:.0f}s, 프레임당 최대 {g.max_per_frame}장")
    print(f"  디렉터리 상한 개수 {rot.file_caps}")
    print(f"  디렉터리 상한 MB   {({k: v // 1048576 for k, v in rot.byte_caps.items()})}"
          "   ← 개수만으로는 5.9GB를 못 막았다(store.py 첫머리)")

    if cfg.get("ovd2", {}).get("enabled", False):
        try:
            STATE["ovd2"] = WorldOvd(cfg)
            n = len(STATE["ovd2"].vocab)
            print(f"OVD2 로드됨 — YOLO-World 텍스트 어휘 {n}종"
                  if n else "OVD2 어휘 없음 → 합의 비활성(단일 모델로 축소)")
        except Exception as exc:
            STATE["ovd2"], STATE["ovd2_error"] = None, repr(exc)[:200]
            print(f"OVD2 없음 → 합의 비활성: {STATE['ovd2_error']}")

    if cfg.get("clip", {}).get("enabled", True):
        try:
            backend = cfg["clip"].get("backend", "torch")
            STATE["clip"] = (TorchClipScorer(cfg) if backend == "torch"
                             else ClipScorer(cfg))
            print(f"CLIP 로드됨 — EP {STATE['clip'].providers}")
        except Exception as exc:
            STATE["clip"], STATE["clip_error"] = None, repr(exc)[:200]
            print(f"CLIP 없음 → 해당 단계만 비활성: {STATE['clip_error']}")

    # Depth — MoGe2-Aerial. 적재 실패와 "적재됐지만 못 쓴다"를 구분해서 알린다.
    if cfg.get("depth", {}).get("enabled", False):
        # provider를 고른다. zipdepth는 metric이 아니라 깊이 맵만 내는 임시
        # 경로다 — 거리는 클래스 크기 사전이 맡는다(models.ZipDepth 참고).
        prov = cfg["depth"].get("provider", "moge2_aerial")
        d = (ZipDepth(cfg) if prov == "zipdepth" else AerialDepth(cfg))
        STATE["depth"] = d
        if d.available():
            kind = "metric(m)" if getattr(d, "metric", True) else "상대값(미터 아님)"
            print(f"Depth 로드됨 — {prov} [{kind}], device {d.device}, fp16 {d.fp16}"
                  + (f", 미매칭 키 {d.unmatched}개" if getattr(d, "unmatched", 0) else ""))
            if getattr(d, "unmatched", 0):
                print("  ⚠ LoRA 키가 일부 안 붙었다 — MoGe 포크가 맞는지 확인한다"
                      "(fetch_models.sh가 AerialMetric의 MoGe/를 쓴다)")
        else:
            print(f"Depth 없음 → 거리 산출만 비활성: {d.error}")
    else:
        STATE["depth"] = None

    # 엣지 RPN — 미검출 분석 전용이다. 시점 정렬에는 쓰지 않는다(그건 말단의
    # KLT 플로우 몫). 없으면 하단 패널만 빠진다(AI-C-05).
    if cfg.get("unknown", {}).get("enabled", True):
        try:
            STATE["edge_rpn"] = EdgeRpn(cfg)
            print(f"엣지 RPN 로드됨 — {Path(cfg['edge_rpn']['model']).name} "
                  f"(미검출 분석 전용, 비동기)")
        except Exception as exc:
            STATE["edge_rpn"], STATE["edge_rpn_error"] = None, repr(exc)[:200]
            print(f"엣지 RPN 없음 → 미검출 패널만 비활성: {STATE['edge_rpn_error']}")

    # 확정 기준을 **켜진 OVD 수**에 맞춘다. "두 소스 그룹의 지지"는 모델이 둘일
    # 때의 조건이고, 하나만 켜면 어떤 객체도 확정되지 않아 장애물이 0개가 된다
    # (2026-09-21에 실제로 겪었다). config에 명시했으면 그것이 이긴다.
    n_ovd = sum(1 for k in ("ovd", "ovd2") if STATE.get(k) is not None)
    ot_cfg = cfg.setdefault("object_tracks", {})
    if "confirm_groups" not in ot_cfg:
        ot_cfg["confirm_groups"] = max(1, n_ovd)
    print(f"확정 기준: 소스 그룹 {ot_cfg['confirm_groups']}개 이상 "
          f"(가동 OVD {n_ovd}종)")

    # 특징 사전은 **CLIP 앵커 전용**이다. CLIP이 꺼져 있으면 올릴 이유가 없다 —
    # 디스크를 읽고 임베딩 자리를 잡아 두지만 아무도 쓰지 않는다.
    if cfg.get("clip", {}).get("enabled", True):
        info = apply_domain(cfg, cfg["clip"].get("active_domain", "generic"))
        print(f"도메인 '{info['domain']}' — 사전 {len(info['classes'])}개 "
              f"(배경 {len(info['background_classes'])}개) ← {info['dictionary']}")
    else:
        STATE["active_domain"] = cfg.get("clip", {}).get("active_domain", "generic")
        print("특징 사전 건너뜀 — CLIP이 꺼져 있다(사전은 CLIP 앵커 전용)")

    if cfg.get("llm", {}).get("enabled"):
        try:
            o = Ollama(cfg["llm"]["host"], float(cfg["llm"].get("timeout_s", 120)))
            want = cfg["llm"]["model"]
            if want not in o.available_models():
                raise RuntimeError(f"ollama에 {want} 없음")
            STATE["llm"] = LlmFeatureGenerator(o, want)
            print(f"LLM 로드됨 — {want} (가시화 명령 시에만 실행)")
        except Exception as exc:
            STATE["llm"], STATE["llm_error"] = None, repr(exc)[:200]
            print(f"LLM 없음 → 사전 즉석생성만 비활성: {STATE['llm_error']}")

    if cfg.get("vlm", {}).get("enabled"):
        try:
            want = (cfg["vlm"].get("model") or "").strip()
            if not want:
                raise RuntimeError("vlm.model 미지정 — 예: ollama pull qwen2.5vl:3b")
            o = Ollama(cfg["vlm"]["host"], float(cfg["vlm"].get("timeout_s", 120)))
            if want not in o.available_models():
                raise RuntimeError(f"ollama에 {want} 없음")
            # 사전 도메인을 따른다 — VLM만 다른 도메인을 보고 있으면 안 된다.
            dom = STATE.get("active_domain") or cfg["vlm"].get("active_domain", "generic")
            cfg["vlm"]["active_domain"] = dom
            STATE["vlm"] = VlmDescriber(
                o, want, scene_prompt=cfg["vlm"].get("scene_prompts", {}).get(dom))
            print(f"VLM 로드됨 — {want}, 장면 서술 도메인 {dom}")
        except Exception as exc:
            STATE["vlm"], STATE["vlm_error"] = None, repr(exc)[:200]
            print(f"VLM 없음 → 환경 서술만 비활성: {STATE['vlm_error']}")


CTL_COMMANDS = {
    "fetch":      "fetch_image",       # 말단에서 이미지 한 장 가져오기
    "infer-on":   "start_inference",   # 추론 시작
    "infer-off":  "stop_inference",    # 추론 종료
    "push-on":    "start_push",        # 추론 결과 말단으로 보내기 시작
    "push-off":   "stop_push",         # 보내기 정지
    "pull-on":    "start_pull",        # 연속 당기기 시작
    "pull-off":   "stop_pull",         # 연속 당기기 정지
    "status":     "terminal_status",
}


def ctl(args) -> int:
    """돌고 있는 엣지에 명령을 보낸다. 별도 파일을 두지 않은 이유는 이게 서버의
    `/api/command`를 그대로 치는 얇은 껍데기이기 때문이다 — 두 벌이 되면 한쪽만
    고치게 된다.

        python3 server.py ctl fetch
        python3 server.py ctl infer-on
        python3 server.py ctl push-off --terminal <말단이름>
    """
    cfg = load_config(args.config)
    scfg = cfg.get("server", {})
    host = args.host or scfg.get("host", "127.0.0.1")
    # 0.0.0.0은 바인드 주소지 접속 주소가 아니다 — 명령은 루프백으로 보낸다.
    host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    port = int(args.port or scfg.get("port", 8891))
    url = f"http://{host}:{port}/api/command"
    payload = {"command": CTL_COMMANDS[args.action]}
    if args.terminal:
        payload["terminal"] = args.terminal
    try:
        r = requests.post(url, json=payload, timeout=(3.0, 30.0))
    except Exception as exc:
        print(f"엣지에 연결할 수 없다 ({url}): {exc!r}")
        print("서버가 떠 있는지 확인한다:  python3 server.py")
        return 1
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))
    return 0 if r.ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="mode")
    c = sub.add_parser("ctl", help="돌고 있는 엣지에 명령을 보낸다")
    c.add_argument("action", choices=sorted(CTL_COMMANDS))
    c.add_argument("--terminal", default=None, help="말단 이름(여럿일 때)")
    c.add_argument("--config", default=None)
    # ctl도 포트를 받아야 한다 — 엣지를 기본이 아닌 포트로 띄웠으면 명령도
    # 거기로 가야 하는데, 안 받으면 config만 보고 엉뚱한 곳을 친다.
    c.add_argument("--port", type=int, default=None)
    c.add_argument("--host", default=None)
    ap.add_argument("--host", default=None,
                    help="바인드 주소(기본: config server.host = 127.0.0.1)")
    ap.add_argument("--port", type=int, default=None,
                    help="스트리밍/명령 포트(기본: config server.port = 8891)")
    ap.add_argument("--no-pull", action="store_true",
                    help="말단을 당기지 않고 스트리밍·운영 창구만 연다")
    args = ap.parse_args()

    if args.mode == "ctl":
        raise SystemExit(ctl(args))

    cfg = load_config(args.config)
    build_state(cfg)

    for _ in range(int(cfg.get("server", {}).get("workers", 1))):
        threading.Thread(target=worker, args=(cfg,), daemon=True).start()

    pullers: list[Puller] = []
    STATE["pullers"] = pullers
    if not args.no_pull:
        for node in cfg.get("terminals", []):
            if not node.get("enabled", True):
                continue
            pl = Puller(cfg, node)
            pullers.append(pl)
            threading.Thread(target=pl.run, daemon=True).start()
            print(f"말단 당기기: {pl.name} → {pl.base} cam{pl.cams}")
        if not pullers:
            print("⚠ config의 terminals가 비었다 — 당길 말단이 없다")

    scfg = cfg.get("server", {})
    # 인자가 config를 이긴다. 같은 코드로 두 엣지를 동시에 띄워 비교할 때
    # config를 복사하지 않아도 되게 하려는 것이다.
    host = args.host or scfg.get("host", "127.0.0.1")
    port = int(args.port or scfg.get("port", 8891))
    # 당기는 쪽/화면이 같은 포트를 쓰므로 STATE에도 남긴다.
    scfg["port"], scfg["host"] = port, host
    shown = host if host not in ("0.0.0.0", "") else "127.0.0.1"
    print(f"\n저장소 {resolve(cfg, cfg['store']['root'])}")
    print(f"**실시간 화면: http://{shown}:{port}/**")
    on = [n for n in ("ovd", "ovd2", "clip", "depth", "llm", "vlm")
          if STATE.get(n) is not None
          and (not hasattr(STATE[n], "available") or STATE[n].available())]
    dmode = cfg.get("depth", {}).get("mode", "sync")
    print(f"  파이프라인: 말단 KLT 플로우 정렬 → YOLOE + depth"
          f"({'동기, 한 판정에 함께' if dmode == 'sync' else '비동기'}) → 말단 반영")
    print(f"  가동 단계: {', '.join(on) or '없음 — fetch_models.sh를 먼저 돌린다'}")
    print("  제어: python3 server.py ctl "
          "{fetch|infer-on|infer-off|push-on|push-off|pull-on|pull-off|status}"
          f" --port {port}")
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    # server_close()가 요청 스레드를 join하지 않게 한다. daemon_threads만으로는
    # 부족하다 — join은 인터프리터 종료 **전에** 걸리므로 여기서 꺼야 한다.
    # SHUTDOWN 신호로 스트림 루프를 이미 깨우고 있으니 기다릴 이유도 없다.
    httpd.block_on_close = False

    def _bye(signum, _frame):
        # SIGTERM도 같이 받는다 — `docker stop`이 보내는 것이 이쪽이고,
        # 안 받으면 10초 뒤 SIGKILL로 죽으면서 정리 절차가 통째로 건너뛰어진다.
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _bye)
        except (ValueError, OSError):
            pass                       # 메인 스레드가 아니면 무시(테스트 등)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n중지 요청 — 정리 중", flush=True)
    finally:
        # 순서가 중요하다. 신호를 먼저 올려야 스트리밍 스레드가 빠져나오고,
        # 그 다음에야 소켓을 닫아도 매달리지 않는다.
        SHUTDOWN.set()
        httpd.shutdown()               # serve_forever 루프 정지(수락 중단)
        for pl in pullers:
            pl.stop()
        STATE["async_q"].put(None)
        httpd.server_close()           # 리슨 소켓 반환 — 여기서 포트가 풀린다
        print(f"포트 {port} 반환 완료", flush=True)


if __name__ == "__main__":
    main()
