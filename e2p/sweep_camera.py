#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_camera.py — 지금 360 카메라 화면 한 장으로 u 값 24개를 한눈에 본다 (슬라이스 각도 미리 고르기)
================================================================================
2026-11-11 · 바탕화면 바로가기 「e2p sweep」

무엇을 하나
  360 카메라에서 **지금** 파노라마 한 장을 가져와, 데이터 스트림의 slice360_live.py 와 같은 식으로
  u = 0, 15, 30, … 345 (24장) 를 잘라, 원본 1장 + 슬라이스 24장(칸 아래에 u 값)을 **이미지 한 장**으로 만들고
  윈도우 사진 앱으로 연다. 그 그림을 보고 데이터 스트림 실행기의 슬라이스 u 칸에 적는다.
    자르기   slice360_common.slice_pano — slice360_live.py 와 같은 함수 · 기본 v -30 · 시야 90 · 1:1
             (v -30 은 바탕화면 데이터 스트림 실행기의 고정값과 같다)
    u        0 = 파노라마 가운데, + 오른쪽. 원본 위 눈금도 같은 숫자다

한 장을 가져오는 곳 (--from, 기본 auto) — 360 카메라가 살아 있을 때만 된다
  1) camera  카메라(--ip)에 직접 연결 → 프리뷰(인카메라 스티칭 RTMP) → 한 장 → 프리뷰 끄고 연결 해제.
             카메라 제어 함수는 360camera_work/stream_to_server.py 의 것을 그대로 쓴다.
  2) stream  「360 카메라 실행」(stream_to_server.py)이 이미 카메라를 쥐고 있으면 카메라가 새 연결을 받지 않는다
             ("already connected by another"). 그때는 데이터 스트림 Redis camera_stream:cam360 의 최신 장을 쓴다
             — 단, --max-age 초(기본 10) 안에 들어온 장만. 같은 카메라의 지금 화면이다.
  auto = 카메라가 응답하면 1), 세션이 막혀 있거나 카메라가 안 보이면 2). 둘 다 안 되면 실패로 끝낸다
  (저장된 옛 파노라마로 대신하지 않는다 — 지금 화면이 아니면 각도를 잘못 고를 수 있어서).

저장
  e2p/e2p_sweep_out/<YYMMDD_HHMMSS>_camera/   (.gitignore 대상)
    sweep.jpg   원본 + 24장 한눈 그림  ← 바로가기를 누르면 열리는 그림
    pano.jpg    가져온 원본 파노라마 그대로

사용 (저장소 루트, venv)
  python e2p/sweep_camera.py                     카메라에서 한 장 → 그림 열기
  python e2p/sweep_camera.py --from stream       데이터 스트림의 최신 장으로만
  python e2p/sweep_camera.py --v -15             세로 각도 바꾸기
  python e2p/sweep_camera.py --pano <파일>        이 파노라마 파일로 (시험용)
  python e2p/sweep_camera.py --no-open           그림을 열지 않는다
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from slice360_common import is_pano, iso_ms, pano_xy, slice_pano  # noqa: E402

GRAB_FRAMES = 8          # 스트림을 연 뒤 이만큼 읽고 마지막 장을 쓴다 (첫 장들은 키프레임 전이라 깨질 수 있다)
GRAB_SECONDS = 6.0       # 그 안에 다 못 읽으면 읽은 것 중 마지막 장


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# ---------------------------------------------------------------- 1) 카메라에서 직접
def grab_from_camera(ip, width, height, bitrate):
    """(BGR 사진, 설명, 그림용 영문 설명) 또는 예외. 세션이 막혀 있으면 BusyError, 안 보이면 Unreachable."""
    import requests
    sys.path.insert(0, str(REPO / "360camera_work"))
    import stream_to_server as cam                      # 카메라 제어 함수 (연결 · 프리뷰 · RTMP 열기)

    cam.CMD_URL = "http://%s:20000/osc/commands/execute" % ip
    cam.STATE_URL = "http://%s:20000/osc/state" % ip
    try:                                                # 카메라가 보이나 (3초) — 안 보이면 연결 재시도로 30초씩 쓰지 않는다
        requests.post(cam.STATE_URL, json={}, timeout=3)
    except Exception as e:
        raise Unreachable("카메라 %s 가 응답하지 않음 (%s)" % (ip, type(e).__name__))

    body = requests.post(cam.CMD_URL, json={"name": "camera._connect", "parameters": {}}, timeout=10).json()
    fp = (body.get("results") or {}).get("Fingerprint")
    if not fp:
        desc = (body.get("error") or {}).get("description")
        if desc == "already connected by another":
            raise BusyError("카메라 세션을 다른 프로그램(360 카메라 실행 등)이 쥐고 있음")
        raise RuntimeError("카메라 연결 실패: %s" % desc)
    cam.fingerprint = fp
    log("카메라 연결됨 (%s)" % ip)
    cam.hb_stop.clear()
    threading.Thread(target=cam.heartbeat, daemon=True).start()
    cap = None
    try:
        url = cam.start_preview(SimpleNamespace(ip=ip, width=width, height=height, bitrate=bitrate))
        if not url:
            raise RuntimeError("카메라 프리뷰가 켜지지 않음")
        t0 = time.time()
        cap = cam.open_capture(url)
        if cap is None:
            raise RuntimeError("프리뷰 스트림을 못 엶: %s" % url)
        log("스트림 열림 %.1f초 — %d장 읽고 마지막 장을 씀" % (time.time() - t0, GRAB_FRAMES))
        frame, n, t1 = None, 0, time.time()
        while n < GRAB_FRAMES and time.time() - t1 < GRAB_SECONDS:
            ok, f = cap.read()
            if ok and f is not None:
                frame, n = f, n + 1
        if frame is None:
            raise RuntimeError("스트림에서 사진을 못 읽음")
        return frame, "카메라 %s 직접 (%s)" % (ip, iso_ms()), "camera %s (direct)" % ip
    finally:
        if cap is not None:
            cap.release()
        cam.send("camera._stopPreview", {}, timeout=5)
        cam.send("camera._disconnect", {}, timeout=5)    # 다음에 「360 카메라 실행」이 바로 연결하도록 (없는 명령이면 무시됨)
        cam.hb_stop.set()
        log("프리뷰 끄고 연결 해제")


class BusyError(RuntimeError):
    pass


class Unreachable(RuntimeError):
    pass


# ---------------------------------------------------------------- 2) 데이터 스트림 최신 장
def grab_from_stream(redis_url, source, max_age):
    import json
    import redis
    r = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=5)
    key = "camera_stream:" + source
    e = r.xrevrange(key, "+", "-", count=1)
    if not e:
        raise RuntimeError("%s 가 비어 있음" % key)
    eid, f = e[0]
    age = time.time() - int(eid.split(b"-")[0]) / 1000.0
    if age > max_age:
        raise RuntimeError("%s 최신 장이 %.0f초 전 것 — 360 이 지금 들어오고 있지 않음 (%g초 안만 씀)"
                           % (key, age, max_age))
    img = cv2.imdecode(np.frombuffer(f.get(b"image") or b"", np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("%s 최신 장을 못 풂" % key)
    try:
        h = json.loads(f.get(b"header") or b"{}")
    except ValueError:
        h = {}
    return img, "데이터 스트림 %s n %s (%.1f초 전)" % (key, h.get("n"), age), \
           "data stream %s n %s (%.1fs ago)" % (key, h.get("n"), age)


# ---------------------------------------------------------------- 한눈 그림
def text(img, s, org, scale, color, th=2, center=False):
    (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = org
    if center:
        x -= tw // 2
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def build_sheet(equ, us, slices, v, fov, tag, cols=6, tile=320, gap=8, band=44):
    """원본(u 눈금) 위 + 슬라이스 칸(아래 띠에 u 값) 아래 → 한 장"""
    rows = (len(slices) + cols - 1) // cols
    W = cols * tile + (cols + 1) * gap
    # 원본
    ph = int(round(equ.shape[0] * (W - 2 * gap) / float(equ.shape[1])))
    pano = cv2.resize(equ, (W - 2 * gap, ph), interpolation=cv2.INTER_AREA)
    pw = pano.shape[1]
    _, vy = pano_xy(0, v, pw, ph)
    cv2.line(pano, (0, int(vy)), (pw - 1, int(vy)), (0, 200, 255), 1, cv2.LINE_AA)
    text(pano, "v%+g" % v, (6, max(18, int(vy) - 6)), 0.55, (0, 200, 255), 1)
    for u in us:
        x, _ = pano_xy(u, 0, pw, ph)
        col = (0, 255, 0) if u % 90 == 0 else (255, 255, 255)
        cv2.line(pano, (int(x), 0), (int(x), 14), col, 2)
        cv2.line(pano, (int(x), ph - 15), (int(x), ph - 1), col, 2)
        text(pano, "%d" % u, (min(max(int(x), 18), pw - 18), 34), 0.55, col, 1, center=True)

    head = 56
    H = head + ph + gap + 30 + rows * (tile + band + gap) + gap
    sheet = np.full((H, W, 3), 32, np.uint8)
    text(sheet, "360 sweep  v %+g  fov %g  (u: 0 = center, + = right)" % (v, fov), (gap, 26), 0.75, (0, 255, 255), 2)
    text(sheet, "%s   %s" % (iso_ms()[:19].replace("T", " "), tag),        # cv2 는 한글을 못 그린다 — 영문
         (gap, 48), 0.5, (200, 200, 200), 1)
    sheet[head:head + ph, gap:gap + pw] = pano
    y0 = head + ph + gap + 30
    text(sheet, "slices (u)", (gap, y0 - 8), 0.6, (0, 255, 255), 1)
    for i, (u, sl) in enumerate(zip(us, slices)):
        r, c = divmod(i, cols)
        x = gap + c * (tile + gap)
        y = y0 + r * (tile + band + gap)
        sheet[y:y + tile, x:x + tile] = cv2.resize(sl, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.rectangle(sheet, (x, y + tile), (x + tile - 1, y + tile + band - 1), (16, 16, 16), -1)
        col = (0, 255, 0) if u % 90 == 0 else (255, 255, 255)
        text(sheet, "u = %d" % u, (x + tile // 2, y + tile + 31), 0.9, col, 2, center=True)
    return sheet


def windows_path(path):
    try:
        r = subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def open_image(path):
    """윈도우 기본 사진 앱으로 연다 (WSL). 리눅스만이면 xdg-open."""
    wp = windows_path(path)
    try:
        if wp:
            # start 가 사진 앱을 띄우고 돌아올 때까지 기다린다 — 바로가기 창이 먼저 닫혀 cmd 가 같이 죽지 않게
            subprocess.run(["cmd.exe", "/c", "start", "", wp], cwd="/mnt/c", timeout=15,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.SubprocessError) as e:
        log("그림을 못 열었습니다 (%s) — 직접 여세요: %s" % (e, wp or path))
        return False


# ---------------------------------------------------------------- 실행
def main():
    ap = argparse.ArgumentParser(description="지금 360 카메라 한 장 → u 24개 한눈 그림",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--from", dest="src", choices=["auto", "camera", "stream"], default="auto",
                    help="한 장을 가져올 곳 (기본 auto: 카메라 → 안 되면 데이터 스트림)")
    ap.add_argument("--pano", default=None, help="이 파노라마 파일로 (시험용, --from 무시)")
    ap.add_argument("--ip", default="192.168.100.124", help="360 카메라 주소 (기본 %(default)s)")
    ap.add_argument("--redis", default="redis://127.0.0.1:6380/0", help="데이터 스트림 Redis (기본 %(default)s)")
    ap.add_argument("--source", default="cam360", help="데이터 스트림 파노라마 소스 (기본 cam360)")
    ap.add_argument("--max-age", type=float, default=10.0, help="데이터 스트림 장은 이 초 안의 것만 (기본 10)")
    ap.add_argument("--v", type=float, default=-30.0, help="세로 각도 (기본 -30 — 실행기 고정값과 같음)")
    ap.add_argument("--fov", type=float, default=90.0, help="가로 시야 (기본 90)")
    ap.add_argument("--step", type=float, default=15.0, help="u 간격 (기본 15 → 24장)")
    ap.add_argument("--out", default=str(HERE / "e2p_sweep_out"), help="저장 폴더 (기본 e2p/e2p_sweep_out)")
    ap.add_argument("--no-open", action="store_true", help="그림을 열지 않는다")
    args = ap.parse_args()
    if args.step <= 0 or abs(360.0 / args.step - round(360.0 / args.step)) > 1e-6:
        sys.exit("--step 은 360 을 나누어떨어지게 하는 양수여야 합니다 (예: 15 · 30)")

    t0 = time.perf_counter()
    equ, where, tag = None, None, None
    if args.pano:
        equ = cv2.imread(args.pano, cv2.IMREAD_COLOR)
        if equ is None:
            sys.exit("파일을 못 읽었습니다: %s" % args.pano)
        where, tag = "파일 %s" % args.pano, "file %s" % Path(args.pano).name
    else:
        errs = []
        if args.src in ("auto", "camera"):
            try:
                equ, where, tag = grab_from_camera(args.ip, 3840, 1920, 10240)
            except (BusyError, Unreachable) as e:
                errs.append("카메라: %s" % e)
                log(str(e) + ("  → 데이터 스트림의 최신 장을 봅니다" if args.src == "auto" else ""))
            except Exception as e:
                errs.append("카메라: %s" % e)
                log("카메라에서 못 가져옴: %s" % e)
        if equ is None and args.src in ("auto", "stream"):
            try:
                equ, where, tag = grab_from_stream(args.redis, args.source, args.max_age)
            except Exception as e:
                errs.append("데이터 스트림: %s" % e)
        if equ is None:
            print("\n360 카메라 화면을 못 가져왔습니다.")
            for e in errs:
                print("  - " + e)
            print("확인: 카메라 전원 · 랜선 · 이 노트북에서 %s 가 보이는지 (360 카메라 실행 바로가기가 되면 된다)" % args.ip)
            return 1
    log("가져옴: %s · %dx%d" % (where, equ.shape[1], equ.shape[0]))
    if not is_pano(equ.shape):
        log("⚠ 2:1 이 아님 — 360 파노라마가 아닐 수 있음")

    n_u = int(round(360.0 / args.step))
    us = [int(round(i * args.step)) for i in range(n_u)]
    slices = [slice_pano(equ, u, args.v, args.fov, 640, "1:1")[0] for u in us]
    sheet = build_sheet(equ, us, slices, args.v, args.fov, tag)

    out = Path(args.out).expanduser() / (time.strftime("%y%m%d_%H%M%S") + "_camera")
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "pano.jpg"), equ, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(out / "sweep.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    log("만듦 %.1f초 → %s" % (time.perf_counter() - t0, out / "sweep.jpg"))
    print("고른 u 를 데이터 스트림 실행기의 슬라이스 u 칸에 적는다 (v %+g)." % args.v)
    if not args.no_open:
        open_image(out / "sweep.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
