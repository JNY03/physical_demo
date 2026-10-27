# -*- coding: utf-8 -*-
"""
e2p_sweep.py — 360 파노라마 한 장을 가로 방향으로 돌아가며 90도 사진으로 잘라 본다
================================================================================
2026-10-18

무엇을 하나
  파노라마(등장방형, 가로:세로 = 2:1) 한 장의 경로를 받아
  u = 0, 15, 30, ... 345 (기본 15도 간격, 24장) 방향으로 90도 원근 사진을 자른다.
  360도는 0도와 같은 방향이라 만들지 않는다 → 24장.
  자르는 방식은 e2p.py 와 같다: py360convert.e2p, 시야 90x90, 720x720.

각도
  u  가로 방향. 0 = 파노라마 사진의 가운데, + 는 오른쪽 (위에서 보면 시계 방향)
  v  세로 방향. 0 = 수평, + 는 위, - 는 아래 (기본 0 — 수평 각도부터 찾는다)

출력
  <저장 루트>/<실행 시각 YYMMDD_HHMMSS>/
      u+000_v+0.jpg, u+015_v+0.jpg, ... u+345_v+0.jpg
      _info.txt        원본 경로 · 크기 · 각도 · 실행 시각
  저장 루트 기본값 = 이 파일이 있는 폴더 아래 e2p_sweep_out/  (--out 으로 바꾼다)

사용
  python3 e2p_sweep.py                          경로를 물어본다 → 붙여 넣고 Enter
  python3 e2p_sweep.py "<파노라마 경로>"
  python3 e2p_sweep.py "<경로>" --v -35         아래로 35도 숙여서
  python3 e2p_sweep.py "<경로>" --step 10       10도 간격 (36장)
  python3 e2p_sweep.py "<경로>" --open          끝나면 윈도우 탐색기로 폴더를 연다 (WSL)

  경로는 이런 꼴을 다 받는다 (따옴표가 붙어 있어도 된다):
    /home/dg/.../frames/000000001.jpg                         리눅스 · WSL
    C:\\Users\\asdfa\\Downloads\\equirect.jpg                     윈도우 (WSL 에서 /mnt/c/... 로 바꿔 읽는다)
    \\\\wsl.localhost\\Ubuntu\\home\\...\\equirect.jpg              윈도우 탐색기에서 복사한 WSL 경로

필요한 것
  pip install py360convert     (numpy · pillow · scipy 가 같이 깔린다. 서버 venv_image 는 1.0.4)
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
import time

import numpy as np
import py360convert
from PIL import Image

FOV_DEG = 90        # e2p.py 와 같다
OUT_SIZE = 720      # e2p.py 와 같다
JPEG_QUALITY = 92   # stream_to_server.py 기본값과 같다


def clean_path(raw):
    """붙여 넣은 경로를 이 컴퓨터에서 열 수 있는 경로로 바꾼다."""
    p = raw.strip().strip('"').strip("'").strip()
    if os.name == "nt":                                   # 윈도우 파이썬이면 그대로
        return p
    m = re.match(r"^\\\\wsl(?:\.localhost|\$)\\[^\\]+(\\.*)$", p, re.I)
    if m:                                                 # \\wsl.localhost\Ubuntu\home\... → /home/...
        return m.group(1).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:                                                 # C:\Users\... → /mnt/c/Users/...
        return "/mnt/%s/%s" % (m.group(1).lower(), m.group(2).replace("\\", "/"))
    return os.path.expanduser(p)


def windows_path(path):
    """WSL 이면 탐색기 주소창에 붙여 넣을 수 있는 윈도우 경로. 아니면 None."""
    if os.name == "nt":
        return path
    try:
        r = subprocess.run(["wslpath", "-w", path], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="360 파노라마 → 가로 방향으로 돌아가며 90도 사진")
    ap.add_argument("pano", nargs="?", help="파노라마 경로 (안 주면 물어본다)")
    ap.add_argument("--step", type=int, default=15, help="가로 간격 (도, 기본 15 → 24장)")
    ap.add_argument("--v", type=float, default=0.0, help="세로 각도 (도, 아래가 -, 기본 0)")
    ap.add_argument("--out", default=os.path.join(here, "e2p_sweep_out"),
                    help="저장 루트. 이 아래에 실행 시각 폴더를 만든다 (기본: 이 파일 옆 e2p_sweep_out)")
    ap.add_argument("--open", action="store_true", help="끝나면 탐색기로 폴더를 연다 (WSL · 윈도우)")
    args = ap.parse_args()

    if args.step <= 0 or 360 % args.step:
        sys.exit("--step 은 360 을 나누어떨어지게 하는 양수여야 합니다 (예: 5 · 10 · 15 · 30 · 45 · 90)")

    raw = args.pano
    if not raw:
        try:
            raw = input("파노라마 경로를 붙여 넣고 Enter: ")
        except (EOFError, KeyboardInterrupt):
            sys.exit("\n취소")
    src = clean_path(raw)
    if not os.path.isfile(src):
        sys.exit("파일이 없습니다: %s%s" % (src, "" if src == raw.strip() else "   (붙여 넣은 값: %s)" % raw.strip()))

    t0 = time.perf_counter()
    try:
        img = Image.open(src)
        img.load()
    except Exception as e:
        sys.exit("사진을 못 열었습니다: %s — %s" % (src, e))
    equ = np.asarray(img.convert("RGB"))
    h, w = equ.shape[:2]
    ratio = w / float(h)

    run = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.abspath(os.path.expanduser(args.out)), run)
    os.makedirs(out_dir, exist_ok=True)

    us = list(range(0, 360, args.step))
    vt = "%+g" % args.v                                   # +0 · -35 · -36.35
    print("=" * 64)
    print("원본   : %s" % src)
    print("크기   : %dx%d (가로:세로 %.2f)" % (w, h, ratio))
    if abs(ratio - 2.0) > 0.02:
        print("  ⚠ 2:1 이 아닙니다 — 이어 붙인 360 파노라마가 아닐 수 있습니다. 잘린 사진이 이상하면 원본부터 확인하세요.")
    print("자르기 : 시야 %dx%d · %dx%d · v %+g · u 0~%d (%d도 간격, %d장)"
          % (FOV_DEG, FOV_DEG, OUT_SIZE, OUT_SIZE, args.v, us[-1], args.step, len(us)))
    print("저장   : %s" % out_dir)
    print("=" * 64)

    names = []
    for u in us:
        persp = py360convert.e2p(equ, fov_deg=(FOV_DEG, FOV_DEG), u_deg=u, v_deg=args.v,
                                 out_hw=(OUT_SIZE, OUT_SIZE))
        if persp.dtype != np.uint8:
            persp = np.clip(persp, 0, 255).astype(np.uint8)
        name = "u+%03d_v%s.jpg" % (u, vt)
        Image.fromarray(persp).save(os.path.join(out_dir, name), quality=JPEG_QUALITY)
        names.append(name)
        print("  %s" % name)

    with open(os.path.join(out_dir, "_info.txt"), "w", encoding="utf-8") as f:
        f.write("실행     %s\n" % run)
        f.write("원본     %s\n" % src)
        f.write("크기     %dx%d\n" % (w, h))
        f.write("시야     %dx%d 도 · 출력 %dx%d\n" % (FOV_DEG, FOV_DEG, OUT_SIZE, OUT_SIZE))
        f.write("v        %+g 도\n" % args.v)
        f.write("u        0~%d 도, %d 도 간격 (%d장)\n" % (us[-1], args.step, len(us)))
        f.write("방향     u 0 = 파노라마 가운데, + 오른쪽 / v + 위, - 아래\n")
        f.write("다시     e2p.py <원본> <저장> <u> <v>  로 같은 사진 한 장을 만들 수 있다\n")

    print("=" * 64)
    print("끝 — %d장, %.1f초" % (len(names), time.perf_counter() - t0))
    print("폴더   : %s" % out_dir)
    wp = windows_path(out_dir)
    if wp and wp != out_dir:
        print("탐색기 : %s" % wp)
    if args.open and wp:
        try:
            subprocess.run(["explorer.exe", wp], timeout=10)   # 탐색기는 성공해도 종료 코드 1 을 낸다
        except (OSError, subprocess.SubprocessError) as e:
            print("탐색기를 못 열었습니다: %s" % e)


if __name__ == "__main__":
    main()
