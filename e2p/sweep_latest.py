#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_latest.py — 데이터 스트림이 지금 받는 360 파노라마로 각도 확인용 조각을 만든다 (각도 고르기)
================================================================================
2026-11-11

무엇을 하나
  데이터 스트림(server_stream_multi_source.py)이 받은 **가장 최근 파노라마 한 장**을 가져와
  u = 0, 15, 30, ... 345 (기본 15도 간격, 24장) 방향으로 90도 조각을 잘라 폴더에 저장한다.
  e2p_sweep.py 와 같은 일인데, 파노라마 경로를 손으로 찾지 않고 데이터 스트림에서 바로 가져오고,
  고르기 쉽게 한눈 그림 두 장을 같이 만든다. 원본 파노라마는 읽기만 한다.

  고른 각도는 slice360_live.py 에 준다:  python e2p/slice360_live.py --u <u> --v <v>
  (이미 돌고 있으면 실행 중에 바꾸기:   python e2p/slice360_live.py --set <u> <v>)

파노라마를 가져오는 곳 (--from, 기본 auto = redis 를 먼저, 안 되면 disk)
  redis  Redis camera_stream:cam360 의 최신 항목 (데이터 스트림이 켜져 있을 때)
  disk   <저장 폴더>/cam360/<가장 최근 실행>/frames.jsonl 의 마지막 장 (데이터 스트림이 꺼져 있어도 된다)
  --pano <경로>  그 파일 (윈도우 경로 · WSL 경로 모두. e2p_sweep.py 와 같은 규칙)

저장 위치
  데이터 스트림 실행 폴더의 장이면  <저장 폴더>/cam360/<실행>/sweep/<YYMMDD_HHMMSS>_n<번호>/
      (어느 실행의 몇 번째 파노라마로 잘랐는지가 폴더 이름에 남는다. 원본 frames/ 는 건드리지 않는다)
  그 밖이면                        e2p/e2p_sweep_out/<YYMMDD_HHMMSS>/   (--out 으로 바꿈)

  u+000_v+0.jpg … u+345_v+0.jpg   조각 (v 를 여러 개 주면 v 마다 24장)
  _sheet_v+0.jpg                   조각 24장을 한 장에 (칸마다 u 값) — 이것부터 본다
  _pano_marked.jpg                 파노라마 위에 각 u 의 세로선 · v 의 가로선 — 방향 감 잡기용
  _info.txt                        원본 · 번호 · 시각 · 각도 · 다음 명령
  _source.json                     원본 장의 기록 한 줄 (frames.jsonl 한 줄 = Redis header)

사용 (저장소 루트에서, venv)
  python e2p/sweep_latest.py                       최신 파노라마, v 0, 24장
  python e2p/sweep_latest.py --v 0 -15 -30         v 세 개 (다리 위처럼 내려다볼 때) → 72장
  python e2p/sweep_latest.py --step 10             10도 간격 36장
  python e2p/sweep_latest.py --from disk           데이터 스트림이 꺼져 있을 때
  python e2p/sweep_latest.py --pano "C:\\...\\equirect.jpg"
  python e2p/sweep_latest.py --open                끝나면 윈도우 탐색기로 연다 (WSL)

필요한 것: numpy · opencv-python · py360convert (pip install py360convert) · redis (--from redis 일 때)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from slice360_common import is_pano, iso_ms, pano_xy, slice_pano  # noqa: E402

REPO = HERE.parent
JPEG_QUALITY = 92


def clean_path(raw):
    """e2p_sweep.py 와 같은 규칙 — 윈도우 · WSL 경로를 이 컴퓨터에서 열 수 있게"""
    import re
    p = raw.strip().strip('"').strip("'").strip()
    if os.name == "nt":
        return p
    m = re.match(r"^\\\\wsl(?:\.localhost|\$)\\[^\\]+(\\.*)$", p, re.I)
    if m:
        return m.group(1).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        return "/mnt/%s/%s" % (m.group(1).lower(), m.group(2).replace("\\", "/"))
    return os.path.expanduser(p)


def windows_path(path):
    if os.name == "nt":
        return str(path)
    try:
        r = subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------- 파노라마 가져오기
def from_redis(url, stream):
    import redis
    r = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=5)
    e = r.xrevrange(stream, "+", "-", count=1)
    if not e:
        raise RuntimeError("%s 가 비어 있음 (360 이 아직 안 들어옴)" % stream)
    eid, f = e[0]
    img = f.get(b"image")
    try:
        header = json.loads(f.get(b"header") or b"{}")
    except ValueError:
        header = {}
    age = time.time() - int(eid.split(b"-")[0]) / 1000.0
    return img, header, "redis %s %s (%.1f초 전)" % (stream, eid.decode(), age)


def from_disk(save_root, source):
    d = Path(save_root) / source
    runs = sorted(p for p in d.iterdir() if p.is_dir() and (p / "frames").is_dir()) if d.is_dir() else []
    for run in reversed(runs):
        jl = run / "frames.jsonl"
        last = None
        if jl.is_file():
            with open(jl, "rb") as f:
                for line in f:
                    if line.endswith(b"\n"):
                        last = line
        if last:
            try:
                header = json.loads(last)
            except ValueError:
                header = None
            if header and header.get("frame_file"):
                p = Path(save_root) / header["frame_file"]
                if p.is_file():
                    return p.read_bytes(), header, "disk %s" % p
        jpgs = sorted((run / "frames").glob("*.jpg"))
        if jpgs:
            return jpgs[-1].read_bytes(), {}, "disk %s (frames.jsonl 없이 가장 마지막 파일)" % jpgs[-1]
    raise RuntimeError("%s 아래에 파노라마가 있는 실행 폴더가 없음" % d)


# ---------------------------------------------------------------- 한눈 그림
def put_text(img, text, org, scale, color=(255, 255, 255), th=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def contact_sheet(items, cols=6, tw=300):
    """items = [(이름, 사진)] → 한 장"""
    rows = (len(items) + cols - 1) // cols
    th = None
    tiles = []
    for name, img in items:
        h, w = img.shape[:2]
        t = cv2.resize(img, (tw, int(round(h * tw / float(w)))), interpolation=cv2.INTER_AREA)
        th = t.shape[0]
        put_text(t, name, (6, 22), 0.6, (0, 255, 255), 1)
        tiles.append(t)
    gap = 4
    sheet = np.full((rows * (th + gap) + gap, cols * (tw + gap) + gap, 3), 40, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = gap + r * (th + gap), gap + c * (tw + gap)
        sheet[y:y + th, x:x + tw] = t
    return sheet


def marked_pano(equ, us, vs, fov, width_out=1920):
    h, w = equ.shape[:2]
    s = width_out / float(w)
    img = cv2.resize(equ, (width_out, int(round(h * s))), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    for v in vs:
        _, y = pano_xy(0, v, W, H)
        cv2.line(img, (0, int(y)), (W - 1, int(y)), (0, 200, 255), 1, cv2.LINE_AA)
        put_text(img, "v%+g" % v, (4, max(14, int(y) - 4)), 0.45, (0, 200, 255))
    for u in us:
        x, _ = pano_xy(u, 0, W, H)
        col = (0, 255, 0) if u % 90 == 0 else (255, 255, 255)
        cv2.line(img, (int(x), 0), (int(x), H - 1), col, 1, cv2.LINE_AA)
        put_text(img, "%d" % u, (int(x) + 2, 16 + (14 if (u // 15) % 2 else 0)), 0.42, col)
    put_text(img, "u: 0 = center, + = right (fov %g deg per slice)" % fov, (6, H - 8), 0.5, (0, 255, 255))
    return img


# ---------------------------------------------------------------- 실행
def main():
    ap = argparse.ArgumentParser(description="데이터 스트림 최신 360 파노라마 → 각도 확인용 조각",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--from", dest="src", choices=["auto", "redis", "disk"], default="auto",
                    help="파노라마를 가져올 곳 (기본 auto: redis → disk)")
    ap.add_argument("--pano", default=None, help="이 파일로 (주면 --from 무시)")
    ap.add_argument("--redis", default="redis://127.0.0.1:6380/0", help="데이터 스트림 Redis (기본 %(default)s)")
    ap.add_argument("--source", default="cam360", help="파노라마 소스 이름 (기본 cam360)")
    ap.add_argument("--save", default=str(REPO / "stream_data"),
                    help="데이터 스트림 저장 폴더 (서버 --save 와 같게, 기본 저장소 루트의 stream_data)")
    ap.add_argument("--step", type=float, default=15.0, help="가로 간격 (도, 기본 15 → 24장)")
    ap.add_argument("--v", type=float, nargs="+", default=[0.0], help="세로 각도들 (도, 아래가 -, 기본 0)")
    ap.add_argument("--fov", type=float, default=90.0, help="가로 시야 (도, 기본 90)")
    ap.add_argument("--width", type=int, default=720, help="조각 가로 픽셀 (기본 720 — e2p_sweep 와 같음)")
    ap.add_argument("--aspect", default="1:1", help="조각 가로:세로 (기본 1:1)")
    ap.add_argument("--out", default=None, help="저장 폴더를 직접 정한다")
    ap.add_argument("--open", action="store_true", help="끝나면 탐색기로 연다 (WSL · 윈도우)")
    args = ap.parse_args()

    if args.step <= 0 or abs(360.0 / args.step - round(360.0 / args.step)) > 1e-6:
        sys.exit("--step 은 360 을 나누어떨어지게 하는 양수여야 합니다 (예: 5 · 10 · 15 · 30 · 45)")

    t0 = time.perf_counter()
    header = {}
    if args.pano:
        p = clean_path(args.pano)
        if not os.path.isfile(p):
            sys.exit("파일이 없습니다: %s" % p)
        data, where = open(p, "rb").read(), "file %s" % p
    else:
        data = None
        errs = []
        if args.src in ("auto", "redis"):
            try:
                data, header, where = from_redis(args.redis, "camera_stream:" + args.source)
            except Exception as e:
                errs.append("redis: %s" % e)
        if data is None and args.src in ("auto", "disk"):
            try:
                data, header, where = from_disk(args.save, args.source)
            except Exception as e:
                errs.append("disk: %s" % e)
        if data is None:
            sys.exit("파노라마를 못 가져왔습니다 — " + " / ".join(errs))

    equ = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if equ is None:
        sys.exit("사진을 못 풀었습니다 (%s)" % where)
    h, w = equ.shape[:2]

    # 저장 위치
    stamp = time.strftime("%y%m%d_%H%M%S")
    run_dir = None
    ff = header.get("frame_file") if isinstance(header, dict) else None
    if ff:
        cand = (Path(args.save) / ff).parent.parent
        if (cand / "frames").is_dir():
            run_dir = cand
    if args.out:
        out = Path(clean_path(args.out)) / stamp
    elif run_dir is not None:
        out = run_dir / "sweep" / ("%s_n%s" % (stamp, header.get("n", "x")))
    else:
        out = HERE / "e2p_sweep_out" / stamp
    out.mkdir(parents=True, exist_ok=False)

    n_u = int(round(360.0 / args.step))
    us = [round(i * args.step, 6) for i in range(n_u)]
    us = [int(u) if float(u).is_integer() else u for u in us]
    print("=" * 66)
    print("원본   : %s" % where)
    print("크기   : %dx%d%s" % (w, h, "" if is_pano(equ.shape) else "   ⚠ 2:1 이 아님 — 360 파노라마가 아닐 수 있음"))
    snd = (header.get("sender") if isinstance(header, dict) else None) or {}
    if header:
        print("기록   : n %s · sequence_id %s · 찍은 시각 %s · 받은 시각 %s"
              % (header.get("n"), snd.get("sequence_id"), snd.get("timestamp"), header.get("received_at")))
    print("자르기 : 가로 시야 %g° · %s · %dpx · u 0~%s (%g도 간격, %d장) × v %s"
          % (args.fov, args.aspect, args.width, us[-1], args.step, len(us), " ".join("%+g" % v for v in args.v)))
    print("저장   : %s" % out)
    print("=" * 66)

    for v in args.v:
        items = []
        for u in us:
            sl, fv = slice_pano(equ, u, v, args.fov, args.width, args.aspect)
            name = "u+%03d_v%+g" % (u, v) if isinstance(u, int) else "u+%07.3f_v%+g" % (u, v)
            cv2.imwrite(str(out / (name + ".jpg")), sl, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            items.append(("u=%s v=%+g" % (u, v), sl))
        cv2.imwrite(str(out / ("_sheet_v%+g.jpg" % v)), contact_sheet(items), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print("  v %+g : %d장 + _sheet_v%+g.jpg" % (v, len(items), v))
    cv2.imwrite(str(out / "_pano_marked.jpg"), marked_pano(equ, us, args.v, args.fov),
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    with open(out / "_source.json", "w", encoding="utf-8") as f:
        json.dump({"from": where, "header": header}, f, ensure_ascii=False, indent=1)
    with open(out / "_info.txt", "w", encoding="utf-8") as f:
        f.write("실행     %s (%s)\n" % (stamp, iso_ms()))
        f.write("원본     %s\n" % where)
        if ff:
            f.write("원본 장  %s  (n %s · sequence_id %s · 찍은 시각 %s)\n"
                    % (ff, header.get("n"), snd.get("sequence_id"), snd.get("timestamp")))
        f.write("크기     %dx%d\n" % (w, h))
        f.write("자르기   가로 시야 %g도 · %s · 가로 %dpx\n" % (args.fov, args.aspect, args.width))
        f.write("u        0~%s 도, %g 도 간격 (%d장)\n" % (us[-1], args.step, len(us)))
        f.write("v        %s\n" % " ".join("%+g" % v for v in args.v))
        f.write("방향     u 0 = 파노라마 가운데, + 오른쪽 / v + 위, - 아래\n")
        f.write("다음     python e2p/slice360_live.py --u <u> --v <v>      (처음 띄울 때)\n")
        f.write("         python e2p/slice360_live.py --set <u> <v>        (돌고 있을 때 바꾸기)\n")
    print("=" * 66)
    print("끝 — %.1f초.  먼저 볼 것: _sheet_v*.jpg · _pano_marked.jpg" % (time.perf_counter() - t0))
    print("폴더   : %s" % out)
    wp = windows_path(out)
    if wp and wp != str(out):
        print("탐색기 : %s" % wp)
    print("고른 뒤: python e2p/slice360_live.py --u <u> --v <v>   (돌고 있으면 --set <u> <v>)")
    if args.open and wp:
        try:
            subprocess.run(["explorer.exe", wp], timeout=10)
        except (OSError, subprocess.SubprocessError) as e:
            print("탐색기를 못 열었습니다: %s" % e)


if __name__ == "__main__":
    main()
