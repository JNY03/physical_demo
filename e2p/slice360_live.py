#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slice360_live.py — 데이터 스트림이 받는 360 파노라마를 정한 각도로 잘라 슬라이스 소스 cam360_slice(10006)로 넣는다
================================================================================
2026-11-11

하는 일
  같은 기기의 데이터 스트림(server_stream_multi_source.py)이 Redis camera_stream:cam360 에 흘리는 파노라마를
  새로 들어올 때마다(가장 최근 장만) 정한 각도로 잘라, 같은 데이터 스트림의 슬라이스 포트로 올린다.
      POST http://127.0.0.1:10006/upload   (file = 슬라이스 JPEG, meta = 프레임 헤더 JSON)
  그 뒤는 데이터 스트림이 다른 소스와 똑같이 한다 — stream_data/cam360_slice/<실행>/frames 저장 · frames.jsonl ·
  Redis camera_stream:cam360_slice · GET :10006/stream · :10006/vision · :10006/health.
  그래서 vision_infer.py(--devices cam360_slice) · 시각화(카메라 칸 · 추론 스트림 포트)가
  슬라이스를 다른 소스처럼 받는다. 파노라마 원본(10001, stream_data/cam360)은 읽기만 한다.

  데이터 스트림 안에 넣지 않고 따로 도는 이유
    - 데이터 스트림은 「받은 것을 손대지 않고 저장·흘리기」만 하고 표준 라이브러리로 돈다(윈도우 파이썬에서도).
      자르기(numpy · opencv · py360convert)를 넣으면 그 성격이 바뀐다.
    - 각도를 바꾸거나 이 파일이 죽어도 데이터 스트림의 받기·저장은 끊기지 않는다.

각도 — 실행 인자 + 실행 중 바꾸기 (각도 파일)
  처음:       python e2p/slice360_live.py --u 230 --v -30
  돌고 있을 때: python e2p/slice360_live.py --set 245 -30        ← 다른 터미널에서. 각도 파일만 고치고 끝난다
  각도 파일(--angle-file, 기본 e2p/slice_angle.json)이 지금 각도를 늘 들고 있다.
    - 시작할 때 --u 를 주면 그 값을 각도 파일에 쓰고 시작한다. --u 를 안 주면 각도 파일 값으로 시작한다.
    - 도는 동안 각도 파일이 바뀌면(장마다 수정 시각을 본다) 다음 장부터 새 각도로 자른다. 화면에 한 줄 찍는다.
    - 장마다 그 장을 자른 각도가 meta 에 실리므로, 중간에 바꿔도 어느 장이 어느 각도인지 기록에 남는다.
  각도 고르기는 e2p/sweep_latest.py (최신 파노라마로 24장).

슬라이스 한 장의 meta (가칭 — 칸 이름은 이름 사전에서 확정)
  원본 파노라마의 프레임 헤더(sender: camera_id · sequence_id · session_id · timestamp(찍은 시각) …)를 그대로 싣고,
  width · height 만 슬라이스 크기로 바꾸고, slice 칸 하나를 더한다:
    slice = {source_stream, source_n, source_frame_file, source_entry_id,   ← 어느 파노라마에서 잘랐나
             u_deg, v_deg, fov_deg, fov_v_deg, width, height, sliced_at}   ← 어떻게 잘랐나
  timestamp 를 원본 그대로 두는 이유: 드론 사진과 짝을 지을 때 "찍은 시각"이 같아야 한다.

늦게 도착한 옛 장은 자르지 않는다 (2026-11-11)
  360 노트북의 업로드가 「응답 없음」으로 실패한 장이 망이 돌아온 뒤 늦게 도착해 데이터 스트림에 저장될 수 있다
  (작업 공간 시험: 14번이 38번 뒤에 도착). 데이터 스트림은 받은 것을 다 저장하므로 그대로 두고, 이 파일은
  같은 session_id 안에서 이미 자른 sequence_id 보다 작거나 같은 장을 건너뛴다 — 실시간 슬라이스(와 그것을 쓰는 비전)가
  옛 장면을 지금 장면으로 쓰지 않게. 10초 보고의 「늦게 온 옛 장 건너뜀」.

빠르기 (Claude 작업 공간 CPU 측정, 실제 노트북은 다를 수 있음)
  자르기 3840×1920 → 960×960 약 20~25ms (같은 각도면 좌표표를 다시 만들지 않는다). 디코딩 · 인코딩이 더 든다.
  밀리면 가장 최근 파노라마만 자른다 — 오래된 장을 쌓지 않는다(실시간 우선). 빠진 장은 원본에 그대로 있다.

사용 (저장소 루트, venv, 데이터 스트림 · Redis 가 떠 있을 때)
  python e2p/slice360_live.py --u 230 --v -30                 기본 960×960 (90도 → 원본 3840 가로의 제 해상도)
  python e2p/slice360_live.py --u 230 --v -30 --width 720     작게 (시각화 망 부담 ↓)
  python e2p/slice360_live.py --set 245 -30                    돌고 있는 것의 각도 바꾸기
  python e2p/slice360_live.py --show                           지금 각도 파일 보기

필요한 것: numpy · opencv-python · py360convert · redis · requests (DS venv 에 py360convert 만 더 깔면 된다)
"""
import argparse
import json
import os
import signal
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from slice360_common import (iso_ms, is_pano, read_angle_file, slice_pano,  # noqa: E402
                             write_angle_file)

DEF_ANGLE_FILE = str(HERE / "slice_angle.json")
STOP = False


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def build_form(boundary, jpg, meta_json):
    """multipart 본문 — requests 없이 만든다 (stream_to_server 가 보내는 것과 같은 모양: file + meta)"""
    b = boundary.encode()
    parts = [
        b"--" + b + b'\r\nContent-Disposition: form-data; name="meta"\r\nContent-Type: application/json\r\n\r\n'
        + meta_json.encode("utf-8") + b"\r\n",
        b"--" + b + b'\r\nContent-Disposition: form-data; name="file"; filename="slice.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n",
        b"--" + b + b"--\r\n",
    ]
    return b"".join(parts)


class Poster:
    """연결 하나를 열어 두고 POST (매번 새로 열지 않는다)"""

    def __init__(self, url, timeout=5.0):
        from urllib.parse import urlsplit
        u = urlsplit(url)
        self.host, self.port = u.hostname, u.port or 80
        self.path = u.path or "/upload"
        self.timeout = timeout
        self.conn = None

    def post(self, jpg, meta_json):
        import http.client
        boundary = "slice360%x" % int(time.time() * 1e6)
        body = build_form(boundary, jpg, meta_json)
        for attempt in (1, 2):                       # 끊긴 연결을 다시 열어 한 번 더
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
                self.conn.request("POST", self.path, body=body,
                                  headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
                r = self.conn.getresponse()
                ans = r.read()
                if r.status != 200:
                    return False, "HTTP %d %s" % (r.status, ans[:120])
                return True, json.loads(ans or b"{}").get("status", "?")
            except Exception as e:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
                if attempt == 2:
                    return False, "%s: %s" % (type(e).__name__, e)
        return False, "?"


def main():
    global STOP
    ap = argparse.ArgumentParser(description="360 파노라마 → 정한 각도 슬라이스 → 데이터 스트림 슬라이스 포트",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--u", type=float, default=None, help="가로 각도 (도). 안 주면 각도 파일 값")
    ap.add_argument("--v", type=float, default=None, help="세로 각도 (도, 아래가 -). 안 주면 각도 파일 값, 그것도 없으면 0")
    ap.add_argument("--fov", type=float, default=None, help="가로 시야 (도, 기본 90)")
    ap.add_argument("--width", type=int, default=None, help="슬라이스 가로 픽셀 (기본 960)")
    ap.add_argument("--aspect", default=None, help="가로:세로 (기본 1:1)")
    ap.add_argument("--quality", type=int, default=92, help="JPEG 품질 (기본 92 — stream_to_server 와 같음)")
    ap.add_argument("--angle-file", default=DEF_ANGLE_FILE, help="각도 파일 (기본 %(default)s)")
    ap.add_argument("--set", nargs=2, type=float, metavar=("U", "V"),
                    help="돌고 있는 slice360_live 의 각도를 바꾼다 (각도 파일만 고치고 끝)")
    ap.add_argument("--show", action="store_true", help="각도 파일을 보여 주고 끝")
    ap.add_argument("--redis", default="redis://127.0.0.1:6380/0", help="데이터 스트림 Redis (기본 %(default)s)")
    ap.add_argument("--source", default="cam360", help="파노라마 소스 (기본 cam360)")
    ap.add_argument("--upload", default="http://127.0.0.1:10006/upload", help="슬라이스를 올릴 곳 (기본 %(default)s)")
    ap.add_argument("--max-age", type=float, default=5.0, help="이보다 오래된 파노라마는 자르지 않는다 (초, 기본 5)")
    ap.add_argument("--report", type=float, default=10.0, help="보고 간격 (초, 기본 10)")
    args = ap.parse_args()

    # ---- 각도 파일만 다루고 끝나는 두 가지
    if args.show:
        a = read_angle_file(args.angle_file)
        print(json.dumps(a, ensure_ascii=False) if a else "각도 파일이 없거나 못 읽음: %s" % args.angle_file)
        return 0
    if args.set:
        cur = read_angle_file(args.angle_file) or {}
        cur.update(u=args.set[0], v=args.set[1])
        for k in ("fov", "width", "aspect"):
            if getattr(args, k) is not None:
                cur[k] = getattr(args, k)
        write_angle_file(args.angle_file, cur, by="--set")
        print("각도 파일 고침 → %s  u %+g · v %+g  (돌고 있는 slice360_live 가 다음 장부터 씀)"
              % (args.angle_file, args.set[0], args.set[1]))
        return 0

    # ---- 시작 각도: 인자 > 각도 파일 > 기본
    base = read_angle_file(args.angle_file) or {}
    if args.u is None and "u" not in base:
        sys.exit("각도가 없습니다 — --u <가로각> (--v <세로각>) 를 주거나, e2p/sweep_latest.py 로 고른 뒤 다시")
    angle = {"u": args.u if args.u is not None else base["u"],
             "v": args.v if args.v is not None else base.get("v", 0.0),
             "fov": args.fov if args.fov is not None else base.get("fov", 90.0),
             "width": args.width if args.width is not None else base.get("width", 960),
             "aspect": args.aspect if args.aspect is not None else base.get("aspect", "1:1")}
    write_angle_file(args.angle_file, angle, by="start")
    angle_mtime = os.stat(args.angle_file).st_mtime_ns

    import redis
    r = None
    key = "camera_stream:" + args.source
    poster = Poster(args.upload)

    def on_signal(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGINT, on_signal)
    try:
        signal.signal(signal.SIGTERM, on_signal)
    except (ValueError, AttributeError, OSError):
        pass

    print("=" * 72)
    print("slice360_live — %s 의 %s → 자르기 → %s" % (args.redis, key, args.upload))
    print("  각도 u %+g · v %+g · 시야 %g · %s · 가로 %dpx  (각도 파일 %s)"
          % (angle["u"], angle["v"], angle["fov"], angle["aspect"], angle["width"], args.angle_file))
    print("  바꾸기: python e2p/slice360_live.py --set <u> <v>   · 끝내기: Ctrl+C")
    print("=" * 72, flush=True)

    last_id = "$"
    st = {"in": 0, "sliced": 0, "old": 0, "late": 0, "bad": 0, "fail": 0}
    last_seq = {}                                   # session_id → 마지막으로 자른 sequence_id
    tm = {"dec": [], "cut": [], "enc": [], "post": [], "lag": []}
    tot = dict(st)
    t_rep = time.time()
    fail_note = None
    redis_err = None
    while not STOP:
        # 각도 파일이 바뀌었나
        try:
            m = os.stat(args.angle_file).st_mtime_ns
        except OSError:
            m = angle_mtime
        if m != angle_mtime:
            angle_mtime = m
            a = read_angle_file(args.angle_file)
            if a:
                new = dict(angle)
                new.update(a)
                if new != angle:
                    log("각도 바뀜: u %+g → %+g · v %+g → %+g%s"
                        % (angle["u"], new["u"], angle["v"], new["v"],
                           "" if (new["width"], new["aspect"], new["fov"]) == (angle["width"], angle["aspect"], angle["fov"])
                           else "  (크기·시야도 바뀜: %s %dpx 시야 %g)" % (new["aspect"], new["width"], new["fov"])))
                    angle = new
            else:
                log("각도 파일을 못 읽음 — 이전 각도 그대로 (%s)" % args.angle_file)

        # 새 파노라마 기다리기 (가장 최근 장만)
        try:
            if r is None:
                r = redis.Redis.from_url(args.redis, socket_connect_timeout=2, socket_timeout=5)
            got = r.xread({key: last_id}, block=1000)
            if redis_err is not None:
                log("Redis 다시 됨")
                redis_err = None
        except Exception as e:
            if redis_err is None:
                log("Redis 안 됨 (%s) — 2초마다 다시" % e)
            redis_err = e
            r = None
            time.sleep(2.0)
            continue

        if got:
            entries = got[0][1]
            st["in"] += len(entries)
            eid, f = entries[-1]                     # 밀린 것이 있으면 마지막 장만
            last_id = eid
            ent_ms = int(eid.split(b"-")[0])
            if time.time() - ent_ms / 1000.0 > args.max_age:
                st["old"] += 1
            else:
                try:
                    header = json.loads(f.get(b"header") or b"{}")
                except ValueError:
                    header = {}
                snd = header.get("sender") if isinstance(header.get("sender"), dict) else {}
                sid, sq = snd.get("session_id"), snd.get("sequence_id")
                late = isinstance(sq, int) and sid is not None and sid in last_seq and sq <= last_seq[sid]
                if late:
                    st["late"] += 1
                    continue
                t0 = time.perf_counter()
                equ = cv2.imdecode(np.frombuffer(f.get(b"image") or b"", np.uint8), cv2.IMREAD_COLOR)
                t1 = time.perf_counter()
                if equ is None or not is_pano(equ.shape):
                    st["bad"] += 1
                    fail_note = "파노라마가 아님/못 풂 (%s)" % (None if equ is None else "x".join(map(str, equ.shape[1::-1])))
                else:
                    sl, fv = slice_pano(equ, angle["u"], angle["v"], angle["fov"], angle["width"], angle["aspect"])
                    t2 = time.perf_counter()
                    ok, buf = cv2.imencode(".jpg", sl, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
                    t3 = time.perf_counter()
                    sender = header.get("sender") if isinstance(header.get("sender"), dict) else {}
                    meta = dict(sender)
                    meta["width"], meta["height"] = int(sl.shape[1]), int(sl.shape[0])
                    meta["slice"] = {"source_stream": args.source, "source_n": header.get("n"),
                                     "source_frame_file": header.get("frame_file"),
                                     "source_entry_id": eid.decode(),
                                     "u_deg": angle["u"], "v_deg": angle["v"],
                                     "fov_deg": angle["fov"], "fov_v_deg": round(fv, 3),
                                     "width": int(sl.shape[1]), "height": int(sl.shape[0]),
                                     "sliced_at": iso_ms()}
                    okp, said = poster.post(buf.tobytes(), json.dumps(meta, ensure_ascii=False))
                    t4 = time.perf_counter()
                    if okp:
                        st["sliced"] += 1
                        if isinstance(sq, int) and sid is not None:
                            last_seq[sid] = sq
                        tm["dec"].append(t1 - t0); tm["cut"].append(t2 - t1)
                        tm["enc"].append(t3 - t2); tm["post"].append(t4 - t3)
                        tm["lag"].append(time.time() - ent_ms / 1000.0)
                    else:
                        st["fail"] += 1
                        fail_note = "올리기 실패 %s — 데이터 스트림에 %s 가 켜져 있나?" % (said, args.upload)

        now = time.time()
        if now - t_rep >= args.report:
            dt = now - t_rep
            t_rep = now
            med = lambda v: statistics.median(v) * 1000 if v else float("nan")  # noqa: E731
            line = ("받음 %d · 자름 %d (%.1f장/초) · 오래돼 건너뜀 %d%s · 실패 %d%s"
                    % (st["in"], st["sliced"], st["sliced"] / dt, st["old"],
                       " · 늦게 온 옛 장 건너뜀 %d" % st["late"] if st["late"] else "", st["fail"] + st["bad"],
                       "" if not st["sliced"] else
                       "  | 중앙 디코딩 %.0f · 자르기 %.0f · 인코딩 %.0f · 올리기 %.0f ms · 파노라마→올림 %.0f ms"
                       % (med(tm["dec"]), med(tm["cut"]), med(tm["enc"]), med(tm["post"]), med(tm["lag"]))))
            if st["in"] == 0:
                line = "새 파노라마 없음 (%s) — 360 이 들어오고 있나? 데이터 스트림 cam360 장/초 확인" % key
            log(line + ("" if not fail_note else "\n           마지막 문제: " + fail_note))
            for k in st:
                tot[k] += st[k]
                st[k] = 0
            for k in tm:
                tm[k] = []
            fail_note = None

    for k in st:
        tot[k] += st[k]
    print("\n끝 — 받은 파노라마 %d · 자른 장 %d · 오래돼 건너뜀 %d · 늦게 온 옛 장 건너뜀 %d · 실패 %d"
          % (tot["in"], tot["sliced"], tot["old"], tot["late"], tot["fail"] + tot["bad"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
