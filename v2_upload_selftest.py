#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2_upload_selftest.py — 장비 없이 360 업로드 입구(/upload_pano)에 v2 모양 사진을 몇 장 올려 본다.

2026-10-14 · 통합데이터규약 v2 적용 시험용. 표준 라이브러리만 쓴다(그림은 이 파일 안의 작은 JPEG).
360 카메라 · stream_to_server.py 없이 「데이터 스트림 → 저장 → 비전」 길을 먼저 확인하는 데 쓴다.
보내는 폼은 v2 stream_to_server.py 와 같은 꼴이다: capture_id · camera_id 글자 칸 + meta(JSON, 프레임 헤더) + file.

쓰는 법
  python3 v2_upload_selftest.py                                   127.0.0.1:10001 로 5장
  python3 v2_upload_selftest.py --url http://<주소>:10001/upload_pano --count 10
  python3 v2_upload_selftest.py --old                             옛 폼(seq · timestamp 글자 칸)으로 — 옛 판 장비가 계속 들어오는지
  python3 v2_upload_selftest.py --jpg <사진 파일>                   그 사진으로 (비전 결과까지 보려면 진짜 사진이 낫다)

보낸 장은 그 실행 폴더(cam360)에 그대로 남는다. capture_id 가 SELFTEST 라 진짜 장과 갈린다.
"""
import argparse
import base64
import json
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
TINY_JPEG = base64.b64decode("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")


def iso_ms(t=None):
    return datetime.fromtimestamp(time.time() if t is None else t, KST).isoformat(timespec="milliseconds")


def jpeg_size(data):
    """JPEG 머리(SOF)에서 (가로, 세로). 못 읽으면 (None, None)."""
    if data[:2] != b"\xff\xd8":
        return None, None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
            i += 1 if marker == 0xFF else 2
            continue
        seg = int.from_bytes(data[i + 2:i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + seg
    return None, None


def multipart(fields):
    """[(name, value bytes|str, content_type|None, filename|None)] → (body, content-type)"""
    b = "selftest" + uuid.uuid4().hex
    out = b""
    for name, value, ctype, filename in fields:
        disp = 'form-data; name="%s"' % name + ('; filename="%s"' % filename if filename else "")
        out += ("--%s\r\nContent-Disposition: %s\r\n" % (b, disp)).encode()
        if ctype:
            out += ("Content-Type: %s\r\n" % ctype).encode()
        out += b"\r\n" + (value if isinstance(value, bytes) else value.encode("utf-8")) + b"\r\n"
    out += ("--%s--\r\n" % b).encode()
    return out, "multipart/form-data; boundary=" + b


def main():
    ap = argparse.ArgumentParser(description="360 업로드 입구에 v2 모양 사진을 올려 본다 (장비 없이)")
    ap.add_argument("--url", default="http://127.0.0.1:10001/upload_pano")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--interval", type=float, default=0.5, help="장 사이 간격(초)")
    ap.add_argument("--camera-id", default="cam360-001_main")
    ap.add_argument("--jpg", default=None, help="보낼 사진 파일 (안 주면 이 파일 안의 작은 JPEG)")
    ap.add_argument("--old", action="store_true", help="옛 폼(seq · timestamp 글자 칸, meta 없음)으로 보낸다")
    a = ap.parse_args()
    data = open(a.jpg, "rb").read() if a.jpg else TINY_JPEG
    w, h = jpeg_size(data)
    if not w:
        print("JPEG 크기를 못 읽었다 — JPEG 파일인지 본다: %s" % a.jpg)
        return 1
    session = uuid.uuid4().hex[:12]
    ok = 0
    for i in range(1, a.count + 1):
        t = time.time()
        if a.old:
            fields = [("capture_id", "SELFTEST", None, None), ("camera_id", "pro2_anchor", None, None),
                      ("seq", str(i), None, None), ("timestamp", "%.6f" % t, None, None)]
        else:
            meta = {"schema_version": "2.0", "source_id": "cam360-001", "node_id": socket.gethostname(),
                    "zone_id": "zoneA", "camera_id": a.camera_id, "sequence_id": i, "session_id": session,
                    "timestamp": iso_ms(t), "encoding": "jpeg", "width": w, "height": h}
            fields = [("capture_id", "SELFTEST", None, None), ("camera_id", a.camera_id, None, None),
                      ("meta", json.dumps(meta, ensure_ascii=False), "application/json", None)]
        fields.append(("file", data, "image/jpeg", "equirect.jpg"))
        body, ctype = multipart(fields)
        req = urllib.request.Request(a.url, data=body, method="POST",
                                     headers={"Content-Type": ctype, "Content-Length": str(len(body))})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                ans = r.read().decode("utf-8", "replace")
                ok += 1
                print("  #%d → %s" % (i, ans))
        except urllib.error.HTTPError as e:
            print("  #%d → HTTP %d %s" % (i, e.code, e.read()[:200]))
        except Exception as e:
            print("  #%d → 못 보냄 (%s: %s)" % (i, type(e).__name__, e))
        time.sleep(a.interval)
    print("보냄 %d / %d  → %s  (%s)" % (ok, a.count, a.url, "옛 폼" if a.old else "v2 폼, session_id=" + session))
    return 0 if ok == a.count else 1


if __name__ == "__main__":
    sys.exit(main())
