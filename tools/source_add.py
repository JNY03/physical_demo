#!/usr/bin/env python3
"""소스 더하기 — 새 카메라 기기를 **시험해 보고** 목록에 넣는다 (2026-11-26).

    venv/bin/python tools/source_add.py --test --host 100.1.2.3 [--cam-port 8090] [--path /stream/1] [--json]
    venv/bin/python tools/source_add.py --save --name pi14 --host 100.1.2.3 [--port 10013] [--cam-port 8090] [--json]
    venv/bin/python tools/source_add.py --list [--json]
    venv/bin/python tools/source_add.py --remove pi14

왜 있나: 기기를 더하려면 지금까지 `server_stream_multi_source.py` 의 표를 고쳐야 했다. 그래서 사람이 카메라
하나를 붙이는 데 코드를 건드려야 했다. 이 도구는 **먼저 연결을 시험하고**(정말 장이 오는지 본다) 되는 것만
`config/sources.json` 에 적는다. 데이터 서버는 그 파일을 읽어 그 소스도 함께 받는다.

무엇을 시험하나: `http://<주소>:<카메라 포트><경로>` 를 열어 **JPEG 표지(FFD8…FFD9)** 가 오는지 보고, 한 장의
크기와 해상도를 돌려준다. 열리기만 하고 장이 안 오는 것(포트는 열렸지만 카메라가 없는 기기)을 가려내려는 것이다.

규칙
  * 읽기와 설정 파일 쓰기만 한다. 데이터 서버를 띄우거나 끄지 않는다.
  * 이름은 이미 있는 소스 · 별칭과 겹칠 수 없다(겹치면 영상이 엉뚱한 기기 것으로 간다).
  * 포트는 비어 있는 것을 고른다(이미 쓰는 포트면 데이터 서버가 못 뜬다).
  * **시험에 실패하면 저장하지 않는다** — `--force` 를 줘야 저장한다(기기를 아직 안 켠 경우).
"""
import argparse
import json
import re
import socket
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CONFIG = REPO / "config" / "sources.json"
NAME_RE = re.compile(r"^[a-z][a-z0-9_\-]{1,20}$")
PORT_LOW, PORT_HIGH = 10013, 10099        # 더하는 소스가 쓸 데이터 서버 포트 범위 (기존은 10001~10012)
CAM_PORT = 8090                            # picam_mjpeg.py · go1-camview · a2-bridge 가 모두 이 포트다


def known_names():
    """이미 쓰는 소스 이름과 별칭 — server_stream_multi_source 를 **글로** 읽는다(cv2 를 끌어오지 않게)"""
    txt = (REPO / "server_stream_multi_source.py").read_text(encoding="utf-8")
    names, ports = set(), {}
    m = re.search(r"^SOURCES\s*=\s*\{(.*?)\}", txt, re.S | re.M)
    if m:
        for k, v in re.findall(r'"([\w\-]+)"\s*:\s*(\d+)', m.group(1)):
            names.add(k)
            ports[k] = int(v)
    m2 = re.search(r"^ALIASES\s*=\s*\{(.*?)\n\}", txt, re.S | re.M)
    if m2:
        names |= set(re.findall(r'"([\w\-\.]+)"\s*:', m2.group(1)))
    return names, ports


def load():
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    except ValueError as e:
        raise SystemExit("%s 을 못 읽었습니다: %s" % (CONFIG, e))


def save(data):
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def free_port(used):
    for p in range(PORT_LOW, PORT_HIGH + 1):
        if p in used:
            continue
        try:
            with socket.create_connection(("127.0.0.1", p), 0.2):
                continue                       # 누가 쓰고 있다
        except OSError:
            return p
    return None


def test_stream(host, cam_port=CAM_PORT, path="/stream/1", timeout=6.0):
    """그 기기에서 정말 장이 오는지 — {ok, why, bytes, jpeg, width, height, ms}"""
    url = "http://%s:%d%s" % (host, cam_port, path)
    out = {"ok": False, "url": url, "why": "", "bytes": 0, "jpeg": 0, "width": None, "height": None, "ms": 0}
    t0 = time.time()
    try:
        with socket.create_connection((host, cam_port), min(3.0, timeout)):
            pass
    except OSError as e:
        out["why"] = "%s:%d 에 닿지 않습니다 (%s)" % (host, cam_port, type(e).__name__)
        return out
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            buf = b""
            while time.time() - t0 < timeout and len(buf) < 4 * 1024 * 1024:
                chunk = r.read(8192)
                if not chunk:
                    break
                buf += chunk
                s = buf.find(b"\xff\xd8")
                e = buf.find(b"\xff\xd9", s + 2) if s >= 0 else -1
                if s >= 0 and e > s:
                    jpg = buf[s:e + 2]
                    out.update(ok=True, bytes=len(buf), jpeg=len(jpg), ms=int((time.time() - t0) * 1000))
                    w, h = _jpeg_size(jpg)
                    out.update(width=w, height=h)
                    return out
            out["bytes"] = len(buf)
            out["why"] = "연결은 됐지만 그림 한 장을 못 받았습니다 (MJPEG 가 아닌 주소일 수 있습니다)"
    except Exception as e:  # noqa: BLE001 — 사람에게 그대로 보여 준다
        out["why"] = "%s: %s" % (type(e).__name__, str(e)[:160])
    out["ms"] = int((time.time() - t0) * 1000)
    return out


def _jpeg_size(data):
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        if data[i + 1] in (0xC0, 0xC1, 0xC2):
            return (data[i + 7] << 8) | data[i + 8], (data[i + 5] << 8) | data[i + 6]
        i += 2 + ((data[i + 2] << 8) | data[i + 3])
    return None, None


def add(name, host, port=None, cam_port=CAM_PORT, path="/stream/1", note="", force=False, timeout=6.0):
    """시험하고 넣는다 → {ok, why, source, test}"""
    names, ports = known_names()
    data = load()
    if not NAME_RE.match(name or ""):
        return {"ok": False, "why": "이름은 영문 소문자로 시작하는 2~21자 (a-z, 0-9, _, -): %r" % name}
    if name in names:
        return {"ok": False, "why": "이미 쓰는 이름입니다: %s" % name}
    if name in data:
        return {"ok": False, "why": "이미 더해 둔 소스입니다: %s (지우려면 --remove)" % name}
    res = test_stream(host, cam_port, path, timeout)
    if not res["ok"] and not force:
        return {"ok": False, "why": res["why"], "test": res}
    used = set(ports.values()) | {int(v.get("port") or 0) for v in data.values()}
    port = int(port) if port else free_port(used)
    if port is None:
        return {"ok": False, "why": "쓸 수 있는 포트가 없습니다 (%d~%d)" % (PORT_LOW, PORT_HIGH), "test": res}
    if port in used:
        return {"ok": False, "why": "그 포트는 이미 씁니다: %d" % port, "test": res}
    one = {"host": host, "port": port, "cam_port": cam_port, "path": path, "note": note}
    data[name] = one
    save(data)
    return {"ok": True, "why": "", "source": dict(one, name=name), "test": res, "file": str(CONFIG)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="소스를 시험해 보고 목록에 넣는다 (config/sources.json)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--test", action="store_true", help="연결만 시험한다 (저장하지 않음)")
    g.add_argument("--save", action="store_true", help="시험한 뒤 목록에 넣는다")
    g.add_argument("--list", action="store_true", help="더해 둔 소스 목록")
    g.add_argument("--remove", metavar="이름", help="더해 둔 소스를 뺀다")
    ap.add_argument("--name")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int, help="데이터 서버가 열 포트 (비우면 빈 것을 고른다)")
    ap.add_argument("--cam-port", type=int, default=CAM_PORT, help="그 기기에서 영상이 나오는 포트 (기본 8090)")
    ap.add_argument("--path", default="/stream/1")
    ap.add_argument("--note", default="")
    ap.add_argument("--force", action="store_true", help="시험에 실패해도 넣는다 (기기를 아직 안 켠 경우)")
    ap.add_argument("--timeout", type=float, default=6.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.list:
        data = load()
        print(json.dumps({"sources": data, "file": str(CONFIG)}, ensure_ascii=False) if a.json
              else ("\n".join("%-12s %s:%d%s  서버 포트 %d" % (k, v["host"], v.get("cam_port", CAM_PORT),
                                                              v.get("path", "/stream/1"), v["port"])
                              for k, v in sorted(data.items())) or "더해 둔 소스가 없습니다."))
        return 0
    if a.remove:
        data = load()
        if a.remove not in data:
            print(json.dumps({"ok": False, "why": "그런 소스가 없습니다"}, ensure_ascii=False) if a.json
                  else "그런 소스가 없습니다: %s" % a.remove)
            return 1
        data.pop(a.remove)
        save(data)
        print(json.dumps({"ok": True}, ensure_ascii=False) if a.json else "뺐습니다: %s" % a.remove)
        return 0
    if not a.host:
        raise SystemExit("--host 가 있어야 합니다")
    if a.test:
        res = test_stream(a.host, a.cam_port, a.path, a.timeout)
        print(json.dumps(res, ensure_ascii=False) if a.json
              else ("장 하나를 받았습니다: %d 바이트 %sx%s (%d ms)" % (res["jpeg"], res["width"], res["height"], res["ms"])
                    if res["ok"] else "안 됩니다: %s" % res["why"]))
        return 0 if res["ok"] else 1
    res = add(a.name, a.host, a.port, a.cam_port, a.path, a.note, a.force, a.timeout)
    print(json.dumps(res, ensure_ascii=False) if a.json
          else ("넣었습니다: %s (서버 포트 %d)" % (a.name, res["source"]["port"]) if res["ok"]
                else "못 넣었습니다: %s" % res["why"]))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
