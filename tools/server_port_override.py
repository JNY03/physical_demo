#!/usr/bin/env python3
"""server_port_override.py — 데이터 서버를 **고치지 않고** 소스마다 카메라 영상 포트를 바꿔 띄운다 (2026-11-26).

왜: server_stream_multi_source.py 는 카메라 파이의 영상을 http://<주소>:8090/stream/<n> 에서만 가져온다
(포트 8090 고정 · 주소 칸에 포트를 적으면 「포트는 쓰지 않습니다」 하고 버린다). 파이 없이 시험할 때
tools/replay_source.py --mode mjpeg 가 낸 http://127.0.0.1:18090/stream/1 을 진짜 서버가 끌어오게 하려고,
서버 모듈을 불러 robot_video_loop 만 감싼 뒤 서버 main() 을 그대로 돈다. 서버 인자는 전부 그대로 넘긴다.

    venv/bin/python tools/server_port_override.py --on pi12 --pi12 127.0.0.1:18090 --save /tmp/ds_replay
    venv/bin/python tools/server_port_override.py --video-port pi12=18090 --on pi12 --pi12 127.0.0.1 ...
    PD_VIDEO_PORT_PI12=18090 venv/bin/python tools/server_port_override.py --on pi12 --pi12 127.0.0.1 ...

포트 고르기 (앞이 이긴다): --video-port 소스=포트 > 소스 주소 칸의 :포트 (--pi12 127.0.0.1:18090) >
  환경변수 PD_VIDEO_PORT_<소스> (- 는 _) > 서버 기본 8090.
바뀌는 것은 MJPEG 로 끌어오는 소스(robot1 · realsense · a2 · pi11~13 · 사람이 더한 소스)뿐이다.
ep-001(H.264 웹소켓) · drone(롱폴) · cam360(밀어 넣기)은 replay_source.py --mode redis|upload 로 흉내 낸다.
서버가 시작할 때 찍는 「영상 ← http://…:8090/…」 줄은 고치기 전 값이다 — 실제로 붙는 주소는 그 아래
「[port-override] …」 줄과 10초 보고의 상태 칸(받는 중 http://…:18090/…)에 나온다.
"""
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parent.parent


def split_overrides(argv, sources):
    """argv → (서버에 넘길 argv, {소스: 포트}). --video-port 를 빼고 소스 주소 칸의 :포트 를 떼어 낸다."""
    out, ports, flag_ports = [], {}, {}
    flags = {"--" + s: s for s in sources}
    flags["--robot2"] = "ep-001"                     # 서버의 옛 이름
    i = 0
    while i < len(argv):
        x = argv[i]
        if x == "--video-port" or x.startswith("--video-port="):
            if "=" in x and x.startswith("--video-port="):
                val = x.split("=", 1)[1]
            else:
                i += 1
                if i >= len(argv):
                    raise SystemExit("--video-port 소스=포트")
                val = argv[i]
            name, _, port = val.partition("=")
            if not port.isdigit():
                raise SystemExit("--video-port 소스=포트 (받은 것 %r)" % val)
            flag_ports[name.strip()] = int(port)
            i += 1
            continue
        key, eq, inline = x.partition("=")
        if key in flags and (eq or i + 1 < len(argv)):
            val = inline if eq else argv[i + 1]
            u = urlsplit(val if "://" in val else "http://" + val)
            try:
                port = u.port
            except ValueError:
                port = None
            if port and u.hostname:
                ports[flags[key]] = port
                val = u.hostname
            out += [key, val]
            i += 1 if eq else 2
            continue
        out.append(x)
        i += 1
    for s in sources:
        v = os.environ.get("PD_VIDEO_PORT_" + s.upper().replace("-", "_"))
        if v and v.isdigit():
            ports.setdefault(s, int(v))
    ports.update(flag_ports)
    return out, ports


def patch(module, ports):
    """module.robot_video_loop 를 소스 이름으로 포트를 고르는 것으로 바꾼다 (main 이 부를 때 전역을 찾으므로 먹힌다)."""
    orig = module.robot_video_loop
    default = module.ROBOT_VIDEO_PORT

    def robot_video_loop(src, host, cam, port=default, path=None):
        if port == default and src.name in ports:
            port = ports[src.name]
            module.log("[port-override] %s 영상 ← http://%s:%d%s" % (src.name, host, port, path or "/stream/%d" % cam))
        return orig(src, host, cam, port, path)

    robot_video_loop.__wrapped__ = orig
    module.robot_video_loop = robot_video_loop
    return module


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    sys.path.insert(0, str(REPO))
    import server_stream_multi_source as ds
    srv_argv, ports = split_overrides(argv, list(ds.DEFAULT_HOST))
    patch(ds, ports)
    if ports:
        print("[port-override] 영상 포트: %s" % " · ".join("%s=%d" % kv for kv in sorted(ports.items())), flush=True)
    sys.argv = [str(REPO / "server_stream_multi_source.py")] + srv_argv
    return ds.main()


if __name__ == "__main__":
    sys.exit(main())
