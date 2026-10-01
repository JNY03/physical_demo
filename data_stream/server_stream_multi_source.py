#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server_stream_multi_source.py — 로봇·드론·360 카메라 데이터 스트림 (서버·엣지 공통)
================================================================================
2026-10-01 · 하천 실측 대비

하는 일
  소스(로봇 2대 · 드론 · 360 카메라)가 주는 영상과 상태를 받아서
    1) 이 파일을 실행한 기기의 Redis Stream 으로 흘린다  (AI 는 이 스트림을 읽는다)
    2) 같은 기기 디스크에 전부 파일로 저장한다
  받은 데이터는 손대지 않는다. 시각·번호 같은 정보는 보내는 쪽이 붙인 것을 그대로 옮기고,
  안 붙어 와도 그대로 흘린다. 이 파일이 붙이는 것은 저장 번호뿐이다.

포트 = 소스  (어느 경로로 들어오든 그 포트로 들어오면 그 소스다)
  10001 cam360  영상: 카메라 노트북의 stream_to_server.py 가 POST 로 보낸다 (/upload_pano)
  10002 drone   영상: pi3 :8890/api/frame 을 계속 받아 온다   상태: pi3 MQTT :1883 전체 구독
  10003 robot1  영상: pi7 :8090/stream/1 을 계속 읽는다        상태: pi7 MQTT :1883 전체 구독
  10004 robot2  영상: pi1 :8090/stream/1 을 계속 읽는다        상태: pi1 MQTT :1883 전체 구독

각 포트에 열리는 주소
  POST /upload_pano   360 업로드 (pano_receiver 와 같은 모양: 폼 칸 + file)
  POST /upload        server_stream_depth_v2 와 같은 모양 (file + camera_id 등)
  POST /push          다른 기기에서 도는 이 파일이 넘겨 준 영상·상태 묶음 (--forward)
  GET  /              원본 영상 + 비전 결과를 한 화면에
  GET  /stream        그 소스 영상을 브라우저로 본다 (MJPEG)
  GET  /vision?model=yoloe   비전 오버레이 (MJPEG). vision_infer.py live 가 Redis 에 넣은 것을 보여 준다
  GET  /health        받은 개수 · 연결 상태 · 비전 모델별 결과 수/지연 (JSON)

Redis Stream (최근 --maxlen 개만 유지, 기본 300)
  camera_stream:<소스>  image   = 받은 이미지 바이트 그대로
                        header  = 헤더 JSON (아래)
                        그 밖에  보내는 쪽이 준 정보 한 칸씩 (예: 360 의 capture_id · camera_id ·
                                seq · timestamp, 드론 meta 의 frame_seq 등)
  state_stream:<소스>   topic, payload = 받은 바이트 그대로, header = 헤더 JSON
  AI 기존 코드(unidepth_dual_yolo.py)는 Redis 주소와 스트림 이름만 바꾸면 읽는다 (image · timestamp 칸.
  timestamp 는 보내는 쪽이 붙여 줄 때만 있다).
  vision_stream:<소스>:<모델> · vision_depth:<소스>:<모델>   추론 결과 (아래 "추론 결과 스트림")
  Redis 메모리: 한도가 없으면 시작할 때 --redis-maxmemory (기본 4gb, allkeys-lru) 를 걸고,
               Redis 가 재시작해 한도가 풀리면 10초 보고 때 다시 건다. 이미 걸린 한도는 건드리지 않는다.
               전부 디스크에 저장하므로 Redis 는 최근 것만 들고 있으면 된다 (넘치면 오래 안 쓴 키부터 버림).

추론 결과 스트림 (vision)
  만드는 쪽  vision/vision_infer.py (기본 live 모드, GPU 기기에서 별도 프로세스)
             camera_stream:<소스> 에서 소스마다 가장 최근 프레임만 꺼내 추론한다. 추론하는 사이 들어온
             프레임은 건너뛴다 (실시간 우선, 건너뛴 프레임은 나중에 --runs <실행> 으로 디스크에서 처리).
             모델: yoloe (YOLOE prompt-free 검출·분할) · unidepth · moge2_aerial (metric depth, m)
             예) python vision/vision_infer.py                                  전 소스 yoloe (기본)
                 python vision/vision_infer.py --devices drone --models unidepth  드론에 depth 추가
  이 파일    vision_stream 을 읽어 /vision · / · /health · 10초 보고로 보여 주기만 한다 (쓰지 않는다).
             비전이 안 돌거나 못 읽어도 받기·저장·흘리기에는 영향이 없다. --no-vision 이면 읽지도 않는다.

  vision_stream:<소스>:<모델>   한 장 = 한 항목 (최근 300개, vision_config.yaml live.maxlen)
      result   JSON — 디스크 <실행>/vision/<모델>/results.jsonl 한 줄과 같은 내용
                 공통   n · frame("frames/<번호>.jpg") · source · run · model · image_wh · infer_ms
                        header (camera_stream 헤더의 n · file · via)
                 live   source_id = 원본 camera_stream:<소스> 항목 ID (이 값으로 원본 이미지를 찾는다)
                        recv_ms   = 원본이 Redis 에 들어간 시각 (ms, 항목 ID 의 앞부분)
                        done_ms   = 추론을 마친 시각 (ms)      lag_ms = done_ms - recv_ms
                 yoloe  detections: [{cls, name, conf, xyxy:[x1,y1,x2,y2], polygon:[[x,y],...]}]
                 depth  depth_stats {min, p5, median, p95, max, valid_ratio} (m) · intrinsics (픽셀 K 3x3)
                        fov_x_deg · fov_x_given · detections: [{cls, name, conf, xyxy, depth_m}]
                        (yoloe 와 같이 돌 때만 검출별 거리가 붙는다)
      overlay  오버레이 JPEG 바이트 (yoloe: 박스·마스크 / depth: 컬러맵 + 검출별 거리 + 색 막대)
      header   {"source", "model", "n", "run", "file", "source_id"}
  vision_depth:<소스>:<모델>    depth 모델만, 한 장 = 한 항목 (최근 30개, live.depth_maxlen — 한 장 약 1.8MB)
      depth    depth 원본 바이트 (미터, 0 = 무효 픽셀). 디스크 vision/<모델>/depth/<번호>.npy 와 같은 값
      dtype    "<f2" (float16)   shape  "H,W"   unit  "m"   header  vision_stream 과 같은 요약

  읽는 예 (파이썬)
      r = redis.Redis(port=6380)      # 다른 기기: redis.Redis(host="100.114.96.78", port=6380)
      eid, f = r.xrevrange("vision_stream:drone:yoloe", count=1)[0]          # 최신 결과
      res = json.loads(f[b"result"])                                          # 검출 목록 등
      img = r.xrange("camera_stream:drone", res["source_id"], res["source_id"])  # 그 결과의 원본 프레임
      eid, f = r.xrevrange("vision_depth:drone:unidepth", count=1)[0]
      depth = np.frombuffer(f[b"depth"], f[b"dtype"].decode()).reshape(
          [int(x) for x in f[b"shape"].split(b",")])                         # (H, W) 미터
      새 결과를 기다리며 받기: r.xread({"vision_stream:drone:yoloe": "$"}, block=1000)

  보기  GET /vision?model=<모델>   그 소스의 최신 오버레이 MJPEG (model 을 빼면 결과가 있는 첫 모델)
        GET /                     원본 + 결과가 있는 모델 오버레이를 한 화면에 (새 모델은 새로고침)
        GET /health  "vision": {모델: {results, n, run, lag_ms, infer_ms, detections, age_s}}
                     "vision_error": 결과 스트림을 못 읽을 때 이유
        10초 보고    "비전 yoloe 6.4장/초 지연 0.07s" (30초 넘게 결과가 없는 모델은 안 찍는다)

헤더 (영상 한 장마다)
  {"source": "robot1", "n": 15, "file": "robot1/20261001_123152/frames/000000015.jpg",
   "via": "pull:http://.../stream/1", "host": "<받은 기기 이름>",
   "sender": { 보내는 쪽이 준 정보 그대로 },
   "edge": { 엣지에서 넘어온 경우 엣지 쪽 번호·파일 } }

저장 (전부, 지우지 않는다)
  <--save>/<소스>/<실행 시작 YYYYMMDD_HHMMSS>/
      frames/
          000000001.jpg, 000000002.jpg, ...  받은 JPEG 바이트를 그대로 저장
      frames.jsonl                           영상 메타데이터 한 장 = 한 줄
      states.jsonl                           상태 한 건 = 한 줄
  - 실행할 때마다 새 폴더를 만든다. 같은 이름이 있으면 _2, _3 을 붙인다.
    파일은 새로 만들기만 하고 덮어쓰지 않는다.
  - frames.jsonl 의 file 필드는 같은 실행 폴더 아래 frames/<번호>.jpg 를 가리킨다.
  - --unpack 은 예전 .frm 저장본을 변환할 때만 남겨 둔 호환 기능이다.

실행 (같은 파일, 값만 다르게)
  말단 → 서버          python3 server_stream_multi_source.py
  말단 → 엣지(노트북)   python server_stream_multi_source.py
  말단 → 엣지 → 서버    노트북: ... --forward <서버 주소>      서버: ... --no-pull
  소스 골라 켜기       ... --on robot1 drone              (노트북 3대 폴백도 이걸로)
  현장 IP 로 바꾸기     ... --robot1 192.168.0.11 --drone 192.168.0.13
  360 카메라 노트북     stream_to_server.py --upload http://<이 파일이 도는 기기>:10001/upload_pano
                       (같은 노트북이면 http://127.0.0.1:10001/upload_pano)
  실시간 비전 (GPU 서버, 다른 터미널)
                       /home/dg/capstone-db/venv_image/bin/python vision/vision_infer.py   (기본: 전 기기 yoloe)

준비
  pip install redis paho-mqtt        (파이썬 3.8 이상)
  Redis 5.0 이상 (Stream 기능). 기본으로 이 기기 127.0.0.1:6380 의 전용 Redis 에 넣는다 (--redis 로 바꿈).
  tailnet 의 다른 기기도 구독하려면 Redis 를 tailscale 주소에도 묶는다 (띄우는 명령은 vision/vision_infer.py 맨 위).
    다른 기기에서: redis.Redis.from_url("redis://100.114.96.78:6380/0")   (100.114.96.78 = 이 PC tailscale 주소)
  Redis 나 paho-mqtt 가 없어도 멈추지 않는다. 저장은 계속하고 경고만 낸다.

주소 설정: 기기 주소(테일넷 IP)는 코드에 두지 않는다. hosts.example.json 을 hosts.local.json 으로
  복사해 채우거나 환경변수 PD_HOST_ROBOT1 / PD_HOST_ROBOT2 / PD_HOST_DRONE 을 쓴다.
  hosts.local.json 은 .gitignore 대상 — 공개 저장소에 올리지 않는다.
"""
import argparse
import base64
import http.client
import json
import os
import signal
import socket
import socketserver
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

try:
    import redis
except ImportError:
    redis = None

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


# ---------------------------------------------------------------- 고정값
SOURCES = {"cam360": 10001, "drone": 10002, "robot1": 10003, "robot2": 10004}
ALIASES = {"cam360": "cam360", "camera": "cam360", "360": "cam360",
           "drone": "drone", "robot1": "robot1", "robot2": "robot2"}
HOSTS_FILE = Path(__file__).resolve().parent / "hosts.local.json"   # 실제 주소는 여기에 (깃에 올리지 않음)


def load_default_hosts():
    """기본 주소: 환경변수 PD_HOST_<소스> > hosts.local.json > 없음. 명령행 --robot1 / --robot2 / --drone 이 가장 우선."""
    hosts = {"robot1": None,         # pi7 (go1-camview 8090 · mosquitto 1883)
             "robot2": None,         # pi1 (go1-camview 8090 · mosquitto 1883)
             "drone": None}          # pi3 (drone-agent 8890 · mosquitto 1883)
    if HOSTS_FILE.is_file():
        try:
            data = json.loads(HOSTS_FILE.read_text(encoding="utf-8"))
        except ValueError as e:
            raise SystemExit("%s 을 못 읽었습니다: %s" % (HOSTS_FILE.name, e))
        hosts.update((k, data[k]) for k in hosts if data.get(k))
    for k in hosts:
        v = os.environ.get("PD_HOST_" + k.upper())
        if v:
            hosts[k] = v
    return hosts


DEFAULT_HOST = load_default_hosts()
ROBOT_VIDEO_PORT = 8090      # go1_cam_view.py — GET /stream/<카메라 번호> (MJPEG, VZ 카드가 쓰는 주소)
DRONE_VIDEO_PORT = 8890      # 드론 agent    — GET /api/frame?cam=&since=&wait= (meta JSON + JPEG)
MQTT_PORT = 1883             # 파이 mosquitto (프로그램용 입구. 9001 은 같은 브로커의 브라우저용 입구)
DRONE_CAM = 0
DRONE_WAIT_S = 2             # 드론 롱폴: 새 장이 없으면 이만큼 기다렸다가 204 로 답한다
MAX_BODY = 64 * 1024 * 1024  # POST 한 번에 받는 최대 크기
FORWARD_QUEUE = 300          # 넘기기 대기열. 넘치면 오래된 것부터 버린다 (이 기기 디스크에는 남아 있다)
FORWARD_BATCH = 4 * 1024 * 1024   # 밀린 것을 한 번에 묶어 보내는 최대 크기
RESERVED = ("image", "header")

HOSTNAME = socket.gethostname()
STOP = threading.Event()


# ---------------------------------------------------------------- 출력
_warn_at = {}
_warn_lock = threading.Lock()
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:                       # 여러 스레드의 줄이 섞이지 않게
        sys.stdout.write(time.strftime("[%H:%M:%S] ") + msg + "\n")
        sys.stdout.flush()


def warn(key, msg, every=10.0):
    """같은 종류의 경고는 every 초에 한 번만 찍는다."""
    now = time.time()
    with _warn_lock:
        if now - _warn_at.get(key, 0.0) < every:
            return
        _warn_at[key] = now
    log(msg)


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return ("%d%s" % (n, unit)) if unit == "B" else ("%.1f%s" % (n, unit))
        n /= 1024.0


def describe(e):
    return ("%s: %s" % (type(e).__name__, e))[:200]


# ---------------------------------------------------------------- 영상 파일 형식 (.frm)
def pack_frame(header, image):
    """[헤더 길이 4바이트][헤더 JSON][이미지 바이트 그대로]"""
    h = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return len(h).to_bytes(4, "big") + h + image


def unpack_frame(data):
    if len(data) < 4:
        raise ValueError("너무 짧음")
    n = int.from_bytes(data[:4], "big")
    if 4 + n > len(data):
        raise ValueError("헤더 길이가 맞지 않음")
    return json.loads(data[4:4 + n].decode("utf-8")), data[4 + n:]


def image_ext(image):
    if image[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if image[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return ".bin"


# ---------------------------------------------------------------- multipart
def header_value(headers, name):
    name = name.lower()
    for k, v in headers.items():
        if k.lower() == name:
            return v
    return None


def content_type_param(ctype, name):
    """'multipart/form-data; boundary=xyz' → 'xyz'"""
    for piece in (ctype or "").split(";")[1:]:
        k, _, v = piece.strip().partition("=")
        if k.strip().lower() == name:
            return v.strip().strip('"')
    return None


def disposition(value):
    """'form-data; name="meta"; filename="a.jpg"' → {'name': 'meta', 'filename': 'a.jpg'}"""
    out = {}
    for piece in (value or "").split(";")[1:]:
        k, _, v = piece.strip().partition("=")
        out[k.strip().lower()] = v.strip().strip('"')
    return out


def parse_header_lines(block):
    headers = {}
    for line in block.split(b"\r\n"):
        k, sep, v = line.decode("utf-8", "replace").partition(":")
        if sep:
            headers[k.strip()] = v.strip()
    return headers


def split_multipart(body, boundary):
    """multipart 본문 → [(칸 머리 dict, 칸 내용 bytes)]"""
    delim = b"\r\n--" + boundary.encode("latin-1")
    pieces = (b"\r\n" + body).split(delim)
    parts = []
    for piece in pieces[1:]:
        if piece.startswith(b"--"):              # 끝 표시
            break
        nl = piece.find(b"\r\n")
        if nl < 0:
            continue
        rest = piece[nl + 2:]
        if rest.startswith(b"\r\n"):             # 머리 없는 칸
            parts.append(({}, rest[2:]))
            continue
        end = rest.find(b"\r\n\r\n")
        if end < 0:
            continue
        parts.append((parse_header_lines(rest[:end]), rest[end + 4:]))
    return parts


def form_to_frame(parts):
    """multipart 칸들 → (보내는 쪽 정보, 이미지 바이트). 이미지는 첫 파일 칸(또는 image/* 칸)."""
    sender, image = {}, None
    for headers, content in parts:
        disp = disposition(header_value(headers, "content-disposition"))
        name = disp.get("name") or ""
        ctype = header_value(headers, "content-type") or ""
        if image is None and ("filename" in disp or ctype.lower().startswith("image/")):
            image = content
            sender["_file"] = {"field": name, "filename": disp.get("filename"), "content_type": ctype or None}
            continue
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            sender[name] = {"base64": base64.b64encode(content).decode("ascii")}
            continue
        if ctype.lower().startswith("application/json"):
            try:
                value = json.loads(text)
            except ValueError:
                value = text
            if isinstance(value, dict):
                for k, v in value.items():       # 드론 meta 처럼 JSON 칸이면 안의 항목을 그대로 펼친다
                    sender.setdefault(str(k), v)
                continue
            sender[name] = value
        else:
            sender[name] = text
    return sender, image


# ---------------------------------------------------------------- Redis
def frame_fields(image, header):
    """Redis 항목: image, header, 그리고 보내는 쪽 정보 한 칸씩"""
    fields = {"image": image, "header": json.dumps(header, ensure_ascii=False)}
    for k, v in (header.get("sender") or {}).items():
        k = str(k)
        if not k or k in RESERVED:
            continue
        fields[k] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return fields


class RedisSink:
    """Redis 에 못 넣어도 예외를 올리지 않는다 (저장이 먼저다). 실패하면 2초 동안은 다시 시도하지 않는다."""

    def __init__(self, url, maxlen):
        self.url, self.maxlen = url, maxlen
        self.maxmemory = None                    # 걸어 둘 메모리 한도 (--redis-maxmemory). Redis 가 재시작해도 다시 건다
        self.client, self.err = None, None
        self.pause_until = 0.0
        if redis is None:
            self.err = "redis 패키지가 없음 (pip install redis)"
            return
        try:
            self.client = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=3)
        except Exception as e:
            self.err = describe(e)

    def ping(self):
        if self.client is None:
            return False
        try:
            self.client.ping()
            self.err = None
            return True
        except Exception as e:
            self.err = describe(e)
            return False

    def ensure_maxmemory(self, limit, policy="allkeys-lru"):
        """Redis 에 메모리 한도가 없으면 건다 (이미 걸려 있으면 그대로 둔다). 결과 문구를 돌려준다.
        전부 디스크에 저장하므로 Redis 는 최근 것만 있으면 된다 — 넘치면 오래 안 쓴 키부터 버린다.
        noeviction 은 쓰지 않는다: 한도에 닿으면 XADD 가 실패해 스트림이 줄지도 않고 멈춘다."""
        if self.client is None or str(limit).strip().lower() in ("", "0", "off", "none"):
            return "한도 안 건드림"
        self.maxmemory = limit
        try:
            cur = int(self.client.config_get("maxmemory").get("maxmemory", 0))
            if cur:
                return "한도 %s (이미 걸려 있음, %s)" % (human_bytes(cur),
                                                  self.client.config_get("maxmemory-policy").get("maxmemory-policy"))
            self.client.config_set("maxmemory", limit)
            self.client.config_set("maxmemory-policy", policy)
            return "한도 %s 걸었음 (%s)" % (limit, policy)
        except Exception as e:
            return "한도를 못 걸었음 (%s) — Redis 설정에서 maxmemory 를 직접 거세요" % describe(e)

    def memory(self):
        """(사용, 한도) 바이트. 못 읽으면 None"""
        if self.client is None or time.time() < self.pause_until:
            return None
        try:
            m = self.client.info("memory")
            return int(m.get("used_memory", 0)), int(m.get("maxmemory", 0))
        except Exception:
            return None

    def add(self, key, fields):
        if self.client is None:
            return False
        now = time.time()
        if now < self.pause_until:
            return False
        try:
            self.client.xadd(key, fields, maxlen=self.maxlen, approximate=True)
        except Exception as e:
            self.err = describe(e)
            self.pause_until = now + 2.0
            warn("redis", "Redis 에 못 넣음 (%s) — 파일 저장은 계속합니다" % self.err)
            return False
        if self.err is not None:
            self.err = None
            log("Redis 다시 됨 — 스트림 재개")
        return True


# ---------------------------------------------------------------- 비전 결과 보기 (vision_infer.py live 가 Redis 에 넣은 것)
class VisionView:
    """vision_stream:<소스>:<모델> 을 읽어 모델마다 최신 오버레이와 요약을 들고 있는다 (/vision · /health · 보고용).
    비전이 안 돌거나 Redis 가 없어도 받기·저장·흘리기에는 아무 영향이 없다 (따로 도는 스레드, 따로 연결)."""

    def __init__(self, url, names):
        self.url, self.names = url, list(names)
        self.client, self.err = None, None
        self.ids = {}                            # 읽는 키 → 마지막 항목 ID
        self.cv = threading.Condition()
        self.latest = {}                         # (소스, 모델) → (번호, JPEG)
        self.info = {}                           # (소스, 모델) → 요약
        self.seq = 0
        if redis is None:
            self.err = "redis 패키지가 없음"
            return
        threading.Thread(target=self._run, daemon=True).start()

    def _take(self, key, entries):
        _, src, model = key.split(":", 2)
        eid, f = entries[-1]
        try:
            res = json.loads(f.get(b"result") or b"{}")
        except ValueError:
            res = {}
        with self.cv:
            i = self.info.setdefault((src, model), {"results": 0})
            i["results"] += len(entries)
            i.update(n=res.get("n"), run=res.get("run"), lag_ms=res.get("lag_ms"), infer_ms=res.get("infer_ms"),
                     detections=len(res.get("detections") or []), at=time.time())
            jpg = f.get(b"overlay")
            if jpg:
                self.seq += 1
                self.latest[(src, model)] = (self.seq, jpg)
                self.cv.notify_all()

    def _run(self):
        next_scan = 0.0
        while not STOP.is_set():
            try:
                if self.client is None:
                    self.client = redis.Redis.from_url(self.url, socket_connect_timeout=2, socket_timeout=5)
                if time.time() >= next_scan:     # 새로 생긴 모델 스트림 찾기
                    for nm in self.names:
                        for k in self.client.scan_iter(match="vision_stream:%s:*" % nm, count=100):
                            k = k.decode()
                            if k not in self.ids:
                                last = self.client.xrevrange(k, "+", "-", count=1)
                                self.ids[k] = last[0][0] if last else "0"
                                if last:
                                    self._take(k, last)
                    next_scan = time.time() + 3.0
                if not self.ids:
                    STOP.wait(1.0)
                    continue
                for key, entries in self.client.xread(self.ids, block=1000) or []:
                    key = key.decode()
                    self.ids[key] = entries[-1][0]
                    self._take(key, entries)
                if self.err is not None:
                    self.err = None
                    log("비전 결과 다시 읽음")
            except Exception as e:
                self.err = describe(e)
                self.client = None
                warn("vision", "비전 결과를 못 읽음 (%s) — 받기·저장은 계속합니다" % self.err, every=30)
                STOP.wait(2.0)

    def models(self, src):
        with self.cv:
            return sorted(m for s, m in self.info if s == src)

    def wait_latest(self, src, model, last, timeout):
        with self.cv:
            cur = self.latest.get((src, model), (0, None))
            if cur[0] == last:
                self.cv.wait(timeout)
                cur = self.latest.get((src, model), (0, None))
            return cur

    def health(self, src):
        now = time.time()
        with self.cv:
            out = {}
            for (s, m), i in self.info.items():
                if s == src:
                    d = dict(i)
                    d["age_s"] = round(now - d.pop("at"), 1)
                    out[m] = d
            return out

    def counts(self):
        with self.cv:
            return dict((k, v["results"]) for k, v in self.info.items())


# ---------------------------------------------------------------- 넘기기 (엣지 → 서버)
def pack_records(items):
    """넘기기 묶음: ([종류 1바이트 F=영상 · S=상태][길이 4바이트][내용]) 반복"""
    return b"".join(kind + len(data).to_bytes(4, "big") + data for kind, data in items)


def unpack_records(body):
    i = 0
    while i < len(body):
        kind, n = body[i:i + 1], int.from_bytes(body[i + 1:i + 5], "big")
        data = body[i + 5:i + 5 + n]
        if len(data) != n:
            raise ValueError("묶음이 잘림")
        yield kind, data
        i += 5 + n


class Forwarder:
    """받은 것을 다른 기기의 같은 포트(POST /push)로 넘긴다.
    연결 하나를 열어 두고, 밀린 것은 한 번에 묶어 보낸다 — 한 장씩 주고받으면 왕복 지연만큼 느려진다."""

    def __init__(self, host, port):
        self.host, self.port = host, port
        self.q = deque()
        self.cv = threading.Condition()
        self.inflight = 0
        self.sent = self.dropped = self.rejected = 0
        self.err = None
        threading.Thread(target=self._run, daemon=True).start()

    def put(self, kind, data):
        with self.cv:
            self.q.append((kind, data))
            while len(self.q) > FORWARD_QUEUE:        # 넘치면 오래된 것부터 버린다 (이 기기 디스크에는 있다)
                self.q.popleft()
                self.dropped += 1
            self.cv.notify()

    def pending(self):
        with self.cv:
            return len(self.q) + self.inflight

    def _take(self):
        with self.cv:
            while not self.q and not STOP.is_set():
                self.cv.wait(1.0)
            batch, size = [], 0
            while self.q and (not batch or (len(batch) < 200 and size + len(self.q[0][1]) <= FORWARD_BATCH)):
                item = self.q.popleft()
                batch.append(item)
                size += len(item[1])
            self.inflight = len(batch)
            return batch

    def _give_back(self, batch):
        with self.cv:
            self.q.extendleft(reversed(batch))
            while len(self.q) > FORWARD_QUEUE:
                self.q.popleft()
                self.dropped += 1
            self.inflight = 0

    def _run(self):
        conn, fails = None, 0
        while not STOP.is_set():
            batch = self._take()
            if not batch:
                continue
            try:
                if conn is None or conn.sock is None:
                    conn = http.client.HTTPConnection(self.host, self.port, timeout=15)
                    conn.connect()
                    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.request("POST", "/push", body=pack_records(batch),
                             headers={"Content-Type": "application/octet-stream"})
                resp = conn.getresponse()
                answer = resp.read()
                status = resp.status
            except Exception as e:
                if conn is not None:
                    conn.close()
                conn = None
                fails += 1
                self.err = describe(e)
                warn("fwd-%d" % self.port, "넘기기 안 됨 → %s:%d (%s) — 이 기기 저장은 계속합니다"
                     % (self.host, self.port, self.err))
                self._give_back(batch)             # 다시 보낸다
                STOP.wait(min(5.0, 0.5 * fails))
                continue
            with self.cv:
                self.inflight = 0
            fails = 0
            if status == 200:
                self.sent += len(batch)
                if self.err is not None:
                    self.err = None
                    log("넘기기 다시 됨 → %s:%d" % (self.host, self.port))
            else:
                self.rejected += len(batch)
                warn("fwd-rej-%d" % self.port, "넘긴 곳이 거절함 %s:%d — HTTP %d %s"
                     % (self.host, self.port, status, answer[:200]))


# ---------------------------------------------------------------- 소스 (포트 하나)
class Source:
    def __init__(self, name, port, folder, rel, sink, forwarder):
        self.name, self.port = name, port
        self.dir, self.rel = folder, rel        # 저장 폴더 / 헤더에 적는 상대 경로
        self.sink, self.fwd = sink, forwarder
        self.lock = threading.Lock()
        self.n_frame = 0                        # 마지막으로 쓴 영상 번호
        self.n_state = 0
        self.got_frames = self.got_states = 0
        self.save_fail = self.redis_fail = 0
        self.status = {}
        self.frame_fh = None
        self.state_fh = None
        self.cv = threading.Condition()
        self.latest = (0, None)                 # /stream 으로 보여 줄 최신 한 장

    def _open_new(self):
        """다음 번호의 frames/<번호>.jpg 를 새로 만든다. 기존 파일은 절대 덮어쓰지 않는다."""
        frames_dir = self.dir / "frames"
        made_dir = False
        while True:
            self.n_frame += 1
            fname = "%09d.jpg" % self.n_frame
            try:
                return open(frames_dir / fname, "xb"), fname, None
            except FileExistsError:
                continue
            except FileNotFoundError as e:      # 실행 중에 폴더가 지워졌으면 다시 만든다
                if made_dir:
                    return None, fname, e
                made_dir = True
                self.n_frame -= 1
                try:
                    frames_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e2:
                    self.n_frame += 1
                    return None, fname, e2
            except OSError as e:
                return None, fname, e

    def add_frame(self, image, sender, via, edge=None):
        with self.lock:
            fh, fname, err = self._open_new()
            n = self.n_frame
            header = {"source": self.name, "n": n,
                      "file": self.rel + "/frames/" + fname,
                      "via": via, "host": HOSTNAME, "sender": sender}
            if edge:
                header["edge"] = edge

            image_saved = False
            meta_saved = False
            if fh is not None:
                try:
                    with fh:
                        fh.write(image)                 # 받은 JPEG 바이트를 그대로 저장
                    image_saved = True
                except OSError as e:
                    err = e
                    try:
                        os.remove(str(self.dir / "frames" / fname))
                    except OSError:
                        pass

            if image_saved:
                try:
                    if self.frame_fh is None:
                        self.dir.mkdir(parents=True, exist_ok=True)
                        self.frame_fh = open(self.dir / "frames.jsonl", "ab")
                    line = (json.dumps(header, ensure_ascii=False) + "\n").encode("utf-8")
                    self.frame_fh.write(line)
                    self.frame_fh.flush()
                    meta_saved = True
                except OSError as e:
                    err = e
                    try:
                        self.frame_fh.close()
                    except Exception:
                        pass
                    self.frame_fh = None

            saved = image_saved and meta_saved
            if not saved:
                self.save_fail += 1
                warn("save-" + self.name, "%s 영상 저장 실패 (%s) — 스트림은 계속합니다" % (self.name, describe(err)))
                if not image_saved:
                    header["file"] = None

            # 엣지→서버 전송 형식은 기존 .frm 패킹을 그대로 사용하되, 디스크에는 저장하지 않는다.
            packed = pack_frame(header, image)
            if not self.sink.add("camera_stream:" + self.name, frame_fields(image, header)):
                self.redis_fail += 1
            self.got_frames += 1
        with self.cv:
            self.latest = (n, image)
            self.cv.notify_all()
        if self.fwd is not None and not via.startswith("push"):   # 넘겨받은 것은 다시 넘기지 않는다 (고리 방지)
            self.fwd.put(b"F", packed)
        return {"status": "saved" if saved else "not_saved", "source": self.name, "n": n}

    def add_state(self, topic, payload, qos, retain, via, edge=None):
        with self.lock:
            self.n_state += 1
            rec = {"n": self.n_state, "topic": topic, "qos": qos, "retain": retain, "via": via, "host": HOSTNAME}
            if edge:
                rec["edge"] = edge
            try:
                rec["payload"], rec["enc"] = payload.decode("utf-8"), "utf8"
            except UnicodeDecodeError:
                rec["payload"], rec["enc"] = base64.b64encode(payload).decode("ascii"), "base64"
            line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
            saved = False
            try:
                if self.state_fh is None:
                    self.dir.mkdir(parents=True, exist_ok=True)
                    self.state_fh = open(self.dir / "states.jsonl", "ab")
                self.state_fh.write(line)
                self.state_fh.flush()
                saved = True
            except OSError as e:
                self.save_fail += 1
                warn("save-" + self.name, "%s 상태 저장 실패 (%s) — 스트림은 계속합니다" % (self.name, describe(e)))
                try:
                    self.state_fh.close()
                except Exception:
                    pass
                self.state_fh = None
            header = {"source": self.name}
            header.update((k, v) for k, v in rec.items() if k not in ("payload", "enc"))
            if not self.sink.add("state_stream:" + self.name,
                                 {"topic": topic, "payload": payload,
                                  "header": json.dumps(header, ensure_ascii=False)}):
                self.redis_fail += 1
            self.got_states += 1
        if self.fwd is not None and not via.startswith("push"):
            self.fwd.put(b"S", line)
        return {"status": "saved" if saved else "not_saved", "source": self.name, "n": rec["n"]}

    def wait_latest(self, last, timeout):
        with self.cv:
            if self.latest[0] == last:
                self.cv.wait(timeout)
            return self.latest

    def set_status(self, key, text):
        self.status[key] = text

    def close(self):
        with self.lock:
            if self.frame_fh is not None:
                try:
                    self.frame_fh.close()
                except Exception:
                    pass
                self.frame_fh = None
            if self.state_fh is not None:
                try:
                    self.state_fh.close()
                except Exception:
                    pass
                self.state_fh = None

    def health(self):
        h = {"source": self.name, "port": self.port, "host": HOSTNAME, "folder": str(self.dir),
             "frames": self.got_frames, "states": self.got_states,
             "save_fail": self.save_fail, "redis_fail": self.redis_fail,
             "redis_error": self.sink.err, "status": dict(self.status)}
        if self.fwd is not None:
            h["forward"] = {"to": "%s:%d" % (self.fwd.host, self.fwd.port), "sent": self.fwd.sent,
                            "pending": self.fwd.pending(), "dropped": self.fwd.dropped,
                            "rejected": self.fwd.rejected, "error": self.fwd.err}
        return h


# ---------------------------------------------------------------- 받는 쪽 (포트마다 HTTP)
def read_exact(f, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"     # 윈도우에서는 같은 포트를 두 프로그램이 잡게 되므로 끈다

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        socketserver.TCPServer.server_bind(self)  # HTTPServer 의 getfqdn(느릴 수 있음)을 건너뛴다
        host, port = self.server_address[:2]
        self.server_name, self.server_port = str(host), port


def make_handler(src, vision=None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "data-stream"
        sys_version = ""
        disable_nagle_algorithm = True          # 작은 응답이 40ms 씩 묶여 늦어지지 않게

        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            path, _, query = self.path.partition("?")
            path = path.rstrip("/") or "/"
            if path == "/stream":
                return self._mjpeg()
            if path == "/vision":
                return self._vision(parse_qs(query).get("model", [None])[0])
            if path == "/health":
                h = src.health()
                if vision is not None:
                    h["vision"] = vision.health(src.name)
                    h["vision_error"] = vision.err
                return self._json(200, h)
            if path == "/":
                models = vision.models(src.name) if vision is not None else []
                cells = ["<figure style='margin:4px'><figcaption>원본</figcaption>"
                         "<img src='/stream' style='max-width:100%%'></figure>"]
                cells += ["<figure style='margin:4px'><figcaption>%s</figcaption>"
                          "<img src='/vision?model=%s' style='max-width:100%%'></figure>" % (m, m) for m in models]
                note = "" if models else " · 비전 결과 없음 (vision_infer.py 가 돌면 새로고침)"
                page = ("<!doctype html><meta charset='utf-8'><title>%s</title>"
                        "<body style='margin:0;background:#111;color:#ddd;font-family:sans-serif'>"
                        "<p style='margin:8px'>%s :%d · <a style='color:#9cf' href='/health'>health</a>%s</p>"
                        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr))'>%s</div>"
                        "</body>" % (src.name, src.name, src.port, note, "".join(cells)))
                return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            self._json(404, {"status": "error", "detail": "없는 주소"})

        def _vision(self, model):
            """비전 오버레이 MJPEG. model 을 안 주면 결과가 있는 첫 모델"""
            if vision is None or vision.err == "redis 패키지가 없음":
                return self._json(503, {"status": "error", "detail": "Redis 를 못 써서 비전 결과를 못 읽음"})
            models = vision.models(src.name)
            model = model or (models[0] if models else None)
            if model is None:
                return self._json(404, {"status": "error", "source": src.name,
                                        "detail": "비전 결과 없음 — vision_infer.py 가 이 소스를 처리하고 있어야 함"})
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            last = -1
            try:
                while not STOP.is_set():
                    seq, img = vision.wait_latest(src.name, model, last, 1.0)
                    if img is None or seq == last:
                        continue
                    last = seq
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(img)).encode() + b"\r\n\r\n" + img + b"\r\n")
            except OSError:
                pass                              # 보던 쪽이 닫음

        def _mjpeg(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            last = -1
            try:
                while not STOP.is_set():
                    seq, img = src.wait_latest(last, 1.0)
                    if img is None or seq == last:
                        continue
                    last = seq
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(img)).encode() + b"\r\n\r\n" + img + b"\r\n")
            except OSError:
                pass                              # 보던 쪽이 닫음

        def do_POST(self):
            path = self.path.split("?", 1)[0].rstrip("/")
            length = self.headers.get("Content-Length")
            if length is None or not length.strip().isdigit():
                self.close_connection = True
                return self._json(411, {"status": "error", "detail": "Content-Length 필요"})
            length = int(length)
            if length > MAX_BODY:
                self.close_connection = True
                return self._json(413, {"status": "error", "detail": "너무 큼"})
            body = read_exact(self.rfile, length)
            if body is None:
                self.close_connection = True
                return
            try:
                if path in ("/upload_pano", "/upload"):
                    ctype = self.headers.get("Content-Type", "")
                    boundary = content_type_param(ctype, "boundary")
                    if not ctype.lower().startswith("multipart/") or not boundary:
                        return self._json(400, {"status": "error", "detail": "multipart/form-data 로 보내야 함"})
                    sender, image = form_to_frame(split_multipart(body, boundary))
                    if image is None:
                        return self._json(400, {"status": "error", "detail": "파일 칸이 없음"})
                    key = "video" if src.name == "cam360" else "upload"   # 로봇·드론 포트는 가져오기 상태를 덮지 않는다
                    text = "받는 중 (%s ← %s)" % (path, self.client_address[0])
                    if src.status.get(key) != text:
                        src.set_status(key, text)
                        log("%s 영상 받는 중 — %s 로 %s 에서 올라옴" % (src.name, path, self.client_address[0]))
                    res = src.add_frame(image, sender, "upload:" + path)
                elif path == "/push":                 # 다른 기기의 이 파일이 넘겨 준 묶음
                    res = {"status": "saved", "source": src.name, "frames": 0, "states": 0, "bad": 0}
                    for kind, data in unpack_records(body):
                        try:
                            if kind == b"F":
                                header, image = unpack_frame(data)
                                edge = dict((k, header.get(k)) for k in ("n", "file", "via", "host"))
                                src.add_frame(image, header.get("sender") or {}, "push", edge=edge)
                                res["frames"] += 1
                            elif kind == b"S":
                                rec = json.loads(data.decode("utf-8"))
                                raw = rec.get("payload") or ""
                                payload = base64.b64decode(raw) if rec.get("enc") == "base64" else raw.encode("utf-8")
                                edge = dict((k, rec.get(k)) for k in ("n", "via", "host"))
                                src.add_state(rec.get("topic", ""), payload, rec.get("qos"), rec.get("retain"),
                                              "push", edge=edge)
                                res["states"] += 1
                            else:
                                res["bad"] += 1
                        except Exception as e:
                            res["bad"] += 1
                            warn("push-bad-" + src.name, "%s 넘겨받은 것 중 못 읽은 것 (%s)" % (src.name, describe(e)))
                else:
                    return self._json(404, {"status": "error", "detail": "없는 주소"})
            except Exception as e:
                return self._json(400, {"status": "error", "detail": describe(e)})
            self._json(200, res)

    return Handler


# ---------------------------------------------------------------- 가져오는 쪽: 로봇 영상 (MJPEG)
def read_mjpeg(resp, boundary):
    """multipart/x-mixed-replace 를 한 장씩 (칸 머리, 바이트) 로 낸다. 칸 머리에 길이가 없어도 읽는다."""
    b = boundary.encode("latin-1")
    markers = (b"--" + b, b) if b.startswith(b"--") else (b"--" + b,)

    def is_marker(line):
        s = line.strip()
        return any(s.startswith(m) for m in markers)

    while True:                                   # 첫 경계까지 건너뛴다
        line = resp.readline()
        if not line:
            return
        if is_marker(line):
            break
    while not STOP.is_set():
        if any(line.strip() == m + b"--" for m in markers):
            return                                # 끝 표시
        headers = {}
        while True:
            h = resp.readline()
            if not h:
                return
            h = h.rstrip(b"\r\n")
            if not h:
                break
            k, sep, v = h.decode("latin-1").partition(":")
            if sep:
                headers[k.strip()] = v.strip()
        length = header_value(headers, "content-length")
        if length is not None and length.strip().isdigit():
            data = read_exact(resp, int(length))
            if data is None:
                return
            yield headers, data
            while True:                           # 다음 경계 줄까지
                line = resp.readline()
                if not line:
                    return
                if is_marker(line):
                    break
        else:
            chunks = []
            while True:
                line = resp.readline()
                if not line:
                    return
                if is_marker(line):
                    break
                chunks.append(line)
            data = b"".join(chunks)
            if data.endswith(b"\r\n"):            # 경계 앞의 줄바꿈은 경계에 속한다
                data = data[:-2]
            elif data.endswith(b"\n"):
                data = data[:-1]
            yield headers, data


def robot_video_loop(src, host, cam):
    path = "/stream/%d" % cam
    url = "http://%s:%d%s" % (host, ROBOT_VIDEO_PORT, path)
    fails, receiving = 0, False
    src.set_status("video", "붙는 중 " + url)
    while not STOP.is_set():
        conn = None
        try:
            conn = http.client.HTTPConnection(host, ROBOT_VIDEO_PORT, timeout=10)
            conn.request("GET", path)
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError("HTTP %d" % resp.status)
            ctype = resp.getheader("Content-Type", "")
            boundary = content_type_param(ctype, "boundary")
            if not boundary:
                raise RuntimeError("MJPEG 가 아님: " + ctype)
            for part_headers, data in read_mjpeg(resp, boundary):
                if STOP.is_set():
                    break
                if not receiving:
                    receiving, fails = True, 0
                    src.set_status("video", "받는 중 " + url)
                    log("%s 영상 받는 중 — %s" % (src.name, url))
                src.add_frame(data, part_headers, "pull:" + url)
            if not STOP.is_set():
                raise RuntimeError("영상 스트림이 끝남")
        except Exception as e:
            if STOP.is_set():
                break
            receiving = False
            fails += 1
            src.set_status("video", "끊김 (%s) — 다시 붙는 중" % describe(e))
            warn(src.name + "-video", "%s 영상 안 됨 %s (%s) — 다시 붙는 중" % (src.name, url, describe(e)))
            STOP.wait(min(5.0, 0.5 * fails))
        finally:
            if conn is not None:
                conn.close()


# ---------------------------------------------------------------- 가져오는 쪽: 드론 영상 (/api/frame 롱폴)
def drone_video_loop(src, host, cam):
    base = "http://%s:%d" % (host, DRONE_VIDEO_PORT)
    via = "pull:%s/api/frame?cam=%d" % (base, cam)
    since, last_seq, last_img = 0, None, None
    fails, receiving = 0, False
    conn = None
    src.set_status("video", "붙는 중 " + base)
    while not STOP.is_set():
        path = "/api/frame?cam=%d&since=%d&wait=%d" % (cam, since, DRONE_WAIT_S)
        try:
            if conn is None:                       # 연결은 열어 두고 계속 쓴다 (매번 새로 열면 느려진다)
                conn = http.client.HTTPConnection(host, DRONE_VIDEO_PORT, timeout=DRONE_WAIT_S + 8)
            conn.request("GET", path)
            resp = conn.getresponse()
            body = resp.read()
            if resp.status == 204:                 # 기다리는 동안 새 장이 없었다
                fails = 0
                since = 0                          # 다음엔 최신 장을 그냥 묻는다 — 드론이 다시 켜져 번호가
                continue                           # 처음부터 시작해도 받게 (같은 장이면 아래에서 거른다)
            if resp.status != 200:
                raise RuntimeError("HTTP %d" % resp.status)
            boundary = content_type_param(resp.getheader("Content-Type", ""), "boundary")
            if not boundary:
                raise RuntimeError("multipart 가 아님: %s" % resp.getheader("Content-Type", ""))
            sender, image = form_to_frame(split_multipart(body, boundary))
            if image is None:
                raise RuntimeError("응답에 이미지 칸이 없음")
            fails = 0
            try:
                seq = int(sender.get("frame_seq"))
            except (TypeError, ValueError):
                seq = None
            if seq is not None:
                if seq == last_seq:                # since 를 0 으로 되돌렸을 때 받은 같은 장
                    since = seq
                    continue
                since = last_seq = seq
            elif image == last_img:                # 번호가 없으면 같은 바이트만 거른다
                STOP.wait(0.05)
                continue
            last_img = image
            if not receiving:
                receiving = True
                src.set_status("video", "받는 중 " + base)
                log("%s 영상 받는 중 — %s/api/frame?cam=%d" % (src.name, base, cam))
            src.add_frame(image, sender, via)
        except Exception as e:
            if conn is not None:
                conn.close()
            conn = None
            if STOP.is_set():
                break
            receiving = False
            fails += 1
            src.set_status("video", "끊김 (%s) — 다시 붙는 중" % describe(e))
            warn(src.name + "-video", "%s 영상 안 됨 %s (%s) — 다시 붙는 중" % (src.name, base, describe(e)))
            STOP.wait(min(5.0, 0.5 * fails))
    if conn is not None:
        conn.close()


# ---------------------------------------------------------------- 가져오는 쪽: 상태 (파이 MQTT 전체 구독)
def _rc_failed(rc):
    failed = getattr(rc, "is_failure", None)
    if failed is not None:
        return bool(failed)
    return rc != 0


def start_mqtt(src, host):
    url = "mqtt://%s:%d" % (host, MQTT_PORT)
    if mqtt is None:
        src.set_status("state", "paho-mqtt 패키지 없음 (pip install paho-mqtt)")
        log("%s 상태: paho-mqtt 가 없어 상태는 받지 않습니다 (pip install paho-mqtt)" % src.name)
        return None
    cid = "stream-%s-%s-%d" % (HOSTNAME, src.name, os.getpid())
    if hasattr(mqtt, "CallbackAPIVersion"):       # paho-mqtt 2.x
        cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid, clean_session=True)
    else:                                         # paho-mqtt 1.x
        cli = mqtt.Client(client_id=cid, clean_session=True)
    via = "pull:" + url

    def on_connect(client, userdata, flags, rc, *rest):
        try:
            if _rc_failed(rc):
                src.set_status("state", "접속 거부 (%s)" % rc)
                warn(src.name + "-mqtt", "%s 상태 %s 접속 거부 (%s)" % (src.name, url, rc))
                return
            client.subscribe("#", qos=1)
            src.set_status("state", "받는 중 " + url)
            log("%s 상태 받는 중 — %s 전체 토픽(#)" % (src.name, url))
        except Exception as e:
            log("%s 상태 구독 오류 (%s)" % (src.name, describe(e)))

    def on_disconnect(client, userdata, *rest):
        if not STOP.is_set():
            src.set_status("state", "끊김 — 다시 붙는 중")
            warn(src.name + "-mqtt", "%s 상태 %s 끊김 — 다시 붙는 중" % (src.name, url))

    def on_message(client, userdata, msg):
        try:
            src.add_state(msg.topic, bytes(msg.payload), msg.qos, bool(msg.retain), via)
        except Exception as e:
            warn(src.name + "-mqtt-msg", "%s 상태 처리 오류 (%s)" % (src.name, describe(e)))

    cli.on_connect = on_connect
    cli.on_disconnect = on_disconnect
    cli.on_message = on_message
    cli.reconnect_delay_set(min_delay=1, max_delay=10)
    src.set_status("state", "붙는 중 " + url)
    try:
        cli.connect_async(host, MQTT_PORT, keepalive=30)
        cli.loop_start()
    except Exception as e:
        src.set_status("state", "시작 실패 (%s)" % describe(e))
        log("%s 상태 %s 시작 실패 (%s)" % (src.name, url, describe(e)))
        return None
    return cli


# ---------------------------------------------------------------- 저장 폴더
def make_run_dirs(save_root, names):
    """<저장폴더>/<소스>/<YYYYMMDD_HHMMSS>/ — 이미 있으면 _2, _3 … (켜진 소스 모두 같은 이름)"""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    k = 1
    while True:
        run = stamp if k == 1 else "%s_%d" % (stamp, k)
        dirs = dict((nm, save_root / nm / run) for nm in names)
        if not any(d.exists() for d in dirs.values()):
            try:
                for d in dirs.values():
                    d.mkdir(parents=True, exist_ok=False)
                return run, dirs
            except FileExistsError:
                pass
        k += 1


def unpack(folder):
    """.frm → 같은 이름의 이미지 + .json (원본은 그대로 두고 <폴더>_unpacked 에 쓴다)"""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        print("폴더가 아닙니다: %s" % root)
        return 1
    out_root = root.parent / (root.name + "_unpacked")
    done = skipped = bad = 0
    for p in sorted(root.rglob("*.frm")):
        out_dir = out_root / p.parent.relative_to(root)
        try:
            header, image = unpack_frame(p.read_bytes())
        except Exception as e:
            bad += 1
            print("  읽기 실패 %s (%s)" % (p, describe(e)))
            continue
        img_path = out_dir / (p.stem + image_ext(image))
        json_path = out_dir / (p.stem + ".json")
        if img_path.exists() or json_path.exists():
            skipped += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(image)
        json_path.write_text(json.dumps(header, ensure_ascii=False, indent=1), encoding="utf-8")
        done += 1
    print("풀기 끝 — %d장 → %s  (이미 있어서 건너뜀 %d, 읽기 실패 %d)" % (done, out_root, skipped, bad))
    return 0


# ---------------------------------------------------------------- 실행
def clean_host(text, what):
    s = text.strip()
    u = urlsplit(s if "://" in s else "http://" + s)
    if not u.hostname:
        raise SystemExit("%s 주소를 못 읽었습니다: %r" % (what, text))
    try:
        has_port = u.port is not None
    except ValueError:
        has_port = True
    if has_port:
        print("참고: %s 에 적은 포트는 쓰지 않습니다 (포트는 정해져 있음) — 주소만 씁니다: %s" % (what, u.hostname))
    return u.hostname


def report_loop(sources, sink, vision=None, every=10.0):
    prev = dict((s.name, (s.got_frames, s.got_states)) for s in sources)
    prev_v = vision.counts() if vision is not None else {}
    t_prev = time.time()
    while not STOP.wait(every):
        now = time.time()
        dt, t_prev = max(now - t_prev, 1e-6), now
        lines = []
        for s in sources:
            f0, s0 = prev[s.name]
            prev[s.name] = (s.got_frames, s.got_states)
            line = "  %-6s 영상 %5.1f장/초 (누적 %d)" % (s.name, (s.got_frames - f0) / dt, s.got_frames)
            if s.name != "cam360":                  # 360 은 상태가 없다
                line += "  상태 %5.1f건/초 (누적 %d)" % ((s.got_states - s0) / dt, s.got_states)
            extra = []
            if s.save_fail:
                extra.append("저장 실패 %d" % s.save_fail)
            if s.redis_fail:
                extra.append("Redis 못 넣음 %d" % s.redis_fail)
            if s.fwd is not None:
                extra.append("넘김 %d · 대기 %d · 버림 %d" % (s.fwd.sent, s.fwd.pending(), s.fwd.dropped))
            if extra:
                line += "  [" + " · ".join(extra) + "]"
            lines.append(line)
            if vision is not None:                  # vision_infer.py live 결과
                vh = vision.health(s.name)
                parts = []
                for m in sorted(vh):
                    v = vh[m]
                    if v["age_s"] > 30:
                        continue                    # 멈춘 모델은 안 찍는다 (/health 에는 남음)
                    rate = (v["results"] - prev_v.get((s.name, m), 0)) / dt
                    parts.append("%s %.1f장/초 지연 %s" % (m, rate, "%.2fs" % (v["lag_ms"] / 1000.0)
                                                         if v.get("lag_ms") is not None else "-"))
                if parts:
                    lines.append("  %-6s 비전 %s" % ("", " · ".join(parts)))
        if vision is not None:
            prev_v = vision.counts()
        mem = sink.memory()
        if mem and not mem[1] and sink.maxmemory:     # Redis 가 재시작하면 한도가 풀린다 → 다시 건다
            log("Redis 메모리 " + sink.ensure_maxmemory(sink.maxmemory))
            mem = sink.memory()
        if mem:
            lines.append("  Redis  %s / %s" % (human_bytes(mem[0]), human_bytes(mem[1]) if mem[1] else "한도 없음"))
        log("받은 양\n" + "\n".join(lines))


def main():
    try:
        sys.stdout.reconfigure(errors="replace")     # 윈도우 콘솔에서 못 찍는 글자 때문에 멈추지 않게
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        description="로봇·드론·360 카메라 데이터를 받아 Redis Stream 으로 흘리고 전부 파일로 저장한다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="포트: cam360 10001 · drone 10002 · robot1 10003 · robot2 10004\n"
               "자세한 설명은 파일 맨 위 주석.")
    ap.add_argument("--on", nargs="+", metavar="소스",
                    help="켤 소스만 고른다: cam360 drone robot1 robot2 (안 주면 전부)")
    ap.add_argument("--robot1", default=DEFAULT_HOST["robot1"], metavar="주소",
                    help="로봇1 파이(pi7) 주소. 기본 hosts.local.json / PD_HOST_* 값")
    ap.add_argument("--robot2", default=DEFAULT_HOST["robot2"], metavar="주소",
                    help="로봇2 파이(pi1) 주소. 기본 hosts.local.json / PD_HOST_* 값")
    ap.add_argument("--drone", default=DEFAULT_HOST["drone"], metavar="주소",
                    help="드론 파이(pi3) 주소. 기본 hosts.local.json / PD_HOST_* 값")
    ap.add_argument("--robot-cam", type=int, default=1, choices=range(1, 6), metavar="번호",
                    help="로봇 카메라: 1 정면 · 2 턱 · 3 왼쪽 · 4 오른쪽 · 5 배 (기본 1)")
    ap.add_argument("--save", default=str(Path(__file__).resolve().parent / "stream_data"), metavar="폴더",
                    help="저장 폴더 (기본: 이 파일 옆 stream_data)")
    ap.add_argument("--redis", default="redis://127.0.0.1:6380/0", metavar="주소",
                    help="흘려 넣을 Redis (기본 %(default)s)")
    ap.add_argument("--maxlen", type=int, default=300, help="Redis 스트림에 남길 개수 (기본 300)")
    ap.add_argument("--redis-maxmemory", default="4gb", metavar="크기",
                    help="Redis 에 메모리 한도가 없으면 시작할 때 건다 (기본 %(default)s, 넘치면 오래 안 쓴 키부터 버림). "
                         "0 = 안 건드림. 이미 걸려 있으면 그대로 둔다")
    ap.add_argument("--no-vision", action="store_true",
                    help="vision_infer.py 결과(vision_stream:*)를 읽지 않는다 (/vision 끔)")
    ap.add_argument("--forward", metavar="서버주소",
                    help="받은 것을 그 기기의 같은 포트로도 넘긴다 (말단 → 엣지 → 서버)")
    ap.add_argument("--no-pull", action="store_true",
                    help="파이에서 가져오지 않고 포트로 들어오는 것만 받는다 (엣지가 넘겨 줄 때 서버 쪽)")
    ap.add_argument("--bind", default="0.0.0.0", metavar="주소", help="포트를 열 주소 (기본 0.0.0.0 = 전부)")
    ap.add_argument("--unpack", metavar="폴더", help="저장된 .frm 을 .jpg + .json 으로 풀고 끝낸다")
    args = ap.parse_args()

    if args.unpack:
        return unpack(args.unpack)

    if args.on:
        names = []
        for x in args.on:
            nm = ALIASES.get(x.strip().lower())
            if nm is None:
                raise SystemExit("모르는 소스: %s  (cam360 · drone · robot1 · robot2)" % x)
            if nm not in names:
                names.append(nm)
        names.sort(key=lambda nm: SOURCES[nm])
    else:
        names = sorted(SOURCES, key=lambda nm: SOURCES[nm])
    hosts = {}
    for k in DEFAULT_HOST:
        if getattr(args, k):
            hosts[k] = clean_host(getattr(args, k), "--" + k)
        elif k in names and not args.no_pull:
            raise SystemExit("%s 주소가 없습니다: --%s 로 주거나 hosts.local.json / PD_HOST_%s 에 적으세요"
                             % (k, k, k.upper()))
    fwd_host = clean_host(args.forward, "--forward") if args.forward else None

    sink = RedisSink(args.redis, args.maxlen)
    redis_ok = sink.ping()
    redis_mem = sink.ensure_maxmemory(args.redis_maxmemory) if redis_ok else None
    if not redis_ok and str(args.redis_maxmemory).strip().lower() not in ("", "0", "off", "none"):
        sink.maxmemory = args.redis_maxmemory      # 나중에 붙으면 보고 때 건다
    sources = [Source(nm, SOURCES[nm], None, None, sink, None) for nm in names]

    vision = None if args.no_vision else VisionView(args.redis, names)
    servers = []                                   # 포트부터 잡는다 — 못 잡으면 폴더를 만들기 전에 끝낸다
    for s in sources:
        try:
            servers.append(Server((args.bind, s.port), make_handler(s, vision)))
        except OSError as e:
            for srv in servers:
                srv.server_close()
            raise SystemExit("포트 %d (%s) 를 못 엽니다: %s\n이미 이 파일이 떠 있거나 다른 프로그램이 쓰고 있습니다."
                             % (s.port, s.name, describe(e)))

    save_root = Path(args.save).expanduser().resolve()
    try:
        run, dirs = make_run_dirs(save_root, names)
    except OSError as e:
        raise SystemExit("저장 폴더를 못 만듭니다: %s (%s)" % (save_root, describe(e)))
    for s in sources:
        s.dir, s.rel = dirs[s.name], s.name + "/" + run
        (s.dir / "frames").mkdir(parents=True, exist_ok=True)
        (s.dir / "frames.jsonl").touch(exist_ok=True)
        (s.dir / "states.jsonl").touch(exist_ok=True)
        if fwd_host:
            s.fwd = Forwarder(fwd_host, s.port)
    for srv in servers:
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True).start()

    try:
        signal.signal(signal.SIGTERM, lambda *a: STOP.set())
    except (ValueError, AttributeError, OSError):
        pass

    bar = "=" * 72
    print(bar)
    print("데이터 스트림 — 실행 %s · 이 기기 %s" % (run, HOSTNAME))
    print("  저장   %s" % (save_root / "<소스>" / run))
    print("  Redis  %s  %s" % (args.redis, ("연결됨 · " + redis_mem) if redis_ok
                                else "안 됨 — 저장만 합니다 (%s)" % sink.err))
    if fwd_host:
        print("  넘기기 %s 의 같은 포트로 (/push)" % fwd_host)
    print("  " + "-" * 70)

    pulls = []                                     # 요약을 다 찍은 뒤에 시작한다
    for s in sources:
        if s.name == "cam360":
            s.set_status("video", "기다림 (stream_to_server.py 가 보내야 옴)")
            print("  cam360 :%d  영상 ← stream_to_server.py --upload http://<이 기기>:%d/upload_pano"
                  % (s.port, s.port))
        elif args.no_pull:
            s.set_status("video", "기다림 (--no-pull: 넘겨 오는 것만 받음)")
            print("  %-6s :%d  받기만 (--no-pull)" % (s.name, s.port))
        else:
            host = hosts[s.name]
            if s.name == "drone":
                video = "http://%s:%d/api/frame?cam=%d" % (host, DRONE_VIDEO_PORT, DRONE_CAM)
                pulls.append((drone_video_loop, (s, host, DRONE_CAM), s, host))
            else:
                video = "http://%s:%d/stream/%d" % (host, ROBOT_VIDEO_PORT, args.robot_cam)
                pulls.append((robot_video_loop, (s, host, args.robot_cam), s, host))
            print("  %-6s :%d  영상 ← %s" % (s.name, s.port, video))
            print("  %-6s %6s  상태 ← mqtt://%s:%d (전체 토픽)" % ("", "", host, MQTT_PORT))
    print("  " + "-" * 70)
    print("  보기   http://<이 기기>:<포트>/  (원본 + 비전)  ·  /stream  ·  /vision?model=<모델>  ·  /health")
    print("Ctrl+C 로 끝냅니다. 10초마다 받은 양을 찍습니다.")
    print(bar, flush=True)

    clients = []
    for loop, loop_args, s, host in pulls:
        threading.Thread(target=loop, args=loop_args, daemon=True).start()
        clients.append(start_mqtt(s, host))
    threading.Thread(target=report_loop, args=(sources, sink, vision), daemon=True).start()
    try:
        while not STOP.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    STOP.set()
    print("\n끝내는 중…", flush=True)
    for cli in clients:
        if cli is not None:
            try:
                cli.disconnect()
                cli.loop_stop()
            except Exception:
                pass
    for srv in servers:
        try:
            srv.shutdown()
            srv.server_close()
        except Exception:
            pass
    print(bar)
    print("끝 — 실행 %s" % run)
    for s in sources:
        s.close()
        tail = ""
        if s.fwd is not None:
            tail = " · 넘김 %d (못 넘기고 버림 %d, 남은 대기 %d)" % (s.fwd.sent, s.fwd.dropped, s.fwd.pending())
        print("  %-6s 영상 %d장 · 상태 %d건%s  → %s" % (s.name, s.got_frames, s.got_states, tail, s.dir))
    print(bar, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
