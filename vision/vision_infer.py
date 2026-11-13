#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision_infer.py — 스트림 프레임에 비전 모델 추론 (YOLOE · UniDepth · MoGe2-Aerial)
================================================================================
2026-10-01
2026-11-11 · 기기에 cam360_slice(DS 포트 10006 — e2p/slice360_live.py 가 자른 360 슬라이스)를 더했다.
  슬라이스는 90도 원근 사진이라 vision_config.yaml devices 의 cam360_slice: {fov_x_deg: 90.0} 을 depth 모델이 쓴다.
2026-10-14 · 통합데이터규약 v2 (DS-06) — 결과를 파일·Redis 로 낼 때만 v2 비전 결과(규약 §3-6)로 바꾼다(v2_record).
  안쪽 계산(오버레이 · 거리 결합 · 지리 계산)은 옛 칸 이름(cls · name · conf · xyxy · image_wh · frame · source)을 그대로 쓴다.
  그 장의 기록(frames.jsonl 한 줄 = Redis header, sender 포함)을 ModelOutput.write(…, header=) 로 넘겨
  공통 머리 · frame_ref · timestamp 를 채운다. 읽는 쪽은 새 이름을 먼저, 없으면 옛 이름(frame_file → file, frame_file → frame).

두 가지 모드 (--runs 로 고른다)
  live (기본)  server_stream_multi_source.py 가 흘리는 Redis camera_stream:<기기> 에서
               기기마다 가장 최근 프레임만 꺼내 추론하고, 결과를 바로 Redis 로 흘리면서 디스크에도 저장한다.
               추론하는 사이에 들어온 프레임은 건너뛴다 (실시간 우선).
  저장된 이미지  --runs all | latest | <실행 폴더>
               <data_root>/<기기>/<실행 YYYYMMDD_HHMMSS>/frames/<번호>.jpg 를 전부 처리한다.
               live 가 건너뛴 프레임을 나중에 채우거나, 다른 모델을 뒤늦게 적용할 때 쓴다.
  원본 frames/ · frames.jsonl · states.jsonl 은 읽기만 한다.

모델 (--models, 여러 개 동시에 가능)
  yoloe          YOLOE 검출+분할. 기본 prompt-free (내장 어휘 4585종, exclude/include 로 조정)
                 config models.yoloe.prompt.mode: text 면 classes 에 적은 클래스만 텍스트 프롬프트로 찾는다
  river          하천 떠내려가는 물체 5종. **학습 없이** 제로샷 모델의 어휘·임계값으로만.
                 models.river.prompts 는 클래스당 문구를 여럿 받고 **나가는 이름은 클래스 하나**다.
                 임계값 세 단계: arm_conf(arm 하한) · prompt_conf(문구별, 병합 전) · class_conf(클래스별, 병합 후).
                 arms 가 여럿이면 동의(min_arms)도 요구한다 — 정확하지만 느리다. 없는 arm 은 빼고 돈다
  unidepth       UniDepth V2 metric depth (m)
  moge2_aerial   MoGe2-Aerial — MoGe-2 ViT-L + 항공 LoRA (AerialMetric, ECCV 2026), metric depth (m)
  depth 모델을 고르면 depth 결과에 검출별 거리가 붙는다 (config fusion. yoloe 를 안 골라도 결합용으로 적재).
  config geo.devices 에 있는 기기(드론)는 검출별 GPS 좌표도 붙는다 (geo.py — 위치·자세는 MQTT state).

Redis 출력 (live)
  vision_stream:<기기>:<모델>   (최근 live.maxlen 개)
      result   JSON — results.jsonl 한 줄과 같음 (v2 비전 결과)
      overlay  오버레이 JPEG 바이트
      header   {"stream","model","n","run","frame_file","stream_entry_id","lag_ms"}
               stream_entry_id = 원본 camera_stream 항목 ID · lag_ms = 결과가 나온 순간 - 원본이 Redis 에 들어간 순간 (ms)
  vision_depth:<기기>:<모델>    depth 모델만 (최근 live.depth_maxlen 개)
      depth    depth 원본 바이트 (미터, 0 = 무효)   dtype ("<f2" 등) · shape ("H,W") · unit ("m") · header
      읽기: np.frombuffer(f[b"depth"], f[b"dtype"].decode()).reshape(map(int, f[b"shape"].split(b",")))
  server_stream_multi_source.py 의 GET /vision?model=<모델> 로 오버레이를 브라우저에서 본다.

저장 (두 모드 같은 형식)
  <실행 폴더>/vision/<모델>/
      results.jsonl       한 장 = 한 줄
      overlay/<번호>.jpg   오버레이 시각화
      depth/<번호>.npy     depth 모델만. 미터 단위 (float16 기본), 0 = 무효 픽셀
  - 이미 results.jsonl 에 있는 번호는 건너뛴다 (중간에 끊어도 이어서 한다).
    --overwrite 를 주면 그 모델 결과를 처음부터 다시 쓴다.
  - 여러 터미널에서 동시에 돌려도 된다. 실행 폴더·모델마다 잠금(<모델>/.lock)을 잡아서
    같은 모델은 한 프로세스만 디스크에 쓰고, 다른 모델은 각자 처리한다.
    live 가 쓰는 중인 실행 폴더·모델은 저장된 이미지 처리에서 건너뛴다 (live 를 끈 뒤 돌리면 채워진다).

results.jsonl 한 줄 (v2 비전 결과, 규약 §3-6)
  머리   schema_version "2.0", source_id "ds-vision", node_id (이 기기 이름), zone_id, timestamp (원본 프레임 시각),
         frame_ref {camera_id, sequence_id, timestamp} (보낸 쪽이 v2 일 때), emitted_at (결과가 나온 순간),
         producer {tier "server", kind "precise"}
  공통   n, frame_file ("frames/<번호>.jpg" — 실행 폴더 기준), stream, run, model, model_file, width, height,
         infer_ms, overlay_file, live (live 일 때)
  river  arms, min_arms_declared, min_arms_applied, missing_arms, arm_ms, min_prompts,
         prompts_declared, class_conf, prompt_conf, prompt_dropped, detections: 위와 같고
         검출마다 agree(동의한 arm 수)·arms(arm 이름)·prompts(말한 문구들) 추가.
         polygon 은 없다 (arm 이 마스크를 내지 않는다)
  yoloe  prompt (free | text), detections: [{class_id, label, confidence, bbox:[x1,y1,x2,y2], polygon?,
         track_id, held?, label_raw?}]   (config stabilize — 박스 EMA · 이름 다수결 · held = 못 잡은 장을 마지막 박스로 유지)
         depth 결합 검출은 depth_m 이 최근 몇 장의 중앙값, depth_m_raw 가 그 장 값
  depth  depth_file, depth_stats{min,p5,median,p95,max,valid_ratio}, intrinsics(픽셀 K), fov_x_deg,
         fov_x_given, detections?: [{class_id, label, confidence, bbox, depth_m, uv}]
         geo 기기면  pose{timestamp, lat, lon, fix_type, amsl_m, relative_m, roll/pitch/yaw_deg, dt_s ...},
                    geo_status (ok | no_fix | no_attitude | no_state | stale_state),
                    detections 마다 range_m, cam_xyz_m, ned_m[북,동,아래], lat, lon (도), alt_m (해발)
  옛 결과 파일(frame · source · image_wh · cls/name/conf/xyxy …)도 이어 하기에서 읽는다 (frame_file 이 없으면 frame).
  오버레이에 쓸 항목은 config overlay (박스 라벨 줄 · 제목 · 드론 위치 줄)

실행 순서 (저장소 루트에서. 2026-10-01 이 PC 에서 실측 확인)
  # 0) 전용 Redis :6380. WSL 을 재시작하면 꺼지므로 그때마다 다시 띄운다
  #    이 PC 의 서버·비전은 127.0.0.1 로, tailnet 의 다른 기기는 100.114.96.78(이 PC tailscale 주소)로 구독한다
  #    비밀번호가 없어 protected-mode 를 끈다 — tailscale0 에만 6380 을 열었다 (ufw). 인터넷 쪽에는 열지 말 것
  redis-server --port 6380 --bind "127.0.0.1 100.114.96.78" --protected-mode no --save "" --appendonly no --daemonize yes
  #    다른 기기에서 읽기: redis.Redis.from_url("redis://100.114.96.78:6380/0")

  # 1) 터미널 1 — 스트림 서버 (로봇·드론 주소는 hosts.local.json)
  source venv/bin/activate
  python server_stream_multi_source.py

  # 2) 터미널 2 — 실시간 비전
  source venv/bin/activate
  python vision/vision_infer.py

모델·기기를 골라 실행 (venv 활성화 상태)
  python vision/vision_infer.py                                     live · 전 기기 · yoloe (config 기본)
  python vision/vision_infer.py --devices drone --models unidepth   live · 드론에 unidepth (다른 터미널에서 추가)
  python vision/vision_infer.py --devices drone robot1 --models yoloe unidepth moge2_aerial
  python vision/vision_infer.py --devices drone --models moge2_aerial --runs 20261001_062842   저장된 이미지
  python vision/vision_infer.py --devices all --models yoloe --runs latest --watch              저장되는 대로 계속
  python vision/vision_infer.py --list                              기기·실행·프레임 수만 본다

준비
  YOLOE    가중치 경로만 맞으면 된다 (prompt-free 는 텍스트 인코더 불필요).
  UniDepth repo_path 를 sys.path 에 넣어 쓴다 (pip 설치 불필요).
  MoGe2-Aerial  external/AerialMetric/ 에 MoGe 코드 · 가중치(Moge2-Aerial.pt) · pydeps(utils3d, peft) 를 둔다.
               공유 venv 는 건드리지 않는다.
  live     config data_root 가 서버의 --save 폴더와 같아야 디스크에 저장된다 (다르면 Redis 로만 낸다).
          Redis 메모리 한도는 서버가 시작할 때 건다 (--redis-maxmemory). 전부 디스크에 있으므로 Redis 는 최근 것만.
"""
import argparse
import contextlib
import fcntl
import json
import signal
import socket
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geo import DiskStates, GeoLocator, frame_ts, redis_lookup, to_epoch  # noqa: E402
DEVICES = ("cam360", "drone", "robot1", "robot2", "realsense", "cam360_slice",   # cam360_slice 2026-11-11 (DS 10006)
           "a2",                                                                 # a2 2026-11-12 (DS 10007, Unitree A2)
           "pi11", "pi12")                                                        # 모듈 카메라 파이 2026-11-12 (DS 10008·10009)
MODELS = ("yoloe", "river", "unidepth", "moge2_aerial")
DEVICE_ALIASES = {"camera": "cam360", "360": "cam360",
                  "pi6": "realsense", "rs": "realsense", "d455": "realsense",
                  "slice": "cam360_slice"}
MODEL_ALIASES = {"yoloe-pf": "yoloe", "yoloe_pf": "yoloe",
                 "moge": "moge2_aerial", "moge2": "moge2_aerial", "moge2-aerial": "moge2_aerial",
                 "moge2aerial": "moge2_aerial",
                 "river-zeroshot": "river", "river_zeroshot": "river", "하천": "river"}
DEPTH_MODELS = ("unidepth", "moge2_aerial")
#: 검출 모델. depth 결합(fusion)은 이 중 **먼저 오는 하나**의 검출을 쓴다 —
#: 둘을 같이 돌리면 결과는 둘 다 저장되지만 거리·GPS 는 한쪽에만 붙는다.
DETECT_MODELS = ("yoloe", "river")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
COLORMAPS = {"turbo": cv2.COLORMAP_TURBO, "jet": cv2.COLORMAP_JET, "inferno": cv2.COLORMAP_INFERNO,
             "magma": cv2.COLORMAP_MAGMA, "viridis": cv2.COLORMAP_VIRIDIS, "plasma": cv2.COLORMAP_PLASMA}

STOP = False
HOSTNAME = socket.gethostname()
KST = timezone(timedelta(hours=9))


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


# ---------------------------------------------------------------- v2 비전 결과 (규약 §3-6 · DS-06)
def iso_ms(t=None):
    """epoch 초 → '2026-10-12T14:10:31.086+09:00'. 밀리초 아래는 버린다 (규약 §2-2)."""
    return datetime.fromtimestamp(time.time() if t is None else t, KST).isoformat(timespec="milliseconds")


DET_RENAME = {"xyxy": "bbox", "name": "label", "cls": "class_id", "conf": "confidence", "name_raw": "label_raw"}
V2_DROP = ("header", "source_id", "recv_ms", "done_ms", "lag_ms")   # 결과에서 뺀다 (P1 · R9 — lag_ms 는 Redis header 로)


def frame_time(s):
    """보낸 쪽 sender → 결과 머리 timestamp(ISO). v2 프레임 헤더면 그 글자 그대로(R12 — frame_ref.timestamp 와 같다).
    옛 기록이면 같은 뜻의 옛 칸(360 폼 timestamp(epoch 글자) · 드론 meta ts · captured_at · observed_at)을 바꿔 쓴다(§3-8).
    GO1 의 옛 X-Timestamp 는 「보낸 순간」이라 쓰지 않는다(D7-1)."""
    v = s.get("timestamp")
    if isinstance(v, str):
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            pass
    for k in ("timestamp", "ts", "captured_at", "observed_at"):
        t = to_epoch(s.get(k))
        if t is not None:
            return iso_ms(t)
    return None


def v2_record(rec, header):
    """안쪽 결과 사전 → v2 비전 결과 (§3-6). header = Redis/frames.jsonl 의 그 장 기록(sender 포함). rec 은 바꾸지 않는다."""
    s = (header or {}).get("sender") or {}
    if not isinstance(s, dict):
        s = {}
    ts = frame_time(s)
    out = {"schema_version": "2.0", "source_id": "ds-vision", "node_id": HOSTNAME,
           "zone_id": s.get("zone_id"), "timestamp": ts}
    cam, seq = s.get("camera_id"), s.get("sequence_id")
    if isinstance(cam, str) and cam and isinstance(seq, int) and not isinstance(seq, bool):
        out["frame_ref"] = {"camera_id": cam, "sequence_id": seq, "timestamp": ts}
    out["emitted_at"] = rec.get("_emitted_at")                 # 두 번 불려도 같게 — pop 하지 않는다
    out["producer"] = {"tier": "server", "kind": "precise"}
    for k, v in rec.items():
        if k.startswith("_") or k in V2_DROP:
            continue
        if k == "image_wh":
            out["width"], out["height"] = v
        elif k in ("weight", "checkpoint"):
            out["model_file"] = v
        elif k == "source":
            out["stream"] = v
        elif k == "frame":
            out["frame_file"] = v
        elif k == "frame_ts":
            continue                                           # = timestamp
        elif k == "pose" and isinstance(v, dict):
            p = dict(v)
            t = p.pop("ts", None)
            out["pose"] = dict({"timestamp": iso_ms(t) if isinstance(t, (int, float)) else t}, **p)
        elif k == "detections":
            out["detections"] = [{DET_RENAME.get(dk, dk): dv for dk, dv in d.items() if not str(dk).startswith("_")}
                                 for d in (v or [])]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------- 입력 (저장 폴더)
def list_runs(data_root, device):
    d = data_root / device
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_dir() and (p / "frames").is_dir())


def pick_runs(runs, want):
    if want == ["all"]:
        return runs
    if want == ["latest"]:
        # 프레임이 있는 마지막 실행. 하나도 없으면 그냥 마지막 실행
        with_frames = [r for r in runs if any((r / "frames").iterdir())]
        return (with_frames or runs)[-1:]
    names = set(want)
    return [r for r in runs if r.name in names]


def read_frame_list(run):
    """frames.jsonl 에 적힌 장만 처리한다 (이미지를 다 쓴 뒤 적히므로 쓰는 중인 파일을 피한다).
    frames.jsonl 이 비어 있으면 frames/ 폴더를 직접 훑는다."""
    items = []
    seen = set()
    jl = run / "frames.jsonl"
    if jl.exists():
        with open(jl, "rb") as f:
            for line in f:
                if not line.endswith(b"\n"):
                    break                       # 쓰는 중인 마지막 줄
                try:
                    h = json.loads(line)
                except ValueError:
                    continue
                ff = h.get("frame_file") or h.get("file")      # v2 DS 기록 · 옛 DS 기록
                if not ff:
                    continue
                name = Path(ff).name
                p = run / "frames" / name
                if name not in seen and p.exists():
                    seen.add(name)
                    items.append((p, h))
    if not items:
        items = [(p, None) for p in sorted((run / "frames").iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return items


def frame_number(path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def fov_hint(dev_cfg, header):
    v = (dev_cfg or {}).get("fov_x_deg")
    if v == "meta":
        try:
            return float(header["sender"]["camera"]["hfov_deg"])
        except (TypeError, KeyError, ValueError):
            return None
    return float(v) if v is not None else None


# ---------------------------------------------------------------- 출력
class ModelOutput:
    """<실행>/vision/<모델>/ — results.jsonl + overlay/ (+ depth/)"""

    def __init__(self, run, dirname, model, overwrite, jpeg_quality):
        self.dir = run / dirname / model
        self.overlay_dir = self.dir / "overlay"
        self.depth_dir = self.dir / "depth"
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        if model in DEPTH_MODELS:
            self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.jpeg = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        self.results = self.dir / "results.jsonl"
        self.done = set()
        if overwrite:
            self.results.write_bytes(b"")
        elif self.results.exists():
            line = b""
            with open(self.results, "rb") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        k = r.get("frame_file") or r.get("frame")   # v2 결과 · 옛 결과
                    except (ValueError, AttributeError):
                        continue
                    if k:
                        self.done.add(k)
            if line and not line.endswith(b"\n"):
                with open(self.results, "ab") as f:  # 강제 종료로 끊긴 마지막 줄 뒤에 이어 붙지 않게
                    f.write(b"\n")
        self.fh = open(self.results, "ab")

    def write(self, rec, overlay=None, depth=None, depth_dtype="float16", overlay_jpg=None, header=None):
        """overlay_jpg 를 주면 (live 에서 Redis 용으로 이미 인코딩한 JPEG) 그 바이트를 그대로 쓴다.
        header = 그 장의 기록(frames.jsonl 한 줄 = Redis header). 파일에는 v2_record(rec, header) 를 쓴다."""
        stem = Path(rec["frame"]).stem
        if overlay_jpg is not None:
            (self.overlay_dir / (stem + ".jpg")).write_bytes(overlay_jpg)
            rec["overlay_file"] = "overlay/%s.jpg" % stem
        elif overlay is not None:
            cv2.imwrite(str(self.overlay_dir / (stem + ".jpg")), overlay, self.jpeg)
            rec["overlay_file"] = "overlay/%s.jpg" % stem
        if depth is not None:
            np.save(self.depth_dir / (stem + ".npy"), depth.astype(depth_dtype))
            rec["depth_file"] = "depth/%s.npy" % stem
        self.fh.write((json.dumps(v2_record(rec, header), ensure_ascii=False) + "\n").encode("utf-8"))
        self.fh.flush()
        self.done.add(rec["frame"])

    def close(self):
        self.fh.close()


# ---------------------------------------------------------------- 모델: YOLOE (prompt-free / 텍스트 프롬프트)
class YoloeRunner:
    """models.yoloe.prompt.mode
         free  prompt-free 가중치(weight)의 내장 어휘 4585종. include 를 주면 그 클래스만, exclude 는 뺀다
               (predict 의 classes= 로 넘겨 NMS 전에 거른다 — 빠진 클래스가 max_det 를 차지하지 않는다)
         text  text_weight(일반 YOLOE 가중치)에 classes 를 텍스트 프롬프트로 넣는다.
               MobileCLIP 텍스트 인코더(mobileclip_blt.ts)로 시작할 때 한 번만 임베딩을 만들고, 추론 중에는 안 쓴다"""
    name = "yoloe"

    def __init__(self, cfg, device):
        from ultralytics import YOLOE
        self.cfg = cfg
        self.device = device
        pr = cfg.get("prompt") or {}
        self.mode = str(pr.get("mode", "free")).lower()
        self.keep = None                                 # predict classes= (클래스 번호). None = 전부
        if self.mode == "text":
            classes = [str(c).strip() for c in pr.get("classes") or [] if str(c).strip()]
            if not classes:
                raise SystemExit("models.yoloe.prompt.mode 가 text 인데 classes 가 비었습니다")
            self.weight = pr.get("text_weight") or cfg["weight"]
            if Path(self.weight).stem.endswith("-pf"):
                raise SystemExit("텍스트 프롬프트에는 prompt-free(-pf) 가 아닌 가중치가 필요합니다 (prompt.text_weight)")
            self.model = YOLOE(self.weight)
            # ultralytics 는 텍스트 인코더를 작업 폴더나 weights_dir 에서 찾는다 (없으면 받는다 ~570MB)
            with contextlib.chdir(pr.get("text_encoder_dir") or Path(self.weight).parent):
                pe = self.model.get_text_pe(classes)
            self.model.set_classes(classes, pe)
            log("  yoloe 텍스트 프롬프트 %d종: %s" % (len(classes), ", ".join(classes)))
        elif self.mode == "free":
            self.weight = cfg["weight"]
            self.model = YOLOE(self.weight)              # 내장 어휘 그대로, set_classes 안 함
            names = self.model.names
            by_name = {}
            for i, n in names.items():
                by_name.setdefault(str(n).lower(), []).append(int(i))

            def ids(key):
                out = []
                for n in pr.get(key) or []:
                    found = by_name.get(str(n).strip().lower())
                    if found:
                        out += found
                    else:
                        log("  yoloe prompt.%s: 내장 어휘에 없는 이름 '%s' — 무시" % (key, n))
                return set(out)
            include, exclude = ids("include"), ids("exclude")
            heads = {str(h).strip().lower() for h in pr.get("exclude_heads") or []}
            n_head = 0
            if heads:                                    # 마지막 단어(중심 명사)로 뺀다: dance floor·night view 는 빠지고 floor fan 은 남는다
                by_head = {int(i) for i, n in names.items() if str(n).split()[-1].lower() in heads}
                n_head = len(by_head - exclude)
                exclude |= by_head
            if include or exclude:
                self.keep = sorted((include or set(int(i) for i in names)) - exclude)
                if not self.keep:
                    raise SystemExit("models.yoloe.prompt: include/exclude 로 남는 클래스가 없습니다")
            log("  yoloe prompt-free %d종%s%s" % (len(self.keep) if self.keep else len(names),
                                                 " · include %d" % len(include) if include else "",
                                                 " · exclude %d (중심 명사 %d)" % (len(exclude), n_head) if exclude else ""))
        else:
            raise SystemExit("models.yoloe.prompt.mode 는 free 또는 text: %s" % self.mode)
        self.half = bool(cfg.get("half", True)) and device.startswith("cuda")
        self.max_area = cfg.get("max_area_ratio")       # 박스가 화면의 이 비율보다 크면 버린다 (전체 화면 검출)
        self.area_exempt = {str(n).lower() for n in cfg.get("area_exempt") or []}
        self(np.zeros((480, 640, 3), np.uint8))  # 첫 장 지연을 미리 치른다

    def __call__(self, bgr):
        c = self.cfg
        t0 = time.perf_counter()
        r = self.model.predict(bgr, conf=c.get("conf", 0.25), iou=c.get("iou", 0.5),
                               imgsz=c.get("imgsz", 640), max_det=c.get("max_det", 100), classes=self.keep,
                               device=self.device, quantize=16 if self.half else None, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        dets = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            polys = r.masks.xy if r.masks is not None else [None] * len(cls)
            img_area = float(bgr.shape[0] * bgr.shape[1])
            for b, s, k, poly in zip(xyxy, conf, cls, polys):
                if (self.max_area and (b[2] - b[0]) * (b[3] - b[1]) > float(self.max_area) * img_area
                        and str(r.names[int(k)]).lower() not in self.area_exempt):
                    continue
                d = {"cls": int(k), "name": r.names[int(k)], "conf": round(float(s), 4),
                     "xyxy": [round(float(v), 1) for v in b]}
                if poly is not None and len(poly):
                    d["_poly"] = poly                       # 결합용 (저장 전에 뺀다)
                    if c.get("save_polygons", True):
                        d["polygon"] = np.round(poly, 1).tolist()
                dets.append(d)
        return {"infer_ms": round(ms, 1), "weight": Path(self.weight).name, "prompt": self.mode}, dets


# ---------------------------------------------------------------- 검출 안정화 (config stabilize)
def box_iou(a, b):
    """a (N,4) · b (M,4) xyxy → (N,M)"""
    a, b = np.asarray(a, np.float64).reshape(-1, 4), np.asarray(b, np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


#: `LabelStabilizer` 가 받는 값. 어느 트래커의 것도 아니다.
LABEL_KEYS = ("label_policy", "label_margin", "label_max_missed", "label_iou_match")

#: `Stabilizer` 가 받는 값. 기성 트래커 설정에 섞여 있어도 거절하지 않기 위해 둔다.
BUILTIN_KEYS = ("iou_match", "center_dist", "box_alpha", "min_hits", "hold_missed",
                "max_missed", "label_decay", "depth_window")


class VendorTracker:
    """ultralytics 에 들어 있는 트래커를 그대로 쓴다 — 우리가 다시 구현하지 않는다.

    8.4.155 기준 여섯이 **이미 설치되어 있다**(bytetrack · botsort · ocsort ·
    deepocsort · fasttrack · tracktrack). 가중치를 받지 않는다 (AI-C-16); 기본값
    yaml 도 패키지 안의 것을 읽는다. `with_reid` 는 따로 켜야 하고 그때는 외모
    인코더가 필요하므로 폐쇄망에서 기본으로 두지 않는다.

    왜 우리 `Stabilizer` 대신인가: 이것들은 **잘 잡힌 검출에서만 track 을 시작한다**
    (`new_track_thresh`). 약한 검출은 이미 있는 track 을 **되찾는 데만** 쓰고
    (`track_low_thresh` — ByteTrack 의 요점이다) 새 물체로 만들지 않는다. 우리
    Stabilizer 는 conf 0.05 짜리 하나하나에 track 을 만들어서, 쓰레기를 이어 붙인
    뒤 그것을 **더 오래** 띄웠다.

    박스는 기본적으로 **탐지기 것을 그대로** 낸다(`keep_boxes: detection`).
    칼만이 낸 박스를 쓰면 수십 px 짜리가 반 박자 늦어 IoU 0.5 문턱을 넘지 못하는 것을
    쟀다(macro 0.290 → 0.082). 트래커는 **번호를 붙이는 데** 쓴다.
    """

    #: 우리처럼 **밖에서 만든 검출**을 먹일 수 있는 것만 둔다. `tracktrack` 은
    #: NMS 이전의 원시 예측을 predictor 훅으로 가져가므로 이 경로로 쓸 수 없다.
    BACKENDS = ("bytetrack", "botsort", "ocsort", "deepocsort", "fasttrack")

    def __init__(self, sc):
        backend = str(sc.get("backend") or "").lower()
        if backend not in self.BACKENDS:
            raise SystemExit("stabilize backend 를 모릅니다: %s (가능: builtin, %s)"
                             % (backend, ", ".join(self.BACKENDS)))
        try:
            import yaml as _yaml
            from ultralytics.trackers.track import TRACKER_MAP
            from ultralytics.utils import IterableSimpleNamespace
            import ultralytics
        except Exception as exc:                       # 설치가 없으면 트래킹만 끈다
            raise _ArmMissing("ultralytics 트래커를 쓸 수 없다 (%s)" % type(exc).__name__) from exc
        try:
            import lap                                 # noqa: F401  연관(linear assignment)에 쓴다
        except ImportError as exc:
            # ultralytics 는 없으면 **런타임에 받는다**. 폐쇄망에서 조용히 나가는
            # 연결이 생기므로 먼저 거절한다 (AI-C-16). `pip install lap` 이 선행이다
            raise _ArmMissing("lap 이 없다 — ultralytics 가 런타임에 받으려 한다. "
                              "폐쇄망에서는 미리 설치한다 (pip install 'lap>=0.5.12')") from exc
        spec = Path(ultralytics.__file__).parent / "cfg" / "trackers" / ("%s.yaml" % backend)
        if not spec.is_file():
            raise _ArmMissing("트래커 기본값이 없다: %s" % spec)
        args = _yaml.safe_load(spec.read_text(encoding="utf-8"))
        # 모르는 값은 거절한다(오타가 조용히 무시되면 안 된다). 다만 **다른 backend 의
        # 값**은 거절하지 않는다 — 한 설정 블록에 builtin 용 값과 기성 트래커 용 값이
        # 같이 적혀 있고, backend 를 바꿔 가며 쓸 수 있어야 한다. 쓰이지 않는다는 것만
        # 알린다.
        # 이름 안정화(LabelStabilizer)의 값은 트래커의 것이 아니다 — 같은 블록에 적히므로
        # 거절하지 않는다. backend 를 바꿔도 이름 정책은 그대로 돈다.
        shared = (("backend", "enabled", "keep_boxes", "reset_gap_s", "on_untracked")
                  + LABEL_KEYS)
        ignored = sorted(k for k in sc if k in BUILTIN_KEYS and k not in args and k not in shared)
        unknown = [k for k in sc if k not in args and k not in shared and k not in BUILTIN_KEYS]
        if unknown:
            raise SystemExit("stabilize.%s 가 모르는 값: %s (이 트래커가 받는 것: %s · "
                             "builtin 이 받는 것: %s)"
                             % (backend, ", ".join(sorted(unknown)), ", ".join(sorted(args)),
                                ", ".join(sorted(BUILTIN_KEYS))))
        if ignored:
            log("  %s 는 다음 값을 쓰지 않습니다 (backend: builtin 용): %s"
                % (backend, ", ".join(ignored)))
        args.update({k: v for k, v in sc.items() if k in args})
        if args.get("with_reid"):
            # 외모 인코더를 받아야 할 수 있다. 폐쇄망에서 조용히 나가지 않는다 —
            # 쓰려면 사람이 자리를 적어 넣는다
            if str(args.get("model", "auto")) == "auto":
                log("  ** %s with_reid 가 켜져 있고 model 이 auto 입니다 — 탐지기 특징을 "
                    "씁니다. 외부 가중치를 받지 않는지 확인하십시오 (AI-C-16) **" % backend)
        self.backend = backend
        # 트래커가 잇지 못한 검출을 어떻게 할 것인가. **기본은 남긴다.**
        # 기성 트래커는 `new_track_thresh`(0.25) 위의 검출로만 track 을 시작한다 —
        # 우리 작동점(conf 0.05)에서 그 조건을 넘는 것은 1/8 뿐이고, 못 넘은 것을
        # 버리면 재현율이 0.640 → 0.024 로 무너진다(쟀다). 그래서 트래킹은
        # **거르는 층이 아니라 번호를 붙이는 층**이다: 잘 잡힌 것은 번호를 받고
        # 나머지는 번호 없이 그대로 나간다. 이러면 탐지 점수가 구조적으로 안 바뀐다.
        self.on_untracked = str(sc.get("on_untracked", "keep")).lower()
        if self.on_untracked not in ("keep", "drop"):
            raise SystemExit("stabilize.on_untracked 는 keep 또는 drop: %s" % self.on_untracked)
        self.keep_boxes = str(sc.get("keep_boxes", "detection")).lower()
        if self.keep_boxes not in ("detection", "tracker"):
            raise SystemExit("stabilize.keep_boxes 는 detection 또는 tracker: %s" % self.keep_boxes)
        self.reset_gap = float(sc.get("reset_gap_s", 2.0))
        self._make = lambda: TRACKER_MAP[backend](IterableSimpleNamespace(**dict(args, device="cpu")))
        self.tracker = self._make()
        self.args = args
        self.last_ts = None

    class _Boxes:
        """ultralytics 트래커가 보는 모양.

        `conf`·`cls`·`xywh`·`xyxy` 네 칸과 **불리언 마스크 인덱싱**이 필요하다 —
        2단 연관이 `results[mask_high]` / `results[mask_low]` 로 검출을 가른다.
        """

        def __init__(self, dets=None, arrays=None):
            if arrays is not None:
                self.xyxy, self.conf, self.cls, self.xywh = arrays
                return
            dets = dets or []
            self.xyxy = np.asarray([d["xyxy"] for d in dets], np.float32).reshape(-1, 4)
            self.conf = np.asarray([d["conf"] for d in dets], np.float32).reshape(-1)
            self.cls = np.asarray([d["cls"] for d in dets], np.float32).reshape(-1)
            x1, y1, x2, y2 = (self.xyxy[:, i] for i in range(4))
            self.xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1)

        def __getitem__(self, mask):
            return type(self)(arrays=(self.xyxy[mask], self.conf[mask],
                                      self.cls[mask], self.xywh[mask]))

        def __len__(self):
            return len(self.conf)

    def update(self, dets, ts, img=None):
        if ts is not None and self.last_ts is not None and not (0 <= ts - self.last_ts <= self.reset_gap):
            self.tracker = self._make()                # 끊긴 구간 — 처음부터
        self.last_ts = ts if ts is not None else self.last_ts
        if not dets:
            self.tracker.update(self._Boxes([]), img)
            return []
        rows = np.asarray(self.tracker.update(self._Boxes(dets), img)).reshape(-1, 8)
        out = [dict(d) for d in dets]
        for row in rows:
            idx = int(row[7])
            if not 0 <= idx < len(out):
                continue
            out[idx]["track_id"] = int(row[4])
            if self.keep_boxes == "tracker":
                out[idx]["xyxy"] = [round(float(v), 1) for v in row[:4]]
        if self.on_untracked == "drop":
            return [d for d in out if d.get("track_id") is not None]
        return out                                     # 번호 없는 것도 그대로 나간다

    def smooth_depth(self, dets, model):               # 거리 평활은 하지 않는다
        return dets


class LabelStabilizer:
    """같은 물체의 **이름이 장마다 뒤집히는 것**을 줄인다. 박스는 건드리지 않는다.

    왜 따로 있나: 기성 트래커는 `new_track_thresh`(0.25) 위의 **잘 잡힌** 검출만 잇는다.
    그런데 이름이 뒤집히는 것은 바로 **신뢰도가 낮은** 검출들이다 — 그것들은 번호를
    못 받으므로 그 쪽으로는 손댈 수 없다. 그래서 이름 안정화는 모든 검출을 보는
    자기 짝짓기(IoU)를 따로 쓴다.

    쟀더니 지금 나가는 것은 **장 사이 전이의 29.4%** 에서 이름이 바뀐다(연속 촬영
    drone+dji, track 2947). 가장 잦은 쌍은 styrofoam_box↔water_jug(302)이고
    styrofoam_box↔wooden_box 는 70 이다.

    `policy`
      raw      그대로 둔다 (아무것도 하지 않음)
      hyst-k   지금 이름을 유지하고 **다른 이름이 k 장 연속** 이겨야 바꾼다
               (측정: k=2 면 깜빡임 81% 감소 · k=3 이면 95% 감소)

    **대가를 숨기지 않는다.** 이름을 붙들면 바뀌어야 할 때도 늦는다 — 라벨 150장에서
    hyst-2 는 macro F1 -0.046 · 재현율 -0.110 이다. 그래서 **버리는 것이 없게** 한다:
    이 장의 원래 이름은 `label_raw` 로, 흔들린다는 사실은 `label_stable: false` 로,
    후보는 `label_candidates` 로 함께 낸다. 프레임별 진실이 필요한 소비자는 `label_raw`
    를 보면 된다(AI-S-03 — 확신과 근거 충분도는 다른 축이다).
    """

    def __init__(self, sc):
        policy = str(sc.get("label_policy", "raw")).lower()
        self.k = 0
        if policy != "raw":
            if not policy.startswith("hyst-") or not policy[5:].isdigit():
                raise SystemExit("stabilize.label_policy 는 raw 또는 hyst-<수>: %s" % policy)
            self.k = int(policy[5:])
            if self.k < 1:
                raise SystemExit("stabilize.label_policy 의 수는 1 이상입니다: %s" % policy)
        self.policy = policy
        self.iou = float(sc.get("label_iou_match", sc.get("iou_match", 0.3)))
        self.margin = float(sc.get("label_margin", 0.6))
        self.decay = float(sc.get("label_decay", 0.8))
        self.reset_gap = float(sc.get("reset_gap_s", 2.0))
        self.max_gap = int(sc.get("label_max_missed", sc.get("max_missed", 5)))
        self.tracks, self.last_ts = [], None

    def update(self, dets, ts):
        if self.policy == "raw":
            return dets
        # **검출이 없는 장에서도 나이를 먹인다.** 건너뛰면 사라진 물체가 영원히 남아
        # 한참 뒤에 그 자리에 들어온 다른 물체에 옛 이름을 얹는다.
        if ts is not None and self.last_ts is not None and not (0 <= ts - self.last_ts <= self.reset_gap):
            self.tracks = []
        self.last_ts = ts if ts is not None else self.last_ts
        used = set()
        if self.tracks and dets:
            iou = box_iou([t["box"] for t in self.tracks], [d["xyxy"] for d in dets])
            pairs = sorted(((float(iou[i, j]), i, j) for i in range(len(self.tracks))
                            for j in range(len(dets)) if iou[i, j] >= self.iou), reverse=True)
        else:
            pairs = []
        taken_t, taken_d = set(), set()
        for _v, i, j in pairs:
            if i in taken_t or j in taken_d:
                continue
            taken_t.add(i)
            taken_d.add(j)
            self._carry(self.tracks[i], dets[j])
            used.add(j)
        # 이번 장에 짝이 없던 것은 나이를 먹고, 너무 오래된 것은 잊는다. 잊지 않으면
        # 사라진 물체의 이름이 뒤에 들어온 다른 물체에 그대로 얹힌다.
        alive = []
        for i, trk in enumerate(self.tracks):
            if i in taken_t:
                alive.append(trk)                      # _carry 가 age 를 0 으로 되돌렸다
            else:
                trk["age"] += 1
                if trk["age"] <= self.max_gap:
                    alive.append(trk)
        for j, d in enumerate(dets):
            if j not in used:
                trk = {"box": list(d["xyxy"]), "name": d["name"], "votes": {}, "cls": {},
                       "streak": 0, "last": None, "age": 0}
                self._carry(trk, d)
                alive.append(trk)
        self.tracks = alive
        return dets

    def _carry(self, t, d):
        t["box"] = list(d["xyxy"])
        t["age"] = 0
        for k in t["votes"]:
            t["votes"][k] *= self.decay
        t["votes"][d["name"]] = t["votes"].get(d["name"], 0.0) + d["conf"]
        t["cls"][d["name"]] = d["cls"]
        raw = d["name"]
        if raw == t["name"]:
            t["streak"], t["last"] = 0, None
        else:
            t["streak"] = t["streak"] + 1 if raw == t["last"] else 1
            t["last"] = raw
            if t["streak"] >= self.k:                  # k 장 연속 이겨야 바꾼다
                t["name"], t["streak"], t["last"] = raw, 0, None
        total = sum(t["votes"].values()) or 1.0
        top = max(t["votes"], key=t["votes"].get)
        d["label_stable"] = bool(t["votes"][top] / total >= self.margin and t["name"] == top)
        if not d["label_stable"]:
            d["label_candidates"] = sorted(t["votes"], key=t["votes"].get, reverse=True)[:3]
        if t["name"] != raw:
            d["label_raw"] = raw                       # 이 장의 원래 이름 — 버리지 않는다
            d["name"] = t["name"]
            d["cls"] = t["cls"].get(t["name"], d["cls"])


def stabilize_cfg(cfg, model):
    """`stabilize` 의 바탕값 위에 `stabilize.<model>` 을 얹는다.

    모델마다 흔들리는 모양이 다르다 — river 는 수십 px 짜리가 움직이는 드론에서
    장마다 들락거리고, yoloe prompt-free 는 큰 물체를 꾸준히 잡는다. 하나의
    값으로 둘을 맞추면 한쪽은 너무 질기고 다른 쪽은 너무 성기다.
    """
    sc = cfg.get("stabilize") or {}
    base = {k: v for k, v in sc.items() if not isinstance(v, dict)}
    return {**base, **(sc.get(model) or {})}


class Stabilizer:
    """기기·실행 폴더마다 하나. 프레임 사이 같은 물체(IoU, 이름 무관)를 이어 붙여 흔들림을 줄인다.
      track_id   같은 물체는 같은 번호 (Unity 등 받는 쪽이 물체를 이어 갈 수 있게)
      박스       EMA (box_alpha)                  이름  감쇠 누적 신뢰도로 다수결 (매 장 바뀌지 않게)
      등장       min_hits 장 연속 잡혀야 내보낸다      소멸  hold_missed 장까지는 마지막 박스로 유지 (held)
      거리       depth 모델마다 최근 depth_window 개의 중앙값 (depth_m, 원래 값은 depth_m_raw)
      이음       IoU >= iou_match, **또는** 중심이 상자 대각선 평균의 center_dist 배 안
                 (center_dist 0 이면 IoU 만 — 겹치지 않으면 track 이 끊긴다)
      reset_gap_s 보다 프레임 시각이 벌어지거나 거꾸로 가면 처음부터"""

    def __init__(self, sc):
        self.iou = float(sc.get("iou_match", 0.3))
        # IoU 만으로는 **상자가 겹칠 때만** 이어진다. 수십 px 짜리가 자기 폭보다 멀리
        # 움직이면(느린 프레임·고정 카메라에서 흔하다) 겹침이 0 이고 track 이 끊긴다.
        # center_dist 는 그때의 보조 기준이다 — 두 상자 대각선 평균의 이 배 안에
        # 중심이 있으면 같은 물체로 본다. 0 이면 끈다(옛 동작 그대로).
        self.center = float(sc.get("center_dist", 0.0))
        self.alpha = float(sc.get("box_alpha", 0.5))
        self.min_hits = int(sc.get("min_hits", 2))
        self.hold = int(sc.get("hold_missed", 2))
        self.max_missed = max(self.hold, int(sc.get("max_missed", 5)))
        self.decay = float(sc.get("label_decay", 0.8))
        self.window = int(sc.get("depth_window", 5))
        self.reset_gap = float(sc.get("reset_gap_s", 2.0))
        self.tracks, self.next_id, self.last_ts = [], 1, None

    def update(self, dets, ts):
        if ts is not None and self.last_ts is not None and not (0 <= ts - self.last_ts <= self.reset_gap):
            self.tracks = []
        self.last_ts = ts if ts is not None else self.last_ts
        pairs = []
        if self.tracks and dets:
            tb = [t["box"] for t in self.tracks]
            db = [d["xyxy"] for d in dets]
            iou = box_iou(tb, db)
            gate = self._center_gate(tb, db) if self.center > 0 else None
            for i in range(len(tb)):
                for j in range(len(db)):
                    near = gate is not None and gate[i][j] is not None
                    if iou[i, j] >= self.iou or near:
                        # IoU 가 있는 쌍이 먼저, 없는 쌍은 중심이 가까운 순서로
                        pairs.append((float(iou[i, j]), -(gate[i][j] if near else 0.0), i, j))
            pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, _d, i, j in pairs:                              # IoU 큰 쌍부터 하나씩
            if i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            t, d = self.tracks[i], dets[j]
            t["box"] = [self.alpha * n + (1 - self.alpha) * o for n, o in zip(d["xyxy"], t["box"])]
            t["conf"] = self.alpha * d["conf"] + (1 - self.alpha) * t["conf"]
            self._vote(t, d)
            t["hits"] += 1
            t["missed"] = 0
            t["det"] = t["last"] = d
        for i, t in enumerate(self.tracks):
            if i not in used_t:
                t["missed"] += 1
                t["det"] = None
        for j, d in enumerate(dets):
            if j not in used_d:
                t = {"id": self.next_id, "box": list(d["xyxy"]), "conf": d["conf"], "votes": {}, "cls": {},
                     "hits": 1, "missed": 0, "det": d, "last": d, "depth": {}}
                self.next_id += 1
                self._vote(t, d)
                self.tracks.append(t)
        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed]
        out = []
        for t in self.tracks:
            if t["hits"] < self.min_hits or t["missed"] > self.hold:
                continue
            name = max(t["votes"], key=t["votes"].get)
            if t["det"] is not None:
                d = dict(t["det"])
            else:
                # 이번 장에 못 잡음 — 마지막 박스로 유지한다. 마스크·외곽선은 **버린다**
                # (이 장의 픽셀이 아니다). 나머지 근거(arms·prompts·agree)는 들고 간다:
                # 비우면 held 검출이 "근거 없이 나타난 박스" 가 되어 읽는 쪽이 설명할 수 없다.
                d = {k: v for k, v in (t.get("last") or {}).items()
                     if k not in ("polygon", "mask", "depth_m", "depth_m_raw")}
                d["held"] = True
            if d.get("name") != name:                          # 다수결 이름과 다른 마스크·외곽선은 그대로 둔다
                d["name_raw"] = d.get("name")
            d.update(cls=t["cls"][name], name=name, conf=round(t["conf"], 4), track_id=t["id"],
                     xyxy=[round(float(v), 1) for v in t["box"]])
            out.append(d)
        return out

    def _center_gate(self, tb, db):
        """[i][j] = 중심거리 / (대각선 평균). 문을 통과하지 못하면 None."""
        import math
        out = []
        for a in tb:
            row = []
            for b in db:
                ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
                bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                da = math.hypot(a[2] - a[0], a[3] - a[1])
                db_ = math.hypot(b[2] - b[0], b[3] - b[1])
                scale = (da + db_) / 2 or 1.0
                rel = math.hypot(ax - bx, ay - by) / scale
                row.append(rel if rel <= self.center else None)
            out.append(row)
        return out

    def _vote(self, t, d):
        for k in t["votes"]:
            t["votes"][k] *= self.decay
        t["votes"][d["name"]] = t["votes"].get(d["name"], 0.0) + d["conf"]
        t["cls"][d["name"]] = d["cls"]

    def smooth_depth(self, dets, model):
        by_id = {t["id"]: t for t in self.tracks}
        for d in dets:
            t = by_id.get(d.get("track_id"))
            if t is None or d.get("depth_m") is None:
                continue
            h = t["depth"].setdefault(model, [])
            h.append(d["depth_m"])
            del h[:-self.window]
            d["depth_m_raw"] = d["depth_m"]
            d["depth_m"] = round(float(np.median(h)), 3)
        return dets


# ---------------------------------------------------------------- 모델: depth 공통
def resize_long(bgr, long_side):
    h, w = bgr.shape[:2]
    if not long_side or max(h, w) <= long_side:
        return bgr, 1.0
    s = long_side / float(max(h, w))
    return cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA), s


def k_from_fov(fov_x_deg, w, h):
    f = 0.5 * w / np.tan(np.radians(fov_x_deg) / 2)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float32)


def fov_from_k(K, w):
    return float(np.degrees(2 * np.arctan(0.5 * w / K[0, 0]))) if K is not None and K[0, 0] > 0 else None


def _unidepth_get_paddings(original_shape, aspect_ratio_range):
    """UniDepth get_paddings 와 같지만 int() 대신 round().
    원본은 464×400 처럼 int(400 * 1.16) = 463 이 되면 pad = -1 이 되어 depth 폭이 1 로 잘린다."""
    H_ori, W_ori = original_shape
    min_ratio, max_ratio = aspect_ratio_range
    target = min(max_ratio, max(min_ratio, W_ori / H_ori))
    H_new, W_new = max(H_ori, round(W_ori / target)), max(W_ori, round(H_ori * target))
    pad_t, pad_l = (H_new - H_ori) // 2, (W_new - W_ori) // 2
    return (pad_l, W_new - W_ori - pad_l, pad_t, H_new - H_ori - pad_t), (H_new, W_new)


class UniDepthRunner:
    name = "unidepth"

    def __init__(self, cfg, device):
        import torch
        if cfg.get("repo_path") and cfg["repo_path"] not in sys.path:
            sys.path.insert(0, cfg["repo_path"])
        from unidepth.models import UniDepthV2
        import unidepth.models.unidepthv2.unidepthv2 as udv2
        udv2.get_paddings = _unidepth_get_paddings
        self.torch = torch
        self.cfg = cfg
        self.device = device
        self.model = UniDepthV2.from_pretrained(cfg["model_name"]).to(device).eval()
        if cfg.get("resolution_level") is not None:
            self.model.resolution_level = int(cfg["resolution_level"])
        self.fp16 = bool(cfg.get("use_fp16", True)) and device.startswith("cuda")

    def __call__(self, bgr, fov_x=None):
        torch = self.torch
        h, w = bgr.shape[:2]
        img, s = resize_long(bgr, self.cfg.get("infer_long_side"))
        ih, iw = img.shape[:2]
        rgb = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).contiguous()
        camera = torch.from_numpy(k_from_fov(fov_x, iw, ih)) if fov_x else None
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            out = self.model.infer(rgb, camera=camera)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        depth = out["depth"][0, 0].float().cpu().numpy()
        # 화각을 주면 그 광선으로 depth 를 만든다. out["intrinsics"] 는 그때도 모델 추정값이라 준 K 를 남긴다
        K = camera.numpy().copy() if camera is not None else out["intrinsics"][0].float().cpu().numpy().copy()
        if (ih, iw) != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            K[:2] /= s
        return {"infer_ms": round(ms, 1), "checkpoint": Path(self.cfg["model_name"]).name}, depth, K


class MogeAerialRunner:
    name = "moge2_aerial"

    def __init__(self, cfg, device):
        import torch
        for p in list(cfg.get("extra_sys_path") or []) + [cfg["repo_path"]]:
            if p not in sys.path:
                sys.path.insert(0, p)
        from peft import LoraConfig, get_peft_model
        from moge.model import import_model_class_by_version
        from moge.lora_model_config import (MODEL_CONFIG, MODEL_VERSION, LORA_TARGET_MODULES,
                                            LORA_MODULES_TO_SAVE, LORA_ALPHA_MULTIPLIER)
        self.torch = torch
        self.cfg = cfg
        self.device = device
        rank = int(cfg.get("lora_rank", 96))
        model = import_model_class_by_version(MODEL_VERSION)(**MODEL_CONFIG)
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=LORA_ALPHA_MULTIPLIER * rank, bias="none",
                                                 target_modules=LORA_TARGET_MODULES,
                                                 modules_to_save=LORA_MODULES_TO_SAVE))
        ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        # 체크포인트 키를 PEFT 모델 키로 맞춘다 (AerialMetric a_infer_lora96_norm.py 와 같은 규칙)
        keys = set(model.state_dict().keys())
        mapped = {}
        for k, v in state.items():
            if k in keys:
                mapped[k] = v
                continue
            pk = "base_model.model." + k
            if pk in keys:
                mapped[pk] = v
                continue
            parts = pk.split(".")
            bk = ".".join(parts[:-1] + ["base_layer", parts[-1]])
            if parts[-1] in ("weight", "bias") and bk in keys:
                mapped[bk] = v
                continue
            for head in LORA_MODULES_TO_SAVE:
                tk = "base_model.model.%s.modules_to_save.default.%s" % (head, k[len(head) + 1:])
                if k.startswith(head + ".") and tk in keys:
                    mapped[tk] = v
                    break
        missing, _ = model.load_state_dict(mapped, strict=False)
        n_lora = sum(1 for k in mapped if "lora_" in k)
        if n_lora == 0 or len(mapped) < len(keys) * 0.9:
            raise RuntimeError("MoGe2-Aerial 가중치가 제대로 안 맞음 (맞은 키 %d/%d, LoRA %d)"
                               % (len(mapped), len(keys), n_lora))
        log("  moge2_aerial: 키 %d/%d 적재 (LoRA %d, 빠짐 %d) → LoRA 병합" % (len(mapped), len(keys), n_lora, len(missing)))
        self.model = model.merge_and_unload().to(device).eval()
        self.fp16 = bool(cfg.get("use_fp16", True)) and device.startswith("cuda")
        if self.fp16:
            self.model.half()

    def __call__(self, bgr, fov_x=None):
        torch = self.torch
        c = self.cfg
        h, w = bgr.shape[:2]
        img, s = resize_long(bgr, c.get("infer_long_side"))
        ih, iw = img.shape[:2]
        t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(self.device).permute(2, 0, 1).float().div(255)
        if self.fp16:
            t = t.half()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.infer(t, resolution_level=int(c.get("resolution_level", 9)),
                                   force_projection=bool(c.get("force_projection", True)),
                                   apply_mask=bool(c.get("apply_mask", True)),
                                   fov_x=fov_x, use_fp16=self.fp16)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        depth = out["depth"].float().cpu().numpy()
        depth[~np.isfinite(depth)] = 0
        K = out["intrinsics"].float().cpu().numpy().copy()     # 정규화 K (cx≈0.5)
        K[0] *= iw
        K[1] *= ih
        if (ih, iw) != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            K[:2] /= s
        return {"infer_ms": round(ms, 1), "checkpoint": Path(c["checkpoint"]).name}, depth, K


# ---------------------------------------------------------------- 모델: river (하천 떠내려가는 물체, 제로샷)
#: 미끼 문구의 주인. 어휘에는 들어가지만 결과로 나가지 않는다.
DECOY = "__decoy__"

def _expand_vocab(classes, spec, decoys=None):
    """클래스 → 찾을 문구들. 평면 목록(클래스당 문구 1개)도 그대로 받는다.

    `decoys` 는 **어휘에만 넣고 출력에서는 버리는** 문구다. 열린어휘 탐지기는 어휘
    전체에 대한 argmax 로 한 박스에 이름 하나를 붙인다 — 배경에 실제로 있는 것
    (유목·판자·돌)을 어휘에 넣어 두면 둔치의 판자가 `wooden_box` 가 아니라
    `wooden plank` 로 불려 **사라진다**. 모델을 바꾸지 않고 어휘만 바꾸는 수다.

    문구가 여럿이어도 **나가는 이름은 그 클래스 하나**다 — `polystyrene foam block`
    으로 잡혀도 결과에는 `styrofoam_box` 가 적힌다. 상위 계약의 클래스는 늘지 않는다.

    같은 문구를 두 클래스가 쓰면 거절한다: 돌아온 문구로 클래스를 되짚을 수 없고
    조용히 한쪽으로 접으면 다른 쪽이 말없이 사라진다.
    """
    classes = [str(c) for c in classes or []]
    if not classes:
        raise SystemExit("models.river: classes 가 비었습니다")
    if spec is None:
        # 적지 않으면 **클래스 이름 자체가 문구**다. 프리 프롬프트 arm 만 쓰는 구성은
        # 텍스트 문구가 필요 없지만 하위 계층(임계값·기록)은 문구 자리를 쓴다.
        # 빈 dict/목록은 그대로 거절한다 — 비워 놓은 것과 안 적은 것은 다른 뜻이다.
        groups = [(c, [c]) for c in classes]
    elif isinstance(spec, dict):
        unknown = [k for k in spec if k not in classes]
        if unknown:
            raise SystemExit("models.river.prompts 에 classes 에 없는 이름이 있습니다: %s"
                             % ", ".join(map(str, unknown)))
        groups = []
        for c in classes:
            if c in spec and spec[c] is None:
                # **문구를 두지 않는 클래스** — 프리 프롬프트 arm 만 채우는 가설 칸이다.
                # 텍스트 어휘에 넣으면 그 낱말이 어휘 전체 argmax 에서 표를 가져가
                # 진짜 물체를 지운다(미끼 문구가 현장에서 진 자리가 바로 그것이다).
                # 그래서 어휘에 **넣지 않는다**. 빈 목록은 그대로 거절한다 —
                # 비워 둔 것(실수)과 안 두겠다고 적은 것(null)은 다른 뜻이다.
                continue
            groups.append((c, [str(p) for p in (spec.get(c) or [])]))
        empty = [c for c, words in groups if not words]
        if empty:
            raise SystemExit("models.river.prompts: 문구가 없는 클래스가 있습니다: %s "
                             "(문구를 두지 않을 뜻이면 null 로 적습니다)" % ", ".join(empty))
    else:
        flat = [str(p) for p in spec or []]
        if len(flat) != len(classes):
            raise SystemExit("models.river: prompts 를 평면 목록으로 적으면 classes 와 같은 "
                             "길이여야 합니다 (%d vs %d). 클래스당 문구를 여럿 두려면 "
                             "prompts 를 클래스 → 문구목록 으로 적습니다."
                             % (len(classes), len(flat)))
        groups = [(c, [p]) for c, p in zip(classes, flat)]
    prompts, owners, seen = [], [], {}
    for cls, words in groups:
        for word in words:
            key = word.strip().lower()
            if key in seen:
                raise SystemExit("models.river.prompts: 같은 문구를 두 클래스가 씁니다 "
                                 "— %r (%s, %s)" % (word, seen[key], cls))
            seen[key] = cls
            prompts.append(word)
            owners.append(cls)
    for word in (decoys or []):
        key = str(word).strip().lower()
        if key in seen:
            raise SystemExit("models.river.decoys: 클래스가 이미 쓰는 문구입니다 — %r (%s)"
                             % (word, seen[key]))
        seen[key] = DECOY
        prompts.append(str(word))
        owners.append(DECOY)
    return prompts, owners


def _prompt_class_map(prompts, owners):
    """gdino 가 돌려주는 문구 → (클래스, 문구). 부분 일치로 짝짓는다 (문구를 쪼개 돌려주기도 한다)."""
    pairs = list(zip(prompts, owners))

    def match(label):
        low = str(label).strip().lower()
        if not low:
            return None, None
        for prompt, cls in pairs:
            if low in prompt.lower() or prompt.lower() in low:
                return cls, prompt
        return None, None
    return match


class _ArmMissing(Exception):
    """arm 의 가중치·스냅샷이 없다. 런타임에 받지 않는다 (AI-C-16 폐쇄망)."""


class _ArmUltralytics:
    """YOLOE / YOLO-World — ultralytics 텍스트 프롬프트."""

    def __init__(self, spec, prompts, owners, device, shared):
        weight = Path(str(spec["weight"])).expanduser()
        if not weight.is_file():
            raise _ArmMissing("가중치가 없다: %s" % weight)
        self.prompts = list(prompts)
        self.owners = list(owners)
        kind = spec["kind"]
        if kind == "yoloe":
            from ultralytics import YOLOE
            if weight.stem.endswith("-pf"):
                raise SystemExit("river arm yoloe 에는 prompt-free(-pf) 가 아닌 가중치가 필요합니다: %s" % weight)
            self.model = YOLOE(str(weight))
            # ultralytics 는 텍스트 인코더(mobileclip_blt.ts)를 **작업 폴더**에서 찾는다.
            # 없으면 인터넷에서 받으므로(≈570MB) 폐쇄망에서 깨진다 — 미리 확인하고 거절한다.
            enc_dir = Path(str(spec.get("text_encoder_dir") or weight.parent)).expanduser()
            if not (enc_dir / "mobileclip_blt.ts").is_file():
                raise _ArmMissing("텍스트 인코더가 없다: %s/mobileclip_blt.ts" % enc_dir)
            with contextlib.chdir(enc_dir):
                pe = self.model.get_text_pe(list(prompts))
            self.model.set_classes(list(prompts), pe)
        else:
            from ultralytics import YOLO
            self.model = YOLO(str(weight))
            self.model.set_classes(list(prompts))
        self.device = device
        self.imgsz = int(spec.get("imgsz") or shared["imgsz"])
        self.conf = float(spec.get("conf", shared["base_conf"]))
        self.max_det = int(spec.get("max_det") or shared["max_det"])
        self.name = weight.name

    def __call__(self, bgr):
        r = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, max_det=self.max_det,
                               device=self.device, verbose=False)[0]
        out = []
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                index = int(b.cls)
                if 0 <= index < len(self.owners):
                    out.append({"cls_name": self.owners[index], "prompt": self.prompts[index],
                                "conf": float(b.conf),
                                "xyxy": [float(v) for v in b.xyxy[0].tolist()]})
        return out


class _ArmGroundingDino:
    """Grounding DINO — 문구 하나로 묶어 넣고 돌려받은 문구를 클래스로 되짚는다."""

    def __init__(self, spec, prompts, owners, device, shared):
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
        model_id = str(spec.get("model") or "IDEA-Research/grounding-dino-tiny")
        kw = {"local_files_only": True}
        if spec.get("cache_dir"):
            kw["cache_dir"] = str(Path(str(spec["cache_dir"])).expanduser())
        try:
            self.proc = AutoProcessor.from_pretrained(model_id, **kw)
            self.model = GroundingDinoForObjectDetection.from_pretrained(model_id, **kw).to(device).eval()
        except Exception as exc:                       # 스냅샷이 없다 — 받지 않는다
            raise _ArmMissing("%s 스냅샷이 없다 (%s)" % (model_id, type(exc).__name__)) from exc
        self.device = device
        self.text = ". ".join(prompts) + "."
        self.match = _prompt_class_map(prompts, owners)
        self.conf = float(spec.get("conf", shared["base_conf"]))
        self.text_threshold = float(spec.get("text_threshold", 0.2))
        self.name = model_id

    def __call__(self, bgr):
        import torch
        from PIL import Image
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = self.proc(images=image, text=self.text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            raw = self.model(**inputs)
        post = self.proc.post_process_grounded_object_detection(
            raw, inputs.input_ids, threshold=self.conf, text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]])[0]
        out = []
        for score, label, box in zip(post["scores"], post["text_labels"], post["boxes"]):
            cls, prompt = self.match(label)
            if cls:
                out.append({"cls_name": cls, "prompt": prompt, "conf": float(score),
                            "xyxy": [float(v) for v in box.tolist()]})
        return out


class _ArmOwlv2:
    """OWLv2 — 문구를 그대로 넣고 번호로 돌려받는다."""

    def __init__(self, spec, prompts, owners, device, shared):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        model_id = str(spec.get("model") or "google/owlv2-large-patch14-ensemble")
        kw = {"local_files_only": True}
        if spec.get("cache_dir"):
            kw["cache_dir"] = str(Path(str(spec["cache_dir"])).expanduser())
        try:
            self.proc = Owlv2Processor.from_pretrained(model_id, **kw)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_id, **kw).to(device).eval()
        except Exception as exc:
            raise _ArmMissing("%s 스냅샷이 없다 (%s)" % (model_id, type(exc).__name__)) from exc
        self.device = device
        self.prompts = [list(prompts)]
        self.words = list(prompts)
        self.owners = list(owners)
        self.conf = float(spec.get("conf", shared["base_conf"]))
        self.name = model_id

    def __call__(self, bgr):
        import torch
        from PIL import Image
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = self.proc(text=self.prompts, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            raw = self.model(**inputs)
        post = self.proc.post_process_grounded_object_detection(
            raw, threshold=self.conf,
            target_sizes=torch.tensor([image.size[::-1]]).to(self.device))[0]
        out = []
        for score, label, box in zip(post["scores"], post["labels"], post["boxes"]):
            index = int(label)
            if 0 <= index < len(self.owners):
                out.append({"cls_name": self.owners[index], "prompt": self.words[index],
                            "conf": float(score),
                            "xyxy": [float(v) for v in box.tolist()]})
        return out


class _ArmYoloeFree:
    """YOLOE **프리 프롬프트**(내장 어휘 4585종) — 자리를 내고 이름은 **가설로** 남긴다.

    왜 이것을 더하는가 (2026-11-12 측정):

    * 5종 텍스트 프롬프트는 실내 검수 27 정답에서 **자리를 0 개** 잡았다. 같은 장을
      프리 프롬프트로 보면 **27/27**(IoU>=0.5 · 26/27 at 0.7)이다. 자리를 못 찾는
      것은 어휘를 5개로 좁힌 탓이지 모델이 못 보는 탓이 아니었다.
    * 그런데 프리 어휘가 붙이는 이름은 장마다 다르다 — 같은 나무판에 breadbox 7 ·
      hassock 6 · plywood 5 · podium 4 · harpsichord 2 · exhaust hood 2 · block 1.
      **자리는 믿고 이름은 믿지 않는다.**

    그래서 `vocab_map` 은 프리 이름 → 우리 클래스의 **선언**이고, 지도에 없는 이름은
    버리지 않고 `unconfirmed_class` 로 낸다. 1 위를 고르지 않고 가설로 남기는 것이다
    (AI-S-04 — 기존 클래스에 강제 매핑하지 않는다). 어느 프리 이름이 말했는지는
    `prompt` 로 남아 뒤에서 설명할 수 있다.

    **위험물은 재질 낱말이 있을 때만 받는다.** `wooden_box` 는 `plywood` `wood`
    `hardwood` 처럼 **재질을 말하는** 이름에서만 승격하고, `breadbox` `crate` `box`
    처럼 모양만 말하는 이름은 `unconfirmed` 로 남는다 — 나무 상자와 골판지 상자·책을
    가르는 것은 모양이 아니라 재질이기 때문이다. 이 둘을 섞으면 하천에서 측정된
    wooden_box 정밀도 0.158 이 그대로 따라온다.

    프리 프롬프트는 장당 72.5 상자를 낸다(5종 텍스트는 6.5). 이 비용을 숨기지 않는다.
    """

    def __init__(self, spec, prompts, owners, device, shared):
        weight = Path(str(spec["weight"])).expanduser()
        if not weight.is_file():
            raise _ArmMissing("가중치가 없다: %s" % weight)
        if not weight.stem.endswith("-pf"):
            raise SystemExit("river arm yoloe_free 에는 prompt-free(-pf) 가중치가 "
                             "필요합니다: %s" % weight)
        from ultralytics import YOLOE
        self.model = YOLOE(str(weight))
        self.device = device
        self.imgsz = int(spec.get("imgsz") or shared["imgsz"])
        self.conf = float(spec.get("conf", shared["base_conf"]))
        self.max_det = int(spec.get("max_det") or shared["max_det"])
        self.name = weight.name
        self.free_vocab = True          # 텍스트 문구를 쓰지 않는다 — 어휘는 가중치 안에 있다
        known = set(shared.get("classes") or owners)
        self.unconfirmed = spec.get("unconfirmed_class")
        if self.unconfirmed is not None:
            self.unconfirmed = str(self.unconfirmed)
            if self.unconfirmed not in known:
                raise SystemExit("river arm yoloe_free.unconfirmed_class 가 classes 에 "
                                 "없습니다: %s" % self.unconfirmed)
        self.vocab_map = self._read_map(spec.get("vocab_map"), known)
        vocab = {str(v).lower() for v in (self.model.names or {}).values()}
        if vocab:
            unknown = sorted(n for n in self.vocab_map if n not in vocab)
            if unknown:
                # 어휘에 없는 이름을 적어 두면 **아무 일도 하지 않는다**. 조용히
                # 지나가면 그 클래스가 왜 안 나오는지 설명할 수 없다.
                raise SystemExit("river arm yoloe_free.vocab_map 에 내장 어휘에 없는 "
                                 "이름이 있습니다(아무 일도 하지 않습니다): %s"
                                 % ", ".join(unknown))
        log("  river arm %s 프리 어휘 %d종 · 지도 %d개 → %d클래스%s"
            % (self.name, len(vocab), len(self.vocab_map),
               len(set(self.vocab_map.values())),
               (" · 지도 밖은 %s" % self.unconfirmed) if self.unconfirmed else " · 지도 밖은 버린다"))

    @staticmethod
    def _read_map(spec, known):
        """{우리 클래스: [프리 이름...]} 을 {프리 이름: 우리 클래스} 로 뒤집는다."""
        if not spec:
            raise SystemExit("river arm yoloe_free 에는 vocab_map 이 필요합니다")
        if not isinstance(spec, dict):
            raise SystemExit("river arm yoloe_free.vocab_map 은 클래스 → 이름 목록입니다")
        out = {}
        for cls, names in spec.items():
            cls = str(cls)
            if cls not in known:
                raise SystemExit("river arm yoloe_free.vocab_map 에 classes 에 없는 "
                                 "클래스가 있습니다: %s" % cls)
            for n in (names or []):
                n = str(n).lower()
                if n in out and out[n] != cls:
                    # 한 이름이 두 클래스를 가리키면 **어느 쪽도 믿을 수 없다**
                    raise SystemExit("river arm yoloe_free.vocab_map: %r 이 %s 와 %s "
                                     "둘을 가리킵니다" % (n, out[n], cls))
                out[n] = cls
        return out

    def __call__(self, bgr):
        r = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, max_det=self.max_det,
                               device=self.device, verbose=False)[0]
        names = self.model.names or {}
        out = []
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                free = str(names.get(int(b.cls), int(b.cls))).lower()
                cls = self.vocab_map.get(free, self.unconfirmed)
                if cls is None:
                    continue
                out.append({"cls_name": cls, "prompt": free, "conf": float(b.conf),
                            "xyxy": [float(v) for v in b.xyxy[0].tolist()]})
        return out


ARM_KINDS = {"yoloe": _ArmUltralytics, "yoloworld": _ArmUltralytics,
             "yoloe_free": _ArmYoloeFree,
             "gdino": _ArmGroundingDino, "owlv2": _ArmOwlv2}


class RiverRunner:
    """하천 떠내려가는 물체 — **학습 없이**. 문구 어휘와 클래스별 임계값으로 조인다.

    왜 학습하지 않는가: 안동 자료는 현장 1곳·날짜 1일·촬영 1회다. 가중치를 학습하면
    그 한 장면을 외운다(인접 프레임 평균차가 go1 50프레임 간격에도 0.30).

    **어휘와 출력 이름은 다르다.** `prompts` 는 클래스당 문구를 여럿 받고(`water_jug`
    를 `plastic water jug` · `water container` 로 찾는 식), 어느 문구로 잡혀도 결과에
    적히는 이름은 클래스 하나다. 검출마다 어느 문구가 말했는지(`prompts`)를 남긴다 —
    버리면 어휘를 왜 그렇게 뒀는지 뒤에서 설명할 수 없다.

    **임계값은 세 단계다.** `arm_conf` 가 arm 이 내놓는 하한, `prompt_conf` 가 문구별
    하한(병합 전 — 문구는 raw 검출의 성질이다), `class_conf` 가 클래스별 하한(병합 후
    — 클래스는 병합군의 성질이다). 적지 않은 자리는 `conf` 를 쓴다. 위험물처럼 다른
    조치를 부르는 클래스는 재현율을 위해 낮추고, 덤불·판자처럼 헛검출이 쏟아지는
    문구는 올린다.

    arm 이 여럿이면 **동의**(`min_arms`)도 요구할 수 있다: 학습 자료가 겹치지 않는
    모델들이 같은 자리에 같은 이름을 말할 때만 받는다. 느리다 — 측정값은 4 arm 에서
    한 장 3.33 s, yoloe 단독 0.166 s 다. `arms` 를 하나로 두면 그 모델 단독이다.

    없는 arm: `on_missing_arm: degrade` 면 그 arm 만 빼고 돌리고 `min_arms` 를
    남은 수로 줄인다. **줄인 사실을 기록에 적는다**(`arms`, `min_arms_applied`) —
    작동점이 조용히 바뀌면 숫자가 어느 작동점의 것인지 알 수 없다. `refuse` 면
    세우지 않는다. 런타임에 가중치를 받지 않는다 (AI-C-16 폐쇄망).
    """
    name = "river"

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.classes = [str(c) for c in cfg.get("classes") or []]
        self.decoys = [str(d) for d in (cfg.get("decoys") or [])]
        self.prompts, self.owners = _expand_vocab(self.classes, cfg.get("prompts"), self.decoys)
        self.conf = float(cfg.get("conf", 0.25))
        self.merge_iou = float(cfg.get("merge_iou", 0.5))
        self.max_area = cfg.get("max_area_ratio")
        # **크기에 따른 하한.** 멀리 있는 작은 물체와 코앞의 큰 물체는 같은 자로 잴 수
        # 없다. 상수 하나로 두면 그 상수를 고른 현장에서만 맞는다 — 하천 GT 는 가장 큰
        # 것이 화면의 0.006 인데 실내 근접의 나무 상자는 0.386 이다. `max_area_ratio`
        # 0.08 하나로 두면 하천에서는 한 번도 켜지지 않고 실내에서는 **정답의 88% 를
        # 버린다**(실측 2026-11-13). 그래서 "얼마나 큰가" 를 버리는 기준이 아니라
        # **얼마나 확신해야 받는가** 의 기준으로 바꾼다. 선언하지 않으면 지금까지와
        # 똑같이 동작한다: max_area_ratio 로 자르고 conf 하나를 쓴다.
        self.area_bands = self._area_bands(cfg.get("area_bands"))
        if self.area_bands and self.max_area is not None:
            raise SystemExit("models.river: area_bands 와 max_area_ratio 를 함께 적었습니다 — "
                             "상한의 주인이 둘이 됩니다. 마지막 band 의 upto 가 상한입니다")
        self.area_cap = self.area_bands[-1][0] if self.area_bands else self.max_area
        self.max_det = int(cfg.get("max_det", 100))

        # 클래스 합침 — **병합 전**에 이름을 바꾼다. 한 물체에 캔과 부탄이 따로 붙은
        # 자리가 하나로 접히려면 이름이 먼저 같아져야 한다(클래스가 다르면 합치지 않는다).
        self.merge_classes = self._class_merge(cfg.get("merge_classes"))

        # 클래스별·문구별 임계값. 적지 않은 자리는 conf 다. 합쳐진 클래스의 임계값은
        # **합친 뒤 이름**으로 찾는다 — 합치기 전 이름의 임계값은 아무 일도 하지 않는다.
        # **클래스별 근거 요구.** conf 는 "얼마나 확신하나" 하나만 묻는데, 어떤
        # 클래스는 그것으로 모자란다. 부탄은 실측에서 최소변 중앙 32 px(실내)·14 px
        # (하천)에 89~100% 가 64 px 미만이었다 — 그 크기에는 부탄과 음료캔을 가를
        # 형상이 없고, 확신값만 올리면 큰 오검출은 그대로 남고 작은 진짜도 같이 죽는다.
        # 그래서 **크기와 동의**를 따로 요구할 수 있게 한다. 적지 않은 클래스는
        # 지금까지와 똑같다.
        self.class_rules = self._class_rules(cfg.get("class_rules"))
        self.class_conf = self._thresholds(cfg.get("class_conf"), self.classes, "class_conf")
        self.prompt_conf = self._thresholds(cfg.get("prompt_conf"), self.prompts, "prompt_conf")

        shared = {"imgsz": int(cfg.get("imgsz", 1920)),
                  "base_conf": float(cfg.get("arm_conf", 0.05)),
                  "max_det": self.max_det,
                  # arm 이 이름을 검사할 때 보는 것은 **선언된 classes** 다. owners 는
                  # '문구를 가진 클래스' 라서 문구 없는 가설 칸이 거기 없다
                  "classes": tuple(self.classes)}
        floor = shared["base_conf"]
        low = [(n, v) for n, v in list(self.class_conf.items()) + list(self.prompt_conf.items())
               + [("area_band<=%g" % u, c) for u, c in self.area_bands]
               if v < floor]
        if low:
            raise SystemExit("models.river: arm_conf(%.3f) 보다 낮은 임계값은 아무 일도 하지 "
                             "않습니다 — arm 이 먼저 버립니다: %s"
                             % (floor, ", ".join("%s %.3f" % t for t in low)))
        if floor > self.conf:
            raise SystemExit("models.river: arm_conf(%.3f) 가 conf(%.3f) 보다 큽니다 — arm 이 먼저 "
                             "버린 검출은 동의에 쓸 수 없습니다" % (floor, self.conf))
        specs = list(cfg.get("arms") or [])
        if not specs:
            raise SystemExit("models.river: arms 가 비었습니다")
        on_missing = str(cfg.get("on_missing_arm", "degrade")).lower()
        if on_missing not in ("degrade", "refuse"):
            raise SystemExit("models.river.on_missing_arm 은 degrade 또는 refuse: %s" % on_missing)

        self.arms, self.missing = [], []
        for spec in specs:
            kind = str(spec.get("kind") or "").lower()
            if kind not in ARM_KINDS:
                raise SystemExit("models.river arm kind 를 모릅니다: %s (가능: %s)"
                                 % (kind, ", ".join(sorted(ARM_KINDS))))
            label = str(spec.get("name") or kind)
            try:
                arm = ARM_KINDS[kind](dict(spec, kind=kind), self.prompts, self.owners,
                                      device, shared)
            except _ArmMissing as exc:
                if on_missing == "refuse":
                    raise SystemExit("models.river arm %s: %s (on_missing_arm: refuse)" % (label, exc))
                self.missing.append((label, str(exc)))
                log("  river arm %s 를 뺍니다 — %s" % (label, exc))
                continue
            self.arms.append((label, arm))
        if not self.arms:
            raise SystemExit("models.river: 쓸 수 있는 arm 이 없습니다 — "
                             + "; ".join("%s: %s" % m for m in self.missing))

        # **가설 칸** — 프리 프롬프트 arm 이 지도 밖 이름을 여기로 보낸다(AI-S-04).
        # 텍스트 문구를 가진 이름은 가설이 될 수 없다: 문구가 있으면 텍스트 arm 도
        # 그 이름을 내고, 그러면 "이름을 고르지 않았다" 가 거짓이 된다.
        # **위험물 후보 검증** — 선언이 없으면 세우지 않는다. 한 모델이 없어도 나머지로
        # 돌고, 전부 없으면 검증만 비활성이 된다(AI-C-05). 검출을 지우지 않는다.
        self.verifier = None
        vspec = cfg.get("verify") or {}
        if vspec.get("enabled"):
            try:
                from hazard_verify import HazardVerifier
            except ImportError:
                import importlib.util as _ilu
                _sp = _ilu.spec_from_file_location(
                    "hazard_verify", Path(__file__).with_name("hazard_verify.py"))
                _m = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_m)
                HazardVerifier = _m.HazardVerifier
            self.verifier = HazardVerifier(vspec, log=log)
            if not self.verifier.ready:
                for name, why in self.verifier.missing:
                    log("    verify 빠짐: %s — %s" % (name, why))

        # **거리 — 주기로 재고 트랙으로 나른다.** 선언이 없으면 세우지 않는다.
        self.depth = None
        dspec = cfg.get("depth") or {}
        if dspec.get("enabled"):
            import importlib.util as _ilu
            _sp = _ilu.spec_from_file_location(
                "depth_track", Path(__file__).with_name("depth_track.py"))
            _m = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_m)
            self.depth = _m.DepthTracker(dspec, log=log)

        self.hypothesis = {a.unconfirmed for _, a in self.arms
                           if getattr(a, "unconfirmed", None)}
        text_arms = [l for l, a in self.arms if not getattr(a, "free_vocab", False)]
        named = sorted(h for h in self.hypothesis
                       if text_arms and h in self.owners)
        if named:
            raise SystemExit("models.river: 가설 칸에 텍스트 문구가 있습니다 — %s "
                             "(prompts 에 null 로 적습니다)" % ", ".join(named))
        want = int(cfg.get("min_arms", 2))
        if want < 1:
            raise SystemExit("models.river.min_arms 는 1 이상입니다: %d" % want)
        self.min_prompts = int(cfg.get("min_prompts", 1))
        if self.min_prompts < 1:
            raise SystemExit("models.river.min_prompts 는 1 이상입니다: %d" % self.min_prompts)
        if self.min_prompts > 1 and len(self.arms) == 1:
            # 한 모델은 한 박스에 **이름 하나**를 붙인다(어휘 전체에 대한 argmax).
            # 그래서 문구 둘이 같은 자리를 가리키는 일이 사실상 없다 — tune 117장
            # 에서 TP 0 · FP 6 이었다. 세우면 아무것도 검출하지 않는다.
            raise SystemExit("models.river.min_prompts(%d) 는 arm 이 하나일 때 쓸 수 없습니다 "
                             "— 한 모델은 한 박스에 문구 하나만 붙입니다(측정: tune 에서 "
                             "검출 0). 동의를 원하면 arms 를 늘립니다." % self.min_prompts)
        self.min_arms_declared = want
        self.arms_declared = len(specs)
        self.min_arms = min(want, len(self.arms))
        self.arm_names = [label for label, _ in self.arms]
        log("  river %d종 · 문구 %d(미끼 %d) · arm %d/%d (%s) · 동의 %d/%d"
            % (len(self.classes), len(self.prompts) - len(self.decoys), len(self.decoys),
               len(self.arms), self.arms_declared,
               "+".join(self.arm_names), self.min_arms, len(self.arms)))
        if self.merge_classes:
            log("  river 클래스 합침: "
                + ", ".join("%s → %s" % t for t in sorted(self.merge_classes.items()))
                + " (합쳐진 이름은 결과에 나오지 않습니다)")
        tuned = {k: v for k, v in self.class_conf.items() if v != self.conf}
        log("  river 임계 conf %.2f%s%s"
            % (self.conf,
               (" · 클래스별 " + ", ".join("%s %.2f" % t for t in sorted(tuned.items()))) if tuned else "",
               (" · 문구별 " + ", ".join("%r %.2f" % t for t in sorted(self.prompt_conf.items())))
               if self.prompt_conf else ""))
        if self.area_bands:
            log("  river 크기별 하한: "
                + " · ".join("<=%g %.2f" % b for b in self.area_bands)
                + " (그 위는 장면 검출로 버린다)")
        elif self.max_area:
            log("  river 면적 상한 %.3f — 크기별 하한을 쓰지 않습니다" % float(self.max_area))
        if self.missing:
            # arm 이 빠지면 같은 min_arms 가 **다른 작동점**이 된다. 3/4 와 3/3 은
            # 다른 기준이고, 3/3 은 만장일치다 — 선언한 것보다 엄해진다. 조용히
            # 지나가면 숫자가 어느 작동점의 것인지 알 수 없다.
            log("  ** river 작동점이 바뀌었습니다: 선언 %d/%d → 적용 %d/%d%s. 빠진 arm: %s **"
                % (want, self.arms_declared, self.min_arms, len(self.arms),
                   " (만장일치)" if self.min_arms == len(self.arms) else "",
                   ", ".join("%s(%s)" % m for m in self.missing)))

    def _class_merge(self, spec):
        """합칠 클래스 → 받을 클래스. 둘 다 classes 안에 있어야 하고 사슬은 만들지 않는다.

        사슬(a→b, b→c)을 허용하면 적용 순서에 따라 결과가 달라진다. 자기 자신으로
        보내는 것도 거절한다 — 아무 일도 하지 않는 선언은 읽는 사람을 속인다.
        """
        if not spec:
            return {}
        if not isinstance(spec, dict):
            raise SystemExit("models.river.merge_classes 는 이름 → 이름 이어야 합니다")
        known = set(self.classes)
        out = {}
        for src, dst in spec.items():
            src, dst = str(src), str(dst)
            for name in (src, dst):
                if name not in known:
                    raise SystemExit("models.river.merge_classes 에 classes 에 없는 이름이 "
                                     "있습니다: %r (가능: %s)" % (name, ", ".join(self.classes)))
            if src == dst:
                raise SystemExit("models.river.merge_classes: %r 를 자기 자신으로 보냅니다" % src)
            out[src] = dst
        chain = [(s, d) for s, d in out.items() if d in out]
        if chain:
            raise SystemExit("models.river.merge_classes 에 사슬이 있습니다 (%s) — 적용 순서에 "
                             "따라 결과가 달라집니다. 한 단계로 적습니다"
                             % ", ".join("%s→%s→%s" % (s, d, out[d]) for s, d in chain))
        return out

    @staticmethod
    def _thresholds(spec, allowed, field):
        """클래스·문구 → 임계값. 모르는 이름은 거절한다 (오타가 조용히 무시되면 안 된다)."""
        if not spec:
            return {}
        if not isinstance(spec, dict):
            raise SystemExit("models.river.%s 는 이름 → 값 이어야 합니다" % field)
        lower = {str(a).strip().lower(): str(a) for a in allowed}
        out = {}
        for key, value in spec.items():
            canon = lower.get(str(key).strip().lower())
            if canon is None:
                raise SystemExit("models.river.%s 에 없는 이름이 있습니다: %r (가능: %s)"
                                 % (field, key, ", ".join(map(str, allowed))))
            out[canon] = float(value)
        return out

    @staticmethod
    def _area_bands(spec):
        """`[{upto, conf}, ...]` — 화면에서 차지하는 비율이 upto 이하인 상자에 요구하는
        하한. 마지막 upto 를 넘는 상자는 **장면 검출**로 버린다(옛 max_area_ratio 의 일).
        upto 는 커지는 차례여야 한다 — 뒤섞이면 어느 칸이 이겼는지 설명할 수 없다."""
        out = []
        for i, band in enumerate(spec or []):
            if not isinstance(band, dict) or "upto" not in band or "conf" not in band:
                raise SystemExit("models.river.area_bands[%d] 는 {upto, conf} 입니다: %r" % (i, band))
            upto, conf = float(band["upto"]), float(band["conf"])
            if not 0 < upto <= 1:
                raise SystemExit("models.river.area_bands[%d].upto 는 0 초과 1 이하입니다: %g" % (i, upto))
            if out and upto <= out[-1][0]:
                raise SystemExit("models.river.area_bands 의 upto 는 커지는 차례여야 합니다: "
                                 "%g 뒤에 %g" % (out[-1][0], upto))
            out.append((upto, conf))
        return out

    def _class_rules(self, spec):
        """`{클래스: {min_short_px, min_arms}}` — 그 클래스를 받기 위한 **추가** 조건.
        conf 는 class_conf 가 소유하므로 여기서 다시 받지 않는다(주인을 둘로 두지 않는다)."""
        out = {}
        known = {c.strip().lower(): c for c in self.classes}
        for key, rule in (spec or {}).items():
            canon = known.get(str(key).strip().lower())
            if canon is None:
                raise SystemExit("models.river.class_rules 에 없는 클래스가 있습니다: %r "
                                 "(가능: %s)" % (key, ", ".join(self.classes)))
            if not isinstance(rule, dict):
                raise SystemExit("models.river.class_rules.%s 는 {min_short_px, min_arms} "
                                 "입니다: %r" % (canon, rule))
            if "conf" in rule:
                # 먼저 묻는다 — "모르는 항목" 으로 떨어뜨리면 어디에 적어야 하는지를
                # 말해 주지 못한다
                raise SystemExit("models.river.class_rules.%s.conf 는 두지 않습니다 — "
                                 "하한의 주인은 class_conf 입니다" % canon)
            bad = set(rule) - {"min_short_px", "min_arms"}
            if bad:
                raise SystemExit("models.river.class_rules.%s 에 모르는 항목: %s "
                                 "(가능: min_short_px, min_arms)" % (canon, ", ".join(sorted(bad))))
            out[canon] = {"min_short_px": float(rule.get("min_short_px") or 0),
                          "min_arms": int(rule.get("min_arms") or 0)}
        return out

    def area_threshold(self, area_ratio):
        """그 크기의 상자에 요구하는 하한. 상한 밖이면 None — 받지 않는다."""
        for upto, conf in self.area_bands:
            if area_ratio <= upto:
                return conf
        return None

    def threshold(self, cls_name):
        return self.class_conf.get(cls_name, self.conf)

    @staticmethod
    def _said(det):
        """어느 문구가 말했는가. 문구를 돌려주지 않는 arm 은 **클래스 이름 자체**로 센다 —
        빈 집합으로 두면 min_prompts 1 이 그 검출을 말없이 버린다."""
        return det.get("prompt") or det["cls_name"]

    def _merge(self, raw, img_area):
        """같은 클래스가 merge_iou 이상 겹치면 하나로 보고 **어느 arm·문구가** 말했는지 모은다.

        클래스가 다른 겹침은 합치지 않는다 — 캔과 부탄은 서로 다른 조치를 부르고,
        둘을 한 칸에 합치면 그 구분이 사라진다.
        """
        kept = []
        for det in sorted(raw, key=lambda d: -d["conf"]):
            x1, y1, x2, y2 = det["xyxy"]
            if self.area_cap and (x2 - x1) * (y2 - y1) > float(self.area_cap) * img_area:
                continue                               # 장면 전체를 가리킨 검출
            dup = None
            for other in kept:
                if other["cls_name"] != det["cls_name"]:
                    continue
                if float(box_iou([other["xyxy"]], [det["xyxy"]])[0, 0]) >= self.merge_iou:
                    dup = other
                    break
            if dup is None:
                kept.append({"cls_name": det["cls_name"], "conf": det["conf"],
                             "xyxy": det["xyxy"], "arms": {det["arm"]},
                             # **이긴 낱말.** raw 는 conf 내림차순이므로 묶음을 연 것이
                             # 가장 확신이 높은 검출이다. `prompts` 는 집합이라 누가
                             # 이겼는지가 사라진다 — 가설 칸은 이름이 전부 `unconfirmed`
                             # 라서 이 낱말이 없으면 사람이 볼 것이 아무것도 없다
                             "lead": self._said(det),
                             # **낱말별 확신을 들고 간다.** 집합만 남기면 "무엇이
                             # 얼마나 그렇게 불렀는지" 가 사라져 사람이 후보를 견줄 수
                             # 없다. 같은 낱말이 여러 번이면 가장 높은 것을 남긴다.
                             "votes": {self._said(det): det["conf"]},
                             "prompts": {self._said(det)},
                             "merged_from": ({det["merged_from"]} if det.get("merged_from")
                                             else set())})
            else:
                dup["arms"].add(det["arm"])
                word = self._said(det)
                dup["prompts"].add(word)
                if det["conf"] > dup["votes"].get(word, 0.0):
                    dup["votes"][word] = det["conf"]
                if det.get("merged_from"):
                    dup["merged_from"].add(det["merged_from"])
        return kept

    def __call__(self, bgr):
        t0 = time.perf_counter()
        raw, per_arm_ms, dropped, folded, decoyed = [], {}, 0, 0, 0
        for label, arm in self.arms:
            a0 = time.perf_counter()
            try:
                found = arm(bgr)
            except Exception as exc:                   # 한 arm 의 실패가 나머지를 멈추지 않는다 (AI-C-05)
                log("  river arm %s 실패: %s: %s" % (label, type(exc).__name__, exc))
                found = []
                per_arm_ms[label] = None
            else:
                per_arm_ms[label] = round((time.perf_counter() - a0) * 1000, 1)
            for det in found:
                # 문구별 하한은 **병합 전**에 쓴다 — 문구는 raw 검출의 성질이고,
                # 병합 뒤에는 어느 문구의 conf 였는지가 최대값에 가려진다.
                bar = self.prompt_conf.get(det.get("prompt"))
                if bar is not None and det["conf"] < bar:
                    dropped += 1
                    continue
                if det["cls_name"] == DECOY:
                    # 배경이 가져간 것 — 어휘에 둔 이유가 이것이다. 세기만 하고 버린다
                    decoyed += 1
                    continue
                merged_to = self.merge_classes.get(det["cls_name"])
                if merged_to:
                    det = dict(det, cls_name=merged_to, merged_from=det["cls_name"])
                    folded += 1
                raw.append(dict(det, arm=label))
        img_area = float(bgr.shape[0] * bgr.shape[1])
        dets, by_area, by_rule = [], 0, 0
        for group in self._merge(raw, img_area):
            # **적은 것이 안 적은 것을 이긴다.** class_conf 는 운영자가 그 클래스를
            # 두고 한 말이고 크기별 하한은 아무 말이 없는 자리의 기본값이다. 위험물에
            # 낮은 class_conf 를 적어 두면 코앞의 큰 위험물도 그 낮은 하한으로 받는다 —
            # 크기가 그것을 덮으면 재현율을 위해 적은 값이 조용히 무효가 된다.
            bar, declared = self.threshold(group["cls_name"]), group["cls_name"] in self.class_conf
            if self.area_bands:
                x1, y1, x2, y2 = group["xyxy"]
                band = self.area_threshold((x2 - x1) * (y2 - y1) / img_area)
                if band is None:
                    continue                           # 상한 밖 (상자는 _merge 에서 이미 걸린다)
                if not declared:
                    bar = band
            rule = self.class_rules.get(group["cls_name"])
            if rule:
                x1, y1, x2, y2 = group["xyxy"]
                if min(x2 - x1, y2 - y1) < rule["min_short_px"]:
                    by_rule += 1                       # 볼 수 없는 크기다
                    continue
                if len(group["arms"]) < rule["min_arms"]:
                    by_rule += 1                       # 한 arm 만으로는 올리지 않는다
                    continue
            if (len(group["arms"]) < self.min_arms
                    or len(group["prompts"]) < self.min_prompts
                    or group["conf"] < bar):
                by_area += bar > self.conf and group["conf"] >= self.conf and not declared
                continue
            dets.append({"cls": self.classes.index(group["cls_name"]), "name": group["cls_name"],
                         "conf": round(float(group["conf"]), 4),
                         "xyxy": [round(float(v), 1) for v in group["xyxy"]],
                         "agree": len(group["arms"]), "arms": sorted(group["arms"]),
                         "lead": group["lead"],
                         "votes": {k: round(float(v), 4) for k, v in
                                   sorted(group["votes"].items(), key=lambda t: -t[1])},
                         "prompts": sorted(group["prompts"])})
            if group.get("merged_from"):
                # 어느 이름으로 잡혀 합쳐졌는지 남긴다 — 합침을 되돌려 보려면 필요하고,
                # 남기지 않으면 "부탄" 이 부탄으로 잡힌 것인지 캔이 넘어온 것인지 모른다
                dets[-1]["merged_from"] = sorted(group["merged_from"])
        # **같은 자리에 확정이 있으면 가설은 내보내지 않는다.** 클래스가 다른 겹침은
        # 병합되지 않으므로(캔과 부탄을 한 칸에 합치지 않기 위한 규칙이다) 이 걸음이
        # 없으면 텍스트 arm 이 이름을 댄 물체마다 프리 arm 의 `unconfirmed` 가 하나씩
        # 더 붙어 **한 물체가 둘로 세어진다**. 가설은 이름을 모르는 자리에만 남는다.
        covered = 0
        if self.hypothesis and len(dets) > 1:
            firm = [d for d in dets if d["name"] not in self.hypothesis]
            if firm:
                keep = []
                for det in dets:
                    if det["name"] in self.hypothesis and max(
                            float(box_iou([det["xyxy"]], [f["xyxy"]])[0, 0]) for f in firm
                    ) >= self.merge_iou:
                        covered += 1
                        continue
                    keep.append(det)
                dets = keep
        dets.sort(key=lambda d: (-d["agree"], -d["conf"]))
        del dets[self.max_det:]
        # **잘라낸 뒤에 검증한다** — 내보내지 않을 상자를 임베딩할 이유가 없다(AI-C-13)
        # 거리가 먼저다 — 검증 근거에 거리를 쓸 수 있어야 한다
        depth_ms = self.depth(bgr, dets) if self.depth else 0.0
        verify_ms = self.verifier(bgr, dets) if self.verifier else 0.0
        ms = (time.perf_counter() - t0) * 1000
        return {"infer_ms": round(ms, 1), "arms": list(self.arm_names),
                "arm_ms": per_arm_ms, "missing_arms": [m[0] for m in self.missing],
                "arms_declared": self.arms_declared,
                "min_arms_declared": self.min_arms_declared,
                "min_arms_applied": self.min_arms,
                "unanimous": self.min_arms == len(self.arms), "conf": self.conf,
                # 어느 작동점의 숫자인지 뒤에서 알 수 있어야 한다. 조용히 바뀌면
                # 같은 설정 이름의 두 측정이 다른 것을 잰 것이 된다
                "area_bands": [{"upto": u, "conf": c} for u, c in self.area_bands],
                "area_cap": self.area_cap,
                "dropped_by_area": by_area,
                "verify_ms": verify_ms,
                "depth_ms": depth_ms,
                "depth_space": getattr(self.depth, "space", None) if self.depth else None,
                "verify_reused": getattr(self.verifier, "last_reused", 0) if self.verifier else 0,
                "class_rules": {k: dict(v) for k, v in self.class_rules.items()},
                "dropped_by_class_rule": by_rule,
                "class_conf": dict(self.class_conf), "prompt_conf": dict(self.prompt_conf),
                "min_prompts": self.min_prompts, "prompts_declared": len(self.prompts),
                "prompt_dropped": dropped, "merge_classes": dict(self.merge_classes),
                "decoys": list(self.decoys), "decoy_dropped": decoyed,
                "merged_dets": folded,
                "hypothesis": sorted(self.hypothesis),
                "hypothesis_covered": covered,
                "prompt": "text"}, dets

RUNNERS = {"yoloe": YoloeRunner, "river": RiverRunner, "river_open": RiverRunner,
           "unidepth": UniDepthRunner, "moge2_aerial": MogeAerialRunner}


def active_detector(names):
    """돌릴 모델 목록·적재된 runner 중 검출 모델 하나를 고른다. 없으면 None."""
    return next((m for m in DETECT_MODELS if m in names), None)



# ---------------------------------------------------------------- depth 통계 · 시각화 · 결합
def depth_stats(depth):
    v = depth[(depth > 0) & np.isfinite(depth)]
    if v.size == 0:
        return {"valid_ratio": 0.0}
    p = np.percentile(v, [5, 50, 95])
    return {"min": round(float(v.min()), 3), "p5": round(float(p[0]), 3), "median": round(float(p[1]), 3),
            "p95": round(float(p[2]), 3), "max": round(float(v.max()), 3),
            "valid_ratio": round(v.size / float(depth.size), 4)}


def colorize(depth, vis):
    valid = (depth > 0) & np.isfinite(depth)
    lo_p, hi_p = vis.get("range_percentile", [2, 98])
    if valid.any():
        lo, hi = np.percentile(depth[valid], [lo_p, hi_p])
    else:
        lo, hi = 0.0, 1.0
    hi = max(hi, lo + 1e-3)
    norm = np.clip((depth - lo) / (hi - lo), 0, 1)
    if vis.get("near_is_warm", True):
        norm = 1 - norm
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), COLORMAPS.get(vis.get("colormap", "turbo"), cv2.COLORMAP_TURBO))
    color[~valid] = 0
    return color, float(lo), float(hi)


def colorbar(width, lo, hi, vis):
    bar_h = max(18, width // 40)
    ramp = np.linspace(0, 1, width)
    if vis.get("near_is_warm", True):
        ramp = 1 - ramp
    bar = cv2.applyColorMap((ramp * 255).astype(np.uint8)[None, :].repeat(bar_h, 0),
                            COLORMAPS.get(vis.get("colormap", "turbo"), cv2.COLORMAP_TURBO))
    fs = bar_h / 30.0
    for i in range(5):
        x = int((width - 1) * i / 4)
        label = ("%.2fm" if hi - lo < 2 else "%.1fm") % (lo + (hi - lo) * i / 4)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        tx = min(max(0, x - tw // 2), width - tw)
        cv2.putText(bar, label, (tx, bar_h - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bar, label, (tx, bar_h - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


# ---------------------------------------------------------------- 오버레이 (config overlay)
LABEL_TOKENS = ("name", "conf", "cls", "id", "depth", "range", "gps", "alt", "ned")


def check_overlay_cfg(cfg):
    for kind in DETECT_MODELS + ("depth",):
        for line in label_spec(cfg.get("overlay") or {}, kind):
            bad = [t for t in line if t not in LABEL_TOKENS]
            if bad:
                raise SystemExit("overlay.%s.label 에 모르는 항목 %s (가능: %s)" % (kind, bad, ", ".join(LABEL_TOKENS)))


def label_spec(ocfg, kind):
    """overlay.<kind>.label → 줄 목록 [[항목, ...], ...]. 한 줄짜리 평평한 목록도 받는다"""
    spec = (ocfg.get(kind) or {}).get("label")
    if spec is None and kind in DETECT_MODELS:
        spec = (ocfg.get("yoloe") or {}).get("label")       # 검출 오버레이는 yoloe 설정을 같이 쓴다
    if spec is None:
        spec = ([["name", "conf"]] if kind in DETECT_MODELS
                else [["name", "conf", "range"], ["gps"]])
    if spec and not isinstance(spec[0], (list, tuple)):
        spec = [spec]
    return [list(line) for line in spec]


def label_token(tok, d, dec):
    """검출 하나의 표시 문자열. 값이 없으면 None (그 항목을 뺀다)"""
    if tok == "name":
        return d["name"]
    if tok == "conf":
        return "%.2f" % d["conf"]
    if tok == "cls":
        return "#%d" % d["cls"]
    if tok == "id":
        return "id%d" % d["track_id"] if d.get("track_id") is not None else None
    if tok == "depth":
        return "Z%.2fm" % d["depth_m"] if d.get("depth_m") is not None else None
    if tok == "range":
        v = d.get("range_m", d.get("depth_m"))         # GPS 계산을 안 하면 range_m 이 없다 → Z 로
        return "%.2fm" % v if v is not None else None
    if tok == "gps":
        return "%.*f,%.*f" % (dec, d["lat"], dec, d["lon"]) if d.get("lat") is not None else None
    if tok == "alt":
        return "alt%.1fm" % d["alt_m"] if d.get("alt_m") is not None else None
    if tok == "ned":
        return "N%.1f E%.1f D%.1f" % tuple(d["ned_m"]) if d.get("ned_m") else None
    return None


def class_color(cls):
    from ultralytics.utils.plotting import colors
    return colors(int(cls), True)                      # BGR, ultralytics 와 같은 색


def draw_dets(img, dets, ocfg, kind):
    """박스 + config 로 고른 라벨 줄들. overlay.min_conf 미만은 그림에서만 뺀다"""
    h, w = img.shape[:2]
    fs = float(ocfg.get("font_scale") or max(0.4, w / 1600.0))
    lw = int(ocfg.get("line_width") or max(1, round(w / 640)))
    min_conf = ocfg.get("min_conf")
    dec = int(ocfg.get("gps_decimals", 6))
    spec = label_spec(ocfg, kind)
    for d in dets or []:
        if min_conf is not None and d["conf"] < float(min_conf):
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        color = class_color(d["cls"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
        lines = [" ".join(t for t in (label_token(k, d, dec) for k in line) if t) for line in spec]
        lines = [s for s in lines if s]
        if not lines:
            continue
        sizes = [cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0] for s in lines]
        lh = max(th for _, th in sizes) + 6
        y = y1 - lh * len(lines)
        if y < 0:                                       # 위에 자리가 없으면 박스 안쪽 위
            y = y1
        text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
        for s, (tw, _) in zip(lines, sizes):
            x = min(max(0, x1), max(0, w - tw - 4))
            cv2.rectangle(img, (x, y), (x + tw + 4, y + lh), color, -1)
            cv2.putText(img, s, (x + 2, y + lh - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, text_color, 1, cv2.LINE_AA)
            y += lh
    return img


def pose_line(rec, dec):
    """depth 오버레이 위쪽에 드론 위치·자세 한 줄"""
    st, p = rec.get("geo_status"), rec.get("pose")
    if st is None:
        return None
    if not p:
        return "pose: %s" % st
    if p.get("lat") is not None and st != "no_fix":
        g = "GPS %.*f,%.*f fix%s" % (dec, p["lat"], dec, p["lon"], p.get("fix_type"))
    else:
        g = "GPS no fix(%s)" % p.get("fix_type")
    parts = [g]
    if p.get("relative_m") is not None:
        parts.append("rel %.1fm" % p["relative_m"])
    if p.get("yaw_deg") is not None:
        parts.append("rpy %.1f %.1f %.1f" % (p["roll_deg"], p["pitch_deg"], p["yaw_deg"]))
    if p.get("dt_s") is not None:
        parts.append("dt %.2fs" % p["dt_s"])
    if st != "ok":
        parts.append("[%s]" % st)
    return "  ".join(parts)


def add_header(img, lines, ocfg):
    """제목·드론 위치 줄을 영상 위 검은 띠에 쓴다 (영상 위에 쓰면 위쪽 박스 라벨을 가린다)"""
    lines = [s for s in lines if s]
    if not lines:
        return img
    fs = float(ocfg.get("font_scale") or max(0.4, img.shape[1] / 1600.0))
    lh = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][1] + 10
    band = np.zeros((lh * len(lines) + 4, img.shape[1], 3), np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(band, s, (6, lh * (i + 1) - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, img])


def detect_overlay(bgr, dets, title, ocfg, kind="yoloe"):
    out = draw_dets(bgr.copy(), dets, ocfg, kind)
    return add_header(out, [title if ocfg.get("title", True) else None], ocfg)


def depth_overlay(bgr, depth, dets, title, vis, ocfg, rec):
    color, lo, hi = colorize(depth, vis)
    if vis.get("mode", "blend") == "side":
        out = np.hstack([bgr, color])
    else:
        a = float(vis.get("alpha", 0.6))
        out = cv2.addWeighted(color, a, bgr, 1 - a, 0)
    draw_dets(out, dets, ocfg, "depth")
    out = add_header(out, [title if ocfg.get("title", True) else None,
                           pose_line(rec, int(ocfg.get("gps_decimals", 6)))
                           if (ocfg.get("depth") or {}).get("pose", True) else None], ocfg)
    if vis.get("colorbar", True):
        out = np.vstack([out, colorbar(out.shape[1], lo, hi, vis)])
    return out


def attach_depth(dets, depth, fusion):
    """검출마다 마스크(없으면 박스 가운데) 안 depth 의 백분위를 거리(Z)로, 마스크 무게중심을 대표 픽셀(uv)로 붙인다.
    박스 영역만 잘라서 계산한다 (검출이 많아도 전체 크기 마스크를 만들지 않는다)."""
    h, w = depth.shape
    out = []
    erode = int(fusion.get("mask_erode_px", 3))
    pct = float(fusion.get("depth_percentile", 40))
    min_ratio = float(fusion.get("min_valid_ratio", 0.2))
    kernel = np.ones((2 * erode + 1, 2 * erode + 1), np.uint8) if erode > 0 else None
    for d in dets:
        e = {k: v for k, v in d.items() if not k.startswith("_") and k != "polygon"}
        e["depth_m"], e["uv"] = None, None
        out.append(e)
        bx1, by1, bx2, by2 = d["xyxy"]
        # 잘라 낸 영역 가장자리에서도 깎이도록 erode 만큼 여유를 둔다
        x0, y0 = max(0, int(bx1) - erode - 1), max(0, int(by1) - erode - 1)
        x3, y3 = min(w, int(np.ceil(bx2)) + erode + 2), min(h, int(np.ceil(by2)) + erode + 2)
        if x3 <= x0 or y3 <= y0:
            continue
        m = np.zeros((y3 - y0, x3 - x0), np.uint8)
        poly = d.get("_poly")
        if poly is not None and len(poly) >= 3:
            cv2.fillPoly(m, [np.round(np.asarray(poly) - (x0, y0)).astype(np.int32)], 1)
            if kernel is not None:
                eroded = cv2.erode(m, kernel)
                if eroded.any():
                    m = eroded
        else:
            cx, cy, bw, bh = (bx1 + bx2) / 2 - x0, (by1 + by2) / 2 - y0, (bx2 - bx1) / 4, (by2 - by1) / 4
            m[max(0, int(cy - bh)):int(cy + bh) + 1, max(0, int(cx - bw)):int(cx + bw) + 1] = 1
        ys, xs = np.nonzero(m)
        if not xs.size:
            continue
        e["uv"] = [round(float(xs.mean()) + x0, 1), round(float(ys.mean()) + y0, 1)]
        region = depth[y0:y3, x0:x3][ys, xs]
        valid = region[(region > 0) & np.isfinite(region)]
        if valid.size / float(region.size) >= min_ratio:
            e["depth_m"] = round(float(np.percentile(valid, pct)), 3)
    return out


def build_geo(cfg):
    """기기 → GeoLocator. config geo.enabled 이고 geo.devices 에 장착 정보가 있는 기기만"""
    g = cfg.get("geo") or {}
    if not g.get("enabled", True):
        return {}
    return {dv: GeoLocator(dg or {}, g) for dv, dg in (g.get("devices") or {}).items()}


# ---------------------------------------------------------------- 한 장 추론 (오프라인 · live 공통)
def infer_frame(bgr, base, header, device, save_detect, depth_models, runners, cfg, pose_fn=None):
    """한 장에 모델을 적용해 [(모델, 기록, 오버레이, depth 또는 None)] 을 돌려준다.
    save_detect  : 검출 결과를 돌려줄지 (False 여도 depth 결합에 필요하면 검출은 돌린다).
                   True/False 또는 검출 모델 이름 집합 — 집합이면 그 안에 있는 것만 저장한다
    depth_models : 돌릴 depth 모델
    pose_fn      : pose_fn(프레임 ts) → (드론 위치·자세, ts 차이 s). 검출 GPS 계산에 쓴다 (geo.py)"""
    fusion = cfg.get("fusion", {}) or {}
    vis = cfg.get("depth_vis", {}) or {}
    ocfg = cfg.get("overlay", {}) or {}
    dev_cfg = (cfg.get("devices", {}) or {}).get(device, {})
    locator = (cfg.get("_geo") or {}).get(device)
    w = bgr.shape[1]
    out = []
    dets = None
    stab = None
    det_model = active_detector(runners)
    if isinstance(save_detect, (set, frozenset, list, tuple)):
        wanted = det_model in set(save_detect)
    else:
        wanted = bool(save_detect)
    if det_model and (wanted or (depth_models and fusion.get("enabled", True))):
        info, dets = runners[det_model](bgr)
        sc = stabilize_cfg(cfg, det_model)
        if sc.get("enabled", True):
            key = (device, base.get("run"), det_model)
            stab = cfg.setdefault("_stab", {}).get(key)
            if stab is None:
                backend = str(sc.get("backend", "builtin")).lower()
                try:
                    stab = (Stabilizer(sc) if backend == "builtin" else VendorTracker(sc))
                except _ArmMissing as exc:             # 트래커만 끈다 (AI-C-05)
                    log("  트래킹을 끕니다 — %s" % exc)
                    stab = False
                cfg["_stab"][key] = stab
            ts = frame_ts(header)
            lab = cfg.setdefault("_lab", {}).get(key)
            if lab is None:
                lab = cfg["_lab"][key] = LabelStabilizer(sc)
            dets = lab.update(dets, ts if ts is not None else time.time())
            if stab:
                dets = (stab.update(dets, ts if ts is not None else time.time(), bgr)
                        if isinstance(stab, VendorTracker)
                        else stab.update(dets, ts if ts is not None else time.time()))
        if wanted:
            rec = dict(base, model=det_model, **info)
            rec["detections"] = [{k: v for k, v in d.items() if not k.startswith("_")} for d in dets]
            title = "%s  %s/%s #%s" % (det_model, device, base.get("run") or "live", base.get("n"))
            out.append((det_model, rec,
                        detect_overlay(bgr, rec["detections"], title, ocfg, det_model), None))
    fov = fov_hint(dev_cfg, header)
    pose = []                                           # 한 장에 한 번만 찾는다 (depth 모델이 여럿이어도)

    def add_depth_fields(rec, depth, K, w):
        rec["depth_stats"] = depth_stats(depth)
        rec["intrinsics"] = np.round(K.astype(np.float64), 3).tolist()
        fx = fov_from_k(K, w)
        rec["fov_x_deg"] = round(fx, 2) if fx else None
        rec["fov_x_given"] = fov

    def add_geo(rec, dets_, K):
        if locator is None:
            return
        if not pose:
            ts = frame_ts(header)
            p, dt = pose_fn(ts) if pose_fn else (None, None)
            max_dt = float((cfg.get("geo") or {}).get("max_state_dt_s", 0.5))
            stale = p is not None and dt is not None and abs(dt) > max_dt
            pose.append((ts, None if p is None else dict(p, dt_s=None if dt is None else round(dt, 3)), stale))
        ts, p, stale = pose[0]
        st = locator.locate(dets_, K, None if stale else p)
        rec["frame_ts"] = ts
        rec["pose"] = p
        rec["geo_status"] = "stale_state" if stale else st

    for m in depth_models:
        info, depth, K = runners[m](bgr, fov_x=fov)
        rec = dict(base, model=m, **info)
        add_depth_fields(rec, depth, K, w)
        fused = None
        if dets is not None and fusion.get("enabled", True):
            fused = attach_depth(dets, depth, fusion)
            if stab:                                   # 트래킹이 꺼졌거나 못 섰으면 건너뛴다
                stab.smooth_depth(fused, m)
            add_geo(rec, fused, K)
            rec["detections"] = fused
        title = "%s  %s/%s #%s" % (m, device, base.get("run") or "live", base.get("n"))
        out.append((m, rec, depth_overlay(bgr, depth, fused, title, vis, ocfg, rec), depth))
    return out


# ---------------------------------------------------------------- 처리
def try_lock(path):
    """<실행>/vision/<모델>/.lock 을 잡는다. 다른 프로세스가 잡고 있으면 None.
    프로세스가 죽으면 OS 가 풀어 주므로 남은 잠금을 지울 일은 없다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def process_run(device, run, models, runners, cfg, args):
    frames = read_frame_list(run)
    if args.stride > 1:
        frames = frames[::args.stride]
    if not frames:                                  # 빈 실행 폴더에는 결과 폴더를 만들지 않는다
        return 0
    dirname = cfg["output"]["dirname"]
    # 같은 실행 폴더·같은 모델은 한 프로세스만 처리한다. 다른 터미널에서 같은 모델을 돌리고 있으면
    # 그 모델만 이번에 건너뛰고 (--watch 면 다음 확인 때 다시 시도) 나머지 모델은 그대로 처리한다.
    locks = {}
    for m in models:
        lk = try_lock(run / dirname / m / ".lock")
        key = (run, m)
        if lk is not None:
            locks[m] = lk
            args._busy.discard(key)
        elif key not in args._busy:
            args._busy.add(key)
            log("%s/%s %s — 다른 프로세스가 처리 중이라 건너뜀" % (device, run.name, m))
    try:
        return _process_run(device, run, [m for m in models if m in locks], runners, cfg, args, frames)
    finally:
        for lk in locks.values():
            lk.close()


def _process_run(device, run, own, runners, cfg, args, frames):
    """own = 이 프로세스가 잠금을 잡은 모델 (결과를 쓰는 모델)"""
    if not own:
        return 0
    dirname = cfg["output"]["dirname"]
    outs = {m: ModelOutput(run, dirname, m, args.overwrite and m not in args._overwritten.get(run, set()),
                           cfg["output"].get("jpeg_quality", 90))
            for m in own}
    args._overwritten.setdefault(run, set()).update(own)      # --watch 에서 다시 돌 때 또 지우지 않도록
    todo = [(p, h) for p, h in frames if any(p.relative_to(run).as_posix() not in outs[m].done for m in own)]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        for o in outs.values():
            o.close()
        return 0
    busy = [m for m in own if any(p.relative_to(run).as_posix() not in outs[m].done for p, _ in todo)]
    log("%s/%s %s — %d장 처리 (전체 %d) → %s" % (device, run.name, "+".join(busy), len(todo), len(frames),
                                             run / dirname))
    states = DiskStates(run / "states.jsonl") if device in (cfg.get("_geo") or {}) else None
    pose_fn = states.lookup if states is not None else None
    done = 0
    t_start = time.time()
    for i, (path, header) in enumerate(todo):
        if STOP:
            break
        rel = path.relative_to(run).as_posix()
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        base = {"n": frame_number(path), "frame": rel, "source": device, "run": run.name}
        if bgr is None:
            for m in own:
                if rel not in outs[m].done:
                    outs[m].write(dict(base, model=m, error="이미지를 읽지 못함", _emitted_at=iso_ms()), header=header)
            continue
        h, w = bgr.shape[:2]
        base["image_wh"] = [w, h]
        need_depth = [m for m in own if m in DEPTH_MODELS and rel not in outs[m].done]
        save_detect = {m for m in DETECT_MODELS if m in outs and rel not in outs[m].done}
        # 검출을 다른 프로세스가 맡고 있어도 결합용 검출은 여기서 돌린다 (저장은 안 함)
        results = infer_frame(bgr, base, header, device, save_detect, need_depth, runners, cfg, pose_fn)
        emitted_at = iso_ms()                               # 저장본도 결과가 나온 순간을 적는다 (live 는 done_ms 자리)
        for m, rec, overlay, depth in results:
            mcfg = cfg["models"][m]
            rec["_emitted_at"] = emitted_at
            outs[m].write(rec, overlay, depth if mcfg.get("save_depth", True) else None,
                          mcfg.get("depth_dtype", "float16"), header=header)
        done += 1
        if done % 20 == 0 or i == len(todo) - 1:
            el = time.time() - t_start
            log("  %s/%s %d/%d  (%.2f장/s)" % (device, run.name, done, len(todo), done / max(el, 1e-6)))
    for o in outs.values():
        o.close()
    return done


# ---------------------------------------------------------------- live (실시간)
class LiveRunner:
    """Redis camera_stream:<기기> 에서 기기마다 가장 최근 프레임만 꺼내 추론하고
    결과를 Redis vision_stream:<기기>:<모델> (+ depth 는 vision_depth:<기기>:<모델>) 에 넣고
    디스크 <실행>/vision/<모델>/ 에도 오프라인과 같은 형식으로 저장한다.
    처리하는 사이에 들어온 프레임은 건너뛴다 (그 프레임은 나중에 --runs 로 오프라인 처리)."""

    def __init__(self, devices, models, runners, cfg, data_root):
        import redis
        self.redis_mod = redis
        lc = cfg.get("live", {}) or {}
        self.lc = lc
        self.url = lc.get("redis", "redis://127.0.0.1:6380/0")
        self.r = redis.Redis.from_url(self.url, socket_connect_timeout=2, socket_timeout=10)
        self.devices, self.models, self.runners, self.cfg = devices, models, runners, cfg
        self.data_root = data_root
        self.maxlen = int(lc.get("maxlen", 300))
        self.depth_maxlen = int(lc.get("depth_maxlen", 30))
        self.max_age = float(lc.get("max_age_s", 5.0))
        self.save = bool(lc.get("save", True))
        self.jpeg = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg["output"].get("jpeg_quality", 90))]
        mf = lc.get("max_fps")                          # null | 숫자 | {기기: 숫자}
        self.min_gap = {}
        for dv in devices:
            v = mf.get(dv) if isinstance(mf, dict) else mf
            self.min_gap[dv] = 1.0 / float(v) if v else 0.0
        self.last_id = dict((dv, None) for dv in devices)
        self.last_t = dict((dv, 0.0) for dv in devices)
        self.run_dir = dict((dv, None) for dv in devices)
        self.outs = dict((dv, {}) for dv in devices)      # 기기 → {모델: ModelOutput}
        self.locks = dict((dv, {}) for dv in devices)
        self.lock_retry = {}
        self.warned = set()
        self.stat = dict((dv, [0, 0.0, 0.0]) for dv in devices)   # 처리 장 수, lag 합, 추론 합 (보고 사이)

    def warn_once(self, key, msg):
        if key not in self.warned:
            self.warned.add(key)
            log(msg)

    # -- 디스크: 실행 폴더가 바뀌면 (서버 재시작) 잠금·결과 파일을 새 폴더로 옮긴다
    def _release(self, dv):
        for o in self.outs[dv].values():
            o.close()
        for lk in self.locks[dv].values():
            lk.close()
        self.outs[dv], self.locks[dv] = {}, {}

    def outputs(self, dv, run_dir):
        if not self.save or run_dir is None:
            return {}
        if run_dir != self.run_dir[dv]:
            self._release(dv)
            self.run_dir[dv] = run_dir
            log("%s → 저장 %s" % (dv, run_dir / self.cfg["output"]["dirname"]))
        dirname = self.cfg["output"]["dirname"]
        now = time.time()
        for m in self.models:
            if m in self.outs[dv] or now < self.lock_retry.get((dv, m), 0):
                continue
            lk = try_lock(run_dir / dirname / m / ".lock")
            if lk is None:                              # 오프라인 작업이 이 실행 폴더·모델을 잡고 있음
                self.lock_retry[(dv, m)] = now + 10
                self.warn_once(("lock", run_dir, m), "%s/%s %s — 다른 프로세스가 잡고 있어 디스크 저장은 쉼 "
                               "(Redis 스트림은 계속, 10초마다 다시 시도)" % (dv, run_dir.name, m))
                continue
            self.locks[dv][m] = lk
            self.outs[dv][m] = ModelOutput(run_dir, dirname, m, False, self.cfg["output"].get("jpeg_quality", 90))
        return self.outs[dv]

    def pose(self, dv, ts):
        """Redis state_stream:<기기> 최근 항목 중 프레임 ts 에 가장 가까운 드론 위치·자세"""
        try:
            return redis_lookup(self.r, "state_stream:" + dv, ts, int((self.cfg.get("geo") or {}).get("redis_scan", 64)))
        except self.redis_mod.RedisError as e:
            self.warn_once(("state", dv), "%s state_stream 을 못 읽음 (%s) — GPS 계산 없이 계속" % (dv, e))
            return None, None

    # -- 한 장
    def handle(self, dv, eid, fields):
        image = fields.get(b"image")
        try:
            header = json.loads(fields.get(b"header") or b"{}")
        except ValueError:
            header = {}
        bgr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR) if image else None
        if bgr is None:
            self.warn_once(("decode", dv), "%s 이미지를 못 풀어 건너뜀 (id %s)" % (dv, eid.decode()))
            return
        recv_ms = int(eid.split(b"-")[0])
        run_dir = rel = None
        ff = header.get("frame_file") or header.get("file")    # v2 DS 기록 · 옛 DS 기록
        if ff:
            p = self.data_root / ff
            if p.parent.parent.is_dir():
                run_dir, rel = p.parent.parent, "frames/" + p.name
            else:
                self.warn_once(("nodir", dv), "%s 저장 폴더가 없음 (%s) — config data_root 가 서버 --save 와 같은지 확인. "
                               "Redis 스트림만 냅니다" % (dv, p.parent.parent))
        outs = self.outputs(dv, run_dir)
        h, w = bgr.shape[:2]
        base = {"n": header.get("n"), "frame": rel, "source": dv, "run": run_dir.name if run_dir else None,
                "image_wh": [w, h], "live": True}
        results = infer_frame(bgr, base, header, dv,
                              {m for m in DETECT_MODELS if m in self.models},
                              [m for m in self.models if m in DEPTH_MODELS], self.runners, self.cfg,
                              lambda ts: self.pose(dv, ts))
        done_ms = int(time.time() * 1000)
        pipe = self.r.pipeline(transaction=False)
        infer_ms = 0.0
        emitted_at = iso_ms(done_ms / 1000.0)
        for m, rec, overlay, depth in results:
            rec["_emitted_at"] = emitted_at
            infer_ms += rec.get("infer_ms") or 0
            jpg = cv2.imencode(".jpg", overlay, self.jpeg)[1].tobytes()
            mcfg = self.cfg["models"][m]
            dtype = mcfg.get("depth_dtype", "float16")
            o = outs.get(m)
            if o is not None and rel is not None and rel not in o.done:
                o.write(rec, depth=depth if (depth is not None and mcfg.get("save_depth", True)) else None,
                        depth_dtype=dtype, overlay_jpg=jpg, header=header)
            short = json.dumps({"stream": dv, "model": m, "n": base["n"], "run": base["run"],
                                "frame_file": ff, "stream_entry_id": eid.decode(),
                                "lag_ms": done_ms - recv_ms}, ensure_ascii=False)       # N6 — lag_ms 는 결과가 아니라 여기
            pipe.xadd("vision_stream:%s:%s" % (dv, m),
                      {"result": json.dumps(v2_record(rec, header), ensure_ascii=False), "overlay": jpg, "header": short},
                      maxlen=self.maxlen, approximate=True)
            if depth is not None:                       # depth 원본 (미터, 0 = 무효). 디스크 depth/*.npy 와 같은 값
                d = np.ascontiguousarray(depth.astype(dtype))
                pipe.xadd("vision_depth:%s:%s" % (dv, m),
                          {"depth": d.tobytes(), "dtype": d.dtype.str, "shape": "%d,%d" % d.shape, "unit": "m",
                           "header": short},
                          maxlen=self.depth_maxlen, approximate=True)
        try:
            pipe.execute()
        except self.redis_mod.RedisError as e:
            self.warn_once(("xadd", str(e)[:60]), "Redis 에 결과를 못 넣음 (%s) — 디스크 저장은 계속" % e)
        st = self.stat[dv]
        st[0] += 1
        st[1] += done_ms - recv_ms
        st[2] += infer_ms

    # -- 반복
    def report(self, dt):
        parts = []
        for dv in self.devices:
            n, lag, inf = self.stat[dv]
            if n:
                parts.append("%s %.1f장/s 지연 %.2fs (추론 %.0fms)" % (dv, n / dt, lag / n / 1000.0, inf / n))
            self.stat[dv] = [0, 0.0, 0.0]
        log("live  " + (" · ".join(parts) if parts else "새 프레임 없음 (서버가 보내는 중인지 확인)"))

    def run(self):
        keys = dict((dv, "camera_stream:" + dv) for dv in self.devices)
        every = float(self.lc.get("report_interval_s", 10.0))
        t_rep = time.time()
        redis_err = None
        log("live 시작 — 읽기 %s  쓰기 vision_stream:<기기>:<모델> (최근 %d) · vision_depth:<기기>:<모델> (최근 %d)"
            % (self.url, self.maxlen, self.depth_maxlen))
        try:
            while not STOP:
                now = time.time()
                if now - t_rep >= every:
                    self.report(now - t_rep)
                    t_rep = now
                worked = False
                wait = 0.5
                try:
                    for dv in self.devices:
                        if STOP:
                            break
                        gap = self.min_gap[dv] - (time.time() - self.last_t[dv])
                        if gap > 0:                         # max_fps 상한
                            wait = min(wait, gap)
                            continue
                        e = self.r.xrevrange(keys[dv], "+", "-", count=1)
                        if not e or e[0][0] == self.last_id[dv]:
                            continue
                        eid, fields = e[0]
                        self.last_id[dv] = eid
                        if time.time() - int(eid.split(b"-")[0]) / 1000.0 > self.max_age:
                            continue                        # 오래된 항목 (서버가 멈췄거나 이전 실행의 것)
                        self.last_t[dv] = time.time()
                        self.handle(dv, eid, fields)
                        worked = True
                    if not worked and not STOP:
                        if wait < 0.5:
                            time.sleep(max(wait, 0.005))
                        else:                               # 새 프레임이 올 때까지 기다린다
                            ids = dict((keys[dv], self.last_id[dv] or "$") for dv in self.devices)
                            self.r.xread(ids, count=1, block=500)
                    if redis_err is not None:
                        log("Redis 다시 됨")
                        redis_err = None
                except self.redis_mod.RedisError as e:
                    if redis_err is None:
                        log("Redis 안 됨 (%s) — 2초마다 다시 시도" % e)
                    redis_err = e
                    time.sleep(2.0)
        finally:
            for dv in self.devices:
                self._release(dv)


# ---------------------------------------------------------------- 실행
def normalize(values, allowed, aliases, what):
    out = []
    for v in values:
        k = v.strip().lower()
        if k == "all":
            return list(allowed)
        k = aliases.get(k, k)
        if k not in allowed:
            raise SystemExit("알 수 없는 %s: %s  (가능: %s, all)" % (what, v, ", ".join(allowed)))
        if k not in out:
            out.append(k)
    return out


def load_config(path):
    path = Path(path).expanduser().resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    root = Path(cfg.get("data_root", "../stream_data")).expanduser()
    cfg["data_root"] = root if root.is_absolute() else (path.parent / root).resolve()
    cfg.setdefault("output", {}).setdefault("dirname", "vision")
    return cfg


def main():
    global STOP
    ap = argparse.ArgumentParser(description="스트림 프레임에 YOLOE / UniDepth / MoGe2-Aerial 추론 (기본: 실시간)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--config", default=str(HERE / "vision_config.yaml"), metavar="파일")
    ap.add_argument("--devices", "-d", nargs="+", metavar="기기",
                    help="추론할 기기: %s 또는 all (기본: config defaults.devices)" % " ".join(DEVICES))
    ap.add_argument("--models", "-m", nargs="+", metavar="모델",
                    help="적용할 모델, 여러 개 가능: %s 또는 all (기본: config defaults.models)" % " ".join(MODELS))
    ap.add_argument("--runs", "-r", nargs="+", metavar="실행",
                    help="live = Redis 실시간 스트림의 최신 프레임 (기본) / 저장된 이미지: "
                         "all | latest | 20261001_062842 ... (기본: config defaults.runs)")
    ap.add_argument("--data-root", metavar="폴더", help="stream_data 폴더 (기본: config data_root)")
    ap.add_argument("--gpu", metavar="장치", help="cuda:0 / cpu (기본: config runtime.device)")
    ap.add_argument("--stride", type=int, default=1, help="[저장된 이미지] N장마다 한 장만 처리 (기본 1)")
    ap.add_argument("--limit", type=int, default=0, help="[저장된 이미지] 실행 폴더마다 최대 N장 (0 = 전부)")
    ap.add_argument("--overwrite", action="store_true", help="[저장된 이미지] 고른 모델의 기존 결과를 지우고 처음부터")
    ap.add_argument("--watch", action="store_true",
                    help="[저장된 이미지] 끝나도 멈추지 않고 새로 저장되는 프레임을 계속 처리")
    ap.add_argument("--list", action="store_true", help="기기·실행 폴더·프레임 수만 보여 주고 끝낸다")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else cfg["data_root"]
    defaults = cfg.get("defaults", {}) or {}
    devices = normalize(args.devices or defaults.get("devices", list(DEVICES)), DEVICES, DEVICE_ALIASES, "기기")
    want_runs = args.runs or defaults.get("runs", "live")
    want_runs = [want_runs] if isinstance(want_runs, str) else [str(r) for r in want_runs]
    live = want_runs == ["live"]
    if "live" in want_runs and not live:
        raise SystemExit("--runs live 는 다른 실행 폴더와 같이 줄 수 없습니다")

    if args.list:
        for dv in devices:
            for r in list_runs(data_root, dv):
                done = sorted(p.name for p in (r / cfg["output"]["dirname"]).glob("*")) \
                    if (r / cfg["output"]["dirname"]).is_dir() else []
                print("%-7s %s  프레임 %5d  결과: %s" % (dv, r.name, len(read_frame_list(r)), ", ".join(done) or "-"))
        return 0

    models = normalize(args.models or defaults.get("models", ["yoloe"]), MODELS, MODEL_ALIASES, "모델")
    models = [m for m in MODELS if m in models]           # 검출을 먼저 돌려야 결합이 된다
    picked = [m for m in models if m in DETECT_MODELS]
    if len(picked) > 1:
        log("검출 모델이 둘입니다 (%s) — 결과는 둘 다 저장하지만 depth 결합·GPS 는 %s 검출에만 붙습니다"
            % (" ".join(picked), picked[0]))
    if live:
        ignored = [o for o, on in (("--watch", args.watch), ("--overwrite", args.overwrite),
                                   ("--limit", args.limit), ("--stride", args.stride > 1)) if on]
        if ignored:
            log("live 에서는 %s 를 쓰지 않습니다 (저장된 이미지 처리 --runs all|latest|<실행> 전용)" % " ".join(ignored))
        try:
            import redis  # noqa: F401
        except ImportError:
            raise SystemExit("live 에는 redis 패키지가 필요합니다 (pip install redis)")
    gpu = args.gpu or (cfg.get("runtime", {}) or {}).get("device", "cuda:0")
    if gpu.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            log("CUDA 를 못 찾아 cpu 로 돌립니다")
            gpu = "cpu"

    def on_signal(*_):
        global STOP
        if STOP:
            sys.exit(1)
        STOP = True
        log("멈추는 중 — 지금 장까지 저장하고 끝냅니다 (한 번 더 누르면 바로 종료)")
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    log("data_root=%s  기기=%s  모델=%s  %s  장치=%s"
        % (data_root, " ".join(devices), " ".join(models),
           "실시간(live)" if live else "저장된 이미지=" + " ".join(want_runs), gpu))
    cfg["_geo"] = build_geo(cfg)
    check_overlay_cfg(cfg)
    load = list(models)
    if (active_detector(load) is None and any(m in DEPTH_MODELS for m in load)
            and (cfg.get("fusion") or {}).get("enabled", True)):
        helper = str((cfg.get("fusion") or {}).get("detector") or "yoloe")
        if helper not in DETECT_MODELS:
            raise SystemExit("fusion.detector 는 %s 중 하나입니다: %s"
                             % (" 또는 ".join(DETECT_MODELS), helper))
        load.insert(0, helper)                          # depth 결합·GPS 계산용 검출. 결과 자체는 저장하지 않는다
        log("depth 결합용으로 %s 도 적재 (%s 결과는 저장하지 않음 — 저장하려면 --models 에 추가)"
            % (helper, helper))
    runners = {}
    for m in load:
        t0 = time.time()
        log("모델 적재: %s" % m)
        runners[m] = RUNNERS[m](cfg["models"][m], gpu)
        log("  %s 준비 (%.1fs)" % (m, time.time() - t0))

    if live:
        LiveRunner(devices, models, runners, cfg, data_root).run()
        log("끝")
        return 0

    args._overwritten = {}
    args._busy = set()
    interval = float((cfg.get("runtime", {}) or {}).get("watch_interval_s", 2.0))
    total = 0
    while not STOP:
        n = 0
        for dv in devices:
            for run in pick_runs(list_runs(data_root, dv), want_runs):
                if STOP:
                    break
                n += process_run(dv, run, models, runners, cfg, args)
        total += n
        if not args.watch:
            break
        if n == 0:
            time.sleep(interval)
    log("끝 — 총 %d장 처리" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
