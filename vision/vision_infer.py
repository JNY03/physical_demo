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

  # 1) 터미널 1 — 스트림 서버 (로봇·드론 주소는 config/hosts.json, 옛 이름 hosts.local.json)
  . <venv>/bin/activate          # 저장소의 venv (또는 컨테이너의 python)
  python server_stream_multi_source.py

  # 2) 터미널 2 — 실시간 비전
  . <venv>/bin/activate
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
import math
import os
import re
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if str(HERE.parent) not in sys.path:                      # ds_resources.settings (경로 · 설정 · 기기 이름)
    sys.path.append(str(HERE.parent))
from ds_resources import settings  # noqa: E402
from geo import DiskStates, GeoLocator, frame_ts, redis_lookup, to_epoch  # noqa: E402
DEVICES = ("cam360", "drone", "robot1", "ep-001", "realsense", "cam360_slice",   # cam360_slice 2026-11-11 (DS 10006) · ep-001 2026-11-24 (DS 10004, robot2 자리)
           "a2",                                                                 # a2 2026-11-12 (DS 10007, Unitree A2)
           "pi11", "pi12",                                                        # 모듈 카메라 파이 2026-11-12 (DS 10008·10009)
           "pi13",                                                                # 모듈 카메라 파이 2026-11-13 (DS 10011)
           "drone2")                                                              # 둘째 드론 ds5 x500-002 2026-11-15 (DS 10012)
MODELS = ("yoloe", "river", "targeted", "unidepth", "moge2_aerial")
DEVICE_ALIASES = {"camera": "cam360", "360": "cam360",
                  "pi6": "realsense", "rs": "realsense", "d455": "realsense",
                  "slice": "cam360_slice",
                  "robot2": "ep-001", "ep": "ep-001", "pi1": "ep-001"}
MODEL_ALIASES = {"yoloe-pf": "yoloe", "yoloe_pf": "yoloe",
                 "moge": "moge2_aerial", "moge2": "moge2_aerial", "moge2-aerial": "moge2_aerial",
                 "moge2aerial": "moge2_aerial",
                 "river-zeroshot": "river", "river_zeroshot": "river", "하천": "river",
                 "target": "targeted", "custom": "targeted", "지정": "targeted", "지정클래스": "targeted"}
DEPTH_MODELS = ("unidepth", "moge2_aerial")
#: 검출 모델. depth 결합(fusion)은 이 중 **먼저 오는 하나**의 검출을 쓴다 —
#: 둘을 같이 돌리면 결과는 둘 다 저장되지만 거리·GPS 는 한쪽에만 붙는다.
DETECT_MODELS = ("yoloe", "river", "targeted")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
COLORMAPS = {"turbo": cv2.COLORMAP_TURBO, "jet": cv2.COLORMAP_JET, "inferno": cv2.COLORMAP_INFERNO,
             "magma": cv2.COLORMAP_MAGMA, "viridis": cv2.COLORMAP_VIRIDIS, "plasma": cv2.COLORMAP_PLASMA}

STOP = False
HOSTNAME = settings.host_label()                 # DS_NODE_NAME > 호스트 이름
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
            import torch
            self.model = YOLOE(self.weight)
            learned = self._learned_pe(pr.get("learned_pe"))
            texts = [c for c in classes if c.lower() not in learned]
            text_pe = {}
            if texts:
                # ultralytics 는 텍스트 인코더를 작업 폴더나 weights_dir 에서 찾는다 (없으면 받는다 ~570MB)
                with contextlib.chdir(pr.get("text_encoder_dir") or Path(self.weight).parent):
                    pe = self.model.get_text_pe(texts)
                text_pe = {c: pe[0, i] for i, c in enumerate(texts)}
            dev = next(self.model.model.parameters()).device
            rows = [learned[c.lower()].to(dev) if c.lower() in learned else text_pe[c].to(dev) for c in classes]
            self.model.set_classes(classes, torch.stack(rows)[None])
            log("  yoloe 텍스트 프롬프트 %d종: %s" % (len(classes), ", ".join(
                c + (" (학습 임베딩)" if c.lower() in learned else "") for c in classes)))
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

    def _learned_pe(self, path):
        """prompt.learned_pe — vision/train_custom_pe.py 가 만든 NPZ 를 {소문자 이름: 벡터} 로.

        'go1' · 'insta 360' 처럼 텍스트 인코더가 모르는 고유명은 박스 친 이미지에서 뽑은
        시각 프롬프트 임베딩으로 찾는다. classes 에 그 이름이 있으면 텍스트 임베딩 대신 쓴다.
        임베딩은 만든 가중치 계열에 묶여 있다 — 다른 가중치에 넣으면 조용히 엉뚱한 것을 찾으므로 거절한다."""
        if not path:
            return {}
        import torch
        path = Path(str(path)).expanduser()
        if not path.is_file():
            raise SystemExit("models.yoloe.prompt.learned_pe 파일이 없습니다: %s" % path)
        with np.load(path, allow_pickle=False) as z:
            emb, names, made_for = z["embeddings"], [str(n) for n in z["names"]], str(z["model"])
        mine = Path(self.weight).stem
        mine = mine[:-4] if mine.endswith("-seg") else mine
        if made_for != mine:
            raise SystemExit("learned_pe 는 %s 용인데 text_weight 는 %s 입니다: %s" % (made_for, mine, path))
        return {n.strip().lower(): torch.from_numpy(emb[0, i].copy()).float() for i, n in enumerate(names)}

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


#: features.track 로 켤 때의 기본값. 지정 클래스 · river 와 같은 작동점(번호만 붙이고 거르지 않는다)에
#: **이름 유지(hyst-2)** 를 더한다 — 런처에서 이 기능을 켜는 까닭이 「탐지가 장마다 바뀐다」이기 때문이다.
#: 2026-11-26 pi12 실측: 끄면 라벨 목록이 31% 의 장에서 바뀌었다(close-up ↔ sky). hyst-2 는 지금 이름을 두고
#: 다른 이름이 2장 연속 이겨야 바꾼다. 라벨 자료에서는 macro F1 −0.046 이라 기본값으로 켜지 않는다 —
#: 켜는 것은 사람이 고를 때뿐이고, vision_config 의 stabilize.<모델> 값이 있으면 그쪽이 이긴다.
TRACK_ON = {"backend": "bytetrack", "on_untracked": "keep", "keep_boxes": "detection",
            "reset_gap_s": 2.0, "label_policy": "hyst-2"}


def stabilize_cfg(cfg, model):
    """`stabilize` 의 바탕값 위에 `stabilize.<model>` 을 얹는다.

    모델마다 흔들리는 모양이 다르다 — river 는 수십 px 짜리가 움직이는 드론에서
    장마다 들락거리고, yoloe prompt-free 는 큰 물체를 꾸준히 잡는다. 하나의
    값으로 둘을 맞추면 한쪽은 너무 질기고 다른 쪽은 너무 성기다.
    """
    sc = cfg.get("stabilize") or {}
    base = {k: v for k, v in sc.items() if not isinstance(v, dict)}
    out = {**base, **(sc.get(model) or {})}
    # 런처 기능 「추적」(features.track) — 그 모델에 stabilize 블록이 없어도 켠다. 가중치가 필요 없고
    # (트래커는 ultralytics 안에 있다) 번호만 붙이는 층이라 박스는 탐지기 것을 그대로 낸다
    if (cfg.get("_features") or {}).get("track") and not out.get("enabled"):
        out = {**TRACK_ON, **out, "enabled": True}
    return out


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


# UniDepth get_paddings 의 int() 버그 고침 — 주인은 depth_track.py 하나(river 거리도 같은 것을 쓴다)
from depth_track import unidepth_get_paddings as _unidepth_get_paddings  # noqa: E402


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


def _mask_polys(r):
    """분할 모델 결과의 마스크 윤곽(원본 좌표, float32 N×2) 목록. 마스크가 없으면 빈 목록.
    **형상 단서**(hazard_overlay.shape_cue)가 쓴다 — 상자만으로는 볼록함·채움을 볼 수 없다."""
    m = getattr(r, "masks", None)
    if m is None:
        return []
    try:
        return [np.asarray(p, np.float32) for p in m.xy]
    except Exception:
        return []


def _arm_imgsz(arm, bgr):
    """arm 이 돌 입력 크기. `follow_frame` 이면 imgsz 를 넘지 않는 선에서 프레임 긴 변(32 배수)."""
    if not getattr(arm, "follow_frame", False):
        return arm.imgsz
    long_side = max(bgr.shape[:2])
    size = -(-long_side // 32) * 32
    floor = getattr(arm, "crop_floor", None)       # 구역을 잘라 볼 때만(_upscale_crop) — 작은 crop 을 이만큼까지 키운다
    if floor:
        size = max(size, int(floor))
    low = getattr(arm, "min_size", None)           # 작은 프레임 하한(targeted imgsz_min) — 먼 물체가 후보 하한에 못 미치지 않게
    if low:
        size = max(size, int(low))
    return int(min(arm.imgsz, size))


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
        # **프레임 크기 근처로 돈다**(`imgsz_follow_frame`, 2026-11-13). imgsz 를 상한으로 두고
        # 프레임 긴 변(32 배수로 올림)이 더 작으면 그것을 쓴다. 464×400 GO1 프레임을 1920 으로
        # 키우면 275 ms, 480 이면 그 몇 분의 일이다. 1280×720 은 1920 → 1280.
        self.follow_frame = bool(spec.get("imgsz_follow_frame", False))
        self.conf = float(spec.get("conf", shared["base_conf"]))
        self.max_det = int(spec.get("max_det") or shared["max_det"])
        self.name = weight.name

    def __call__(self, bgr):
        r = self.model.predict(bgr, conf=self.conf, imgsz=_arm_imgsz(self, bgr), max_det=self.max_det,
                               device=self.device, verbose=False)[0]
        out = []
        polys = _mask_polys(r)
        if r.boxes is not None and len(r.boxes):
            for i, b in enumerate(r.boxes):
                index = int(b.cls)
                if 0 <= index < len(self.owners):
                    out.append({"cls_name": self.owners[index], "prompt": self.prompts[index],
                                "conf": float(b.conf),
                                "xyxy": [float(v) for v in b.xyxy[0].tolist()],
                                "poly": polys[i] if i < len(polys) else None})
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
        # **화면을 덮는 상자는 내지 않는다**(`max_area_ratio`, 2026-11-13). 프리 어휘는
        # 바닥·벽(tile wall · floor)을 물체처럼 크게 낸다 — 그것이 후보가 되면 단서
        # 점수가 장면 전체의 색을 재게 된다. 텍스트 arm 의 상한(area_bands)과 따로 둔다.
        mar = spec.get("max_area_ratio")
        self.max_area_ratio = float(mar) if mar is not None else None
        # **장면 낱말**(`scene_words`): 바닥·벽·방 같은 **장소** 이름이 화면의 큰 부분
        # (`scene_min_area` 이상)을 덮으면 물체가 아니라 장면이다 — 버린다. 크기만으로는
        # 가까이 다가간 물체와 구분할 수 없어 **이름과 크기를 함께** 본다. 작은 상자는
        # 장면 낱말이어도 남긴다(작은 '물' 조각에 물체가 숨어 있을 수 있다).
        self.scene_words = {str(w).lower() for w in (spec.get("scene_words") or [])}
        self.scene_min_area = float(spec.get("scene_min_area", 0.10))
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
        if vocab and self.scene_words - vocab:
            raise SystemExit("river arm yoloe_free.scene_words 에 내장 어휘에 없는 이름이 "
                             "있습니다: %s" % ", ".join(sorted(self.scene_words - vocab)))
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
        H, W = bgr.shape[:2]
        cap = self.max_area_ratio * W * H if self.max_area_ratio else None
        polys = _mask_polys(r)
        if r.boxes is not None and len(r.boxes):
            for i, b in enumerate(r.boxes):
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if cap is not None and area > cap:
                    continue
                free = str(names.get(int(b.cls), int(b.cls))).lower()
                if free in self.scene_words and area >= self.scene_min_area * W * H:
                    continue                       # 장소 이름이 화면을 덮었다 — 장면이다
                cls = self.vocab_map.get(free, self.unconfirmed)
                if cls is None:
                    continue
                out.append({"cls_name": cls, "prompt": free, "conf": float(b.conf),
                            "xyxy": [float(v) for v in b.xyxy[0].tolist()],
                            "poly": polys[i] if i < len(polys) else None})
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

        # **점수로 견준다**(`assess`, 2026-11-13). 후보(위험물 이름 + 가설 칸)마다 위험물
        # 클래스 전부에 대해 같은 단서를 재고, 가장 높은 클래스가 하한 이상이면 그 이름으로
        # 의심한다. 검출기가 댄 낱말은 단서 하나일 뿐이다. 식의 주인은 hazard_overlay.assess.
        self.assess = bool(cfg.get("assess", False))
        # **기기마다 따로 기억한다**(`use_stream`, 2026-11-13). 실행기 하나를 여러 기기가 함께
        # 쓰는데 검증 캐시(앞 장과 겹치면 다시 안 잰다)와 거리 기억(트랙으로 나른다)이 하나뿐
        # 이었다 — 기기가 번갈아 올 때마다 캐시가 깨져 매번 다시 쟀고(robot1·realsense·pi13
        # 동시: 장당 ~800 ms), **다른 기기의 같은 자리 결과를 빌려 쓸 수 있었다.**
        self._stream, self._streams = None, {}
        # **부품은 전체에 접는다**(`nest_ratio`, 2026-11-13). 부탄 통 안의 뚜껑·라벨·마개가
        # 따로 잡혀 한 물체에 상자가 여럿 겹쳤다. 작은 상자가 큰 상자 안에 이 비율 이상
        # 들어가 있으면 **큰 것 하나만** 남긴다. null = 끔.
        nr = cfg.get("nest_ratio")
        self.nest_ratio = float(nr) if nr else None
        # **붙은 두 물체를 하나로 잡은 상자는 뺀다**(`group_split`, 2026-11-13). 다른 후보 둘
        # 이상을 품고(각각 이 상자 넓이의 member_min 이상) 그 후보들이 상자의 cover 이상을
        # 덮으면 그 상자는 물체가 아니라 **묶음**이다 — 나무상자와 붙은 통을 감싼 'wooden block',
        # 충전기·케이블·수첩을 감싼 'stick insect' 가 그랬다. 품은 후보들은 남는다.
        # 부품(통 안의 뚜껑 8%)은 member_min 에 못 미쳐 묶음으로 세지 않는다 — 부품 접기의 몫이다.
        gs = cfg.get("group_split")
        self.group_split = dict(gs) if gs else None

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

    def use_stream(self, key):
        """이번 장이 어느 기기(실행)의 것인지 알린다 — 그 기기의 검증 캐시·거리 기억으로 바꾼다."""
        if key == self._stream:
            return
        v, dp = self.verifier, self.depth
        if self._stream is not None:
            self._streams[self._stream] = {
                "v_last": getattr(v, "_last", None), "d_mem": getattr(dp, "_mem", None),
                "d_n": getattr(dp, "_n", None)}
        st = self._streams.get(key) or {}
        if v is not None:
            v._last = st.get("v_last") or []
        if dp is not None:
            dp._mem = st.get("d_mem") or {}
            dp._n = st.get("d_n") or 0
        self._stream = key

    def _reassess_masked(self, dets, bgr, cards):
        """**안에 든 다른 물체를 가리고 다시 잰다**(2026-11-13).

        상자 D 가 자기와 크게 겹친 다른 물체 M(검출기가 그 클래스 이름으로 부름, 겹친 넓이가 D 의
        member_min 이상)의 클래스로 기울어 있으면, D 의 점수는 D 가 아니라 M 을 잰 것이다 — 나무상자 앞에 붙은 통을
        함께 감싼 상자가 '부탄' 이 되고(pi13 장면1 왼쪽 상자 52/192 장), 장면을 감싼 상자가 그 안의
        통 때문에 '부탄' 이 됐다. 그래서 M 자리를 D crop 의 중앙색으로 칠한 그림에서 D 의 단서를
        다시 재고 그 판정을 쓴다. 장면이나 물건 이름에 기대지 않는 규칙이다.
        부품(통 안의 뚜껑·라벨)은 의심되지 않거나(근거 부족) 넓이가 작아 이 걸음에 들지 않는다.
        """
        if not dets:
            return
        rule = self.group_split or {}
        mmin = float(rule.get("member_min", 0.15))
        inside = float(rule.get("inside", 0.8))

        def area(b):
            return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

        def inter(a, b):
            return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                    * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))

        def lean(d):
            a = d.get("assess") or {}
            s = a.get("scores") or {}
            return max((c for c in s if s[c] is not None), key=lambda c: s[c], default=None)
        H, W = bgr.shape[:2]
        for D in dets:
            if not D.get("assess"):
                continue
            c = lean(D)
            # **검출기가 이 클래스로 부른 상자는 가리지 않는다** — 그 상자는 그 물체 전체이고, 안의
            # 의심 후보는 부품(통의 라벨·뚜껑)이다. 이름이 다른데(wooden block 인데 부탄으로 기움)
            # 또는 이름이 없는데(장면을 감싼 상자) 기운 경우만 안의 것이 끌고 간 것으로 본다.
            if c is None or D.get("name") == c:
                continue
            B, A = D["xyxy"], area(D["xyxy"])
            # 가리는 것은 **검출기가 그 클래스 이름으로 부른** 물체뿐이다 — 붙은 다른 물체(앞의 통)는
            # 이름을 받고, 부품(라벨·뚜껑)은 받지 않는다. 이름 없는 라벨을 가리면 통이 제 라벨을 잃었다
            # **안에 들지 않아도** 이 상자의 member_min 이상과 겹치면 센다 — 앞에 붙은 통은 상자 밖으로
            # 삐져나온다(pi13 장면1: 통의 위쪽만 상자에 들어 52 장이 가려지지 않았다). 겹친 곳만 칠한다
            mem = [M for M in dets if M is not D and M.get("name") == c
                   and (M.get("assess") or {}).get("suspect") == c
                   and inter(M["xyxy"], B) >= mmin * A
                   # 거의 같은 상자는 같은 물체다(IoU 0.7 이상) — 크기로 거르지 않는다: 앞의 통이
                   # 뒤 상자보다 클 수 있다(장면1 통 81k px² > 상자 76k px²)
                   and inter(M["xyxy"], B) / (A + area(M["xyxy"]) - inter(M["xyxy"], B)) < 0.7]
            if not c or not mem:
                continue
            x1, y1 = max(0, int(B[0])), max(0, int(B[1]))
            x2, y2 = min(W, int(B[2])), min(H, int(B[3]))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            img = bgr.copy()
            fill = np.median(bgr[y1:y2, x1:x2].reshape(-1, 3), axis=0).astype(np.uint8)
            for M in mem:
                m = M["xyxy"]
                ya, yb = max(y1, int(m[1])), min(y2, int(m[3]))
                xa, xb = max(x1, int(m[0])), min(x2, int(m[2]))
                if yb > ya and xb > xa:
                    img[ya:yb, xa:xb] = fill
            probe = {k: v for k, v in D.items() if k not in ("verify", "verify_by", "assess", "_poly")}
            if self.verifier is not None:
                saved = getattr(self.verifier, "_last", None)
                self.verifier._last = []                       # 캐시를 쓰지도 남기지도 않는다
                self.verifier(img, [probe])
                self.verifier._last = saved
            D["assess"] = dict(cards.assess(probe, img), masked=len(mem))
            if probe.get("verify_by") is not None:
                D["verify_by"], D["verify"] = probe["verify_by"], probe.get("verify")

    def _drop_groups(self, dets):
        """다른 후보 둘 이상으로 채워진 큰 상자(묶음)를 지운다. (남은 것, 지운 수)."""
        mmin = float(self.group_split.get("member_min", 0.15))
        inside = float(self.group_split.get("inside", 0.8))
        cover = float(self.group_split.get("cover", 0.5))

        def area(b):
            return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

        def inter(a, b):
            w = min(a[2], b[2]) - max(a[0], b[0])
            h = min(a[3], b[3]) - max(a[1], b[1])
            return max(0.0, w) * max(0.0, h)
        gone = set()
        for D in dets:
            B, A = D["xyxy"], area(D["xyxy"])
            if A <= 0:
                continue
            mem = [M for M in dets if M is not D and area(M["xyxy"]) >= mmin * A
                   and area(M["xyxy"]) < A and inter(M["xyxy"], B) >= inside * area(M["xyxy"])]
            if len(mem) < 2:
                continue
            # 서로 많이 겹치는 것은 같은 물체의 두 상자다 — 다른 물체 둘이어야 묶음이다
            distinct = [m for i, m in enumerate(mem)
                        if all(inter(m["xyxy"], o["xyxy"]) < 0.5 * min(area(m["xyxy"]), area(o["xyxy"]))
                               for o in mem[:i])]
            if len(distinct) < 2:
                continue
            # 덮는 넓이: 칸으로 나눠 센다(겹친 곳을 두 번 세지 않는다)
            g = 24
            xs = np.linspace(B[0], B[2], g + 1)
            ys = np.linspace(B[1], B[3], g + 1)
            cx, cy = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
            X, Y = np.meshgrid(cx, cy)
            hit = np.zeros_like(X, bool)
            for m in distinct:
                b = m["xyxy"]
                hit |= (X >= b[0]) & (X <= b[2]) & (Y >= b[1]) & (Y <= b[3])
            if hit.mean() >= cover:
                gone.add(id(D))
        return [d for d in dets if id(d) not in gone], len(gone)

    def _drop_nested(self, dets):
        """**부품은 전체에 접는다** — 큰 상자 안에 `nest_ratio` 이상 들어간 상자를 지운다.

        * 의심되지 않은 상자: **의심된** 큰 상자 안에 들면 지운다(통의 라벨·마개). 의심 아닌 큰 상자
          (그물·해먹 같은 장면 상자)는 안의 물체를 지우지 않는다 — 각각 제로샷 검출로 남는다(2026-11-16).
        * 같은 클래스로 의심된 것끼리 포개지면 **한 번에** 고른다(2026-11-13). 의심된 후보를
          (그 클래스 이름으로 불렸는가, 넓이) 순으로 세우고, 앞서 남긴 것과 포개진 것은 지운다 —
          이름 붙은 것과 전체가 부품보다 먼저 자리를 잡는다.
          짝마다 따로 고르면 '장면 상자가 통을 지우고 라벨이 장면 상자를 지워' 라벨만 남았다.
        * 작은 쪽이 **다른** 위험물로 의심되면 남긴다(큰 나무 토막 상자가 옆 통을 덮은 장).
        """
        def area(b):
            return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

        def nested(a, b):                            # 작은 쪽이 큰 쪽 안에 비율 이상 드는가
            small, big = (a, b) if area(a) <= area(b) else (b, a)
            iw = min(small[2], big[2]) - max(small[0], big[0])
            ih = min(small[3], big[3]) - max(small[1], big[1])
            return iw > 0 and ih > 0 and area(small) > 0 and iw * ih / area(small) >= self.nest_ratio

        def sus(d):
            return (d.get("assess") or {}).get("suspect")
        gone = set()
        # 의심 아닌 부품은 **의심된 위험물 안에서만** 접는다(2026-11-16). 예전에는 어떤 큰 상자든
        # 안에 든 것을 지웠다 — 리얼센스 하천에서 그물 · 해먹(화면 1/4) 상자가 그 안의 나무 상자 ·
        # 부탄 · 캔 후보를 장마다 17~28 개 지우고 자신은 크기 상한으로 거부됐다. 의심 아닌 물체도
        # 각각 제로샷 검출로 남아야 한다 — 큰 상자가 전체를 대신하지 않는다.
        for d in dets:
            if not sus(d) and any(o is not d and sus(o) and area(o["xyxy"]) > area(d["xyxy"])
                                  and nested(d["xyxy"], o["xyxy"]) for o in dets):
                gone.add(id(d))
        # (그 이름으로 불렸는가, 넓이) — 이름 붙은 것이 먼저, 그다음 **큰 것**(전체)이 부품보다 먼저.
        # 장면을 감싼 상자는 앞 걸음(_reassess_masked)에서 안의 이름 붙은 통을 가리고 다시 재
        # 의심이 벗겨지므로 여기까지 오지 않는다
        ranked = sorted((d for d in dets if sus(d)),
                        key=lambda d: (d.get("name") == sus(d), area(d["xyxy"])), reverse=True)
        kept = []
        for d in ranked:
            if any(sus(k) == sus(d) and nested(d["xyxy"], k["xyxy"]) for k in kept):
                gone.add(id(d))
            else:
                kept.append(d)
        return [d for d in dets if id(d) not in gone], len(gone)

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
                             "_poly": det.get("poly"),
                             "merged_from": ({det["merged_from"]} if det.get("merged_from")
                                             else set())})
            else:
                dup["arms"].add(det["arm"])
                if dup.get("_poly") is None and det.get("poly") is not None:
                    dup["_poly"] = det["poly"]
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
                         "prompts": sorted(group["prompts"]),
                         "_poly": group.get("_poly")})              # 내보내지 않는다(_ 칸)
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
        assess_ms = 0.0
        if self.assess:
            a0 = time.perf_counter()
            cards = _cards_module()
            for d in dets:
                if d["name"] in cards.HAZARD or d["name"] in self.hypothesis:
                    d["assess"] = cards.assess(d, bgr)
            self._reassess_masked(dets, bgr, cards)
            assess_ms = round((time.perf_counter() - a0) * 1000, 1)
        grouped = 0
        if self.group_split:
            dets, grouped = self._drop_groups(dets)
        nested = 0
        if self.nest_ratio:
            dets, nested = self._drop_nested(dets)
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
                "assess_ms": assess_ms,
                "nested_dropped": nested, "nest_ratio": self.nest_ratio,
                "group_dropped": grouped,
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

class AxisDecider:
    """**축 단위 판정**(지정 클래스, 2026-11-23) — 트랙이 붙은 뒤에.

    전역(외형 점수 · 게이트 전체, target_assess 가 낸 판정) · 국소(부분이 트랙의 절반 이상 장에서 잡히는가) ·
    기하(실물 크기가 선언과 같은 자릿수인가)를 **따로 센다.** 잰 축 가운데 반대하는 축이 없고 동의하는 축이 둘 이상이면
    판정한다. 국소는 그 트랙에서 지금까지 잰 장들의 다수결이다(첫 장부터 판정하고 장이 쌓일수록 다듬어진다).
    전역 근거(탐지기 · 글 설명 · 특징 사전 …)는 모두 같은 crop 의 2D 외형이라 하나로 센다 — 여럿으로 세면 외투를 건
    의자처럼 실루엣만 같은 것이 높은 점수를 받는다(robot1 2026-11-23: 2 명 장면에서 판정 평균 3.37).
    값은 모두 일반 상수다: 부분 하한 = arm 하한(0.05), 다수결 0.5, 창 30 장."""

    WINDOW, MIN_MEASURED, MAJORITY = 30, 3, 0.5

    def __init__(self):
        self.mem, self.n, self.anon = {}, 0, []

    def _key(self, d):
        if d.get("track_id") is not None:
            return ("t", d["track_id"])
        # 번호 없는 검출 — 앞 장의 번호 없는 상자와 IoU 로 잇는다(depth_track 과 같은 차선책)
        best, score = None, 0.3
        for k, box, name in self.anon:
            if name == d["name"]:
                v = float(box_iou([box], [d["xyxy"]])[0, 0])
                if v >= score:
                    best, score = k, v
        return best or ("a", self.n, round(d["xyxy"][0]), round(d["xyxy"][1]))

    def update(self, dets):
        self.n += 1
        anon = []
        for d in dets:
            a = d.get("assess") or {}
            ax = a.get("axes")
            if not ax:
                continue
            k = self._key(d)
            if k[0] != "t":
                anon.append((k, list(d["xyxy"]), d["name"]))
            if k not in self.mem:
                # **새 트랙은 같은 자리의 최근 기록을 이어받는다** — 트래커가 번호를 다시 붙일 때마다 국소 근거를
                # 처음부터 쌓느라 같은 사람이 보류됐다(robot1: 왼쪽 사람 88 장)
                prev = max(((v, float(box_iou([v["box"]], [d["xyxy"]])[0, 0])) for v in self.mem.values()
                            if v.get("name") == d["name"] and self.n - v["n"] <= 30 and v.get("box")),
                           key=lambda t: t[1], default=(None, 0.0))
                self.mem[k] = {"local": list(prev[0]["local"]) if prev[1] >= 0.3 else [], "n": self.n}
            m = self.mem[k]
            m["n"], m["box"], m["name"] = self.n, list(d["xyxy"]), d["name"]
            if ax.get("local") is not None:
                m["local"] = (m["local"] + [ax["local"]])[-self.WINDOW:]
            vals = m["local"]
            local = None
            if ax.get("local_declared") and len(vals) >= 1:
                ratio = sum(1 for v in vals if v >= ta_part_floor()) / len(vals)
                local = ratio >= self.MAJORITY
            else:
                ratio = None
            glob_ = bool(a.get("suspect"))
            geom = ax.get("geometry")
            votes = [v for v in (glob_, local, geom) if v is not None]
            pending = False             # 보류하지 않는다 — 회전 스캔처럼 트랙이 안 이어지는 곳에서는 영영 보류된다
            ok = glob_ and False not in votes and votes.count(True) >= (2 if len(votes) >= 2 else 1) and not pending
            if not ax.get("local_declared") and geom is None:
                ok = glob_                                # 국소도 기하도 없는 대상은 전역만(예전과 같다)
            name = (a.get("scores") and next(iter(a["scores"]))) or d["name"]
            word = lambda v: "-" if v is None else ("yes" if v else "no")
            txt = "global %s · local %s%s · geometry %s" % (
                word(glob_), word(local), "" if ratio is None else " %d%% of %d" % (round(100 * ratio), len(vals)),
                word(geom)) + (" · pending" if pending else "")
            card = (a.get("cards") or {}).get(name)
            if card is not None:
                card["rows"] = [r for r in card["rows"] if r[0] != "axes"] + [["axes", None, txt]]
            d["assess"] = dict(a, suspect=name if ok else None,
                               axes=dict(ax, local_ratio=None if ratio is None else round(ratio, 2),
                                         local_frames=len(vals), decision="axes",
                                         pending=bool(pending)))
        self.anon = anon
        for k in [k for k, v in self.mem.items() if self.n - v["n"] > 60]:
            del self.mem[k]
        return dets


class TrackEvidence(AxisDecider):
    """**트랙에 근거를 쌓는다**(지정 클래스, 2026-11-23). 한 물체를 오래 볼수록 판정과 점수가 근거 전체로 다듬어진다.

    장마다 target_assess 가 낸 근거 값(cue_values)을 트랙에 쌓고(최근 WINDOW 장), 트랙 단위로 다시 셈한다:
      점수 근거  최근 장들의 평균 → 대상 무게로 가중 평균 = **트랙 점수**(장 단위 점수 대신)
      필수 근거  잰 장들의 다수결로 통과 · 탈락 — 한 장의 우연한 실패 · 통과에 흔들리지 않는다
      거부      잰 장의 절반 넘게 거부면 거부
      국소 · 기하 (AxisDecider 와 같다) 다수결
    판정 = 전역(트랙 점수 ≥ 하한 · 근거 충분 · 필수 통과 · 거부 없음) + 축 규칙. 이미 판정된 트랙은 하한 − HYST 까지 유지.
    카드의 % 는 트랙 평균으로 바뀌고 'track' 줄에 몇 장의 근거인지 적는다. 장 단위 값은 assess.frame 에 남긴다.
    새 트랙은 같은 자리의 최근 기록을 잇는다(트래커가 번호를 다시 붙일 때)."""

    HYST = 5.0

    def update(self, dets):
        self.n += 1
        anon = []
        for d in dets:
            a = d.get("assess") or {}
            ax, cv = a.get("axes"), a.get("cue_values")
            if not ax or cv is None:
                continue
            k = self._key(d)
            if k[0] != "t":
                anon.append((k, list(d["xyxy"]), d["name"]))
            if k not in self.mem:
                prev = max(((v, float(box_iou([v["box"]], [d["xyxy"]])[0, 0])) for v in self.mem.values()
                            if v.get("name") == d["name"] and self.n - v["n"] <= 30 and v.get("box")),
                           key=lambda t: t[1], default=(None, 0.0))
                base = prev[0] if prev[1] >= 0.3 else None
                self.mem[k] = {"local": list(base["local"]) if base else [], "geom": list(base["geom"]) if base else [],
                               "cues": {c: list(v) for c, v in base["cues"].items()} if base else {},
                               "veto": list(base["veto"]) if base else [], "frames": base["frames"] if base else 0,
                               "sus": base["sus"] if base else None, "n": self.n}
            m = self.mem[k]
            m["n"], m["box"], m["name"] = self.n, list(d["xyxy"]), d["name"]
            m["frames"] += 1
            W = self.WINDOW
            for c, v in cv.items():
                if v["role"] in ("score", "gate"):
                    val = v["pct"] if v["role"] == "score" else v["passed"]
                    if val is not None:
                        m["cues"][c] = (m["cues"].get(c, []) + [val])[-W:]
            m["veto"] = (m["veto"] + [bool(a.get("veto_text"))])[-W:]
            if ax.get("local") is not None:
                m["local"] = (m["local"] + [ax["local"]])[-W:]
            if ax.get("geometry") is not None:
                m["geom"] = (m["geom"] + [bool(ax["geometry"])])[-W:]
            # -- 트랙 점수
            w = {c: v["w"] for c, v in cv.items() if v["role"] == "score"}
            total = sum(w.values()) or 1.0
            means = {c: sum(m["cues"][c]) / len(m["cues"][c]) for c in w if m["cues"].get(c)}
            wsum = sum(w[c] for c in means)
            score = (sum(means[c] * w[c] for c in means) / wsum) if wsum > 0 else None
            enough = wsum / total >= float(a.get("min_evidence", 0.45))
            gates = {c: (sum(m["cues"][c]) / len(m["cues"][c]) >= 0.5) if m["cues"].get(c) else None
                     for c, v in cv.items() if v["role"] == "gate"}
            failed = [c for c, ok in gates.items() if ok is not True]
            vetoed = sum(m["veto"]) * 2 > len(m["veto"])
            if vetoed and score is not None:
                score = 0.0
            shown = None if score is None else max(0.0, score - float(a.get("lookalike_pen", 0.0)))
            mn = float(a.get("min_score", 50.0))
            bar = mn - self.HYST if m.get("sus") else mn
            glob_ = score is not None and enough and not failed and not vetoed and score >= bar
            local = ratio = None
            if ax.get("local_declared") and m["local"]:
                ratio = sum(1 for v in m["local"] if v >= ta_part_floor()) / len(m["local"])
                local = ratio >= self.MAJORITY
            geom = (sum(m["geom"]) * 2 >= len(m["geom"])) if m["geom"] else None
            votes = [v for v in (glob_, local, geom) if v is not None]
            ok = glob_ and False not in votes and votes.count(True) >= (2 if len(votes) >= 2 else 1)
            name = (a.get("scores") and next(iter(a["scores"]))) or d["name"]
            m["sus"] = name if ok else None
            # -- 카드: 트랙 평균으로, 몇 장의 근거인지
            yn = lambda v: "-" if v is None else ("yes" if v else "no")
            card = (a.get("cards") or {}).get(name)
            if card is not None:
                rows = []
                for r in card["rows"]:
                    c = r[0]
                    if c in means:
                        r = [c, round(means[c], 1), "%s · avg of %d" % (r[2], len(m["cues"][c]))]
                    elif c in gates and gates[c] is not None:
                        k_pass = sum(m["cues"][c])
                        r = [c, round(100.0 * k_pass / len(m["cues"][c]), 1),
                             "%s · passed %d/%d" % (r[2].split(" · ", 1)[-1], k_pass, len(m["cues"][c]))]
                    rows.append(r)
                rows = [r for r in rows if r[0] not in ("axes", "track")]
                rows.append(["axes", None, "global %s · local %s%s · geometry %s" % (
                    yn(glob_), yn(local), "" if ratio is None else " %d%% of %d" % (round(100 * ratio), len(m["local"])),
                    yn(geom))])
                rows.append(["track", None, "evidence over %d frames" % m["frames"]])
                card = dict(card, rows=rows, score=None if shown is None else round(shown, 1))
            d["assess"] = dict(a, frame={"score": a.get("score"), "suspect": a.get("suspect"),
                                         "gates_failed": a.get("gates_failed"), "cards": a.get("cards")},
                               score=None if shown is None else round(shown, 1),
                               scores={name: None if shown is None else round(shown, 1)},
                               gate_scores={name: None if score is None else round(score, 1)},
                               suspect=name if ok else None, enough=bool(enough), gates_failed=failed,
                               vetoed="too big" if vetoed else None, evidence=round(wsum / total, 2),
                               cards={name: card} if card is not None else a.get("cards"),
                               track_frames=m["frames"],
                               axes=dict(ax, local_ratio=None if ratio is None else round(ratio, 2),
                                         local_frames=len(m["local"]), decision="track"))
        self.anon = anon
        for k in [k for k, v in self.mem.items() if self.n - v["n"] > 60]:
            del self.mem[k]
        return dets


def ta_part_floor():
    import target_assess as _ta
    return _ta.PART_FLOOR


class AssessSmoother:
    """**정지한 물체 · 정지한 카메라인데 판정이 흔들리는 것**을 막는다(2026-11-13).

    장마다 점수가 몇 점씩 오르내리면 하한 근처의 물체는 의심 ↔ 아님을 오간다. 그래서
    트랙(track_id)마다 클래스 점수를 **지수 평균**(alpha)하고, 의심은 **들어갈 때**(하한)와
    **나올 때**(하한 − hysteresis) 문턱을 달리한다. 트랙이 없는 검출은 그대로 둔다.
    날 점수는 `assess.raw_scores` 로 남긴다 — 평활이 무엇을 바꿨는지 뒤에서 볼 수 있게.
    """

    def __init__(self, spec):
        self.alpha = float(spec.get("alpha", 0.4))
        self.hyst = float(spec.get("hysteresis", 5.0))
        self.max_missed = int(spec.get("max_missed", 15))
        self.mem, self.n = {}, 0

    def update(self, dets):
        self.n += 1
        for d in dets:
            a, tid = d.get("assess"), d.get("track_id")
            if not a or tid is None:
                continue
            raw = a.get("scores") or {}
            # 판정은 **감점 전 점수**(gate_scores, 닮은꼴 감점 2026-11-16)로, 보이는 점수는 감점 뒤로.
            # 둘 다 같은 식으로 평활한다. gate 가 없는 옛 기록은 scores 를 그대로 쓴다
            rawg = a.get("gate_scores") or raw
            m = self.mem.get(tid)

            def ema(cur, prev):
                if prev is None:
                    return dict(cur)
                out = {}
                for c, v in cur.items():
                    old = prev.get(c)
                    out[c] = old if v is None else (v if old is None else
                                                    self.alpha * v + (1 - self.alpha) * old)
                return out
            sm = ema(raw, m["s"] if m else None)
            sg = ema(rawg, m.get("g") if m else None)
            best = max((c for c in sg if sg[c] is not None), key=lambda c: sg[c], default=None)
            top = sg.get(best) if best else None
            Ts = a.get("min_scores") or {}
            T = float(Ts.get(best, a.get("min_score") or 0))
            prev = m["sus"] if m else None
            if top is None or not a.get("enough", True) or a.get("gates_failed") or a.get("vetoed"):
                # 근거가 모자란 장, **필수 근거에 떨어진 장**(지정 클래스 gates_failed · 거부)은 판정하지 않는다.
                # 평활된 점수만 보면 게이트에 떨어진 외투 건 의자가 다시 사람이 됐다(2026-11-23). river 에는 이 칸이 없다
                sus = None
            elif best == prev:
                sus = best if top >= T - self.hyst else None
            elif top >= T:
                sus = best
            elif prev and sg.get(prev) is not None and sg[prev] >= float(
                    Ts.get(prev, a.get("min_score") or 0)) - self.hyst:
                sus = prev
            else:
                sus = None
            self.mem[tid] = {"s": sm, "g": sg, "sus": sus, "n": self.n}
            shown = sm.get(sus or best)
            d["assess"] = dict(a, raw_scores=raw, smoothed=True, suspect=sus,
                               min_score=float(Ts.get(sus or best, a.get("min_score") or 0)),
                               score=None if shown is None else round(float(shown), 1),
                               gate_scores={c: None if v is None else round(float(v), 1)
                                            for c, v in sg.items()},
                               scores={c: None if v is None else round(float(v), 1)
                                       for c, v in sm.items()})
        for k in [k for k, v in self.mem.items() if self.n - v["n"] > self.max_missed]:
            del self.mem[k]
        return dets


# ---------------------------------------------------------------- 모델: targeted (지정 클래스 탐지, 2026-11-23)
#: 무거운 모델은 프로세스에 하나씩만 — 대상을 바꿔 실행기를 다시 세워도 가중치는 다시 읽지 않는다
_MODEL_CACHE = {}


def _cached(key, make):
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = make()
    return _MODEL_CACHE[key]


def _ultra_clip_dir(models_dir):
    """YOLO-World 의 텍스트 인코더(OpenAI CLIP ViT-B/32)를 **models/clip/** 에서 찾게 한다.
    ultralytics 는 작업 폴더의 weights/clip 을 보고, 없으면 인터넷에서 받는다(폐쇄망에서 깨진다)."""
    try:
        import ultralytics.nn.text_model as tm
        tm.WEIGHTS_DIR = Path(models_dir)
    except Exception:
        pass


class _TextArm:
    """YOLOE 텍스트 프롬프트 — 모델은 캐시, 문구만 바꾼다."""

    def __init__(self, weight, enc_dir, prompts, owners, device, imgsz, conf, max_det, follow):
        from ultralytics import YOLOE
        weight = Path(weight)
        if not weight.is_file():
            raise _ArmMissing("가중치가 없다: %s" % weight)
        if not (Path(enc_dir) / "mobileclip_blt.ts").is_file():
            raise _ArmMissing("텍스트 인코더가 없다: %s/mobileclip_blt.ts" % enc_dir)
        self.model = _cached(("yoloe_text", str(weight)), lambda: YOLOE(str(weight)))
        with contextlib.chdir(enc_dir):
            pe = self.model.get_text_pe(list(prompts))
        self.model.set_classes(list(prompts), pe)
        self.prompts, self.owners = list(prompts), list(owners)
        self.device, self.imgsz, self.conf, self.max_det, self.follow_frame = device, imgsz, conf, max_det, follow
        self.name = weight.name

    def __call__(self, bgr):
        r = self.model.predict(bgr, conf=self.conf, imgsz=_arm_imgsz(self, bgr), max_det=self.max_det,
                               device=self.device, verbose=False)[0]
        out, polys = [], _mask_polys(r)
        if r.boxes is not None and len(r.boxes):
            for i, b in enumerate(r.boxes):
                k = int(b.cls)
                if 0 <= k < len(self.owners):
                    out.append({"cls_name": self.owners[k], "prompt": self.prompts[k], "conf": float(b.conf),
                                "xyxy": [float(v) for v in b.xyxy[0].tolist()],
                                "poly": polys[i] if i < len(polys) else None})
        return out


class _FreeArm:
    """YOLOE prompt-free — 자리를 넓힌다. 대상의 free_vocab 에 든 이름만 그 대상으로, 나머지는 버린다."""

    def __init__(self, weight, vocab_map, device, imgsz, conf, max_det, scene_words, scene_min_area, follow=True):
        """vocab_map: 이름 → (대상, 비슷한 이름인가). 비슷한 이름(중심 명사가 같은 이름)은 탐지기 근거에 0.7 배로 들어간다."""
        from ultralytics import YOLOE
        weight = Path(weight)
        if not weight.is_file():
            raise _ArmMissing("가중치가 없다: %s" % weight)
        self.model = _cached(("yoloe_free", str(weight)), lambda: YOLOE(str(weight)))
        self.vocab_map = dict(vocab_map)
        names = self.model.names or {}
        self.keep = sorted(int(i) for i, n in names.items() if str(n).lower() in self.vocab_map)
        self.device, self.imgsz, self.conf, self.max_det = device, imgsz, conf, max_det
        # **프레임 크기 근처로**(2026-11-23) — 464x400 을 1280 으로 키우면 의자를 engineering · forklift 로 불렀다.
        # 480 에서는 electric chair 0.56 (robot1 실측)
        self.follow_frame = bool(follow)
        self.scene_words = {str(w).lower() for w in scene_words or []}
        self.scene_min_area = float(scene_min_area)
        self.name = weight.name
        self.free_vocab = True

    def __call__(self, bgr):
        if not self.keep:
            return []
        # classes= 로 NMS 전에 거른다 — 대상과 무관한 이름이 max_det 를 차지하지 않는다
        r = self.model.predict(bgr, conf=self.conf, imgsz=_arm_imgsz(self, bgr), max_det=self.max_det,
                               classes=self.keep, device=self.device, verbose=False)[0]
        names, out, polys = self.model.names or {}, [], _mask_polys(r)
        H, W = bgr.shape[:2]
        if r.boxes is not None and len(r.boxes):
            for i, b in enumerate(r.boxes):
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                free = str(names.get(int(b.cls), "")).lower()
                if free in self.scene_words and (x2 - x1) * (y2 - y1) >= self.scene_min_area * W * H:
                    continue
                hit = self.vocab_map.get(free)
                if hit:
                    out.append({"cls_name": hit[0], "prompt": free, "similar": hit[1], "conf": float(b.conf),
                                "xyxy": [x1, y1, x2, y2], "poly": polys[i] if i < len(polys) else None})
        return out


class _OvdArm:
    """YOLO-World — OVD 동의 근거. 대상의 ovd_labels 로 찾고, 상자는 판정에만 쓴다(후보를 내지 않는다)."""

    def __init__(self, weight, clip_dir, labels, owners, device, imgsz, conf):
        from ultralytics import YOLO
        weight = Path(weight)
        if not weight.is_file():
            raise _ArmMissing("가중치가 없다: %s" % weight)
        if not (Path(clip_dir) / "clip" / "ViT-B-32.pt").is_file():
            raise _ArmMissing("YOLO-World 텍스트 인코더가 없다: %s/clip/ViT-B-32.pt" % clip_dir)
        _ultra_clip_dir(clip_dir)
        self.model = _cached(("yolo_world", str(weight)), lambda: YOLO(str(weight)))
        self.model.set_classes(list(labels))
        self.owners = list(owners)
        self.device, self.imgsz, self.conf = device, imgsz, conf
        self.name = "yolo_world"

    def __call__(self, bgr):
        r = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out = []
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                k = int(b.cls)
                if 0 <= k < len(self.owners):
                    out.append({"cls_name": self.owners[k], "conf": float(b.conf),
                                "xyxy": [float(v) for v in b.xyxy[0].tolist()], "arm": self.name})
        return out


class _GdinoArm:
    """GroundingDINO-tiny — main 과 같은 호출(문구를 '. ' 로 잇고 box 0.15 · text 0.15). **돌려받은 라벨 글자를
    그대로 남긴다** — main 의 'door 는 라벨이 정확히 door + 점수 0.25 이상' 규칙이 그 글자를 본다."""

    def __init__(self, path, labels, owners, device, box_thr=0.15, text_thr=0.15):
        path = Path(path)
        if not (path / "model.safetensors").is_file():
            raise _ArmMissing("GroundingDINO 스냅샷이 없다: %s" % path)
        # depth_track 이 UniDepth 를 위해 넣어 둔 빈 `wandb` 모듈(__spec__ 없음)이 있으면 transformers 가
        # find_spec('wandb') 에서 ValueError 로 넘어진다(2026-11-23, 거리 측정을 켠 채 문을 더할 때).
        # 적재하는 동안만 치운다 — UniDepth 쪽은 그대로다
        stub = sys.modules.get("wandb")
        if stub is not None and getattr(stub, "__spec__", None) is None:
            del sys.modules["wandb"]
        else:
            stub = None
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            self.proc, self.model = _cached(("gdino", str(path)), lambda: (
                AutoProcessor.from_pretrained(str(path), local_files_only=True),
                AutoModelForZeroShotObjectDetection.from_pretrained(str(path), local_files_only=True).to(device).eval()))
        finally:
            if stub is not None:
                sys.modules.setdefault("wandb", stub)
        self.labels = [str(l).lower() for l in labels]
        self.owners = list(owners)
        self.text = ". ".join(self.labels) + "."
        self.device, self.box_thr, self.text_thr = device, box_thr, text_thr
        self.name = self.arm = "grounding_dino"

    def __call__(self, bgr):
        import torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.proc(images=rgb, text=self.text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            raw = self.model(**inputs)
        h, w = bgr.shape[:2]
        post = self.proc.post_process_grounded_object_detection(
            raw, input_ids=inputs["input_ids"], threshold=self.box_thr, text_threshold=self.text_thr,
            target_sizes=[(h, w)])[0]
        labels = post.get("text_labels") or post.get("labels")
        out = []
        for score, label, box in zip(post["scores"], labels, post["boxes"]):
            lab = str(label).strip().lower()
            owner = next((o for l, o in zip(self.labels, self.owners) if l == lab), None)
            if owner is None:                          # 부분 일치("door pedestal")는 주인을 첫 일치로
                owner = next((o for l, o in zip(self.labels, self.owners) if l in lab), None)
            if owner:
                out.append({"cls_name": owner, "prompt": lab, "conf": float(score),
                            "xyxy": [float(v) for v in box.tolist()], "arm": self.name})
        return out


def _auto_free_vocab(name, vocab):
    """직접 입력 대상의 prompt-free 어휘 지도 — 같은 이름, 그리고 중심 명사(마지막 낱말)가 같은 이름."""
    name = name.lower().strip()
    head = name.split()[-1]
    out = [n for n in sorted(vocab) if n == name]
    if head != name and head in vocab:
        out.append(head)
    return out


class TargetedRunner:
    """**지정 클래스 탐지** — 대상 · 근거를 프로필에서 받는다(탐지대상_런처_계획_261123.md).

    후보: YOLOE 텍스트 arm(대상 문구 + 미끼) · YOLOE prompt-free arm(대상 free_vocab, 선택) ·
          학습 검출기(demo_box). OVD(YOLO-World · GroundingDINO)는 **후보를 내지 않고** 판정의
          근거(동의 · 짝 IoU)로만 쓴다 — main 의 '서로 다른 provider 동의' 를 점수로 옮긴 것이다.
    판정: target_assess.TargetEvidence (역할: score · gate · veto). 결과 모양은 river 와 같다.
    거리: 거리 측정이 켜졌을 때만 UniDepth(depth_track, 주기로 재고 트랙으로 나른다).
          demo_box 는 Go1 카메라 프레임에서 박스 높이 식(k / h_px − c)을 쓴다.
    """
    name = "targeted"

    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        resolved = cfg.get("_resolved")
        if resolved is None:
            from cue_catalog import resolve_profile
            resolved = resolve_profile({}, {"models": {"targeted": cfg}}, ["targeted"])["targeted"]
        self._build(resolved)

    def _build(self, resolved):
        import cue_catalog as cc
        import target_assess as ta
        cfg, device = self.cfg, self.device
        self.resolved = resolved
        self.targets = list(resolved["targets"])
        self.names = [t["name"] for t in self.targets]
        self.distance = bool(resolved.get("distance", True))
        files = cc.model_files(cfg)
        imgsz = int(cfg.get("imgsz", 1280))
        arm_conf = float(cfg.get("arm_conf", 0.05))
        self.conf = float(cfg.get("conf", 0.15))
        self.merge_iou = float(cfg.get("merge_iou", 0.5))
        self.max_det = int(cfg.get("max_det", 100))
        self.area_cap = float(cfg.get("max_area_ratio", 0.9))
        self.arms, self.missing = [], []
        text_t = [t for t in self.targets if t.get("detector", "yoloe_text") == "yoloe_text"]
        trained_t = [t for t in self.targets if t.get("detector") == "trained"]
        # **같은 문구를 두 대상이 쓰면**(door · demo_door) 문구는 먼저 온 대상이 갖고, 뒤 대상은 그 후보를
        # 복사해 받아 **자기 근거로 따로 판정**한다(alias). 한 문구를 두 이름이 나누면 어휘 argmax 가 둘로 갈린다.
        self.alias = {}
        if text_t:
            spec, owner_of = {}, {}
            for t in text_t:
                mine = []
                for p in (t.get("prompts") or [t["name"]]):
                    k = str(p).strip().lower()
                    if k in owner_of:
                        self.alias.setdefault(owner_of[k], [])
                        if t["name"] not in self.alias[owner_of[k]]:
                            self.alias[owner_of[k]].append(t["name"])
                        continue
                    owner_of[k] = t["name"]
                    mine.append(str(p))
                if mine:
                    spec[t["name"]] = mine
            names = [n for n in (t["name"] for t in text_t) if n in spec]
            want = {p.strip().lower() for ps in spec.values() for p in ps}
            decoys = [d for d in (cfg.get("decoys") or []) if str(d).strip().lower() not in want]
            prompts, owners = _expand_vocab(names, spec, decoys)
            tw = files["yoloe_text"]
            try:
                self.arms.append(("yoloe", _TextArm(tw[0], tw[1].parent, prompts, owners, device, imgsz, arm_conf,
                                                    self.max_det, bool(cfg.get("imgsz_follow_frame", True)))))
            except _ArmMissing as exc:
                raise SystemExit("targeted: YOLOE 텍스트 arm 을 못 세움 — %s" % exc)
            self.decoys = decoys
            free_t = [t for t in text_t if "free_arm" in t["cues"]]
            if free_t:
                fw = files["yoloe_free"][0]
                try:
                    from ultralytics import YOLOE
                    vocab = {str(n).lower() for n in _cached(("yoloe_free", str(fw)),
                                                             lambda: YOLOE(str(fw))).names.values()}
                    vmap = {}
                    for t in free_t:
                        if any(t["name"] in v for v in self.alias.values()):
                            continue                  # alias 대상은 주인의 후보를 받는다
                        fv = t.get("free_vocab")
                        fv = _auto_free_vocab(t["name"], vocab) if fv in (None, "auto") else fv
                        unknown = [n for n in fv if n.lower() not in vocab]
                        if unknown:
                            log("  targeted %s: prompt-free 어휘에 없는 이름 %s — 뺍니다" % (t["name"], unknown))
                        for n in fv:
                            if n.lower() in vocab:
                                vmap.setdefault(n.lower(), (t["name"], False))
                        t["_free_vocab_used"] = [n for n in fv if n.lower() in vocab]
                        # **비슷한 이름** — 중심 명사(마지막 낱말)가 대상의 것과 같은 어휘 이름(chair → electric chair ·
                        # computer chair). 같은 물건을 다르게 부른 것이라 근거로 받되 0.7 배로(target_assess.SIMILAR_WEIGHT)
                        heads = {str(x).lower().split()[-1] for x in (t.get("prompts") or [t["name"]]) + list(fv)}
                        skip = {str(x).lower() for x in (t.get("similar_exclude") or [])}
                        sim = sorted(n for n in vocab if n.split()[-1] in heads and n not in skip
                                     and n not in {x.lower() for x in fv})
                        for n in sim:
                            vmap.setdefault(n, (t["name"], True))
                        t["_similar_vocab"] = sim
                        if not t["_free_vocab_used"]:
                            log("  targeted %s: prompt-free 어휘에 비슷한 이름 없음 — 텍스트 문구로만 찾음" % t["name"])
                    if vmap:
                        fc = cfg.get("free") or {}
                        self.arms.append(("yoloe_free", _FreeArm(
                            fw, vmap, device, int(fc.get("imgsz", 1280)), float(fc.get("conf", arm_conf)),
                            self.max_det, fc.get("scene_words") or [], fc.get("scene_min_area", 0.10),
                            bool(fc.get("imgsz_follow_frame", True)))))
                except _ArmMissing as exc:
                    self.missing.append(("yoloe_free", str(exc)))
        else:
            self.decoys = []
        for _label, a in self.arms:                    # 작은 프레임에서도 이만큼은 키워 본다(imgsz_min)
            a.min_size = int(cfg.get("imgsz_min") or 0) or None
        self.trained = []
        for t in trained_t:
            try:
                from demo_box import TrainedBoxArm
                arm = _cached(("trained", t["name"], str(t["trained"]["weight"])), lambda: TrainedBoxArm(t, device))
                self.trained.append((t["name"], arm))
            except Exception as exc:
                self.missing.append((t["name"], "%s: %s" % (type(exc).__name__, exc)))
                log("  targeted %s 학습 검출기를 뺍니다 — %s" % (t["name"], exc))
        # OVD — 대상이 OVD 근거를 고른 경우만
        self.ovd = []
        ovd_t = [t for t in text_t if "class" in t["cues"] or "ovd_pair_iou" in t["cues"]]
        cue_models = resolved.get("cue_models") or {}       # 런처 근거 표에서 고른 근거 모델 (2026-11-27)
        ok = cc.ovd_available(cfg, files, ovd_t, cue_models.get("ovd"))
        if ovd_t and "yolo_world" in ok:
            labels, owners = [], []
            for t in ovd_t:
                for lab in (t.get("ovd_labels") or [t["name"]]):
                    if str(lab) in labels:            # 같은 낱말 — 주인 하나, 나머지는 alias 로 받는다
                        continue
                    labels.append(str(lab))
                    owners.append(t["name"])
            oc = cfg.get("ovd") or {}
            yw = files["yolo_world"]
            try:
                self.ovd.append(_OvdArm(yw[0], yw[1].parent.parent, labels, owners, device,
                                        int(oc.get("imgsz", 640)), float(oc.get("conf", 0.05))))
            except _ArmMissing as exc:
                self.missing.append(("yolo_world", str(exc)))
        if ovd_t and "grounding_dino" in ok:
            labels, owners = [], []
            for t in ovd_t:
                for lab in (t.get("ovd_labels") or [t["name"]]):
                    if str(lab).lower() not in labels:
                        labels.append(str(lab).lower())
                        owners.append(t["name"])
            try:
                self.ovd.append(_GdinoArm(files["gdino"][0].parent, labels, owners, device))
            except _ArmMissing as exc:
                self.missing.append(("grounding_dino", str(exc)))
        self.ovd_rules = {t["name"]: (t.get("ovd_rules") or {}) for t in ovd_t}
        # 거리
        self.depth = None
        dist_names = [t["name"] for t in self.targets if t.get("distance")]
        if self.distance and dist_names:
            dspec = dict(cfg.get("depth") or {}, enabled=True, classes=dist_names)
            mid = cc.depth_model_id(cue_models, files)
            if mid:                                       # 근거 표에서 고른 거리 모델 (같은 UniDepth 코드, 가중치만)
                dspec["model_id"] = mid
            key = ("depth", json.dumps({k: v for k, v in dspec.items() if k != "classes"}, sort_keys=True))
            import importlib.util as _ilu
            if key not in _MODEL_CACHE:
                _sp = _ilu.spec_from_file_location("depth_track", HERE / "depth_track.py")
                _m = _ilu.module_from_spec(_sp)
                _sp.loader.exec_module(_m)
                _MODEL_CACHE[key] = _m.DepthTracker(dspec, log=log)
            self.depth = _MODEL_CACHE[key]
            self.depth.classes = tuple(dist_names)
        tw = files["yoloe_text"]
        self.evidence = ta.TargetEvidence(self.targets, files["clip"], log=log,
                                          parts_model=(tw[0], tw[1].parent, device))
        TARGET_STYLE.clear()
        for t in self.targets:
            rgb = t.get("colour_rgb") or [0, 200, 255]
            TARGET_STYLE[t["name"]] = {"bgr": (int(rgb[2]), int(rgb[1]), int(rgb[0])), "shown": t.get("shown", t["name"])}
        self.resolve_same_as = ta.resolve_same_as
        self.suppress_duplicates = ta.suppress_duplicates
        self.same_as = {t["name"]: t.get("same_as") for t in self.targets if t.get("same_as")}
        self.arm_names = [l for l, _ in self.arms] + [n for n, _ in self.trained] + [a.name for a in self.ovd]
        self._streams, self._stream = {}, None
        log("  targeted 대상 %d: %s · 거리 %s · arm %s%s"
            % (len(self.names), ", ".join(self.names), "켬" if self.distance else "끔", "+".join(self.arm_names),
               (" · 빠짐 " + ", ".join("%s(%s)" % m for m in self.missing)) if self.missing else ""))
        for t in self.targets:
            log("    %s: %s" % (t["name"], ", ".join("%s=%s" % kv for kv in t["cues"].items())))

    def reconfigure(self, resolved):
        """대상 · 근거 · 거리를 바꾼다. 무거운 모델은 캐시에서 다시 쓴다."""
        self._build(resolved)
        self._streams, self._stream = {}, None

    def prewarm(self, resolved):
        """**새로 필요한 무거운 모델만 미리 적재한다**(다른 스레드에서). 그동안 옛 설정으로 계속 돈다.
        문구를 바꾸는 일(set_classes)은 여기서 하지 않는다 — 돌고 있는 모델을 건드리기 때문이다."""
        import cue_catalog as cc
        import target_assess as ta
        files = cc.model_files(self.cfg)
        need = cc.needed_models(resolved, self.cfg)
        if "clip" in need:
            ta._shared_clip(files["clip"])
        if "yolo_world" in need:
            from ultralytics import YOLO
            _cached(("yolo_world", str(files["yolo_world"][0])), lambda: YOLO(str(files["yolo_world"][0])))
        if "yoloe_free" in need:
            from ultralytics import YOLOE
            _cached(("yoloe_free", str(files["yoloe_free"][0])), lambda: YOLOE(str(files["yoloe_free"][0])))
        if "grounding_dino" in need:
            _GdinoArm(files["gdino"][0].parent, ["x"], ["x"], self.device)     # 모델만 캐시에 올린다
        for t in resolved["targets"]:
            if t.get("detector") == "trained":
                from demo_box import TrainedBoxArm
                _cached(("trained", t["name"], str(t["trained"]["weight"])), lambda: TrainedBoxArm(t, self.device))
        if resolved.get("distance") and any(t.get("distance") for t in resolved["targets"]):
            dspec = dict(self.cfg.get("depth") or {}, enabled=True)
            mid = cc.depth_model_id(resolved.get("cue_models"), files)
            if mid:
                dspec["model_id"] = mid
            key = ("depth", json.dumps({k: v for k, v in dspec.items() if k != "classes"}, sort_keys=True))
            if key not in _MODEL_CACHE:
                import importlib.util as _ilu
                _sp = _ilu.spec_from_file_location("depth_track", HERE / "depth_track.py")
                _m = _ilu.module_from_spec(_sp)
                _sp.loader.exec_module(_m)
                _MODEL_CACHE[key] = _m.DepthTracker(dspec, log=log)

    def forget_device(self, device):
        """그 기기의 거리 기억을 비운다(스캔 각도 표시, 2026-11-24) — 앞 각도의 거리가 carried 로 실려 오지 않게."""
        for k in [k for k in self._streams if k[0] == device]:
            del self._streams[k]
        if self.depth is not None and self._stream is not None and self._stream[0] == device:
            self.depth._mem = {}

    def use_stream(self, key):
        """기기마다 거리 기억을 따로 둔다(river 와 같은 이유)."""
        if key == self._stream or self.depth is None:
            self._stream = key
            return
        dp = self.depth
        if self._stream is not None:
            self._streams[self._stream] = {"d_mem": dp._mem, "d_n": dp._n}
        st = self._streams.get(key) or {}
        dp._mem, dp._n = st.get("d_mem") or {}, st.get("d_n") or 0
        self._stream = key

    def _merge(self, raw, img_area):
        kept = []
        for det in sorted(raw, key=lambda d: -d["conf"]):
            x1, y1, x2, y2 = det["xyxy"]
            if (x2 - x1) * (y2 - y1) > self.area_cap * img_area:
                continue
            dup = next((o for o in kept if o["cls_name"] == det["cls_name"]
                        and float(box_iou([o["xyxy"]], [det["xyxy"]])[0, 0]) >= self.merge_iou), None)
            word = det.get("prompt") or det["cls_name"]
            if dup is None:
                kept.append({"cls_name": det["cls_name"], "conf": det["conf"], "xyxy": det["xyxy"],
                             "arms": {det["arm"]}, "lead": word, "votes": {word: det["conf"]},
                             "by_arm": {det["arm"]: {word: (det["conf"], bool(det.get("similar")))}},
                             "_poly": det.get("poly"), "extra": {k: v for k, v in det.items()
                                                                 if k in ("box_gates", "box_distance",
                                                                          "box_bearing", "camera_profile")}})
            else:
                dup["arms"].add(det["arm"])
                if dup.get("_poly") is None and det.get("poly") is not None:
                    dup["_poly"] = det["poly"]
                if det["conf"] > dup["votes"].get(word, 0.0):
                    dup["votes"][word] = det["conf"]
                ba = dup["by_arm"].setdefault(det["arm"], {})
                if det["conf"] > ba.get(word, (0.0, False))[0]:
                    ba[word] = (det["conf"], bool(det.get("similar")))
        return kept

    def __call__(self, bgr, roi_crop=None):
        """roi_crop = [x1, y1, x2, y2] 이면 텍스트 · 프리 · OVD arm 은 그 안을 잘라 돌리고 좌표를 되돌린다.
        학습 검출기(demo_box)는 **늘 전체 프레임** — 관문 상수가 Go1 테두리 crop 좌표계에 묶여 있다."""
        t0 = time.perf_counter()
        raw, arm_ms, decoyed = [], {}, 0
        if roi_crop is not None:
            ox, oy = roi_crop[0], roi_crop[1]
            sub = np.ascontiguousarray(bgr[roi_crop[1]:roi_crop[3], roi_crop[0]:roi_crop[2]])
        else:
            ox = oy = 0
            sub = bgr

        def back(found):
            if roi_crop is None:
                return found
            for d in found:
                x1, y1, x2, y2 = d["xyxy"]
                d["xyxy"] = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
                if d.get("poly") is not None:
                    d["poly"] = np.asarray(d["poly"], np.float32) + np.array([ox, oy], np.float32)
            return found
        for label, arm in self.arms:
            a0 = time.perf_counter()
            try:
                found = back(arm(sub))
            except Exception as exc:
                log("  targeted arm %s 실패: %s: %s" % (label, type(exc).__name__, exc))
                found = []
            arm_ms[label] = round((time.perf_counter() - a0) * 1000, 1)
            for d in found:
                if d["cls_name"] == DECOY:
                    decoyed += 1
                    continue
                raw.append(dict(d, arm=label))
        rejected, by_gates = [], []
        for name, arm in self.trained:
            try:
                found, rej, byg, ms = arm(bgr)
            except Exception as exc:
                log("  targeted %s 실패: %s: %s" % (name, type(exc).__name__, exc))
                found, rej, byg, ms = [], [], [], None
            arm_ms[name] = ms
            rejected += [dict(r, target=name) for r in rej]
            by_gates += [dict(r, target=name) for r in byg]
            raw += [dict(d, arm=name) for d in found]
        ovd = {}
        for arm in self.ovd:
            a0 = time.perf_counter()
            try:
                found = back(arm(sub))
            except Exception as exc:
                log("  targeted OVD %s 실패: %s: %s" % (arm.name, type(exc).__name__, exc))
                found = []
            arm_ms[arm.name] = round((time.perf_counter() - a0) * 1000, 1)
            for d in found:
                ovd.setdefault(d["cls_name"], []).append(dict(d, arm=getattr(arm, "arm", arm.name)))
        img_area = float(bgr.shape[0] * bgr.shape[1])
        dets = []
        for g in self._merge(raw, img_area):
            if g["conf"] < self.conf and not g["extra"]:
                continue
            d = {"cls": self.names.index(g["cls_name"]), "name": g["cls_name"], "conf": round(float(g["conf"]), 4),
                 "xyxy": [round(float(v), 1) for v in g["xyxy"]], "agree": len(g["arms"]), "arms": sorted(g["arms"]),
                 "lead": g["lead"], "votes": {k: round(float(v), 4) for k, v in
                                              sorted(g["votes"].items(), key=lambda t: -t[1])},
                 "_poly": g.get("_poly"), "_by_arm": g["by_arm"]}
            d.update(g["extra"])
            if g["cls_name"] in self.same_as:
                d["_same_as"] = self.same_as[g["cls_name"]]
            dets.append(d)
        for src, names in self.alias.items():         # 같은 후보를 alias 대상에도 준다(각자 판정)
            for d in [d for d in dets if d["name"] == src]:
                for n in names:
                    c = dict(d, name=n, cls=self.names.index(n))
                    if n in self.same_as:
                        c["_same_as"] = self.same_as[n]
                    dets.append(c)
        for src, names in self.alias.items():
            for n in names:
                if src in ovd:
                    ovd.setdefault(n, []).extend(ovd[src])
        # **받는 대상의 규칙으로 거른다**(main grounding_dino_evidence) — demo_door 는 GDINO 라벨이 정확히
        # door 이고 점수 0.25 이상인 상자만. 문 상자를 복사해 받으므로 거르는 것도 받은 뒤다
        for n in list(ovd):
            rules = self.ovd_rules.get(n) or {}
            keep = []
            for d in ovd[n]:
                rule = rules.get(d["arm"]) or {}
                if rule.get("exact_label") and str(d.get("prompt", "")).strip() not in self._labels(n):
                    continue
                if d["conf"] < float(rule.get("min_score", 0.0)):
                    continue
                keep.append(d)
            ovd[n] = keep
        dets += self._ovd_proposals(dets, raw, ovd, img_area)
        dets.sort(key=lambda d: -d["conf"])
        del dets[self.max_det:]
        depth_ms = 0.0
        if self.distance:
            box_cm = [d for d in dets if (d.get("box_distance") or {}).get("distance_cm") is not None]
            for d in box_cm:
                d["dist"] = {"value": round(d["box_distance"]["distance_cm"] / 100.0, 3), "space": "metric",
                             "source": "box_height", "age": 0}
            if self.depth is not None:
                rest = [d for d in dets if d not in box_cm]
                depth_ms = self.depth(bgr, rest)
        assess_ms = self.evidence(bgr, dets, ovd)
        dets = self.resolve_same_as(dets)
        dets = self.suppress_duplicates(dets)
        for d in dets:
            d.pop("_same_as", None)
        ms = (time.perf_counter() - t0) * 1000
        return {"infer_ms": round(ms, 1), "targets": list(self.names),
                "cues": {t["name"]: dict(t["cues"]) for t in self.targets},
                "provisional": [t["name"] for t in self.targets if t.get("provisional")],
                "distance": self.distance, "arms": list(self.arm_names), "arm_ms": arm_ms,
                "missing_arms": [m[0] for m in self.missing], "depth_ms": depth_ms, "assess_ms": assess_ms,
                "decoy_dropped": decoyed, "rejected_candidates": rejected, "rejected_by_gates": by_gates,
                "ovd_boxes": sum(len(v) for v in ovd.values()), "prompt": "targeted"}, dets

    def _ovd_proposals(self, dets, raw, ovd, img_area):
        """**OVD 가 후보를 낸다**(대상 파일 ovd_propose, 2026-11-24) — 먼 시연 문을 YOLOE 가 후보 하한(conf) 아래로 보는데
        GroundingDINO 는 0.46~0.78 로 잡았다(스캔 45°). main 문 탐지가 GroundingDINO 를 provider 로 쓴 것과 같다.
        조건: 그 arm 의 규칙(min_score · max_area — 어안 테두리 전체를 door 로 잡는 오탐을 넓이로 거른다), 같은 대상의 후보와
        IoU 0.5 미만. 근거가 두 번 세어지지 않게 탐지기 근거는 **하한 아래로 버려진 YOLOE 상자**(_by_arm)와 그 OVD 상자로만,
        짝 IoU 는 자기 자신이 아니라 **YOLOE 상자**(_pair_with)와 잰다 — '서로 다른 탐지기 동의' 의 뜻 그대로."""
        out = []
        for t in self.targets:
            prop = t.get("ovd_propose") or {}
            if not prop:
                continue
            n = t["name"]
            owners = {n} | {src for src, names in self.alias.items() if n in names}
            weak = [r for r in raw if r["cls_name"] in owners and r.get("arm") in ("yoloe", "yoloe_free")]
            for o in sorted(ovd.get(n) or [], key=lambda o: -o["conf"]):
                rule = prop.get(o["arm"])
                if rule is None or o["conf"] < float(rule.get("min_score", 0.3)):
                    continue
                x1, y1, x2, y2 = o["xyxy"]
                if (x2 - x1) * (y2 - y1) > float(rule.get("max_area", 0.5)) * img_area:
                    continue
                if any(d["name"] == n and float(box_iou([d["xyxy"]], [o["xyxy"]])[0, 0]) >= 0.5 for d in dets + out):
                    continue
                by_arm = {}
                for r in weak:
                    if float(box_iou([r["xyxy"]], [o["xyxy"]])[0, 0]) >= 0.5:
                        w = r.get("prompt") or r["cls_name"]
                        prev = by_arm.setdefault(r["arm"], {}).get(w, (0.0, False))
                        if r["conf"] > prev[0]:
                            by_arm[r["arm"]][w] = (float(r["conf"]), bool(r.get("similar")))
                out.append({"cls": self.names.index(n), "name": n, "conf": round(float(o["conf"]), 4),
                            "xyxy": [round(float(v), 1) for v in o["xyxy"]], "agree": 1 + len(by_arm),
                            "arms": sorted({o["arm"]} | set(by_arm)), "lead": str(o.get("prompt") or n),
                            "votes": {str(o.get("prompt") or n): round(float(o["conf"]), 4)},
                            "_poly": None, "_by_arm": by_arm, "proposed_by": o["arm"],
                            "_pair_with": [{"xyxy": r["xyxy"], "conf": r["conf"], "arm": r["arm"]} for r in weak]})
        return out

    def _labels(self, name):
        t = next((t for t in self.targets if t["name"] == name), {})
        return [str(x).lower() for x in (t.get("ovd_labels") or [name])]


RUNNERS = {"yoloe": YoloeRunner, "river": RiverRunner, "river_open": RiverRunner,
           "targeted": TargetedRunner,
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
LABEL_TOKENS = ("name", "lead", "conf", "cls", "id", "depth", "range", "dist", "gps", "alt", "ned")


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
    if tok == "lead":
        # **이긴 낱말, 없으면 클래스 이름.** 가설 칸은 이름이 전부 `unconfirmed` 라
        # 그것을 쓰면 화면에 같은 글자만 스물세 개 뜬다. 그래서 label 에서 `name` 을
        # 빼고 이 토큰만 두면 "unconfirmed" 글자가 사라지고 낱말만 남는다 —
        # 어느 클래스가 가설 칸인지를 여기서 하드코딩하지 않는 방법이기도 하다.
        return d.get("lead") or (d.get("prompts") or [None])[0] or d.get("name")
    if tok == "conf":
        return "%.2f" % d["conf"]
    if tok == "cls":
        return "#%d" % d["cls"]
    if tok == "id":
        return "id%d" % d["track_id"] if d.get("track_id") is not None else None
    if tok == "depth":
        return "Z%.2fm" % d["depth_m"] if d.get("depth_m") is not None else None
    if tok == "dist":
        v = d.get("dist")
        if not v:
            return None
        return "%.1fm" % v["value"] if v.get("space") == "metric" else "%.3f" % v["value"]
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


def draw_dets(img, dets, ocfg, kind, colour_fn=None, drawn=None):
    """박스 + config 로 고른 라벨 줄들. overlay.min_conf 미만은 그림에서만 뺀다.

    `drawn` 에 목록을 주면 그린 박스·라벨 띠의 (x1, y1, x2, y2) 를 담는다 — 근거 카드가
    그 자리를 피해 놓이도록(카드가 검출을 가리지 않게) 넘겨주는 용도다."""
    h, w = img.shape[:2]
    fs = float(ocfg.get("font_scale") or max(0.4, w / 1600.0))
    lw = int(ocfg.get("line_width") or max(1, round(w / 640)))
    min_conf = ocfg.get("min_conf")
    dec = int(ocfg.get("gps_decimals", 6))
    spec = label_spec(ocfg, kind)
    for d in dets or []:
        if (min_conf is not None and d["conf"] < float(min_conf)
                and not (d.get("assess") or {}).get("suspect")):   # 점수로 의심한 것은 늘 그린다
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        color = colour_fn(d) if colour_fn else class_color(d["cls"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
        if drawn is not None:
            drawn.append((x1, y1, x2, y2))
        if d.get("_label") is not None:                 # 부르는 쪽이 정한 라벨(river 판정)
            lines = [d["_label"]] if d["_label"] else []
        else:
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
            if drawn is not None:
                drawn.append((x, y, x + tw + 4, y + lh))
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


def detect_overlay(bgr, dets, title, ocfg, kind="yoloe", rec=None, rects=None):
    blk = (ocfg.get(kind) or {}) if isinstance(ocfg.get(kind), dict) else {}
    lines = [title if ocfg.get("title", True) else None]
    if kind == "targeted":
        out = _targeted_overlay(bgr, dets, ocfg, blk, rec)
    else:
        out = _detect_overlay(bgr, dets, ocfg, kind)
    if rects:
        out = draw_roi(out, rects)
    if blk.get("count") and rec is not None:
        # **탐지 개수는 프레임 왼쪽 위에 클래스별 라벨로**(2026-11-23). 구역이 있으면 구역 안에서만 센 값이다
        out = draw_counts(out, rec.get("counts") or {}, dets, kind, ocfg)
    if any(d.get("reid") for d in dets or []):
        from reid import draw_reid
        out = draw_reid(out, dets)
    return add_header(out, lines, ocfg)


#: 탐지 영역 색(BGR) — 대상 색과 겹치지 않는 청록
ROI_BGR = (200, 200, 0)


def draw_roi(img, rects, alpha=0.18):
    """탐지 영역을 **색으로** — 안을 옅게 칠하고 테두리를 그린다(2026-11-23). 밖은 그대로 둔다."""
    out = img.copy()
    tint = img.copy()
    lw = max(2, round(img.shape[1] / 480))
    for r in rects:
        x1, y1, x2, y2 = [int(round(v)) for v in r]
        cv2.rectangle(tint, (x1, y1), (x2, y2), ROI_BGR, -1)
    cv2.addWeighted(tint, alpha, out, 1 - alpha, 0, out)
    for r in rects:
        x1, y1, x2, y2 = [int(round(v)) for v in r]
        cv2.rectangle(out, (x1, y1), (x2, y2), ROI_BGR, lw)
    return out


def count_colour(name, dets, kind):
    if kind == "targeted":
        st = TARGET_STYLE.get(name)
        if st:
            return st["bgr"]
    if kind == "river":
        try:
            cards = _cards_module()
            if name in cards.HAZARD:
                return tuple(int(v) for v in cards.HAZARD_RGB[::-1])
        except Exception:
            pass
    d = next((d for d in dets or [] if d.get("name") == name and d.get("cls") is not None), None)
    return class_color(d["cls"]) if d is not None else (160, 160, 160)


def draw_counts(img, counts, dets, kind, ocfg):
    """프레임 왼쪽 위에 클래스별 라벨 '이름 개수' 를 세로로 쌓는다(박스 라벨과 같은 모양)."""
    if not counts:
        return img
    h, w = img.shape[:2]
    fs = float(ocfg.get("font_scale") or max(0.4, w / 1600.0))
    pad = 4
    y = pad
    for name, n in counts.items():
        if kind == "targeted" and name in TARGET_STYLE:
            name_s = TARGET_STYLE[name].get("shown", name)
        else:
            name_s = name
        s = "%s %d" % (name_s, n)
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        lh = th + 8
        if y + lh > h:
            break
        color = count_colour(name, dets, kind)
        cv2.rectangle(img, (pad, y), (pad + tw + 8, y + lh), color, -1)
        text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
        cv2.putText(img, s, (pad + 4, y + lh - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, text_color, 1, cv2.LINE_AA)
        y += lh + 3
    return img


#: 지정 클래스 대상의 색 · 보이는 이름 — TargetedRunner 가 채운다
TARGET_STYLE = {}


def _targeted_overlay(bgr, dets, ocfg, blk, rec):
    """판정된 대상은 대상 색 + 고른 라벨, 판정 아닌 검출은 회색(끄면 안 그림), 관문에 떨어진 demo_box 는 빨간 점선."""
    other = (165, 155, 150)
    dec = int(ocfg.get("gps_decimals", 6))
    spec = label_spec(ocfg, "targeted")
    show_other = bool(blk.get("show_non_suspect", True))
    marked = []
    for d in dets or []:
        a = d.get("assess") or {}
        sus = a.get("suspect")
        st = TARGET_STYLE.get(d.get("name")) or {}
        if sus:
            parts = []
            for line in spec:
                toks = []
                for k in line:
                    if k in ("name", "lead"):
                        toks.append(st.get("shown", sus))
                    else:
                        v = label_token(k, d, dec)
                        if v:
                            toks.append(v)
                if toks:
                    parts.append(" ".join(toks))
            if parts and a.get("score") is not None:
                parts[0] += "  %d" % round(a["score"])
            marked.append(dict(d, _label="  ".join(parts), _colour=st.get("bgr", (0, 200, 255))))
        elif show_other:
            marked.append(dict(d, _label=d.get("lead") or "", _colour=other))
    out = bgr.copy()
    drawn = []
    draw_dets(out, marked, ocfg, "targeted", lambda x: x["_colour"], drawn)
    if blk.get("show_rejected") and rec:
        for r in rec.get("rejected_by_gates") or []:
            x1, y1, x2, y2 = [int(round(v)) for v in r["bbox"]]
            for i in range(x1, x2, 10):
                cv2.line(out, (i, y1), (min(i + 5, x2), y1), (60, 60, 230), 2)
                cv2.line(out, (i, y2), (min(i + 5, x2), y2), (60, 60, 230), 2)
            for j in range(y1, y2, 10):
                cv2.line(out, (x1, j), (x1, min(j + 5, y2)), (60, 60, 230), 2)
                cv2.line(out, (x2, j), (x2, min(j + 5, y2)), (60, 60, 230), 2)
    cards = blk.get("cards")
    if cards is True or cards == "always":
        out = _targeted_cards(out, dets, drawn)
    return out


def _targeted_cards(img, dets, obstacles, max_cards=5):
    """'항상' 카드 — 판정된 대상마다 근거 줄을 상자 옆에 그린다(PIL · 한글 글꼴)."""
    sus = [d for d in dets or [] if (d.get("assess") or {}).get("suspect")]
    if not sus:
        return img
    try:
        from PIL import Image, ImageDraw, ImageFont
        font_path = _cards_module().find_font()
        fs = max(12, int(img.shape[1] / 90))
        font = ImageFont.truetype(str(font_path), fs) if font_path else ImageFont.load_default()
    except Exception as exc:
        log("  targeted 카드 실패: %s" % exc)
        return img
    from cue_catalog import CUE_EN
    H, W = img.shape[:2]
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    taken = [tuple(o) for o in obstacles or []]
    sus.sort(key=lambda d: -((d.get("assess") or {}).get("score") or 0))
    lh = fs + 6
    bar_w = int(fs * 5)
    pad = 8
    cards, side = [], []
    for d in sus[:max_cards]:
        a = d["assess"]
        name = a["suspect"]
        st = TARGET_STYLE.get(name) or {}
        card = (a.get("cards") or {}).get(name) or {}
        rows = []
        for k, pct, txt in card.get("rows") or []:
            label = CUE_EN.get(k, k)                   # 카드는 영어로만
            rows.append((label, pct, str(txt)[:72]))
        head = st.get("shown", name)
        score = "-" if a.get("score") is None else "%d/100" % round(a["score"])
        lab_w = int(max([probe.textlength(r[0], font=font) for r in rows] + [0])) + 8
        pct_w = int(probe.textlength("100%", font=font)) + 8
        txt_w = int(max([probe.textlength(r[2], font=font) for r in rows] + [0]))
        cw = max(pad * 2 + lab_w + bar_w + 6 + pct_w + txt_w,
                 pad * 2 + int(probe.textlength(head + "  " + score, font=font)))
        ch = pad * 2 + lh * (len(rows) + 1)
        x1, y1, x2, y2 = d["xyxy"]
        pick = None
        for cx, cy in [(x2 + 8, y1), (x1 - cw - 8, y1), (x1, y2 + 8), (x1, y1 - ch - 8)]:
            cx, cy = int(cx), int(cy)
            if cx < 0 or cy < 0 or cx + cw > W or cy + ch > H:
                continue
            box = (cx, cy, cx + cw, cy + ch)
            if not any(not (box[2] <= o[0] or o[2] <= box[0] or box[3] <= o[1] or o[3] <= box[1]) for o in taken):
                pick = box
                break
        rgb = tuple(int(v) for v in (st.get("bgr") or (0, 200, 255))[::-1])
        item = (head, score, rows, lab_w, pct_w, rgb, a.get("score"))
        if pick is None:                       # **자리가 없으면 오른쪽 패널에 쌓는다** — 장면을 가리지 않는다
            side.append((item, cw, ch, d))
        else:
            taken.append(pick)
            cards.append((pick, item))
    pw = max([c[1] for c in side], default=0)
    canvas = img if not side else np.hstack([img, np.full((H, pw + 8, 3), 17, np.uint8)])
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    y = 4
    for item, cw, ch, d in side:
        if y + ch > H:
            break
        box = (W + 4, y, W + 4 + cw, y + ch)
        cards.append((box, item))
        x1, y1, x2, y2 = d["xyxy"]
        dr.line([(int(x2), int((y1 + y2) / 2)), (box[0], box[1] + pad + lh // 2)], fill=item[5], width=1)
        y += ch + 6
    grey, bar_bg, white = (154, 163, 178), (58, 62, 70), (230, 230, 230)
    for box, (head, score, rows, lab_w, pct_w, rgb, sc) in cards:
        # 이전 카드(river · 웹 화면)와 같은 모양 — 머리: 대상 색 이름 + 점수, 줄: 근거 · 막대 · % · **세부는 회색**
        dr.rectangle(box, fill=(24, 27, 33), outline=rgb, width=2)
        x0, y0 = box[0] + pad, box[1] + pad
        dr.text((x0, y0), head, font=font, fill=rgb)
        hc = (90, 160, 255) if (sc or 0) >= 60 else ((255, 150, 60) if (sc or 0) >= 40 else (255, 110, 100))
        dr.text((x0 + probe.textlength(head + "  ", font=font), y0), score, font=font, fill=hc)
        for i, (label, pct, txt) in enumerate(rows):
            yy = y0 + lh * (i + 1)
            dr.text((x0, yy), label, font=font, fill=white)
            bx = x0 + lab_w
            if pct is None:
                dr.text((bx, yy), txt, font=font, fill=grey)
                continue
            by = yy + lh // 2 - 4
            dr.rectangle((bx, by, bx + bar_w, by + 8), fill=bar_bg)
            dr.rectangle((bx, by, bx + int(bar_w * max(0.0, min(100.0, pct)) / 100.0), by + 8), fill=white)
            p = "%d%%" % round(pct)
            px = bx + bar_w + 6
            dr.text((px + pct_w - 8 - probe.textlength(p, font=font), yy), p, font=font, fill=white)
            dr.text((px + pct_w, yy), txt, font=font, fill=grey)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _detect_overlay(bgr, dets, ocfg, kind):
    # 카드를 쓰는 배치에서는 **색의 주인도 카드 쪽**이다 — 위험물 노랑, 나머지 회색.
    # 두 곳이 색을 따로 정하면 같은 상자가 화면마다 다른 색이 된다.
    use_cards = ((ocfg.get(kind) or {}) if isinstance(ocfg.get(kind), dict) else {}).get("cards")
    use_cards = use_cards is True or use_cards == "always"     # 프로필의 off · hover 는 그림에 안 넣는다
    colour_fn = None
    if use_cards:
        try:
            colour_fn = lambda d: _cards_module().box_colour(d, bgr=True)
        except Exception:
            colour_fn = None
    if any(d.get("assess") for d in dets or []):
        # **판정을 그린다**(2026-11-13). 의심한 위험물은 "이름 거리 점수", 그 밖은 이긴
        # 낱말만. 근거(단서별 점수)는 그림에 넣지 않는다 — 웹 화면(/vision/view)에서
        # 마우스를 올리면 카드로 보인다.
        cards = _cards_module()
        hz = tuple(int(v) for v in cards.HAZARD_RGB[::-1])
        ot = tuple(int(v) for v in cards.OTHER_RGB[::-1])
        marked = []
        for d in dets:
            a = d.get("assess") or {}
            sus = a.get("suspect")
            if sus:
                dist = cards.fmt_distance(d.get("dist"))
                text = "  ".join(s for s in (cards.SHOWN.get(sus, sus), dist,
                                             "%d" % round(a["score"])) if s)
            elif d.get("name") in cards.HAZARD:
                # 검출기는 위험물 낱말로 불렀지만 **점수가 의심에 못 미쳤다** — 회색 라벨에
                # 위험물 이름을 쓰면 위험물처럼 읽힌다(줄자에 'butane gas canister'). 일반 이름으로
                text = "object"
            else:
                text = d.get("lead") or (d.get("prompts") or [None])[0] or d.get("name") or ""
            marked.append(dict(d, _label=text, _colour=hz if sus else ot))
        dets = marked
        colour_fn = lambda d: d.get("_colour") or ot
    drawn = []                                   # 그린 박스·라벨 자리 — 카드가 피한다
    out = draw_dets(bgr.copy(), dets, ocfg, kind, colour_fn, drawn)
    # **근거 카드.** 선언으로 켠다(`overlay.<kind>.cards`). 위험물 후보가 있는 장에만
    # 그려지고 없으면 비용 0 이다(실측 카드 1개 8.4 ms · 3개 12.7 ms).
    # 그리는 코드의 주인은 `hazard_overlay.py` 하나다 — 운영 화면과 검수 화면이
    # 각자 조립하면 어느 날 둘이 다른 말을 한다.
    if use_cards:
        try:
            # **카드 테두리는 박스와 같은 색이다.** 색이 다르면 어느 카드가 어느
            # 상자인지 지시선을 따라가야 안다. 색의 주인은 여기(`class_color`)다.
            out = _cards_module().add_cards(
                out, bgr, dets, cues=(ocfg.get(kind) or {}).get("cues"),
                colour_of=None,                  # 색의 주인은 카드 모듈이다
                obstacles=drawn)                 # **카드는 그려진 박스·라벨을 가리지 않는다**
        except Exception as exc:                 # 카드가 실패해도 오버레이는 나간다
            log("  overlay 카드 실패: %s: %s" % (type(exc).__name__, exc))
    return out


def _cards_module(_m=[]):
    if not _m:
        import importlib.util as _ilu
        sp = _ilu.spec_from_file_location("hazard_overlay",
                                          Path(__file__).with_name("hazard_overlay.py"))
        mod = _ilu.module_from_spec(sp)
        sp.loader.exec_module(mod)
        _m.append(mod)
    return _m[0]


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
    fuse_model = active_detector(runners)
    roi = ((cfg.get("_roi") or {}).get(device)) or None
    # **고른 검출 모델을 모두 돌린다**(2026-11-23). 예전에는 첫 하나만 돌았다 — 로그는 "둘 다 저장"
    # 이라 했지만 infer_frame 은 active_detector 하나만 불렀다. depth 결합·GPS 는 지금처럼 첫 하나에만.
    for det_model in [m for m in DETECT_MODELS if m in runners]:
        if isinstance(save_detect, (set, frozenset, list, tuple)):
            wanted = det_model in set(save_detect)
        else:
            wanted = bool(save_detect)
        fusing = det_model == fuse_model and depth_models and fusion.get("enabled", True)
        if not (wanted or fusing):
            continue
        d_out, d_stab = _detect_one(det_model, bgr, base, header, device, runners, cfg, ocfg, roi, wanted)
        if d_out[0] is not None:
            out.append(d_out[:4])
        if det_model == fuse_model:
            dets, stab = d_out[4], d_stab
    fov = fov_hint(dev_cfg, header)
    pose = []                                           # 한 장에 한 번만 찾는다 (depth 모델이 여럿이어도)
    return _infer_depth(out, dets, stab, bgr, base, header, device, depth_models, runners, cfg, pose_fn,
                        fusion, vis, ocfg, locator, fov, pose, w)


# ---------------------------------------------------------------- 구역(ROI, 웹 화면에서 끌어 지정, 2026-11-23)
def roi_applies(roi, model):
    return bool(roi and roi.get("rects") and (not roi.get("model") or roi.get("model") == model))


def roi_rects_px(roi, w, h):
    out = []
    for r in roi.get("rects") or []:
        x1, y1, x2, y2 = (float(v) for v in r)
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        if (x2 - x1) * w >= 4 and (y2 - y1) * h >= 4:
            out.append([x1 * w, y1 * h, x2 * w, y2 * h])
    return out


def roi_crop_box(rects, w, h, margin=0.10, full_above=0.60):
    """자를 상자 하나 — 구역이 하나이고 화면의 full_above 보다 작을 때만. 아니면 None(전체 프레임)."""
    if len(rects) != 1:
        return None
    x1, y1, x2, y2 = rects[0]
    if (x2 - x1) * (y2 - y1) >= full_above * w * h:
        return None
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return [int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(w, np.ceil(x2 + mx))), int(min(h, np.ceil(y2 + my)))]


def roi_offset(dets, ox, oy):
    """자른 그림에서 낸 검출 좌표를 원래 프레임으로 되돌린다."""
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        d["xyxy"] = [round(x1 + ox, 1), round(y1 + oy, 1), round(x2 + ox, 1), round(y2 + oy, 1)]
        if d.get("_poly") is not None:
            d["_poly"] = np.asarray(d["_poly"], np.float32) + np.array([ox, oy], np.float32)
        if d.get("polygon"):
            d["polygon"] = [[round(px + ox, 1), round(py + oy, 1)] for px, py in d["polygon"]]
    return dets


def roi_keep(d, rects, w, h, need=0.5):
    """**분할 마스크 기준.** 마스크 넓이의 need 이상이 구역 안이어야 남긴다. 마스크가 없으면 상자 중심.
    상자로 거르면 구역에 걸친 큰 물체(벽 · 책상 상판)가 들어온다."""
    poly = d.get("_poly")
    if poly is not None and len(poly) >= 3:
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2 + 1), min(h, y2 + 1)
        if x2 > x1 and y2 > y1:
            m = np.zeros((y2 - y1, x2 - x1), np.uint8)
            cv2.fillPoly(m, [np.round(np.asarray(poly) - (x1, y1)).astype(np.int32)], 1)
            total = int(m.sum())
            if total > 0:
                inside = np.zeros_like(m)
                for r in rects:
                    a, b = int(max(0, r[0] - x1)), int(max(0, r[1] - y1))
                    c, e = int(min(x2 - x1, r[2] - x1)), int(min(y2 - y1, r[3] - y1))
                    if c > a and e > b:
                        inside[b:e, a:c] = 1
                return float((m & inside).sum()) / total >= need
    cx, cy = (d["xyxy"][0] + d["xyxy"][2]) / 2, (d["xyxy"][1] + d["xyxy"][3]) / 2
    return any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rects)


@contextlib.contextmanager
def _upscale_crop(runner, on):
    """구역을 잘라 볼 때는 **키워서** 본다 — 그것이 자르는 이유다. 다만 640 까지만: 실측(robot1 의자, 168x216 crop)
    224 → 0.10 · 640 → 0.30 · 1280 → 0.09. 맥락 없이 너무 키우면 오히려 못 찾는다. 큰 구역은 제 크기대로."""
    arms = [a for _l, a in (getattr(runner, "arms", None) or []) if getattr(a, "follow_frame", False)] if on else []
    for a in arms:
        a.crop_floor = 640
    try:
        yield
    finally:
        for a in arms:
            a.crop_floor = None


# ---------------------------------------------------------------- 객체 재식별 (2026-11-23, vision/reid.py)
def _reid_engine(cfg):
    eng = cfg.get("_reid_engine")
    if eng is None:
        import cue_catalog as cc
        import target_assess as ta
        from reid import MemStore, ReidEngine
        files = cc.model_files((cfg.get("models") or {}).get("targeted") or {})
        clip = ta._shared_clip(files["clip"])
        try:
            from feature_match import FeatureMatcher
            matcher = FeatureMatcher(settings.resolve("models/lightglue_superpoint"), cfg.get("_device", "cuda:0"))
        except Exception as exc:                       # 매칭이 없으면 다시 찾기만 못 한다(추적은 된다)
            log("  재식별: 특징점 매칭을 못 세움 — %s: %s" % (type(exc).__name__, exc))
            matcher = None
        eng = cfg["_reid_engine"] = ReidEngine(cfg.get("_reid_store") or MemStore(), clip, matcher, log=log)
        log("  재식별 준비 (특징점 매칭 %s)" % ("SuperPoint+LightGlue" if matcher else "없음"))
    return eng


def _reid_step(cfg, device, model, bgr, dets, base):
    """고른 물체가 있으면 재식별. 판정된 것(river · targeted) · 그린 것(yoloe)만 후보다."""
    if not (cfg.get("_features") or {}).get("reid"):
        return None
    # from_n(저장된 이미지 --reid): 고른 기기에서는 그 장부터. 다른 기기는 갤러리가 생긴 뒤부터(엔진이 본다)
    items = [it for it in (cfg.get("_reid_items") or [])
             if it.get("from_n") is None or it.get("device") != device
             or (base.get("n") is not None and base["n"] >= it["from_n"])]
    if not items:
        return None
    min_conf = (cfg.get("overlay") or {}).get("min_conf")
    ok = []
    for d in dets:
        a = d.get("assess")
        if model in ("river", "targeted"):
            if a and a.get("suspect"):
                ok.append(d)
        elif min_conf is None or d["conf"] >= float(min_conf):
            ok.append(d)
    eng = _reid_engine(cfg)
    eng.last_try = {}
    ms = eng.step(items, device, model, bgr, ok, base.get("n"))
    cfg["_reid_try"] = {k: v for k, v in eng.last_try.items() if v.get("device") == device}
    return ms


def count_dets(model, dets, ocfg):
    """탐지 개수 — yoloe 는 그린 검출, river · targeted 는 **판정된 것만**(회색은 세지 않는다)."""
    out = {}
    min_conf = ocfg.get("min_conf")
    for d in dets or []:
        a = d.get("assess") or {}
        if model in ("river", "targeted"):
            name = a.get("suspect")
            if not name:
                continue
        else:
            if min_conf is not None and d["conf"] < float(min_conf):
                continue
            name = d["name"]
        out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items(), key=lambda t: (-t[1], t[0])))


def model_ocfg(ocfg, cfg, model, device):
    """yaml overlay 위에 프로필의 그 모델 오버레이(그리고 기기별)를 덮은 것."""
    prefs = dict(((cfg.get("_overlay") or {}).get(model)) or {})
    prefs.update(((cfg.get("_overlay_dev") or {}).get(device) or {}).get(model) or {})
    if not prefs:
        return ocfg, prefs
    o = dict(ocfg)
    blk = dict(ocfg.get(model) or {}) if isinstance(ocfg.get(model), dict) else {}
    for k, v in prefs.items():
        if k == "title":
            o["title"] = v
        else:
            blk[k] = v
    o[model] = blk
    return o, prefs


def _go1_focal(cfg):
    """Go1 카메라 ((W, H), f px) — demo_box 대상 선언의 거리 계수에서. 없으면 None. 한 번만 읽는다."""
    if "_go1_cam" not in cfg:
        try:
            import cue_catalog as cc
            from demo_box import go1_camera
            cfg["_go1_cam"] = go1_camera(cc.load_targets()["demo_box"])
        except Exception as exc:                      # 대상 파일 · 계수 파일이 없는 PC
            log("  Go1 초점 거리를 못 읽음 (%s: %s) — Go1 프레임 방위는 화각으로" % (type(exc).__name__, exc))
            cfg["_go1_cam"] = None
    return cfg["_go1_cam"]


def add_bearing(dets, W, H, cfg, fov):
    """**검출마다 화면 중심 대비 수평각**(스캔 창구, 2026-11-24) — bearing = {offset_deg, f_px, f_source}.
    오른쪽 + . 스캔 창구가 rotation_deg 를 더한다(scan_assess.bearing). 초점 거리:
    Go1 프레임(464x400) → demo_box 거리 계수의 f = k / 39 · 그 밖 → 기기 fov_x_deg · 없으면 offset None."""
    go1 = _go1_focal(cfg)
    if go1 is not None and (W, H) == tuple(go1[0]):
        f, src = go1[1], "go1_box"
    elif fov:
        f, src = 0.5 * W / math.tan(math.radians(float(fov)) / 2), "fov"
    else:
        f, src = None, None
    for d in dets:
        if f is None:
            d["bearing"] = {"offset_deg": None, "f_px": None, "f_source": None}
            continue
        cx = (d["xyxy"][0] + d["xyxy"][2]) / 2.0
        d["bearing"] = {"offset_deg": round(math.degrees(math.atan((cx - W / 2.0) / f)), 2),
                        "f_px": round(f, 1), "f_source": src}
    return dets


def _go1_ground(cfg):
    """Go1 바닥선 거리 상수(demo_box.go1_ground) — 없으면 None. 한 번만 읽는다."""
    if "_go1_ground" not in cfg:
        try:
            import cue_catalog as cc
            from demo_box import go1_ground
            cfg["_go1_ground"] = go1_ground(cc.load_targets()["demo_box"])
        except Exception as exc:
            log("  Go1 바닥선 상수를 못 읽음 (%s: %s) — 바닥선 거리 없음" % (type(exc).__name__, exc))
            cfg["_go1_ground"] = None
    return cfg["_go1_ground"]


def add_ground_distance(dets, W, H, cfg):
    """**Go1 프레임의 검출마다 바닥선 거리**(2026-11-24) — ground_distance = {distance_cm, reason, ...}.
    UniDepth 는 Go1 어안 화면에서 초점 거리를 크게 잡아 거리를 1.75~2.5 배로 부풀린다(시연 박스 높이식과 대조).
    바닥에 선 물체인지는 여기서 가리지 않는다 — 스캔 창구(scan_assess.distance)가 대상 이름으로 가린다."""
    g = _go1_ground(cfg)
    if g is None or (W, H) != tuple(g["frame"]):
        return dets
    from demo_box import ground_distance
    for d in dets:
        d["ground_distance"] = ground_distance(d["xyxy"], g)
    return dets


def _detect_one(det_model, bgr, base, header, device, runners, cfg, ocfg, roi, wanted):
    """검출 모델 하나 — (모델, 기록, 오버레이, None, 결합용 검출) 와 트래커. wanted 가 아니면 기록은 None."""
    runner = runners[det_model]
    if hasattr(runner, "use_stream"):                  # 기기마다 따로 기억하는 실행기
        runner.use_stream((device, base.get("run")))
    H, W = bgr.shape[:2]
    om0, _p = model_ocfg(ocfg, cfg, det_model, device)
    roi_on = ((om0.get(det_model) or {}) if isinstance(om0.get(det_model), dict) else {}).get("roi", True)
    # **탐지 영역**(오버레이 선택지 roi, 기본 켬) — 끄면 이 모델은 웹 화면의 영역을 쓰지도 그리지도 않는다
    rects = roi_rects_px(roi, W, H) if (roi_on and roi_applies(roi, det_model)) else []
    crop = roi_crop_box(rects, W, H) if rects else None
    with _upscale_crop(runner, crop is not None):
        if det_model == "targeted":
            info, dets = runner(bgr, roi_crop=crop)
        elif crop is not None:
            info, dets = runner(np.ascontiguousarray(bgr[crop[1]:crop[3], crop[0]:crop[2]]))
            dets = roi_offset(dets, crop[0], crop[1])
        else:
            info, dets = runner(bgr)
    if det_model == "targeted":
        # 기기마다 다른 대상 (2026-11-27, 예: pi12 의자 · robot1 사람) — 한 번 찾고 그 기기의 대상만 남긴다
        allow = (cfg.get("_targets_dev") or {}).get(device)
        if allow is not None:
            dets = [d for d in dets if d.get("name") in allow]
            info = dict(info, targets_device=list(allow))
    if rects:
        before = len(dets)
        dets = [d for d in dets if roi_keep(d, rects, W, H)]
        info = dict(info, roi={"version": roi.get("version"), "rects": roi.get("rects"),
                               "mode": "crop" if crop is not None else "full", "kept": len(dets),
                               "dropped": before - len(dets)})
    stab = None
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
    # **판정이 장마다 흔들리지 않게**(assess_smooth, 2026-11-13) — 트랙 번호가 붙은 뒤에
    asm = ((cfg.get("models") or {}).get(det_model) or {}).get("assess_smooth")
    if asm and asm.get("enabled", True) and det_model != "targeted":    # 지정 클래스는 TrackEvidence 가 쌓는다
        skey = (device, base.get("run"), det_model)
        sm = cfg.setdefault("_asm", {}).get(skey)
        if sm is None:
            sm = cfg["_asm"][skey] = AssessSmoother(asm)
        dets = sm.update(dets)
    if det_model == "targeted":
        akey = (device, base.get("run"), det_model)
        ad = cfg.setdefault("_axes", {}).get(akey)
        if ad is None:
            ad = cfg["_axes"][akey] = TrackEvidence()
        dets = ad.update(dets)
        add_bearing(dets, W, H, cfg, fov_hint((cfg.get("devices", {}) or {}).get(device, {}), header))
        add_ground_distance(dets, W, H, cfg)
    reid_ms = _reid_step(cfg, device, det_model, bgr, dets, base)
    if not wanted:
        return (None, None, None, None, dets), stab
    om, prefs = model_ocfg(ocfg, cfg, det_model, device)
    rec = dict(base, model=det_model, **info)
    if reid_ms is not None:
        rec["reid_ms"] = reid_ms
        if cfg.get("_reid_try"):
            rec["reid_try"] = cfg["_reid_try"]          # 찾지 못한 장의 시도(매칭 수 · 둘째 · 유사도)
    rec["detections"] = [{k: v for k, v in d.items() if not k.startswith("_")} for d in dets]
    rec["counts"] = count_dets(det_model, rec["detections"], om)
    mblk = om.get(det_model) if isinstance(om.get(det_model), dict) else {}
    cards = mblk.get("cards")
    rec["overlay_prefs"] = {"cards": ("always" if cards is True or cards == "always" else
                                      "off" if cards == "off" else "hover") if det_model in ("river", "targeted")
                            else "off",
                            "count": bool(mblk.get("count", False)),
                            "roi": bool(mblk.get("roi", True)),
                            "reid": bool((cfg.get("_features") or {}).get("reid")),
                            "show_non_suspect": bool(mblk.get("show_non_suspect", True))}
    if cfg.get("_profile_label") is not None or cfg.get("_profile_version") is not None or cfg.get("_scan"):
        rec["profile"] = {"label": cfg.get("_profile_label"), "version": cfg.get("_profile_version")}
        if cfg.get("_scan") and det_model == "targeted":      # 스캔 판의 대상으로 낸 결과 · 몇 번째 각도 표시 뒤인지
            rec["profile"]["scan"] = cfg["_scan"]
            rec["profile"]["mark"] = (cfg.get("_scan_mark") or {}).get(device)
    title = "%s  %s/%s #%s" % (det_model, device, base.get("run") or "live", base.get("n"))
    overlay = detect_overlay(bgr, rec["detections"], title, om, det_model, rec=rec, rects=rects)
    return (det_model, rec, overlay, None, dets), stab


def _infer_depth(out, dets, stab, bgr, base, header, device, depth_models, runners, cfg, pose_fn,
                 fusion, vis, ocfg, locator, fov, pose, w):
    """infer_frame 의 깊이 모델 부분 (그대로 옮겼다)."""
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


# ---------------------------------------------------------------- 프로필 (런처 · 명령줄, 2026-11-23)
def cli_profile(args, models, base=None):
    """--profile 파일 위에 명령줄 지정(--target · --cues · --no-distance · --cards · --label · --count)을 덮는다."""
    import cue_catalog as cc
    prof = json.loads(json.dumps(base or {}))
    pm = prof.setdefault("models", {})
    if args.target:
        cues = cc.parse_cues_arg(args.cues) if args.cues else None
        ts = []
        for t in args.target:
            spec = cc.parse_target_arg(t)
            if cues is not None:
                spec["cues"] = cues
            ts.append(spec)
        pm.setdefault("targeted", {})["targets"] = ts
    elif args.cues:
        raise SystemExit("--cues 는 --target 과 함께 준다")
    if args.no_distance:
        pm.setdefault("targeted", {})["distance"] = False
    over = {}
    if args.cards:
        over["cards"] = args.cards
    if args.label:
        over["label"] = [[t.strip() for t in args.label.split(",") if t.strip()]]
    if args.count:
        over["count"] = True
    if over:
        opts = cc.overlay_options(models, not args.no_distance)
        for m in models:
            o = {k: v for k, v in over.items() if k == "label" or opts[m].get(k) is True}
            if "label" in o:
                o["label"] = [[t for t in o["label"][0] if opts[m].get({"dist": "dist", "range": "dist",
                               "depth": "dist", "gps": "gps", "alt": "gps", "ned": "gps", "conf": "conf",
                               "id": "id"}.get(t, "name")) is True]]
            if o:
                pm.setdefault(m, {}).setdefault("overlay", {}).update(o)
    return prof


def apply_profile(cfg, resolved):
    """확정된 프로필을 cfg 에 건다 — 오버레이는 그대로, targeted 는 _resolved 로."""
    cfg["_overlay"] = resolved.get("overlay") or {}
    cfg["_overlay_dev"] = resolved.get("devices") or {}
    cfg["_targets_dev"] = resolved.get("targets_dev") or {}    # 기기 → 그 기기가 맡을 대상 (없으면 전부)
    cfg["_features"] = resolved.get("features") or {}
    cfg["_profile_label"] = resolved.get("label")
    cfg["_profile_version"] = resolved.get("version")
    cfg["_active"] = resolved                          # 스캔 판이 끝나면 되돌릴 설정(원문은 resolved["_raw"])
    if resolved.get("targeted") is not None:
        cfg["models"].setdefault("targeted", {})["_resolved"] = resolved["targeted"]


def load_profile_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class ProfileWatch:
    """런처가 바꾸는 프로필 파일을 지켜본다(1초마다 수정 시각만). 바뀌면 다시 읽어 확정한다.
    **틀린 프로필은 반영하지 않는다** — 옛 설정을 유지하고 상태 파일에 이유를 쓴다(런처 상태 줄)."""

    def __init__(self, path, args, models, devices, cfg):
        self.path = Path(path)
        self.args, self.models, self.cfg = args, models, cfg
        self.mtime = self._mtime()
        self.t = 0.0
        tag = "%s_%s" % ("-".join(models), "-".join(devices))
        self.status_path = self.path.with_name("status_%s.json" % re.sub(r"[^A-Za-z0-9_\-]", "_", tag)[:120])
        self.pending = None                         # (resolved, thread)

    def _mtime(self):
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return None

    def status(self, ok, msg, version=None):
        try:
            self.status_path.write_text(json.dumps({"ok": ok, "message": msg, "version": version,
                                                    "models": self.models, "at": iso_ms()},
                                                   ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def poll(self):
        """바뀌었으면 확정된 프로필, 아니면 None. 틀렸으면 None 과 상태 기록."""
        now = time.time()
        if now - self.t < 1.0:
            return None
        self.t = now
        m = self._mtime()
        if m is None or m == self.mtime:
            return None
        self.mtime = m
        import cue_catalog as cc
        try:
            prof = cli_profile(self.args, self.models, load_profile_file(self.path))
            resolved = cc.resolve_profile(prof, self.cfg, self.models)
            resolved["_raw"] = prof
            return resolved
        except (cc.ProfileError, ValueError, OSError) as exc:
            log("프로필 반영 안 함 — %s (옛 설정 유지)" % exc)
            self.status(False, str(exc))
            return None


# ---------------------------------------------------------------- live (실시간)
# ---------------------------------------------------------------- 자원 사용량 보고 (2026-11-24)
#: 프로세스마다 Redis vision_telemetry:<호스트>:<pid> (TTL 60초) 에 보고 주기(report_interval_s)마다 쓴다.
#: data_stream 서버의 자원 기록기(ds_resources/recorder.py)가 읽어 모델 조합 사용량으로 기억한다.
#: 가시화용 스트림이 아니다 — 키 하나를 덮어쓴다.
TELEMETRY_TTL_S = 60


def telemetry_cfg_hash(cfg, models):
    """설정 해시 — 레지스트리 파일 + 이 프로세스 모델들의 설정. 바뀌면 기억한 사용량을 새로 잰다.
    배치하는 쪽과 같은 함수를 쓴다(ds_resources.memory.cfg_hash)."""
    sys.path.insert(0, str(HERE.parent))
    from ds_resources.memory import cfg_hash
    return cfg_hash(HERE / "model_registry.yaml", cfg, models)


def telemetry_device_class():
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            return "gpu:" + torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    import platform
    return "cpu:" + platform.machine()


def telemetry_gpu(pid):
    """GPU 메모리 원천별로 따로 (MB) — 어느 것도 못 재면 None 칸.
      proc_mb        nvidia-smi 프로세스별 값 (onnxruntime 몫까지). **WSL 에서는 [N/A] 라 None**
      torch_peak_mb  torch 예약 최대치 (CUDA 컨텍스트 · onnxruntime 몫은 빠진다)
      device_used_mb · device_total_mb  기기 전체 (다른 프로세스 몫까지)
    합치는 규칙은 기록기(ds_resources/recorder.py)가 정한다."""
    import shutil
    import subprocess
    out = {"proc_mb": None, "torch_peak_mb": None, "device_used_mb": None, "device_total_mb": None}
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            txt = subprocess.run([exe, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=2).stdout
            for line in txt.splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) == 2 and p[0] == str(pid) and p[1].isdigit():
                    out["proc_mb"] = float(p[1])
            txt = subprocess.run([exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=2).stdout.split("\n")[0].split(",")
            if len(txt) == 2:
                out["device_used_mb"], out["device_total_mb"] = float(txt[0]), float(txt[1])
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            out["torch_peak_mb"] = round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        pass
    return out


def telemetry_summary(samples):
    """[(infer_ms, w, h)] → {frames, infer_ms_p50, infer_ms_p90, w, h}"""
    if not samples:
        return {"frames": 0}
    ms = sorted(x[0] for x in samples)
    pick = lambda q: ms[min(len(ms) - 1, int(round(q * (len(ms) - 1))))]  # noqa: E731
    w, h = samples[-1][1], samples[-1][2]
    return {"frames": len(ms), "infer_ms_p50": round(pick(0.5), 1), "infer_ms_p90": round(pick(0.9), 1), "w": w, "h": h}


class LiveRunner:
    """Redis camera_stream:<기기> 에서 기기마다 가장 최근 프레임만 꺼내 추론하고
    결과를 Redis vision_stream:<기기>:<모델> (+ depth 는 vision_depth:<기기>:<모델>) 에 넣고
    디스크 <실행>/vision/<모델>/ 에도 오프라인과 같은 형식으로 저장한다.
    처리하는 사이에 들어온 프레임은 건너뛴다 (그 프레임은 나중에 --runs 로 오프라인 처리)."""

    def __init__(self, devices, models, runners, cfg, data_root, watch=None):
        import redis
        from reid import RedisStore
        self.watch = watch
        self.roi_t = {}
        self.job = None                                 # (종류, 확정 프로필, 적재 스레드, 덧붙임) — 반영 한 건
        self.scan = self.deferred = self.before = None  # 스캔 판 · 판 동안 미룬 파일 프로필 · 판 전 설정
        self.scan_t, self.mark_seen, self.scan_failed = 0.0, {}, None
        self.depth_period = None                        # 판 동안 바꾼 거리 주기 (DepthTracker, 원래 주기)
        self.redis_mod = redis
        lc = cfg.get("live", {}) or {}
        self.lc = lc
        self.url = lc.get("redis", "redis://127.0.0.1:6380/0")
        self.r = redis.Redis.from_url(self.url, socket_connect_timeout=2, socket_timeout=10)
        cfg["_reid_store"] = RedisStore(self.r)
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
        self.tel = dict((dv, []) for dv in devices)                # 자원 보고: (추론 ms, w, h) (보고 사이)
        self.tel_seq = 0
        self.tel_key = "vision_telemetry:%s:%d" % (HOSTNAME, os.getpid())

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
        self.read_roi(dv)
        self.read_reid()
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
        self.tel[dv].append((infer_ms, w, h))

    # -- 구역(웹 화면) · 프로필(런처)
    def read_roi(self, dv):
        """Redis vision_roi:<기기> — 웹 화면에서 끌어 지정한 구역. 0.5초마다만 읽는다."""
        now = time.time()
        if now - self.roi_t.get(dv, 0.0) < 0.5:
            return
        self.roi_t[dv] = now
        try:
            raw = self.r.get("vision_roi:" + dv)
        except self.redis_mod.RedisError:
            return
        rois = self.cfg.setdefault("_roi", {})
        if not raw:
            if rois.pop(dv, None) is not None:
                log("%s 구역 해제" % dv)
            return
        try:
            roi = json.loads(raw)
        except ValueError:
            return
        old = rois.get(dv)
        if not old or old.get("version") != roi.get("version"):
            log("%s 구역 %d개 (번호 %s%s)" % (dv, len(roi.get("rects") or []), roi.get("version"),
                                          (", 모델 %s" % roi["model"]) if roi.get("model") else ""))
        rois[dv] = roi

    def read_reid(self):
        """Redis vision_reid — 가시화 · 웹 화면에서 고른 재식별 물체. 0.5초마다."""
        now = time.time()
        if now - getattr(self, "_reid_t", 0.0) < 0.5:
            return
        self._reid_t = now
        try:
            raw = self.r.get("vision_reid")
        except self.redis_mod.RedisError:
            return
        items = (json.loads(raw).get("items") if raw else None) or []
        old = {it["id"] for it in (self.cfg.get("_reid_items") or [])}
        new = {it["id"] for it in items}
        if old != new:
            log("재식별 대상: %s" % (", ".join("%s(%s)" % (it["id"], it.get("label")) for it in items) or "없음"))
        self.cfg["_reid_items"] = items

    def check_profile(self):
        """런처 프로필 파일. **스캔 판 동안 바뀐 파일은 미뤘다가** 판이 끝나면 반영한다(스캔 창구, 2026-11-24)."""
        self._poll_job()
        w = self.watch
        if w is None or self.job is not None:
            return
        resolved = w.poll()
        if resolved is None:
            return
        if self.scan is not None:
            self.deferred = resolved
            log("프로필 바뀜 — 스캔 %s 중이라 끝나면 반영" % self.scan["scan_id"])
            w.status(True, "스캔 중 — 끝나면 반영", resolved.get("version"))
            return
        self._start("file", resolved)

    # -- 반영 한 건: 무거운 모델은 다른 스레드에서 미리 올리고(그동안 옛 설정으로 계속) 끝나면 바꾼다
    def _start(self, kind, resolved, meta=None):
        tr = self.runners.get("targeted")
        th = None
        if tr is not None and resolved.get("targeted") is not None:
            import threading

            def warm():
                try:
                    tr.prewarm(resolved["targeted"])
                except Exception as exc:
                    threading.current_thread().error = "%s: %s" % (type(exc).__name__, exc)
            th = threading.Thread(target=warm, daemon=True)
            th.error = None
            th.start()
            log("%s — 필요한 모델을 미리 적재하는 동안 옛 설정으로 계속" % {
                "file": "프로필 바뀜", "scan": "스캔 %s 대상으로" % (meta or {}).get("scan_id"),
                "restore": "스캔 끝 — 원래 대상으로"}[kind])
        self.job = (kind, resolved, th, meta)
        self._poll_job()

    def _poll_job(self):
        if self.job is None:
            return
        kind, resolved, th, meta = self.job
        if th is not None and th.is_alive():
            return
        self.job = None
        self._finish(kind, resolved, getattr(th, "error", None), meta)

    def _finish(self, kind, resolved, error, meta):
        if not error:
            try:
                self.swap_profile(resolved)
            except Exception as exc:
                error = "%s: %s" % (type(exc).__name__, exc)
        what = {"file": "프로필", "scan": "스캔 대상", "restore": "원래 대상"}[kind]
        if error:
            log("%s 반영 실패 — %s (옛 설정 유지)" % (what, error))
        else:
            log("%s 반영 (번호 %s%s)" % (what, resolved.get("version"),
                                     (", 스캔 %s" % meta["scan_id"]) if kind == "scan" else ""))
        if kind == "file" and self.watch is not None:
            self.watch.status(not error, error or "반영됨", resolved.get("version"))
        elif kind == "scan":
            if not error:
                self.scan = meta
                self.cfg["_scan"] = meta["scan_id"]
                self.cfg["_scan_mark"] = {}
                self.mark_seen = {}
                self._depth_every_frame(True)
            else:
                self.scan_failed = meta["scan_id"]
            self._ack(meta["scan_id"], error, resolved.get("version"))
        elif kind == "restore" and self.watch is not None and not error:
            self.watch.status(True, "반영됨", resolved.get("version"))

    def swap_profile(self, resolved):
        tr = self.runners.get("targeted")
        if tr is not None and resolved.get("targeted") is not None:
            tr.reconfigure(resolved["targeted"])
            for store in ("_asm", "_stab", "_lab", "_axes"):
                for k in [k for k in (self.cfg.get(store) or {}) if k[2] == "targeted"]:
                    del self.cfg[store][k]
        apply_profile(self.cfg, resolved)

    # -- 스캔 판(스캔 창구, 2026-11-24): Redis vision_scan {scan_id, devices, targets} — 서버 scan_desk.py 가 쓰고
    #    판이 사는 동안 TTL 을 늘린다. 내 기기가 들어 있으면 targeted 대상을 그 판 대상으로 바꾸고, 키가 사라지면
    #    (끝 · 취소 · 서버가 죽어 TTL 만료) 판 전 설정(또는 그동안 미룬 파일 프로필)으로 되돌린다.
    def check_scan(self):
        now = time.time()
        if now - self.scan_t < 0.5:
            return
        self.scan_t = now
        if "targeted" not in self.runners:
            return
        try:
            raw = self.r.get("vision_scan")
        except self.redis_mod.RedisError:
            return
        try:
            spec = json.loads(raw) if raw else None
        except ValueError:
            spec = None
        if spec is not None and not (set(spec.get("devices") or []) & set(self.devices)):
            spec = None
        busy = self.job[3] if (self.job is not None and self.job[0] == "scan") else None
        cur = self.scan or busy
        if spec is not None:
            if (cur is not None and cur["scan_id"] == spec.get("scan_id")) or spec.get("scan_id") == self.scan_failed:
                return                                   # 이미 그 판 · 거절한 판은 다시 하지 않는다
            if self.job is not None:
                return                                   # 다른 반영이 끝나면 다음 0.5초에
            if self.scan is None:
                self.before = self.cfg.get("_active")
            try:
                resolved = self.scan_profile(spec)
            except Exception as exc:
                log("스캔 %s 대상 거절 — %s" % (spec.get("scan_id"), exc))
                self._ack(spec.get("scan_id"), str(exc), None)
                self.scan_failed = spec.get("scan_id")
                return
            self._start("scan", resolved, {"scan_id": spec.get("scan_id"), "devices": list(spec.get("devices") or [])})
        elif self.scan is not None and self.job is None:
            back = self.deferred or self.before
            log("스캔 %s 끝 — %s" % (self.scan["scan_id"], "미룬 프로필로" if self.deferred else "원래 대상으로"))
            self.scan, self.deferred, self.before = None, None, None
            self.cfg.pop("_scan", None)
            self.cfg.pop("_scan_mark", None)
            self._depth_every_frame(False)
            if back is not None:
                self._start("restore", back)

    def _depth_every_frame(self, on):
        """판 동안 UniDepth 를 **매 장** 잰다 — 주기(기본 5장)로 재면 각도 표시 0.5초 뒤 잡힌 장의 거리가 앞 장에서
        나른 값(carried)이라 창구가 믿지 않는다(2026-11-24 robot1 재생: 8각도 중 6각도). 끝나면 원래 주기로."""
        dp = getattr(self.runners.get("targeted"), "depth", None)
        if on and dp is not None:
            if self.depth_period is None:
                self.depth_period = (dp, dp.period)
            dp.period = 1
        elif not on and self.depth_period is not None:
            obj, period = self.depth_period
            obj.period = period
            self.depth_period = None

    def scan_profile(self, spec):
        """판 전 프로필(원문) 위에 판 대상만 바꾼 것 → 확정. 판 대상에 맞지 않는 targeted 오버레이는 빼고 다시 해 본다."""
        import copy
        import cue_catalog as cc
        base = (self.deferred or self.cfg.get("_active") or {}).get("_raw") or {}
        targets = spec.get("targets")
        if not isinstance(targets, list) or not targets:
            raise cc.ProfileError("스캔 대상이 없음")
        prof = copy.deepcopy(base)
        tm = prof.setdefault("models", {}).setdefault("targeted", {})
        tm["targets"], tm["distance"] = targets, True
        try:
            resolved = cc.resolve_profile(prof, self.cfg, self.models)
        except cc.ProfileError:
            tm.pop("overlay", None)
            for part in (prof.get("devices") or {}).values():
                ((part or {}).get("models") or {}).pop("targeted", None)
            resolved = cc.resolve_profile(prof, self.cfg, self.models)
        resolved["_raw"] = base                          # 판이 끝나도 되돌릴 원문은 판 전 것
        return resolved

    def _ack(self, scan_id, error, version):
        try:
            self.r.set("vision_scan_ack:%s" % scan_id,
                       json.dumps({"ok": not error, "error": error, "version": version, "devices": self.devices,
                                   "at": iso_ms()}, ensure_ascii=False), ex=600)
        except self.redis_mod.RedisError as e:
            log("스캔 %s 응답을 못 씀 (%s)" % (scan_id, e))

    def check_marks(self):
        """vision_scan_mark:<기기> 번호가 바뀌면(각도 표시) 그 기기의 targeted 트랙 · 이름 · 거리 기억을 비운다
        — 회전 뒤 같은 화면 자리의 상자가 앞 각도의 근거 · 판정 · 거리를 물려받지 않게.
        **판 동안 매 바퀴** 본다(장을 꺼내기 전에 비워야 한다). 결과의 profile.mark 에 번호를 싣는다 — 창구는
        그 번호 뒤의 장만 그 각도의 대표로 잡는다."""
        if self.scan is None:
            return
        for dv in self.devices:
            if dv not in self.scan["devices"]:
                continue
            try:
                v = self.r.get("vision_scan_mark:" + dv)
            except self.redis_mod.RedisError:
                return
            if v is None or self.mark_seen.get(dv) == v:
                continue
            self.mark_seen[dv] = v
            try:
                self.cfg.setdefault("_scan_mark", {})[dv] = int(v)
            except ValueError:
                pass
            for store in ("_asm", "_stab", "_lab", "_axes"):
                for k in [k for k in (self.cfg.get(store) or {}) if k[0] == dv and k[2] == "targeted"]:
                    del self.cfg[store][k]
            tr = self.runners.get("targeted")
            if tr is not None and hasattr(tr, "forget_device"):
                tr.forget_device(dv)

    # -- 반복
    def report(self, dt):
        parts = []
        for dv in self.devices:
            n, lag, inf = self.stat[dv]
            if n:
                parts.append("%s %.1f장/s 지연 %.2fs (추론 %.0fms)" % (dv, n / dt, lag / n / 1000.0, inf / n))
            self.stat[dv] = [0, 0.0, 0.0]
        log("live  " + (" · ".join(parts) if parts else "새 프레임 없음 (서버가 보내는 중인지 확인)"))
        self.publish_telemetry(dt)

    def publish_telemetry(self, dt):
        """보고 주기의 자원 사용량을 vision_telemetry:<호스트>:<pid> 에 덮어쓴다. 실패해도 추론은 계속."""
        streams = {}
        for dv in self.devices:
            sm = telemetry_summary(self.tel[dv])
            if sm["frames"]:
                sm["fps"] = round(sm["frames"] / dt, 2) if dt > 0 else None
            streams[dv] = sm
            self.tel[dv] = []
        self.tel_seq += 1
        if not hasattr(self, "tel_hash"):
            self.tel_hash = telemetry_cfg_hash(self.cfg, self.models)
        doc = {"schema": "vtel/1", "proc": self.tel_key.split(":", 1)[1], "seq": self.tel_seq, "window_s": round(dt, 2),
               "models": sorted(self.models), "cfg_hash": self.tel_hash, "device_class": telemetry_device_class(),
               "gpu": telemetry_gpu(os.getpid()), "streams": streams, "ts": iso_ms(time.time())}
        try:
            self.r.set(self.tel_key, json.dumps(doc, ensure_ascii=False), ex=TELEMETRY_TTL_S)
        except self.redis_mod.RedisError as e:
            self.warn_once(("telemetry", str(e)[:60]), "자원 사용량을 Redis 에 못 넣음 (%s) — 추론은 계속" % e)

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
                self.check_scan()
                self.check_marks()
                self.check_profile()
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
    """설정 읽기 (D1, 2026-11-25). 기본 파일이면 config/vision_config.local.yaml 을 덮어쓴다.
    models/… · external/… 상대 경로는 DS_HOME(기본 저장소 루트) 기준 절대 경로로 푼다.
    data_root 의 상대 경로는 DS_HOME 기준(서버 --save 기본값과 같다). ./ · ../ 로 시작하면 예전처럼 이 config 파일 기준."""
    path = Path(path).expanduser().resolve()
    if path == (HERE / "vision_config.yaml").resolve():
        cfg = settings.load_vision_config(path)
    else:
        cfg = settings.load_yaml(path)
    cfg = settings.resolve_tree(cfg)
    raw = str(cfg.get("data_root") or "stream_data")
    root = Path(raw).expanduser()
    base = path.parent if raw.startswith(("./", "../")) else settings.ds_home()
    cfg["data_root"] = root if root.is_absolute() else (base / root).resolve()
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
    g = ap.add_argument_group("지정 클래스 탐지 · 오버레이 (탐지대상_런처_계획_261123.md)")
    g.add_argument("--profile", metavar="파일", help="프로필 JSON (런처가 만든다). live 에서는 바뀌면 바로 반영")
    ap.add_argument("--max-fps", nargs="+", metavar="기기=fps",
                    help="live 기기별 처리 상한 (config live.max_fps 를 덮는다). 오케스트레이터 배치표의 fps 상한")
    g.add_argument("--target", action="append", metavar="대상",
                   help="지정 클래스 대상. 여러 번. 예) door · 'traffic cone:size=300x300x700:colour=orange'")
    g.add_argument("--cues", metavar="근거", help="--target 들의 근거. default · default,aspect=off · class,semantic")
    g.add_argument("--no-distance", action="store_true", help="지정 클래스 거리 측정 끄기 (UniDepth 를 적재하지 않는다)")
    g.add_argument("--cards", choices=("off", "hover", "always"), help="단서 카드: 끔 · 웹 화면 마우스 · 영상에 항상")
    g.add_argument("--label", metavar="항목", help="박스 라벨 항목 (쉼표): name,conf,id,dist,gps …")
    g.add_argument("--count", action="store_true", help="탐지 개수 표시")
    g.add_argument("--catalog", action="store_true", help="런처용 카탈로그 JSON 을 찍고 끝낸다 (모델을 적재하지 않는다)")
    g.add_argument("--reid", action="append", metavar="기기:장:x1,y1,x2,y2",
                   help="[저장된 이미지] 그 장에서 그 상자의 물체를 재식별 대상으로 (여러 번). live 는 가시화에서 고른다")
    g.add_argument("--reid-on", action="store_true", help="객체 재식별 기능 켜기 (런처는 프로필 features.reid)")
    args = ap.parse_args()

    if args.catalog:
        import cue_catalog as cc
        print(json.dumps(cc.catalog(load_config(args.config)), ensure_ascii=False))
        return 0
    cfg = load_config(args.config)
    if args.max_fps:                                       # 배치표 fps 상한 (2026-11-24)
        mf = {}
        for item in args.max_fps:
            dv, _, v = item.partition("=")
            try:
                mf[dv.strip()] = float(v)
            except ValueError:
                raise SystemExit("--max-fps 는 기기=fps 꼴: %r" % item)
        cfg.setdefault("live", {})["max_fps"] = mf
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
    # 프로필 — 기본값(yaml + 대상 파일) 위에 --profile 파일, 그 위에 명령줄
    import cue_catalog as cc
    try:
        base = load_profile_file(args.profile) if args.profile and Path(args.profile).exists() else {}
        raw = cli_profile(args, models, base)
        resolved = cc.resolve_profile(raw, cfg, models)
        resolved["_raw"] = raw
    except (cc.ProfileError, ValueError, OSError) as exc:
        raise SystemExit("프로필이 틀렸습니다 — %s" % exc)
    apply_profile(cfg, resolved)
    cfg["_device"] = gpu
    if args.reid_on or args.reid:
        cfg.setdefault("_features", {})["reid"] = True
    if args.reid:
        items = []
        for k, spec in enumerate(args.reid, 1):
            try:
                dv, n, box = spec.split(":")
                items.append({"id": k, "device": dv, "from_n": int(n), "n": int(n),
                              "bbox": [float(v) for v in box.split(",")]})
            except ValueError:
                raise SystemExit("--reid 는 기기:장번호:x1,y1,x2,y2 — %r" % spec)
        cfg["_reid_items"] = items
    if resolved.get("overlay"):
        log("오버레이 지정: %s" % json.dumps(resolved["overlay"], ensure_ascii=False))
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
        watch = ProfileWatch(args.profile, args, models, devices, cfg) if args.profile else None
        if watch is not None:
            watch.status(True, "시작", resolved.get("version"))
        LiveRunner(devices, models, runners, cfg, data_root, watch).run()
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
