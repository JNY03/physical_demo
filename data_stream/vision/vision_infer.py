#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision_infer.py — 스트림 프레임에 비전 모델 추론 (YOLOE · UniDepth · MoGe2-Aerial)
================================================================================
2026-10-01

두 가지 모드 (--runs 로 고른다)
  live (기본)  server_stream_multi_source.py 가 흘리는 Redis camera_stream:<기기> 에서
               기기마다 가장 최근 프레임만 꺼내 추론하고, 결과를 바로 Redis 로 흘리면서 디스크에도 저장한다.
               추론하는 사이에 들어온 프레임은 건너뛴다 (실시간 우선).
  저장된 이미지  --runs all | latest | <실행 폴더>
               <data_root>/<기기>/<실행 YYYYMMDD_HHMMSS>/frames/<번호>.jpg 를 전부 처리한다.
               live 가 건너뛴 프레임을 나중에 채우거나, 다른 모델을 뒤늦게 적용할 때 쓴다.
  원본 frames/ · frames.jsonl · states.jsonl 은 읽기만 한다.

모델 (--models, 여러 개 동시에 가능)
  yoloe          YOLOE prompt-free 검출+분할 (내장 어휘 4585종)
  unidepth       UniDepth V2 metric depth (m)
  moge2_aerial   MoGe2-Aerial — MoGe-2 ViT-L + 항공 LoRA (AerialMetric, ECCV 2026), metric depth (m)
  yoloe 와 depth 모델을 함께 고르면 depth 결과에 검출별 거리가 붙는다 (config fusion).

Redis 출력 (live)
  vision_stream:<기기>:<모델>   (최근 live.maxlen 개)
      result   JSON — results.jsonl 한 줄과 같음 + source_id(원본 camera_stream 항목 ID), recv_ms, done_ms, lag_ms
      overlay  오버레이 JPEG 바이트
      header   {"source","model","n","run","file","source_id"}
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

results.jsonl 한 줄
  공통   n, frame, source, run, model, image_wh, infer_ms, (header: frames.jsonl 의 그 장 정보)
         live 면 live, source_id, recv_ms, done_ms, lag_ms 도
  yoloe  detections: [{cls, name, conf, xyxy, polygon?}]
  depth  depth_file, depth_stats{min,p5,median,p95,max,valid_ratio}, intrinsics(픽셀 K), fov_x_deg,
         fov_x_given, detections?: [{..., depth_m}]

실행 순서 (저장소 루트에서. 2026-10-01 이 PC 에서 실측 확인)
  # 0) 전용 Redis :6380. WSL 을 재시작하면 꺼지므로 그때마다 다시 띄운다
  #    이 PC 의 서버·비전은 127.0.0.1 로, tailnet 의 다른 기기는 100.114.96.78(이 PC tailscale 주소)로 구독한다
  #    비밀번호가 없어 protected-mode 를 끈다 — tailscale0 에만 6380 을 열었다 (ufw). 인터넷 쪽에는 열지 말 것
  redis-server --port 6380 --bind "127.0.0.1 100.114.96.78" --protected-mode no --save "" --appendonly no --daemonize yes
  #    다른 기기에서 읽기: redis.Redis.from_url("redis://100.114.96.78:6380/0")

  # 1) 터미널 1 — 스트림 서버 (로봇·드론 주소는 data_stream/hosts.local.json)
  source data_stream/venv/bin/activate
  python data_stream/server_stream_multi_source.py

  # 2) 터미널 2 — 실시간 비전
  source data_stream/venv/bin/activate
  python data_stream/vision/vision_infer.py

모델·기기를 골라 실행 (venv 활성화 상태)
  python data_stream/vision/vision_infer.py                                     live · 전 기기 · yoloe (config 기본)
  python data_stream/vision/vision_infer.py --devices drone --models unidepth   live · 드론에 unidepth (다른 터미널에서 추가)
  python data_stream/vision/vision_infer.py --devices drone robot1 --models yoloe unidepth moge2_aerial
  python data_stream/vision/vision_infer.py --devices drone --models moge2_aerial --runs 20261001_062842   저장된 이미지
  python data_stream/vision/vision_infer.py --devices all --models yoloe --runs latest --watch              저장되는 대로 계속
  python data_stream/vision/vision_infer.py --list                              기기·실행·프레임 수만 본다

준비
  YOLOE    가중치 경로만 맞으면 된다 (prompt-free 는 텍스트 인코더 불필요).
  UniDepth repo_path 를 sys.path 에 넣어 쓴다 (pip 설치 불필요).
  MoGe2-Aerial  external/AerialMetric/ 에 MoGe 코드 · 가중치(Moge2-Aerial.pt) · pydeps(utils3d, peft) 를 둔다.
               공유 venv 는 건드리지 않는다.
  live     config data_root 가 서버의 --save 폴더와 같아야 디스크에 저장된다 (다르면 Redis 로만 낸다).
          Redis 메모리 한도는 서버가 시작할 때 건다 (--redis-maxmemory). 전부 디스크에 있으므로 Redis 는 최근 것만.
"""
import argparse
import fcntl
import json
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
DEVICES = ("cam360", "drone", "robot1", "robot2")
MODELS = ("yoloe", "unidepth", "moge2_aerial")
DEVICE_ALIASES = {"camera": "cam360", "360": "cam360"}
MODEL_ALIASES = {"yoloe-pf": "yoloe", "yoloe_pf": "yoloe",
                 "moge": "moge2_aerial", "moge2": "moge2_aerial", "moge2-aerial": "moge2_aerial",
                 "moge2aerial": "moge2_aerial"}
DEPTH_MODELS = ("unidepth", "moge2_aerial")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
COLORMAPS = {"turbo": cv2.COLORMAP_TURBO, "jet": cv2.COLORMAP_JET, "inferno": cv2.COLORMAP_INFERNO,
             "magma": cv2.COLORMAP_MAGMA, "viridis": cv2.COLORMAP_VIRIDIS, "plasma": cv2.COLORMAP_PLASMA}

STOP = False


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


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
                if not h.get("file"):
                    continue
                name = Path(h["file"]).name
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
                        self.done.add(json.loads(line)["frame"])
                    except (ValueError, KeyError):
                        pass
            if line and not line.endswith(b"\n"):
                with open(self.results, "ab") as f:  # 강제 종료로 끊긴 마지막 줄 뒤에 이어 붙지 않게
                    f.write(b"\n")
        self.fh = open(self.results, "ab")

    def write(self, rec, overlay=None, depth=None, depth_dtype="float16", overlay_jpg=None):
        """overlay_jpg 를 주면 (live 에서 Redis 용으로 이미 인코딩한 JPEG) 그 바이트를 그대로 쓴다."""
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
        self.fh.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
        self.fh.flush()
        self.done.add(rec["frame"])

    def close(self):
        self.fh.close()


# ---------------------------------------------------------------- 모델: YOLOE prompt-free
def overlay_min_conf(yoloe_cfg):
    """오버레이에 그릴 최소 신뢰도 (models.yoloe.overlay.min_conf). null 이면 거르지 않는다.
    yoloe 오버레이와 depth 오버레이의 검출 박스에 같이 쓴다."""
    v = ((yoloe_cfg or {}).get("overlay", {}) or {}).get("min_conf")
    return float(v) if v is not None else None


class YoloeRunner:
    name = "yoloe"

    def __init__(self, cfg, device):
        from ultralytics import YOLOE
        self.cfg = cfg
        self.device = device
        self.weight = cfg["weight"]
        self.model = YOLOE(self.weight)          # prompt-free: 내장 어휘 그대로, set_classes 안 함
        self.half = bool(cfg.get("half", True)) and device.startswith("cuda")
        self(np.zeros((480, 640, 3), np.uint8))  # 첫 장 지연을 미리 치른다

    def __call__(self, bgr):
        c = self.cfg
        t0 = time.perf_counter()
        r = self.model.predict(bgr, conf=c.get("conf", 0.25), iou=c.get("iou", 0.5),
                               imgsz=c.get("imgsz", 640), max_det=c.get("max_det", 100),
                               device=self.device, quantize=16 if self.half else None, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        dets = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            polys = r.masks.xy if r.masks is not None else [None] * len(cls)
            for b, s, k, poly in zip(xyxy, conf, cls, polys):
                d = {"cls": int(k), "name": r.names[int(k)], "conf": round(float(s), 4),
                     "xyxy": [round(float(v), 1) for v in b]}
                if poly is not None and len(poly):
                    d["_poly"] = poly                       # 결합용 (저장 전에 뺀다)
                    if c.get("save_polygons", True):
                        d["polygon"] = np.round(poly, 1).tolist()
                dets.append(d)
        # 오버레이는 박스·라벨만 그린다 (폴리곤 마스크는 안 그림 — results.jsonl 의 polygon 과 depth 결합에는 그대로 쓴다).
        # overlay.min_conf 미만 검출은 그림에서만 뺀다. results.jsonl 에는 모델 conf 이상 검출이 전부 남는다
        ov = c.get("overlay", {}) or {}
        shown = r
        min_conf = overlay_min_conf(c)
        if min_conf is not None and r.boxes is not None and len(r.boxes):
            shown = r[r.boxes.conf >= min_conf]
        overlay = shown.plot(line_width=ov.get("line_width"), masks=False)
        return {"infer_ms": round(ms, 1), "weight": Path(self.weight).name}, overlay, dets


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


class UniDepthRunner:
    name = "unidepth"

    def __init__(self, cfg, device):
        import torch
        if cfg.get("repo_path") and cfg["repo_path"] not in sys.path:
            sys.path.insert(0, cfg["repo_path"])
        from unidepth.models import UniDepthV2
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


RUNNERS = {"yoloe": YoloeRunner, "unidepth": UniDepthRunner, "moge2_aerial": MogeAerialRunner}


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
        label = "%.1fm" % (lo + (hi - lo) * i / 4)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        tx = min(max(0, x - tw // 2), width - tw)
        cv2.putText(bar, label, (tx, bar_h - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bar, label, (tx, bar_h - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def draw_text(img, text, org, fs, color):
    x, y = org
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)


def depth_overlay(bgr, depth, dets, title, vis):
    color, lo, hi = colorize(depth, vis)
    if vis.get("mode", "blend") == "side":
        out = np.hstack([bgr, color])
    else:
        a = float(vis.get("alpha", 0.6))
        out = cv2.addWeighted(color, a, bgr, 1 - a, 0)
    h, w = bgr.shape[:2]
    fs = max(0.4, w / 1600.0)
    lw = max(1, round(w / 640))
    for d in dets or []:
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        dm = d.get("depth_m")
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), lw)
        draw_text(out, "%s %s" % (d["name"], "%.2fm" % dm if dm is not None else "-"), (x1, y1 - 2), fs, (255, 255, 255))
    draw_text(out, title, (4, int(22 * fs / 0.5)), fs, (255, 255, 255))
    if vis.get("colorbar", True):
        out = np.vstack([out, colorbar(out.shape[1], lo, hi, vis)])
    return out


def attach_depth(dets, depth, fusion):
    """검출마다 마스크(없으면 박스 가운데) 안 depth 의 백분위를 거리로 붙인다."""
    h, w = depth.shape
    out = []
    erode = int(fusion.get("mask_erode_px", 3))
    pct = float(fusion.get("depth_percentile", 40))
    min_ratio = float(fusion.get("min_valid_ratio", 0.2))
    for d in dets:
        m = np.zeros((h, w), np.uint8)
        poly = d.get("_poly")
        if poly is not None and len(poly) >= 3:
            cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
            if erode > 0:
                eroded = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
                if eroded.any():
                    m = eroded
        else:
            x1, y1, x2, y2 = d["xyxy"]
            cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 4, (y2 - y1) / 4
            m[max(0, int(cy - bh)):int(cy + bh) + 1, max(0, int(cx - bw)):int(cx + bw) + 1] = 1
        region = depth[m > 0]
        valid = region[(region > 0) & np.isfinite(region)]
        dm = None
        if region.size and valid.size / float(region.size) >= min_ratio:
            dm = round(float(np.percentile(valid, pct)), 3)
        e = {k: v for k, v in d.items() if not k.startswith("_") and k != "polygon"}
        e["depth_m"] = dm
        out.append(e)
    return out


# ---------------------------------------------------------------- 한 장 추론 (오프라인 · live 공통)
def infer_frame(bgr, base, header, device, save_yoloe, depth_models, runners, cfg):
    """한 장에 모델을 적용해 [(모델, 기록, 오버레이, depth 또는 None)] 을 돌려준다.
    save_yoloe   : yoloe 결과를 돌려줄지 (False 여도 depth 결합에 필요하면 검출은 돌린다)
    depth_models : 돌릴 depth 모델"""
    fusion = cfg.get("fusion", {}) or {}
    vis = cfg.get("depth_vis", {}) or {}
    dev_cfg = (cfg.get("devices", {}) or {}).get(device, {})
    w = bgr.shape[1]
    out = []
    dets = None
    if "yoloe" in runners and (save_yoloe or (depth_models and fusion.get("enabled", True))):
        info, overlay, dets = runners["yoloe"](bgr)
        if save_yoloe:
            rec = dict(base, model="yoloe", **info)
            rec["detections"] = [{k: v for k, v in d.items() if not k.startswith("_")} for d in dets]
            out.append(("yoloe", rec, overlay, None))
    fov = fov_hint(dev_cfg, header)
    for m in depth_models:
        info, depth, K = runners[m](bgr, fov_x=fov)
        rec = dict(base, model=m, **info)
        rec["depth_stats"] = depth_stats(depth)
        rec["intrinsics"] = np.round(K.astype(np.float64), 3).tolist()
        fx = fov_from_k(K, w)
        rec["fov_x_deg"] = round(fx, 2) if fx else None
        rec["fov_x_given"] = fov
        fused = None
        if dets is not None and fusion.get("enabled", True):
            fused = attach_depth(dets, depth, fusion)
            rec["detections"] = fused
        title = "%s  %s/%s #%s" % (m, device, base.get("run") or "live", base.get("n"))
        min_conf = overlay_min_conf((cfg.get("models", {}) or {}).get("yoloe"))
        drawn = fused if fused is None or min_conf is None else [d for d in fused if d["conf"] >= min_conf]
        out.append((m, rec, depth_overlay(bgr, depth, drawn, title, vis), depth))
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
    done = 0
    t_start = time.time()
    for i, (path, header) in enumerate(todo):
        if STOP:
            break
        rel = path.relative_to(run).as_posix()
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        base = {"n": frame_number(path), "frame": rel, "source": device, "run": run.name}
        if header is not None:
            base["header"] = {k: header[k] for k in ("n", "file", "via") if k in header}
        if bgr is None:
            for m in own:
                if rel not in outs[m].done:
                    outs[m].write(dict(base, model=m, error="이미지를 읽지 못함"))
            continue
        h, w = bgr.shape[:2]
        base["image_wh"] = [w, h]
        need_depth = [m for m in own if m in DEPTH_MODELS and rel not in outs[m].done]
        save_yoloe = "yoloe" in outs and rel not in outs["yoloe"].done
        # yoloe 를 다른 프로세스가 맡고 있어도 결합용 검출은 여기서 돌린다 (저장은 안 함)
        for m, rec, overlay, depth in infer_frame(bgr, base, header, device, save_yoloe, need_depth, runners, cfg):
            mcfg = cfg["models"][m]
            outs[m].write(rec, overlay, depth if mcfg.get("save_depth", True) else None,
                          mcfg.get("depth_dtype", "float16"))
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
        if header.get("file"):
            p = self.data_root / header["file"]
            if p.parent.parent.is_dir():
                run_dir, rel = p.parent.parent, "frames/" + p.name
            else:
                self.warn_once(("nodir", dv), "%s 저장 폴더가 없음 (%s) — config data_root 가 서버 --save 와 같은지 확인. "
                               "Redis 스트림만 냅니다" % (dv, p.parent.parent))
        outs = self.outputs(dv, run_dir)
        h, w = bgr.shape[:2]
        base = {"n": header.get("n"), "frame": rel, "source": dv, "run": run_dir.name if run_dir else None,
                "image_wh": [w, h], "live": True, "source_id": eid.decode(), "recv_ms": recv_ms}
        base["header"] = {k: header[k] for k in ("n", "file", "via") if k in header}
        results = infer_frame(bgr, base, header, dv, "yoloe" in self.models,
                              [m for m in self.models if m in DEPTH_MODELS], self.runners, self.cfg)
        done_ms = int(time.time() * 1000)
        pipe = self.r.pipeline(transaction=False)
        infer_ms = 0.0
        for m, rec, overlay, depth in results:
            rec["done_ms"], rec["lag_ms"] = done_ms, done_ms - recv_ms
            infer_ms += rec.get("infer_ms") or 0
            jpg = cv2.imencode(".jpg", overlay, self.jpeg)[1].tobytes()
            mcfg = self.cfg["models"][m]
            dtype = mcfg.get("depth_dtype", "float16")
            o = outs.get(m)
            if o is not None and rel is not None and rel not in o.done:
                o.write(rec, depth=depth if (depth is not None and mcfg.get("save_depth", True)) else None,
                        depth_dtype=dtype, overlay_jpg=jpg)
            short = json.dumps({"source": dv, "model": m, "n": base["n"], "run": base["run"],
                                "file": header.get("file"), "source_id": base["source_id"]}, ensure_ascii=False)
            pipe.xadd("vision_stream:%s:%s" % (dv, m),
                      {"result": json.dumps(rec, ensure_ascii=False), "overlay": jpg, "header": short},
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
    models = [m for m in MODELS if m in models]           # yoloe 를 먼저 돌려야 결합이 된다
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
    runners = {}
    for m in models:
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
