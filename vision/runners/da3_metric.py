"""Depth Anything 3 (ByteDance-Seed) — 단안 depth runner.

implements: AI-C-02 (값은 자기 기준을 들고 다닌다), AI-C-05 (없으면 그 기능만 빠진다), AI-C-16 (가중치는 내부에서만)
tests: vision/test_depth_anything_3.py

## 두 가중치, 두 가지 뜻

| 가중치 (models/…)          | 크기   | 내는 것                                   | 라이선스   |
|---------------------------|--------|-------------------------------------------|-----------|
| da3-metric-large (0.35B)  | 1.3 GB | **미터** depth — 단 초점거리(fx)가 있어야 미터가 된다 | Apache-2.0 |
| da3-small (0.08B)         | 137 MB | **상대** depth(축척 모름) + 카메라 내부파라미터 K   | Apache-2.0 |

DA3-Metric 의 출력은 그대로는 미터가 아니다. 공식 정의는 `metric = fx * net / 300` 이다 (fx 는
**모델 입력 해상도**의 픽셀 초점거리, 저장소 utils/alignment.apply_metric_scaling). 그래서 fx 를 어디서
얻느냐가 거리의 기준이 된다 — 이 runner 는 그 출처를 `info["focal_source"]` 로 함께 낸다:

  1. `fov_x`   (호출자가 준 화각 — vision_config devices.<기기>.fov_x_deg)       → "given"
  2. `focal_model` (DA3-Small 의 카메라 디코더가 추정한 fx, 공식 Nested 모델과 같은 발상) → "da3_small"
  3. `fov_x_deg` (설정에 적은 가정값)                                             → "assumed"

셋 다 없으면 적재 때 거절한다 — 기준 없는 값을 미터로 내지 않는다(AI-C-02).

DA3-Small 만 쓰면 축척 없는 depth 다. 그때 `space` 는 "relative" 이고, vision_infer 처럼 미터를
기대하는 곳에 넘기지 말아야 한다(레지스트리 provides: depth_relative).

## 반환값 (vision_infer.UniDepthRunner 와 같은 약속)

`runner(bgr, fov_x=None) -> (info, depth, K)`
  info   {"infer_ms", "checkpoint", "space", "focal_source", "fx"}
  depth  원본 크기 H×W float32, 미터(metric) — 하늘(sky ≥ 0.3)은 0 = 무효
  K      원본 픽셀 기준 3×3 내부파라미터

## 의존

외부 코드 external/Depth-Anything-3 (git 체크아웃, 커밋 고정 — 레지스트리 models.da3_*)의 src 를 sys.path 로 쓴다 —
pip 설치는 하지 않는다(그 pyproject 가 numpy<2 · xformers · open3d 등을 끌고 와 venv 를 깬다).
추론에 필요한 것만 더 깔았다: addict 2.4.0 · omegaconf 2.4.0 (순수 파이썬).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

METRIC, RELATIVE = "metric", "relative"
PATCH = 14
SCALE = 300.0                                    # apply_metric_scaling 의 scale_factor
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
REPO = Path(__file__).resolve().parents[2]


def ds_path(p):
    """상대 경로 → DS_HOME(없으면 저장소 루트) 기준 절대 경로."""
    q = Path(str(p)).expanduser()
    if q.is_absolute():
        return q
    home = os.environ.get("DS_HOME")
    return (Path(home).expanduser() if home else REPO) / q


def model_name(path):
    """가중치 폴더의 config.json model_name (예: da3metric-large). DA3 가 아니면 None."""
    try:
        name = json.loads((ds_path(path) / "config.json").read_text()).get("model_name")
    except (OSError, ValueError, AttributeError):
        return None
    return str(name) if name and str(name).startswith("da3") else None


def is_da3(path):
    return model_name(path) is not None


def proc_size(h, w, res):
    """upper_bound_resize: 긴 변을 res 로 맞추고 각 변을 14 의 배수로 반올림 (저장소 InputProcessor 와 같다)."""
    s = res / float(max(h, w))
    nh, nw = max(PATCH, round(h * s / PATCH) * PATCH), max(PATCH, round(w * s / PATCH) * PATCH)
    return nh, nw


def k_from_fov(fov_x_deg, w, h):
    f = 0.5 * w / np.tan(np.radians(float(fov_x_deg)) / 2)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float32)


class _Net:
    """가중치 폴더 하나 → DepthAnything3Net (api.DepthAnything3 를 거치지 않는다 — 그 모듈은 내보내기
    의존(open3d·gsplat 등)을 import 한다)."""

    def __init__(self, path, device, torch):
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from depth_anything_3.cfg import create_object
        p = ds_path(path)
        meta = json.loads((p / "config.json").read_text())
        self.model_name = str(meta.get("model_name"))
        net = create_object(OmegaConf.create(meta["config"]))
        sd = load_file(str(p / "model.safetensors"))
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        # any-view(da3-small) 의 광선(aux) 머리 마지막 층은 공개 가중치에 없다 — 광선 출력은 쓰지 않고
        # 카메라 디코더(cam_dec)로 K 를 낸다. 그 밖이 빠지면 거절한다
        missing = [k for k in missing if "_aux." not in k]
        if missing or unexpected:
            raise RuntimeError("DA3 %s 가중치가 안 맞음 (빠짐 %d, 남음 %d: %s)"
                               % (self.model_name, len(missing), len(unexpected), (missing or unexpected)[:3]))
        self.net = net.to(device).eval()
        self.path = p
        self.has_cam = getattr(net, "cam_dec", None) is not None


class Runner:
    """Depth Anything 3. cfg 키:

      model_path    가중치 폴더 (기본 models/da3-metric-large). model_id · model_name 도 받는다
      focal_model   fx 를 추정할 DA3 any-view 가중치 (기본: metric 이면 models/da3-small 이 있을 때 그것, false 면 끔)
      fov_x_deg     fx 의 마지막 출처 — 가정 화각(도). 주지 않으면 없음
      repo_path     외부 코드 (기본 external/Depth-Anything-3)
      process_res   모델 입력 긴 변 (기본 504, 14 의 배수로 반올림)
      use_fp16      CUDA 에서 autocast (bf16 이 되면 bf16) — 기본 true
      sky_zero      하늘(sky ≥ 0.3)을 0(무효)으로 — 기본 true
    """

    name = "depth_anything_3"

    def __init__(self, cfg, device="cuda:0"):
        import torch
        cfg = dict(cfg or {})
        repo = ds_path(cfg.get("repo_path") or "external/Depth-Anything-3")
        src = str(repo / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        self.torch = torch
        self.cfg = cfg
        self.device = str(device or cfg.get("device") or "cuda:0")
        path = cfg.get("model_path") or cfg.get("model_id") or cfg.get("model_name") or "models/da3-metric-large"
        self.main = _Net(path, self.device, torch)
        self.space = METRIC if "metric" in self.main.model_name else RELATIVE
        self.focal = None
        fm = cfg.get("focal_model", "auto")
        if self.space == METRIC and fm not in (False, None, "", "off"):
            fpath = "models/da3-small" if fm == "auto" else fm
            if fm != "auto" or is_da3(fpath):
                self.focal = _Net(fpath, self.device, torch)
                if not self.focal.has_cam:
                    raise ValueError("focal_model %s 에 카메라 디코더가 없다" % self.focal.model_name)
        # 셋째 출처. focal_model 도 이것도 없으면 fov_x 를 매번 주는 호출자만 쓸 수 있다(안 주면 __call__ 이 거절)
        self.fov_default = cfg.get("fov_x_deg")
        self.res = int(cfg.get("process_res") or 504)
        self.amp = bool(cfg.get("use_fp16", True)) and self.device.startswith("cuda")
        self.amp_dtype = (torch.bfloat16 if self.amp and torch.cuda.is_bf16_supported() else torch.float16)
        self.sky_zero = bool(cfg.get("sky_zero", True))

    def _tensor(self, bgr, nh, nw):
        rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return self.torch.from_numpy(x.transpose(2, 0, 1).copy())[None, None].to(self.device)

    def _forward(self, net, x):
        torch = self.torch
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp):
            return net.net(x)

    def __call__(self, bgr, fov_x=None):
        torch = self.torch
        h, w = bgr.shape[:2]
        nh, nw = proc_size(h, w, self.res)
        x = self._tensor(bgr, nh, nw)
        t0 = time.perf_counter()
        out = self._forward(self.main, x)
        depth = out["depth"][0, 0].float()
        sky = out.get("sky")
        K_proc, src = None, None
        if fov_x:
            K_proc, src = k_from_fov(fov_x, nw, nh), "given"
        elif self.main.has_cam and "intrinsics" in out:
            K_proc, src = out["intrinsics"][0, 0].float().cpu().numpy(), "self"
        elif self.focal is not None:
            fo = self._forward(self.focal, x)
            K_proc, src = fo["intrinsics"][0, 0].float().cpu().numpy(), "da3_small"
        elif self.fov_default:
            K_proc, src = k_from_fov(self.fov_default, nw, nh), "assumed"
        if self.space == METRIC:
            if K_proc is None:
                raise ValueError("DA3-Metric 은 fx 가 있어야 미터가 된다 — fov_x · focal_model · fov_x_deg 중 하나를 준다")
            fx = 0.5 * (float(K_proc[0, 0]) + float(K_proc[1, 1]))
            depth = depth * (fx / SCALE)
        if self.sky_zero and sky is not None:
            depth = torch.where(sky[0, 0].float() >= 0.3, torch.zeros_like(depth), depth)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        d = depth.cpu().numpy().astype(np.float32)
        d[~np.isfinite(d)] = 0
        if (nh, nw) != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        if K_proc is None:
            K = np.zeros((3, 3), np.float32)
        else:
            K = np.array(K_proc, np.float32).copy()
            K[0] *= w / float(nw)
            K[1] *= h / float(nh)
        info = {"infer_ms": round(ms, 1), "checkpoint": self.main.path.name, "space": self.space,
                "focal_source": src, "fx": round(float(K[0, 0]), 1) if K_proc is not None else None}
        return info, d, K
