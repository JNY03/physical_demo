"""Depth Anything 상대 깊이 CPU 실행기 — ONNX(onnxruntime) (2026-11-27)

기기(파이 · GPU 없음)의 depth_estimate 기기 단계(needs: depth_relative)와 서버 CPU 여유분에서 돈다.
**torch 를 쓰지 않는다** — numpy · cv2 · onnxruntime 만. 저장소 import 없음(기기 컨테이너에 이 파일만 복사).

  DepthAnything3SmallCpuRunner  DA3-Small (onnx-community/depth-anything-v3-small) — 깊이 + 내부 행렬 추정
  DepthAnything2SmallCpuRunner  DA-V2-Small (onnx-community/depth-anything-v2-small) — 상대 역깊이(disparity)

depth 약속(README)과 같은 꼴 (info, depth, K) 로 돌려주되 **미터가 아니다** — info["relative"] = True,
info["units"] = "relative". depth 는 「클수록 멀다」, 화면 가운데 값(중앙값)이 1.0 이 되게 맞춘 상대값이다.
가까움 · 멂의 순서와 근거리 판단에만 쓴다 — 거리값으로 쓰지 않는다(기기 단계 note 와 같다).

  DA3: 모델이 깊이(크기 모름)와 내부 행렬을 함께 낸다 — K 는 모델 추정 (fov_x 를 주면 그것으로)
  DA2: 모델이 역깊이(affine 불변)를 낸다 — 1/역깊이 로 바꾸되 이동(shift)을 몰라 순서만 믿는다. K 는 fov_x
       (없으면 cfg.fov_x, 기본 60°)

cfg (vision_config.yaml models.<키>)
  model     ONNX 경로 (기본: 아래 클래스의 default_model)
  input     긴 변 픽셀 (기본 308 — 14 의 배수로 맞춘다. 320 급 · 518 은 640 급)
  threads   4          fov_x  60 (DA2 의 K · DA3 에서 모델 추정 대신 쓰려면 호출 때 fov_x)
"""
from __future__ import annotations

import math
import os
import time

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def resolve_path(path):
    """상대 경로(models/…)는 DS_HOME(없으면 저장소 루트 — 이 파일의 두 단계 위) 기준. 저장소 import 없이"""
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    home = os.environ.get("DS_HOME") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(home, path)


def k_from_fov(w, h, fov_x):
    f = (w / 2.0) / math.tan(math.radians(float(fov_x)) / 2.0)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float64)


class _DepthAnythingCpu:
    name = "depth_anything_cpu"
    kind = "depth"
    default_model = None
    flavor = "da3"

    def __init__(self, cfg: dict, device: str = "cpu"):
        import onnxruntime as ort
        cfg = dict(cfg or {})
        self.path = resolve_path(cfg.get("model") or self.default_model)
        self.input = int(cfg.get("input", 308))
        self.fov_x = float(cfg.get("fov_x", 60.0))
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(cfg.get("threads", 4))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.five_d = len(self.inp.shape) == 5                      # DA3: (B, N, 3, H, W)

    def _size(self, h, w):
        s = self.input / max(h, w)
        return max(14, int(round(h * s / 14)) * 14), max(14, int(round(w * s / 14)) * 14)

    def __call__(self, bgr: np.ndarray, fov_x: float | None = None):
        t0 = time.perf_counter()
        h0, w0 = bgr.shape[:2]
        h, w = self._size(h0, w0)
        x = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        if self.five_d:
            x = x[:, None]
        outs = self.sess.run(None, {self.inp.name: np.ascontiguousarray(x, dtype=np.float32)})
        pred = np.asarray(outs[0], np.float32).reshape(outs[0].shape[-2:])
        info = {"checkpoint": os.path.basename(self.path), "relative": True, "units": "relative",
                "input": [int(w), int(h)], "flavor": self.flavor}
        if self.flavor == "da2":
            disp = np.maximum(pred, 1e-6)
            depth = 1.0 / disp
        else:
            depth = pred
        med = float(np.median(depth[np.isfinite(depth) & (depth > 0)])) if depth.size else 1.0
        depth = (depth / (med if med > 0 else 1.0)).astype(np.float32)
        depth = cv2.resize(depth, (w0, h0), interpolation=cv2.INTER_LINEAR)
        K = None
        if fov_x is None and self.flavor == "da3" and len(outs) >= 4:
            k = np.asarray(outs[3], np.float64).reshape(-1, 3, 3)[0]
            if np.isfinite(k).all() and k[0, 0] > 0:
                K = k.copy()
                K[0] *= w0 / w
                K[1] *= h0 / h
                info["k_source"] = "model"
        if K is None:
            K = k_from_fov(w0, h0, fov_x if fov_x is not None else self.fov_x)
            info["k_source"] = "fov_x"
        info["infer_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return info, depth, K

    def close(self):
        self.sess = None


class DepthAnything3SmallCpuRunner(_DepthAnythingCpu):
    name = "da3_small_cpu"
    default_model = "models/depth-anything-v3-small-onnx/onnx/model.onnx"
    flavor = "da3"


class DepthAnything2SmallCpuRunner(_DepthAnythingCpu):
    name = "dav2_small_cpu"
    default_model = "models/depth-anything-v2-small/onnx/model.onnx"
    flavor = "da2"
