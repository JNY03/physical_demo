"""CLIP 계열 이미지 임베딩 CPU 실행기 — ONNX(onnxruntime) (2026-11-27)

기기에서 상자마다 작은 임베딩(512 차원)을 뽑아 서버로 보낸다 — reid · multi_cam_track 기기 단계(needs: embed)와
기준 사진 유사도(match 근거)를 기기에서 먼저 거를 때. **torch 를 쓰지 않는다** — numpy · cv2 · onnxruntime 만,
저장소 import 없음(기기 컨테이너에 이 파일만 복사).

  MobileClipS0EmbedRunner   MobileCLIP-S0 이미지 인코더 fp32 (Xenova/mobileclip_s0, 256×256, 512 차원)
                            Xenova 의 int8 판(vision_model_int8.onnx)은 쓰지 않는다 — 실측 fp32 와 코사인 0.07~0.16 으로
                            깨져 있고 x86 에서 4 배 느리다. MatMul 만 양자화해도 빨라지지 않았다(합성곱이 대부분)

글 임베딩(같은 저장소 text_model.onnx)은 서버에서 미리 계산해 기기로 보낸다(「미리 굳힌 프롬프트 임베딩」) —
기기에는 토크나이저가 필요 없다. 같은 모델의 글 · 이미지 임베딩이라 코사인 유사도로 바로 비교된다.

embed 약속: __call__(bgr, boxes) -> (info, (N, D) float32 L2 정규화). boxes 가 비면 그림 전체 한 장.

cfg (vision_config.yaml models.<키>)
  model     ONNX 경로 (기본: default_model)
  size      256 (모델 입력 — 고정)      pad   0.05 (상자를 이만큼 넓혀 자른다)
  mean/std  없으면 0~1 그대로 (MobileCLIP). OpenAI CLIP 이면 [0.4815, 0.4578, 0.4082] · [0.2686, 0.2613, 0.2758]
  threads   4          batch  8 (한 번에 돌리는 상자 수)
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np


def resolve_path(path):
    """상대 경로(models/…)는 DS_HOME(없으면 저장소 루트 — 이 파일의 두 단계 위) 기준. 저장소 import 없이"""
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    home = os.environ.get("DS_HOME") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(home, path)


class ClipEmbedCpuRunner:
    name = "clip_embed_cpu"
    kind = "embed"
    default_model = None

    def __init__(self, cfg: dict, device: str = "cpu"):
        import onnxruntime as ort
        cfg = dict(cfg or {})
        self.path = resolve_path(cfg.get("model") or self.default_model)
        self.size = int(cfg.get("size", 256))
        self.pad = float(cfg.get("pad", 0.05))
        self.batch = max(1, int(cfg.get("batch", 8)))
        self.mean = None if cfg.get("mean") is None else np.array(cfg["mean"], np.float32)
        self.std = None if cfg.get("std") is None else np.array(cfg["std"], np.float32)
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(cfg.get("threads", 4))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.dim = int(self.sess.get_outputs()[0].shape[-1])

    def _crop(self, bgr, box):
        """상자를 조금 넓혀 자르고 → 짧은 변을 size 로 → 가운데 size×size (CLIP 전처리와 같다)"""
        h, w = bgr.shape[:2]
        if box is None:
            x1, y1, x2, y2 = 0, 0, w, h
        else:
            bx1, by1, bx2, by2 = (float(v) for v in box[:4])
            px, py = (bx2 - bx1) * self.pad, (by2 - by1) * self.pad
            x1, y1 = int(max(0, bx1 - px)), int(max(0, by1 - py))
            x2, y2 = int(min(w, bx2 + px)), int(min(h, by2 + py))
            if x2 - x1 < 2 or y2 - y1 < 2:
                return None
        im = bgr[y1:y2, x1:x2]
        ch, cw = im.shape[:2]
        s = self.size / min(ch, cw)
        nw, nh = max(self.size, int(round(cw * s))), max(self.size, int(round(ch * s)))
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        ox, oy = (nw - self.size) // 2, (nh - self.size) // 2
        im = im[oy:oy + self.size, ox:ox + self.size, ::-1].astype(np.float32) / 255.0
        if self.mean is not None:
            im = (im - self.mean) / self.std
        return im.transpose(2, 0, 1)

    def __call__(self, bgr: np.ndarray, boxes=None):
        t0 = time.perf_counter()
        boxes = list(boxes) if boxes else [None]
        crops = [self._crop(bgr, b) for b in boxes]
        out = np.zeros((len(boxes), self.dim), np.float32)
        ok = [i for i, c in enumerate(crops) if c is not None]
        for s in range(0, len(ok), self.batch):
            idx = ok[s:s + self.batch]
            x = np.ascontiguousarray(np.stack([crops[i] for i in idx]), dtype=np.float32)
            out[idx] = self.sess.run(None, {self.inp: x})[0]
        n = np.linalg.norm(out, axis=1, keepdims=True)
        out = np.where(n > 0, out / np.maximum(n, 1e-12), 0.0).astype(np.float32)
        info = {"infer_ms": round((time.perf_counter() - t0) * 1000.0, 2), "n": len(boxes), "dim": self.dim,
                "checkpoint": os.path.basename(self.path), "skipped": len(boxes) - len(ok)}
        return info, out

    def close(self):
        self.sess = None


class MobileClipS0EmbedRunner(ClipEmbedCpuRunner):
    """MobileCLIP-S0 이미지 인코더 fp32 — 512 차원"""
    name = "mobileclip_s0_cpu"
    default_model = "models/mobileclip-s0/onnx/vision_model.onnx"
