"""RF-DETR (Roboflow, Apache-2.0 — N/S/M/L) COCO 80종 탐지 · 분할 runner.

레지스트리 `models.rf_detr_*` 의 `runner: rf_detr:Runner` 가 이 파일이다.

cfg
  variant      small | medium | seg-small | seg-medium (+ nano · large · seg-nano · seg-large).
               없으면 weight 파일 이름(rf-detr-<variant>.pt[h])에서 읽고, 그것도 없으면 small
  weight       가중치 경로 (DS_HOME 기준 상대 경로 허용). 기본 models/rf-detr/<variant 기본 파일>
  conf         문턱 (기본 0.3)
  max_det      남길 최대 개수 (점수 높은 순, 기본 100)
  half         CUDA 에서 fp16 (기본 true)
  compile      false(기본) = eager export forward. "cudagraph" · "torchscript" 면 고정 해상도로 추적/캡처
               (첫 적재가 수십 초 늘고, cudagraph 는 다른 스레드가 돌기 전에 만들어야 한다)
  include      남길 COCO 이름 목록 (비우면 전부)
  preresize    true(기본) = 넣기 전에 cv2 로 res×res 로 줄인다 (rfdetr 의 비율 무시 늘이기와 같은 모양).
               false = 원본을 그대로 넘겨 rfdetr 가 줄인다 (큰 화면에서 느리다)
  save_polygons  분할 변형에서 외곽선을 det["polygon"] 으로도 남긴다 (기본 true)

가중치를 받지 않는다 — 없으면 SystemExit. (rfdetr 는 이름만 주면 repo.roboflow.com 에서 받으려 한다:
폐쇄망에서 조용히 멈추지 않도록 반드시 파일 경로를 넘긴다.)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

#: variant → (rfdetr 클래스 이름, 기본 가중치 파일). 파일 이름은 rfdetr ModelWeights 의 filename 과 같다
VARIANTS = {
    "nano": ("RFDETRNano", "rf-detr-nano.pth"),
    "small": ("RFDETRSmall", "rf-detr-small.pth"),
    "medium": ("RFDETRMedium", "rf-detr-medium.pth"),
    "large": ("RFDETRLarge", "rf-detr-large-2026.pth"),
    "seg-nano": ("RFDETRSegNano", "rf-detr-seg-nano.pt"),
    "seg-small": ("RFDETRSegSmall", "rf-detr-seg-small.pt"),
    "seg-medium": ("RFDETRSegMedium", "rf-detr-seg-medium.pt"),
    "seg-large": ("RFDETRSegLarge", "rf-detr-seg-large.pt"),
}
WEIGHT_DIR = "models/rf-detr"


def _ds_home() -> Path:
    return Path(os.environ.get("DS_HOME") or Path(__file__).resolve().parents[2])


def variant_of(cfg: dict) -> str:
    v = cfg.get("variant")
    if not v and cfg.get("weight"):
        stem = Path(str(cfg["weight"])).stem.lower()
        v = next((k for k, (_c, f) in VARIANTS.items() if Path(f).stem == stem), None)
    return str(v or "small").lower()


def weight_path(cfg: dict) -> Path:
    variant = variant_of(cfg)
    if variant not in VARIANTS:
        raise SystemExit("rf_detr variant 를 모릅니다: %s (가능: %s)" % (variant, ", ".join(VARIANTS)))
    p = Path(str(cfg.get("weight") or Path(WEIGHT_DIR) / VARIANTS[variant][1])).expanduser()
    return p if p.is_absolute() else _ds_home() / p


class Runner:
    name = "rf_detr"
    kind = "detect"

    def __init__(self, cfg: dict, device: str):
        import torch
        import rfdetr

        self.cfg = cfg = dict(cfg or {})
        self.device = device
        self.variant = variant_of(cfg)
        self.weight = weight_path(cfg)
        if not self.weight.is_file():
            raise SystemExit("rf_detr 가중치가 없습니다: %s (tools/fetch_models.py 로 받는다)" % self.weight)
        self.seg = self.variant.startswith("seg")         # 분할 변형: kind 는 detect 그대로, det 에 _poly 가 붙는다
        cls = getattr(rfdetr, VARIANTS[self.variant][0])
        src = cls(pretrain_weights=str(self.weight), device=device)
        self.half = bool(cfg.get("half", True)) and device.startswith("cuda")
        comp = cfg.get("compile") or False
        # inference() 는 가중치 사본으로 export 모드 predictor 를 만든다 (옛 optimize_for_inference 의 후속).
        # 원본은 곧바로 놓아 GPU 에 한 벌만 남긴다.
        self.model = src.inference(compile=bool(comp),
                                   compile_backend=comp if isinstance(comp, str) else None,
                                   dtype=torch.float16 if self.half else torch.float32)
        self.res = int(src.model.resolution)               # 모델 입력 한 변 (정사각형으로 늘려 넣는다)
        self.names = list(src.class_names)                 # COCO 80종 (0-based)
        self._idx = {n: i for i, n in enumerate(self.names)}
        del src
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        self.conf = float(cfg.get("conf", 0.3))
        self.max_det = int(cfg.get("max_det", 100))
        self.preresize = bool(cfg.get("preresize", True))
        inc = {str(n).strip().lower() for n in cfg.get("include") or [] if str(n).strip()}
        self.keep = {i for i, n in enumerate(self.names) if n.lower() in inc} if inc else None
        self(np.zeros((480, 640, 3), np.uint8))           # 첫 장 지연을 미리 치른다

    def __call__(self, bgr):
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]
        if self.preresize and (h, w) != (self.res, self.res):
            # rfdetr 도 정사각형 res×res 로 (비율 무시, antialias 없이) 늘인다. 그것을 CPU 에서 먼저 하면
            # 큰 화면(360 3840×1920)을 통째로 텐서로 옮기지 않는다 — 박스 · 외곽선은 아래에서 원래 크기로 되돌린다
            small = cv2.resize(bgr, (self.res, self.res), interpolation=cv2.INTER_LINEAR)
            sx, sy = w / self.res, h / self.res
        else:
            small, sx, sy = bgr, 1.0, 1.0
        rgb = np.ascontiguousarray(small[:, :, ::-1])
        r = self.model.predict(rgb, threshold=self.conf, include_source_image=False)
        ms = (time.perf_counter() - t0) * 1000
        dets = []
        n = len(r)
        if n:
            names = r.data.get("class_name")
            order = np.argsort(-r.confidence)[: self.max_det]
            for j in order:
                nm = str(names[j]) if names is not None else str(int(r.class_id[j]))
                k = self._idx.get(nm, int(r.class_id[j]))
                if self.keep is not None and k not in self.keep:
                    continue
                d = {"cls": int(k), "name": nm, "conf": round(float(r.confidence[j]), 4),
                     "xyxy": [round(float(v), 1) for v in r.xyxy[j] * (sx, sy, sx, sy)]}
                if self.seg and r.mask is not None:
                    poly = _mask_poly(r.mask[j])
                    if poly is not None:
                        poly = poly * np.float32((sx, sy))
                        d["_poly"] = poly                   # 결합용 (저장 전에 뺀다)
                        if self.cfg.get("save_polygons", True):
                            d["polygon"] = np.round(poly, 1).tolist()
                dets.append(d)
        return {"infer_ms": round(ms, 1), "weight": self.weight.name, "variant": self.variant}, dets

    def close(self):
        self.model = None
        try:
            import torch
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:
            pass


def _mask_poly(mask):
    """이진 마스크 → 가장 큰 외곽선 (N,2) float32. 없으면 None"""
    cs, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    return c.reshape(-1, 2).astype(np.float32) if len(c) >= 3 else None
