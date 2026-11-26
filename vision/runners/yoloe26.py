# -*- coding: utf-8 -*-
"""yoloe26.py — YOLOE-26 (열린 어휘 검출 + 분할) 실행기 (2026-11-26).

YOLOE-11 (vision_infer.YoloeRunner · _TextArm · _FreeArm) 의 뒤를 잇는 것. 부르는 법은 같다:

  FreeRunner   prompt-free 가중치(yoloe-26{s,l}-seg-pf.pt) 의 내장 어휘(4585종) 그대로.
               provides [detect, segment, open_vocab]
  TextRunner   일반 가중치(yoloe-26{s,l}-seg.pt) 에 classes 를 텍스트 프롬프트로 넣는다.
               provides [detect, segment, open_vocab] (어휘 = 준 문구)
  Runner       cfg["mode"] (free | text) 로 둘 중 하나를 고른다 (load_runner 기본 이름)

YOLOE-11 과 다른 점 — **텍스트 인코더가 MobileCLIP2 (mobileclip2_b.ts, 254 MB)** 다. YOLOE-11 의
mobileclip_blt.ts 는 YOLOE-26 에 맞지 않는다(가중치 yaml 의 text_model 이 mobileclip2:b). 인코더 파일은
가중치의 text_model 로 고르므로 YOLOE-11 가중치도 이 실행기로 돈다(그때는 mobileclip_blt.ts).
토크나이저는 ultralytics 가 `clip` 패키지(clip.clip.tokenize)로 부른다 — clip 이 설치되어 있어야 한다.
ultralytics 는 인코더를 **작업 폴더**(또는 weights_dir)에서 찾고 없으면 인터넷에서 받는다 —
폐쇄망에서 조용히 나가지 않도록 파일이 없으면 시작할 때 거절한다.

cfg (vision_config models.<선택지> 와 같은 이름):
  weight            가중치 경로 (필수)
  mode              free | text            (Runner 만 본다)
  classes           text 모드의 문구 목록 (필수)
  text_encoder_dir  mobileclip2_b.ts 가 있는 폴더 (기본: weight 와 같은 폴더)
  include · exclude free 모드에서 남길 · 뺄 내장 어휘 이름 (NMS 전에 classes= 로 거른다)
  conf 0.25 · iou 0.5 · imgsz 640 · max_det 100 · half true
  imgsz_follow_frame  true 면 프레임 긴 변(32 배수 올림)이 imgsz 보다 작을 때 그것을 쓴다
  save_polygons     true 면 det["polygon"] 도 남긴다 (det["_poly"] 는 항상)

의존: ultralytics (>= 8.4.x, YOLOE-26 지원) · torch · numpy. 저장소의 다른 모듈을 import 하지 않는다.
"""
from __future__ import annotations

import contextlib
import time
from pathlib import Path

import numpy as np

#: 가중치 yaml 의 text_model → ultralytics 가 찾는 인코더 파일 이름. YOLOE-26 은 mobileclip2:b,
#: YOLOE-11 · v8 은 text_model 칸이 없다(= mobileclip:blt). 같은 실행기로 두 세대를 다 돌린다 (비교 측정용)
ENCODERS = {"mobileclip2:b": "mobileclip2_b.ts", "mobileclip:blt": "mobileclip_blt.ts"}
ENCODER = ENCODERS["mobileclip2:b"]


def _imgsz(cfg, bgr):
    """imgsz 를 상한으로, imgsz_follow_frame 이면 프레임 긴 변(32 배수로 올림)이 더 작을 때 그것."""
    size = int(cfg.get("imgsz", 640))
    if cfg.get("imgsz_follow_frame"):
        long_side = max(bgr.shape[:2])
        size = min(size, max(32, -(-long_side // 32) * 32))
    return size


class _Base:
    name = "yoloe26"
    kind = "detect"
    provides: tuple = ()

    def __init__(self, cfg: dict, device: str):
        from ultralytics import YOLOE
        self.cfg = dict(cfg or {})
        self.device = str(device)
        weight = Path(str(self.cfg.get("weight") or "")).expanduser()
        if not weight.is_file():
            raise FileNotFoundError("YOLOE-26 가중치가 없다: %s" % weight)
        self.weight = weight
        self.model = YOLOE(str(weight))
        self.half = bool(self.cfg.get("half", True)) and self.device.startswith("cuda")
        self.keep = None                                   # predict classes= (None = 전부)
        self._setup()
        self(np.zeros((480, 640, 3), np.uint8))            # 첫 장 지연을 미리 치른다

    def _setup(self):
        raise NotImplementedError

    def __call__(self, bgr):
        c = self.cfg
        t0 = time.perf_counter()
        r = self.model.predict(bgr, conf=c.get("conf", 0.25), iou=c.get("iou", 0.5), imgsz=_imgsz(c, bgr),
                               max_det=c.get("max_det", 100), classes=self.keep, device=self.device,
                               quantize=16 if self.half else None, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        dets = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            polys = r.masks.xy if r.masks is not None else [None] * len(cls)
            for b, s, k, poly in zip(xyxy, conf, cls, polys):
                d = {"cls": int(k), "name": str(r.names[int(k)]), "conf": round(float(s), 4),
                     "xyxy": [round(float(v), 1) for v in b]}
                if poly is not None and len(poly):
                    d["_poly"] = poly
                    if c.get("save_polygons", False):
                        d["polygon"] = np.round(poly, 1).tolist()
                dets.append(d)
        return {"infer_ms": round(ms, 1), "weight": self.weight.name, "prompt": self.mode}, dets

    def close(self):
        self.model = None


class FreeRunner(_Base):
    """prompt-free — 내장 어휘 그대로. include/exclude 로 거른다."""
    mode = "free"
    provides = ("detect", "segment", "open_vocab")

    def _setup(self):
        if not self.weight.stem.endswith("-pf"):
            raise ValueError("prompt-free 에는 -pf 가중치가 필요하다: %s" % self.weight)
        names = self.model.names or {}
        by_name = {}
        for i, n in names.items():
            by_name.setdefault(str(n).lower(), []).append(int(i))

        def ids(key):
            return {i for n in self.cfg.get(key) or [] for i in by_name.get(str(n).strip().lower(), [])}
        include, exclude = ids("include"), ids("exclude")
        if include or exclude:
            self.keep = sorted((include or {int(i) for i in names}) - exclude)
            if not self.keep:
                raise ValueError("include/exclude 로 남는 클래스가 없다")


class TextRunner(_Base):
    """텍스트 프롬프트 — 시작할 때 MobileCLIP2 로 한 번 임베딩하고, 추론 중에는 인코더를 안 쓴다."""
    mode = "text"
    provides = ("detect", "segment", "open_vocab")

    def _setup(self):
        classes = [str(c).strip() for c in self.cfg.get("classes") or [] if str(c).strip()]
        if not classes:
            raise ValueError("text 모드인데 classes 가 비었다")
        if self.weight.stem.endswith("-pf"):
            raise ValueError("텍스트 프롬프트에는 -pf 가 아닌 가중치가 필요하다: %s" % self.weight)
        enc_dir = Path(str(self.cfg.get("text_encoder_dir") or self.weight.parent)).expanduser()
        variant = str(getattr(self.model.model, "yaml", {}).get("text_model") or "mobileclip:blt")
        enc = ENCODERS.get(variant)
        if enc is None:
            raise ValueError("모르는 텍스트 인코더 %s (%s)" % (variant, self.weight.name))
        if not (enc_dir / enc).is_file():
            # 없으면 ultralytics 가 인터넷에서 받는다 — 먼저 거절한다
            raise FileNotFoundError("텍스트 인코더가 없다: %s/%s" % (enc_dir, enc))
        self.encoder = enc
        with contextlib.chdir(enc_dir):
            pe = self.model.get_text_pe(classes)
        self.model.set_classes(classes, pe)
        self.classes = classes

    def set_classes(self, classes):
        """문구를 바꾼다 (돌고 있는 모델을 건드리므로 추론과 같은 스레드에서만)."""
        self.cfg["classes"] = list(classes)
        self._setup()


def Runner(cfg: dict, device: str):
    """cfg["mode"] 로 고른다 — free(기본) | text."""
    mode = str((cfg or {}).get("mode", "free")).lower()
    if mode == "text":
        return TextRunner(cfg, device)
    if mode == "free":
        return FreeRunner(cfg, device)
    raise ValueError("mode 는 free 또는 text: %s" % mode)
