#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""feature_match.py — 실시간 특징점 매칭 (SuperPoint + LightGlue, 2026-11-23).

가중치: models/lightglue_superpoint (HF ETH-CVG/lightglue_superpoint, transformers 의 LightGlueForKeypointMatching).
런타임에 받지 않는다 — 없으면 매칭만 '못 잼' 이다.

실측(RTX 4050 Laptop, robot1 2026-11-23, crop 256 · fp16 · 점 256개):
  같은 사람 다른 장 31 · 74 쌍, 다른 사람 · 외투 건 의자 5 ~ 7 쌍. 한 쌍 약 48 ms, 8 쌍 묶음 약 200 ms, GPU 0.6 GB.
그래서 재식별은 **전역 특징(CLIP)으로 후보를 1~2 개로 좁힌 뒤에만** 이것을 부른다.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import cv2
import numpy as np

SIZE = 256              # 매칭 입력 한 변(crop 긴 변을 이것으로)
MAX_KEYPOINTS = 256
MATCH_THRESHOLD = 0.2   # LightGlue 매칭 확신 하한(post_process 기본 범위)


def to_rgb_square(bgr, size=SIZE):
    """crop → size x size RGB. **늘려 맞춘다**(검게 채우지 않는다 — 경계에 가짜 특징점이 생긴다). 같은 물체는 종횡비가
    비슷해 늘린 모양도 비슷하다 — 위 실측이 이 방식이다."""
    r = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(r, cv2.COLOR_BGR2RGB)


@contextlib.contextmanager
def no_wandb_stub():
    """depth_track 이 UniDepth 를 위해 넣는 빈 `wandb` 모듈(__spec__ 없음)이 있으면 transformers 가
    find_spec('wandb') 에서 ValueError 로 넘어진다. transformers 를 적재하는 동안만 치운다."""
    stub = sys.modules.get("wandb")
    if stub is not None and getattr(stub, "__spec__", None) is None:
        del sys.modules["wandb"]
    else:
        stub = None
    try:
        yield
    finally:
        if stub is not None:
            sys.modules.setdefault("wandb", stub)


class FeatureMatcher:
    """crop 쌍 → 매칭 수. 한 번에 여러 쌍을 묶어 돈다."""

    def __init__(self, path, device="cuda:0"):
        import torch
        with no_wandb_stub():
            from transformers import AutoConfig, AutoImageProcessor, AutoModelForKeypointMatching
        path = Path(path)
        if not (path / "model.safetensors").is_file():
            raise FileNotFoundError("LightGlue 가중치가 없다: %s" % path)
        with no_wandb_stub():
            self._load(path, device, AutoConfig, AutoImageProcessor, AutoModelForKeypointMatching, torch)

    def _load(self, path, device, AutoConfig, AutoImageProcessor, AutoModelForKeypointMatching, torch):
        cfg = AutoConfig.from_pretrained(str(path), local_files_only=True)
        cfg.keypoint_detector_config.max_keypoints = MAX_KEYPOINTS
        self.proc = AutoImageProcessor.from_pretrained(str(path), local_files_only=True)
        self.proc.size = {"height": SIZE, "width": SIZE}
        self.half = str(device).startswith("cuda")
        self.model = AutoModelForKeypointMatching.from_pretrained(
            str(path), config=cfg, local_files_only=True,
            dtype=torch.float16 if self.half else torch.float32).to(device).eval()
        self.device = device
        self.torch = torch

    def count(self, pairs):
        """pairs: [(rgb_a, rgb_b)] (to_rgb_square 결과) → [매칭 수]."""
        if not pairs:
            return []
        inp = self.proc([[a, b] for a, b in pairs], return_tensors="pt").to(self.device)
        if self.half:
            inp["pixel_values"] = inp["pixel_values"].half()
        with self.torch.no_grad():
            out = self.model(**inp)
        sizes = [[(SIZE, SIZE), (SIZE, SIZE)]] * len(pairs)
        post = self.proc.post_process_keypoint_matching(out, sizes, threshold=MATCH_THRESHOLD)
        return [int(len(p["matching_scores"])) for p in post]
