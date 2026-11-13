"""거리 — **주기로 재고, 트랙으로 나른다.**

implements: AI-C-02 (값은 자기 기준을 들고 다닌다), AI-C-13 (자원 최소화),
AI-S-01 (추적이 유지하는 것), AI-O-01 (관리 비용도 잰다)
관련: AI-C-04 (provider 경계), AI-C-05, AI-C-16
tests: test_depth_track.py

## 왜 매 장 재지 않는가

metric depth 는 한 장에 수십~수백 ms 다. 매 장 돌리면 탐지기보다 비싸진다. 그런데
**거리는 장마다 새로 생기는 사실이 아니다** — 같은 물체는 같은 트랙 위에서 이어지고,
그 사이 거리는 연속으로 변한다. 그래서 주기로 재고 사이는 트랙으로 나른다.

## 나를 때 크기로 고친다

핀홀에서 겉보기 크기는 거리에 반비례한다. 트랙의 상자가 `s` 배 커졌으면 거리는
대략 `d / s` 다. 그대로 들고 있는 것보다 낫고, 가정이 하나뿐이라 설명된다
(정면 평행이동 가정 — 비스듬히 지나가면 틀린다. 그래서 **나이를 함께 낸다**).

## 새 트랙은 기다리지 않는다

주기만 지키면 새로 나타난 위험물이 다음 주기까지 거리 없이 떠 있다. 처음 보는
트랙이 있으면 **그 장에서 바로 잰다**. 위험물은 늦게 아는 것이 비용이다.

## 값은 기준을 들고 다닌다

`space` 가 `metric` 인지 `affine_inverse` 인지 함께 낸다. 상대 깊이를 미터처럼
쓰지 않기 위해서다(AI-C-02). 보정되지 않은 상대값은 미터로 바꾸지 않는다.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

METRIC, RELATIVE = "metric", "affine_inverse"


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)


class _HFMetric:
    """transformers 의 metric depth 모델. 가중치는 **내부 캐시에서만** 온다(AI-C-16)."""

    space = METRIC

    def __init__(self, spec):
        import os
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        mid = str(spec["model_id"])
        self.size = int(spec.get("size") or 266)
        self.device = str(spec.get("device") or "cuda:0")
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(mid)
        dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.model = AutoModelForDepthEstimation.from_pretrained(mid, dtype=dtype)
        self.model = self.model.to(self.device).eval()
        self.dtype = dtype
        self.name = str(spec.get("name") or mid.split("/")[-1])

    def __call__(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        px = self.proc(images=rgb, return_tensors="pt",
                       size={"height": self.size, "width": self.size})
        px = {k: (v.to(self.device).to(self.dtype) if v.dtype == self.torch.float32
                  else v.to(self.device)) for k, v in px.items()}
        with self.torch.no_grad():
            out = self.model(**px)
        return out.predicted_depth[0].float().cpu().numpy()


class _OnnxRelative:
    """ZipDepth 류 — **affine-invariant inverse depth**. 미터가 아니다."""

    space = RELATIVE

    def __init__(self, spec):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(spec.get("threads") or 8)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(spec["path"]), so,
                                         providers=["CPUExecutionProvider"])
        self.i = self.sess.get_inputs()[0].name
        self.o = self.sess.get_outputs()[0].name
        dims = [d for d in self.sess.get_inputs()[0].shape if isinstance(d, int) and d > 4]
        self.size = dims[-1] if dims else 256
        self.name = str(spec.get("name") or "zipdepth")

    def __call__(self, bgr):
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.squeeze(self.sess.run([self.o], {self.i: rgb.transpose(2, 0, 1)[None]})[0])


class _UniDepth:
    """UniDepth V2 — metric depth **와 내부파라미터를 함께** 낸다.

    fx 가 나오는 것이 이 provider 의 값어치다. 거리만으로는 "저게 137 mm 인가
    200 mm 인가" 를 물을 수 없지만, fx 와 거리가 있으면 픽셀 폭이 **실물 mm** 로 바뀐다
    (`mm = px * depth_m / fx * 1000`). 목함지뢰를 가려내는 힘의 대부분이 그 치수다.

    소스는 내부 경로에서 import 한다. `wandb` 는 학습 로깅용인데 패키지 초기화가
    끌고 오므로 **설치하지 않고 더미로 막는다** — 추론에 쓰이지 않는다.
    """

    space = METRIC

    def __init__(self, spec):
        import sys
        import types
        import torch
        src = str(spec.get("source") or "")
        if src and src not in sys.path:
            sys.path.insert(0, src)
        sys.modules.setdefault("wandb", types.ModuleType("wandb"))
        from unidepth.models import UniDepthV2
        self.torch = torch
        self.device = str(spec.get("device") or "cuda:0")
        self.model = UniDepthV2.from_pretrained(str(spec["model_id"])).to(self.device).eval()
        lvl = spec.get("resolution_level")
        if lvl is not None:
            self.model.resolution_level = int(lvl)
        self.name = str(spec.get("name") or "unidepth_v2")
        self.intrinsics = None                   # (fx, fy, cx, cy) — 마지막 프레임의 것

    def __call__(self, bgr):
        rgb = self.torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        with self.torch.no_grad():
            out = self.model.infer(rgb.to(self.device))
        k = out.get("intrinsics")
        if k is not None:
            k = k.squeeze().float().cpu().numpy()
            self.intrinsics = (float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))
        return out["depth"].squeeze().float().cpu().numpy()


KINDS = {"hf_metric": _HFMetric, "onnx_relative": _OnnxRelative, "unidepth": _UniDepth}


class DepthTracker:
    """주기로 재고 트랙으로 나르는 거리 단계."""

    def __init__(self, spec, log=print):
        self.period = max(1, int(spec.get("period") or 5))
        self.percentile = float(spec.get("percentile") or 40)
        self.max_age = int(spec.get("max_age") or 30)
        self.iou_match = float(spec.get("iou_match") or 0.5)
        self.classes = tuple(spec.get("classes") or ())
        self.provider, self.why = None, None
        kind = str(spec.get("kind") or "")
        if spec.get("enabled") and kind in KINDS:
            try:
                self.provider = KINDS[kind](spec)
            except Exception as exc:                 # 없으면 거리만 빠진다 (AI-C-05)
                self.why = "%s: %s" % (type(exc).__name__, str(exc)[:120])
        elif spec.get("enabled"):
            self.why = "kind 를 모릅니다: %r (가능: %s)" % (kind, ", ".join(KINDS))
        self.space = getattr(self.provider, "space", None)
        self.ready = self.provider is not None
        self._mem = {}                               # key -> {d, frame, box}
        self._n = 0
        if log:
            log("  depth %s%s · 주기 %d장 · %s"
                % ("준비됨" if self.ready else "**비활성**",
                   (" (%s)" % self.why) if self.why else "",
                   self.period, self.space or "-"))

    def _key(self, det, prev_keys):
        tid = det.get("track_id")
        if tid is not None:
            return ("t", tid)
        # 트랙이 없으면 앞 장 상자와의 겹침으로 잇는다(문서화된 차선책)
        best, score = None, self.iou_match
        for k in prev_keys:
            box = self._mem[k].get("box")
            if box and self._mem[k].get("cls") == det["name"]:
                s = _iou(box, det["xyxy"])
                if s >= score:
                    best, score = k, s
        return best

    def _sample(self, dmap, box, shape):
        H, W = shape
        dh, dw = dmap.shape[:2]
        x1, y1, x2, y2 = box
        sx, sy = dw / float(W), dh / float(H)
        a, b = max(0, int(x1 * sx)), max(0, int(y1 * sy))
        c, d = min(dw, int(x2 * sx)), min(dh, int(y2 * sy))
        patch = dmap[b:d, a:c]
        if patch.size == 0:
            return None
        good = patch[patch > 0] if self.space == METRIC else patch
        if good.size == 0:
            return None
        return float(np.percentile(good, self.percentile))

    def _add_size(self, det, depth_m):
        """**픽셀을 실물 치수로.** fx 와 거리가 둘 다 있어야 한다 — 없으면 적지 않는다.

        `mm = px * depth_m / fx * 1000` (핀홀, 광축에 수직인 면 가정).
        비스듬히 보면 짧게 나온다 — 그래서 **하한**으로 읽어야 한다.
        """
        k = getattr(self.provider, "intrinsics", None)
        if not k or self.space != METRIC or not depth_m:
            return
        fx = k[0]
        x1, y1, x2, y2 = det["xyxy"]
        det["real_mm"] = {"long": round(max(x2-x1, y2-y1) * depth_m / fx * 1000, 1),
                          "short": round(min(x2-x1, y2-y1) * depth_m / fx * 1000, 1),
                          "fx": round(fx, 1)}

    def __call__(self, bgr, dets):
        """`dist` 를 붙인다. 돌린 ms 를 돌려준다 (안 돌렸으면 0)."""
        if not self.ready:
            return 0.0
        want = [d for d in dets if not self.classes or d["name"] in self.classes]
        if not want:
            self._n += 1
            return 0.0
        prev_keys = list(self._mem)
        keyed = [(d, self._key(d, prev_keys)) for d in want]
        # **새 트랙은 기다리지 않는다** — 처음 보는 위험물이 있으면 주기를 당긴다
        fresh = any(k is None or k not in self._mem for _d, k in keyed)
        due = (self._n % self.period == 0) or fresh
        t0 = time.perf_counter()
        dmap = self.provider(bgr) if due else None
        ms = round((time.perf_counter() - t0) * 1000, 1) if due else 0.0
        H, W = bgr.shape[:2]
        for det, key in keyed:
            if key is None:
                key = ("b", round(det["xyxy"][0]), round(det["xyxy"][1]), det["name"])
            if dmap is not None:
                v = self._sample(dmap, det["xyxy"], (H, W))
                if v is not None:
                    self._mem[key] = {"d": v, "frame": self._n, "box": list(det["xyxy"]),
                                      "cls": det["name"], "w": det["xyxy"][2]-det["xyxy"][0]}
                    det["dist"] = {"value": round(v, 3), "space": self.space, "age": 0,
                                   "source": "measured"}
                    self._add_size(det, v)
                    continue
            mem = self._mem.get(key)
            if not mem or self._n - mem["frame"] > self.max_age:
                det["dist"] = None                   # **모르면 모른다고 한다**
                continue
            # 겉보기 크기로 고친다: 상자가 s 배 커졌으면 거리는 대략 d/s
            w_now = det["xyxy"][2] - det["xyxy"][0]
            s = (w_now / mem["w"]) if mem.get("w") else 1.0
            s = min(3.0, max(0.33, s))               # 터무니없는 배율은 믿지 않는다
            value = mem["d"] / s if self.space == METRIC else mem["d"] * s
            det["dist"] = {"value": round(value, 3), "space": self.space,
                           "age": self._n - mem["frame"], "source": "carried",
                           "scale": round(s, 3)}
            self._add_size(det, value)
            mem["box"] = list(det["xyxy"])
        self._n += 1
        return ms
