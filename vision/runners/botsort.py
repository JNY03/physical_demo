# -*- coding: utf-8 -*-
"""botsort.py — BoT-SORT 트래커 어댑터 (2026-11-26). provides [track]

ultralytics 에 들어 있는 BOTSORT 를 그대로 쓴다 — 다시 구현하지 않는다. vision_infer.VendorTracker
(backend: bytetrack) 와 같은 방식으로 **밖에서 만든 검출**을 먹이고, 검출마다 track_id 를 붙여 돌려준다.
ByteTrack 과 다른 점:
  · GMC (전역 움직임 보정, 기본 sparseOptFlow) — 카메라가 움직이는 로봇 · 드론 영상에서 칼만 예측을 프레임
    움직임만큼 옮긴다. 그래서 **프레임(img) 을 함께 넘겨야** 한다. 없으면 GMC 는 건너뛴다(항등)
  · ReID (선택, 기본 끔) — with_reid: true 이고 reid_model 에 **로컬 파일**(예: models/yolo26n-reid.onnx)을
    줄 때만 켠다. model: auto(탐지기 특징)는 ultralytics predictor 안에서만 되므로 이 경로에서는 쓸 수 없다

스트림(카메라)마다 트래커를 따로 둔다 — update(dets, ts, bgr, stream="pi12") 처럼 부른다.
번호를 붙이는 층이다: 트래커가 잇지 못한 검출도 번호 없이 그대로 나간다(on_untracked: keep, 기본).
박스도 탐지기 것을 그대로 둔다(keep_boxes: detection, 기본) — VendorTracker 의 실측 근거와 같다.

Runner 규약의 __call__(bgr) 은 검출이 없으므로 __call__(bgr, dets=..., stream=..., ts=...) 로 받는다.

cfg: botsort.yaml 의 값 아무거나 (track_high_thresh · new_track_thresh · track_buffer · gmc_method …) +
     with_reid · reid_model (로컬 경로) · on_untracked (keep|drop) · keep_boxes (detection|tracker) ·
     reset_gap_s (이보다 오래 끊기면 그 스트림을 처음부터, 기본 2.0) · reid_device (기본: device)

의존: ultralytics · lap · numpy (· opencv — GMC). 저장소의 다른 모듈을 import 하지 않는다.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

OWN_KEYS = ("on_untracked", "keep_boxes", "reset_gap_s", "reid_model", "reid_device")


class _Boxes:
    """ultralytics 트래커가 보는 모양 — conf · cls · xywh · xyxy 와 불리언 마스크 인덱싱."""

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
        return type(self)(arrays=(self.xyxy[mask], self.conf[mask], self.cls[mask], self.xywh[mask]))

    def __len__(self):
        return len(self.conf)


class Runner:
    name = "botsort"
    kind = "track"
    provides = ("track",)
    #: ultralytics 의 트래커 설정 파일 · 모듈 · 클래스 — bytetrack.py 가 이 셋만 바꿔 같은 어댑터를 쓴다 (2026-11-27)
    TRACKER_YAML = "botsort.yaml"
    TRACKER_MODULE = "ultralytics.trackers.bot_sort"
    TRACKER_CLASS = "BOTSORT"

    def __init__(self, cfg: dict, device: str = "cpu"):
        import importlib
        import yaml
        import ultralytics
        from ultralytics.utils import IterableSimpleNamespace
        BOTSORT = getattr(importlib.import_module(self.TRACKER_MODULE), self.TRACKER_CLASS)
        try:
            import lap  # noqa: F401
        except ImportError as exc:
            # 없으면 ultralytics 가 런타임에 받으려 한다 — 폐쇄망에서 조용히 나가지 않게 먼저 거절
            raise ImportError("lap 이 없다 (pip install 'lap>=0.5.12')") from exc
        cfg = dict(cfg or {})
        spec = Path(ultralytics.__file__).parent / "cfg" / "trackers" / self.TRACKER_YAML
        args = yaml.safe_load(spec.read_text(encoding="utf-8"))
        unknown = sorted(k for k in cfg if k not in args and k not in OWN_KEYS)
        if unknown:
            raise ValueError(self.name + " 가 모르는 값: %s (받는 것: %s)"
                             % (", ".join(unknown), ", ".join(sorted(set(args) | set(OWN_KEYS)))))
        args.update({k: v for k, v in cfg.items() if k in args})
        if args.get("with_reid"):
            reid = cfg.get("reid_model")
            if not reid or not Path(str(reid)).expanduser().is_file():
                # model: auto 는 predictor 안에서만 된다. 이름만 주면 ultralytics 가 받으러 나간다 — 거절
                raise FileNotFoundError("with_reid 에는 로컬 reid_model 파일이 필요하다: %s" % reid)
            args["model"] = str(Path(str(reid)).expanduser())
        self.args = args
        self.device = str(cfg.get("reid_device") or device)
        self.on_untracked = str(cfg.get("on_untracked", "keep")).lower()
        if self.on_untracked not in ("keep", "drop"):
            raise ValueError("on_untracked 는 keep 또는 drop: %s" % self.on_untracked)
        self.keep_boxes = str(cfg.get("keep_boxes", "detection")).lower()
        if self.keep_boxes not in ("detection", "tracker"):
            raise ValueError("keep_boxes 는 detection 또는 tracker: %s" % self.keep_boxes)
        self.reset_gap = float(cfg.get("reset_gap_s", 2.0))
        self._make = lambda: BOTSORT(IterableSimpleNamespace(**dict(args, device=self.device)))
        self._streams = {}                                 # stream → [tracker, last_ts]
        self._encoder = None                               # ReID 인코더는 스트림끼리 나눠 쓴다

    def _tracker(self, stream, ts):
        st = self._streams.get(stream)
        if st is None or (ts is not None and st[1] is not None and not 0 <= ts - st[1] <= self.reset_gap):
            tr = self._make()
            if self.args.get("with_reid"):                 # 인코더를 스트림마다 다시 올리지 않는다
                if self._encoder is None:
                    self._encoder = tr.encoder
                tr.encoder = self._encoder
            st = self._streams[stream] = [tr, None]
        if ts is not None:
            st[1] = ts
        return st[0]

    def update(self, dets, ts=None, bgr=None, stream="default"):
        """track 약속(vision/runners/README.md): update(dets, ts, bgr=None) → 같은 목록의 사본에 track_id.
        bgr 가 없으면 GMC 는 건너뛴다(항등). stream 은 카메라마다 트래커를 가른다."""
        img = bgr
        tr = self._tracker(stream, ts)
        if not dets:
            tr.update(_Boxes([]), img)
            return []
        rows = np.asarray(tr.update(_Boxes(dets), img)).reshape(-1, 8)
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
        return out

    def __call__(self, bgr, dets=None, stream="default", ts=None):
        t0 = time.perf_counter()
        out = self.update(list(dets or []), ts, bgr, stream=stream)
        ms = (time.perf_counter() - t0) * 1000
        return {"track_ms": round(ms, 2), "tracker": self.name, "gmc": self.args.get("gmc_method"),
                "reid": bool(self.args.get("with_reid")), "stream": stream}, out

    def reset(self, stream=None):
        if stream is None:
            self._streams.clear()
        else:
            self._streams.pop(stream, None)

    def close(self):
        self._streams.clear()
        self._encoder = None
