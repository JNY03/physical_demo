#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""demo_box.py — 시연 박스(demo_box) 학습 검출기 arm (2026-11-23).

출처: Box_Detection(khw18033/physical-midterm-demo) `demo/test/mqtt_box_receiver.py` 의 `BoxDetector`
(2026-11-03, run yolov8n_graybox_v9). **식 · 관문 · 문장을 그대로 옮겼다** — 같은 프레임이면 같은
검출 · 관문 · 거리가 나와야 한다(시험: test_demo_box.py 가 원본 함수와 견준다).

다른 것
  * 출력 이름은 `demo_box`. 모델이 내는 `gray box` 는 `source_class` 로 남긴다.
  * **모든 소스에서 돈다.** 관문 상수와 거리식은 Go1 카메라(464x400 → 테두리 crop 237x241)
    높이 · 자세 · 화각에서 잰 값이라, 그 크기 프레임에서만 원본대로 잰다.
    다른 크기 프레임: 전체 프레임을 imgsz 256 으로 검출하고 shape 관문만 잰다. ground · height 는
    판정 보류(`applicable: false`), 거리 · 방위는 내지 않는다(거리는 거리 측정이 켜졌으면 UniDepth 가 붙인다).
  * 근거 문장(words)은 **영어로** 바꿨다(2026-11-23, 카드는 영어로만). 판정 · 값은 원본과 같다.
  * MQTT 판(scan) · 8방향 요약 · 경로 · try_box/ 산출물은 옮기지 않는다 — 이쪽은 프레임 단위 스트림이다.
  * 방위는 rotation_deg 를 모르므로 화면 중앙 대비 오프셋만 낸다(원본도 '참고값').
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from cue_catalog import REPO


def _p(path):
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else REPO / p


def go1_camera(target):
    """Go1 카메라 (프레임 크기 (W, H), 초점 거리 px) — 거리 계수 k 와 박스 앞면 높이로 f = k / 39.
    가중치를 올리지 않는다(스캔 방위가 demo_box 를 고르지 않은 판에도 쓴다, 2026-11-24)."""
    tr = dict(target.get("trained") or {})
    fit = json.loads(_p(tr["fit"]).read_text(encoding="utf-8"))
    k = float(fit["model_B_pinhole_offset"]["k"])
    return tuple(tr.get("go1_frame") or (464, 400)), k / float(tr.get("box_height_cm", 39.0))


class TrainedBoxArm:
    """학습 run 하나 = 가중치 + 그 가중치로 실측에 맞춘 거리 계수 + 그때의 imgsz."""

    name = "demo_box"

    def __init__(self, target, device):
        tr = dict(target.get("trained") or {})
        weights, fit_path = _p(tr["weight"]), _p(tr["fit"])
        if not weights.is_file():
            raise FileNotFoundError("demo_box 가중치가 없다: %s" % weights)
        if not fit_path.is_file():
            raise FileNotFoundError("demo_box 거리 계수가 없다: %s" % fit_path)
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        model_b = fit["model_B_pinhole_offset"]
        self.target = target["name"]
        self.run = tr.get("run")
        self.conf = float(tr.get("conf", 0.25))
        self.candidate_conf = float(tr.get("candidate_conf", 0.05))
        self.max_rejected = int(tr.get("max_rejected", 3))
        self.go1 = tuple(tr.get("go1_frame") or (464, 400))            # (W, H)
        self.crop_box = tuple(tr.get("border_crop") or (114, 80, 351, 321))
        self.edge = float(tr.get("edge_px", 2.0))
        self.box_h_cm = float(tr.get("box_height_cm", 39.0))
        g = tr.get("ground") or {}
        self.ga, self.gb, self.gtol = float(g.get("a", 0.803)), float(g.get("b", 135.7)), float(g.get("tol_px", 12.0))
        self.aspect_min = float(tr.get("aspect_min", 0.40))
        self.aspect_min_edge = float(tr.get("aspect_min_edge_cut", 0.35))
        self.source_class = tr.get("source_class", "gray box")
        self.imgsz = int(fit["imgsz"])
        self.k = float(model_b["k"])
        self.c = float(model_b["c_cm"])
        self.focal_px = self.k / self.box_h_cm
        gt = [row["gt_cm"] for row in model_b.get("lodo_detail") or []]
        self.validated_cm = [min(gt), max(gt)] if gt else None
        self.fit_source = {"file": str(fit_path.relative_to(REPO)) if REPO in fit_path.parents else str(fit_path),
                           "formula": model_b["formula"], "k": self.k, "c_cm": self.c, "imgsz": self.imgsz}
        self.device = device
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        w = self.crop_box[2] - self.crop_box[0]
        h = self.crop_box[3] - self.crop_box[1]
        self.model.predict(np.zeros((h, w, 3), np.uint8), imgsz=self.imgsz, conf=self.conf, verbose=False,
                           device=device)

    # -- 원본 BoxDetector 와 같은 식 (crop 좌표계)
    def distance(self, box):
        x1, y1, x2, y2 = box
        crop_h = self.crop_box[3] - self.crop_box[1]
        h_px = y2 - y1
        if y1 <= self.edge or y2 >= crop_h - self.edge:
            return {"distance_cm": None, "h_px": round(h_px, 2),
                    "reason": "box touches the top/bottom edge - height (39 cm front face) not measurable"}
        d = self.k / h_px - self.c
        return {"distance_cm": round(d, 1), "h_px": round(h_px, 2), "reason": None,
                "formula": "distance_cm = k / h_px − c",
                "substituted": f"{self.k:.1f} / {h_px:.2f} − {self.c:.1f} = {d:.1f}"}

    def bearing(self, box, rotation_deg=0.0):
        cx_orig = (box[0] + box[2]) / 2 + self.crop_box[0]
        offset = math.degrees(math.atan((cx_orig - self.go1[0] / 2) / self.focal_px))
        bearing = (rotation_deg + offset) % 360.0
        return {"bearing_deg": round(bearing, 1), "offset_deg": round(offset, 1),
                "formula": "bearing = rotation_deg + atan((cx − W/2) / f),  f = k / 39",
                "note": "참고값 -- 어안 보정 미정, 화면 가장자리 정확도 미검증 · rotation_deg 모름(0)"}

    def gates(self, box, score, distance, frame_wh=None):
        """원본 `BoxDetector.gates()` 와 같다. frame_wh 가 Go1 이 아니면 ground · height 는 판정 보류."""
        go1 = frame_wh is None
        if go1:
            crop_w, crop_h = self.crop_box[2] - self.crop_box[0], self.crop_box[3] - self.crop_box[1]
        else:
            crop_w, crop_h = frame_wh
        y1, y2 = box[1], box[3]
        h_px = y2 - y1
        gates = {"score_gate": {"required": True, "passed": score >= self.conf, "score": round(score, 3),
                                "threshold_min": self.conf, "formula": "passed = score ≥ conf",
                                "substituted": f"{score:.3f} ≥ {self.conf:g}",
                                "words": f"YOLO score {score:.2f}/{self.conf:g}"}}
        cut_top, cut_bottom = y1 <= self.edge, y2 >= crop_h - self.edge
        if not go1:
            gates["ground_gate"] = {"required": True, "applicable": False, "passed": True, "not_go1": True,
                                    "words": "not a Go1 camera frame - ground check deferred"}
        else:
            expected_y2 = self.ga * h_px + self.gb
            residual = y2 - expected_y2
            if cut_bottom:
                ground = {"applicable": False, "passed": True,
                          "words": "bottom edge touches the frame - ground check deferred (pass)"}
            else:
                ok = abs(residual) <= self.gtol
                ground = {"applicable": True, "passed": ok,
                          "words": (f"on the ground (bottom {y2:.0f}px, expected {expected_y2:.0f}±{self.gtol:g})" if ok else
                                    f"not on the ground (bottom {y2:.0f}px, expected {expected_y2:.0f}±{self.gtol:g}, "
                                    f"{abs(residual):.0f}px {'above' if residual < 0 else 'below'})")}
            gates["ground_gate"] = {
                "required": True, **ground, "y2": round(y2, 1), "h_px": round(h_px, 2),
                "expected_y2": round(expected_y2, 1), "residual_px": round(residual, 1), "tol_px": self.gtol,
                "a": self.ga, "b": self.gb,
                "formula": "passed = |y2 − (a·h_px + b)| ≤ tol  (아랫변이 crop 아래 끝이면 판정 보류)",
                "substituted": (f"y2 {y2:.1f} = crop 아래 끝 → 판정 보류" if cut_bottom else
                                f"|{y2:.1f} − ({self.ga:g}·{h_px:.1f} + {self.gb:g})| = {abs(residual):.1f} "
                                f"{'≤' if abs(residual) <= self.gtol else '>'} {self.gtol:g}")}
        w_px = box[2] - box[0]
        aspect = w_px / h_px if h_px > 0 else 0.0
        cut_lr = box[0] <= self.edge or box[2] >= crop_w - self.edge
        aspect_min = self.aspect_min_edge if cut_lr else self.aspect_min
        shape_ok = aspect >= aspect_min
        if shape_ok:
            shape_words = f"box shape (w/h {aspect:.2f} >= {aspect_min:g})"
        elif cut_lr:
            shape_words = (f"thin piece cut by the {'left' if box[0] <= self.edge else 'right'} edge (w/h {aspect:.2f} < {aspect_min:g}) "
                           f"- a real box shows whole in the next rotation")
        else:
            shape_words = f"too thin for the box (w/h {aspect:.2f} < {aspect_min:g}, side face alone is 0.54)"
        gates["shape_gate"] = {
            "required": True, "passed": shape_ok, "w_px": round(w_px, 2), "h_px": round(h_px, 2),
            "aspect": round(aspect, 3), "edge_cut_lr": cut_lr, "aspect_min": aspect_min,
            "formula": "passed = w_px / h_px ≥ aspect_min  (좌우 끝에 걸치면 aspect_min_edge_cut)",
            "substituted": f"{w_px:.1f} / {h_px:.1f} = {aspect:.2f} {'≥' if shape_ok else '<'} {aspect_min:g}",
            "words": shape_words}
        if not go1:
            gates["height_gate"] = {"required": False, "applicable": False, "passed": False,
                                    "words": "not a Go1 camera frame - box-height distance not used"}
            return gates
        if cut_top or cut_bottom:
            where = "top and bottom" if cut_top and cut_bottom else ("top" if cut_top else "bottom")
            height_words = f"height not measurable - box touches the {where} edge"
        else:
            height_words = f"height {h_px:.1f}px measurable"
        gates["height_gate"] = {
            "required": False, "passed": not (cut_top or cut_bottom),
            "y1": round(y1, 1), "y2": round(y2, 1), "h_px": round(h_px, 2), "crop_h": crop_h, "edge_px": self.edge,
            "formula": "passed = (y1 > edge_px) and (y2 < crop_h − edge_px)",
            "substituted": f"({y1:.1f} > {self.edge:g}) and ({y2:.1f} < {crop_h} − {self.edge:g})",
            "words": height_words}
        d = distance.get("distance_cm")
        if d is not None and self.validated_cm is not None:
            lo, hi = self.validated_cm
            inside = lo <= d <= hi
            gates["calibration_range"] = {
                "required": False, "passed": inside, "distance_cm": d, "validated_cm": [lo, hi],
                "source": self.fit_source["file"],
                "formula": "passed = validated_min ≤ distance_cm ≤ validated_max",
                "substituted": f"{lo:g} ≤ {d:.1f} ≤ {hi:g}",
                "words": (f"inside the measured range {lo:g}-{hi:g} cm ({d:.1f} cm)" if inside
                          else f"outside the measured range {lo:g}-{hi:g} cm ({d:.1f} cm) - reference only")}
        return gates

    def is_go1(self, bgr):
        return (bgr.shape[1], bgr.shape[0]) == self.go1

    def __call__(self, bgr):
        """→ (검출, conf 아래 후보, 필수 관문에 떨어진 검출). 좌표는 **원래 프레임** 좌표."""
        go1 = self.is_go1(bgr)
        if go1:
            x0, y0, x1, y1 = self.crop_box
            img = bgr[y0:y1, x0:x1]
        else:
            x0 = y0 = 0
            img = bgr
        t0 = time.perf_counter()
        result = self.model.predict(img, imgsz=self.imgsz, conf=self.candidate_conf, verbose=False,
                                    device=self.device)[0]
        ms = round((time.perf_counter() - t0) * 1000, 1)
        dets, rejected, by_gates = [], [], []
        boxes = list(result.boxes) if result.boxes is not None else []
        for b in sorted(boxes, key=lambda b: -float(b.conf[0])):
            box = [float(v) for v in b.xyxy[0].tolist()]
            score = float(b.conf[0])
            full = [box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0]
            if score < self.conf:
                if len(rejected) < self.max_rejected:
                    rejected.append({"bbox": [round(v, 1) for v in full], "score": round(score, 3),
                                     "threshold_min": self.conf,
                                     "words": f"candidate {score:.2f} < {self.conf:g} - dropped"})
                continue
            dist = self.distance(box) if go1 else {"distance_cm": None, "reason": "not a Go1 camera frame"}
            gates = self.gates(box, score, dist, None if go1 else (img.shape[1], img.shape[0]))
            det = {"cls_name": self.target, "prompt": self.source_class, "conf": score, "xyxy": full,
                   "poly": None, "box_gates": gates, "box_distance": dist,
                   "camera_profile": "go1" if go1 else "other"}
            if go1:
                det["box_bearing"] = self.bearing(box)
            failed = [n for n, g in gates.items() if g.get("required") and not g["passed"]]
            if failed:
                by_gates.append({"bbox": [round(v, 1) for v in full], "score": round(score, 3),
                                 "failed_gates": failed,
                                 "words": " · ".join(gates[n]["words"] for n in failed) + " - dropped"})
                continue
            dets.append(det)
        return dets, rejected, by_gates, ms
