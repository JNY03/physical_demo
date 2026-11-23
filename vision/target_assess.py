#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""target_assess.py — 지정 클래스 탐지(targeted)의 **근거 계산과 판정** (2026-11-23).

implements: 탐지대상_런처_계획_261123.md §3.0 · §3.1 · §3.4
관련: cue_catalog.py (근거 정의 · 역할) · hazard_overlay.py (river 의 판정 — 이 파일은 그것을 바꾸지 않는다)

river 판정(hazard_overlay.assess)은 위험물 두 클래스를 전제로 짜여 있다. 지정 클래스는 대상이
바뀌므로 판정을 따로 둔다. **결과 모양은 river 와 같다**(`assess` 사전: scores · suspect · cards …)
— 그래야 트랙 평활(AssessSmoother)과 웹 화면 카드(/vision/view)가 그대로 붙는다.

역할
  score  가중 평균에 들어간다. 못 재면(None) 빠지고 남은 무게로 다시 나눈다
  gate   통과하지 못하면 판정하지 않는다. **못 잰 게이트는 통과로 치지 않는다**(main 과 같다)
  veto   걸리면 점수를 0 으로 둔다
  cond   거리 신뢰 조건 — 판정에는 쓰지 않고 카드에만 (demo_box height · calibration)
  on     켜기만 (후보 넓히기 · 닮은꼴 감점)

main(JNY03/Physical_demo) 사전(CLIP) provider 에서 옮긴 것 — 식이 같다
  * CLIP: 양자화 ONNX(vision/text_model_quantized) + main 의 tokenizer(pad = <|endoftext|>), 224 가운데 자르기
  * 특징 사전: target_sim = max_k cos(crop, "a {대상}'s {특징}")  ≥ feature_sim_min
  * CLIP 색: gap = sim(자기 색) − max(경쟁 색) ≥ −tolerance
  * CLIP 모양: gap = max(긍정) − max(부정) ≥ −tolerance
  * 채도 · 색상: crop HSV 중앙값 (S ≥ min · H 범위)
  * 종횡비: 세로/가로 ≥ min (main 의 h_over_w)
  * OVD 짝 IoU: 탐지기 상자와 OVD 상자 중 가장 잘 맞는 쌍의 IoU ≥ pair_iou_min
퍼센트는 **확률이 아니다**. 어느 근거가 약한지 보라고 둔 공통 자다.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from cue_catalog import COLOURS, CUES, REPO

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
IMAGE_SIZE = 224
MIN_CROP_PX = 16          # main MIN_CROP_SIDE_PX — 이보다 작은 crop 은 CLIP 근거를 '못 잼'
SEMANTIC_T = 100.0        # CLIP logit 배율 (river verify.semantic 과 같다)
FEATURE_SPAN = 0.05       # 특징 사전 %: 하한 −0.05 → 0, 하한 → 50, 하한 +0.05 → 100 (선언된 자)
OVD_ASSOC_IOU, OVD_ASSOC_CONTAIN = 0.30, 0.7    # main ASSOCIATION_IOU_THR · CONTAINMENT_THR
DEFAULT_MIN_EVIDENCE = 0.45
LOOKALIKE_FROM, LOOKALIKE_MAX = 0.5, 15.0       # river decision.lookalike 와 같은 모양
#: 비슷한 이름(중심 명사가 같은 prompt-free 이름, 예: chair ← electric chair)의 확신에 곱하는 값. **선언이다** —
#: 같은 물건을 다르게 부른 것이지만 그 이름이 정말 그 물건인지는 확신만큼 믿지 않는다
SIMILAR_WEIGHT = 0.7
ARM_SHOWN = {"yoloe": "text", "yoloe_free": "free", "yolo_world": "yolo_world", "grounding_dino": "gdino"}


def detector_support(d, ovd_boxes):
    """탐지기 근거 — 같은 자리를 이 대상으로 부른 탐지기마다 확신 하나(비슷한 이름은 0.7 배, OVD 는 겹친 상자의 최고)를
    noisy-OR 로 모은다: 1 − Π(1 − c). 여러 탐지기가 동의할수록 오르고, 하나만 말하면 그 확신이다.
    → (%, 설명, OVD 최고 확신을 낸 arm 이름 또는 None)"""
    parts = []
    for arm, words in (d.get("_by_arm") or {}).items():
        exact = [(w, c) for w, (c, sim) in words.items() if not sim]
        similar = [(w, c) for w, (c, sim) in words.items() if sim]
        if exact:
            w, c = max(exact, key=lambda t: t[1])
            parts.append((c, "%s %s %.2f" % (ARM_SHOWN.get(arm, arm), w, c)))
        elif similar:
            w, c = max(similar, key=lambda t: t[1])
            parts.append((c * SIMILAR_WEIGHT, "%s ~%s %.2f×%.1f" % (ARM_SHOWN.get(arm, arm), w, c, SIMILAR_WEIGHT)))
    if not parts:
        parts.append((float(d["conf"]), "%s %.2f" % (d.get("lead") or d["name"], d["conf"])))
    best_ovd = {}
    for o in ovd_boxes:
        if _iou(d["xyxy"], o["xyxy"]) >= OVD_ASSOC_IOU or _contain(d["xyxy"], o["xyxy"]) >= OVD_ASSOC_CONTAIN:
            if o["conf"] > best_ovd.get(o["arm"], 0.0):
                best_ovd[o["arm"]] = o["conf"]
    for arm, c in sorted(best_ovd.items()):
        parts.append((c, "%s %.2f" % (ARM_SHOWN.get(arm, arm), c)))
    miss = 1.0
    for c, _t in parts:
        miss *= (1.0 - max(0.0, min(1.0, c)))
    return 100.0 * (1.0 - miss), " · ".join(t for _c, t in parts), (max(best_ovd, key=best_ovd.get) if best_ovd else None)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _contain(a, b):
    """작은 쪽이 큰 쪽 안에 드는 비율."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small > 0 else 0.0


# ---------------------------------------------------------------- CLIP (main 사전 provider 와 같은 엔진)
def _clip_preprocess(bgr):
    rgb = bgr[:, :, ::-1]
    h, w = rgb.shape[:2]
    s = IMAGE_SIZE / min(h, w)
    r = cv2.resize(np.ascontiguousarray(rgb), (max(IMAGE_SIZE, int(round(w * s))), max(IMAGE_SIZE, int(round(h * s)))),
                   interpolation=cv2.INTER_CUBIC)
    y = (r.shape[0] - IMAGE_SIZE) // 2
    x = (r.shape[1] - IMAGE_SIZE) // 2
    c = r[y:y + IMAGE_SIZE, x:x + IMAGE_SIZE].astype(np.float32) / 255.0
    return ((c - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)


class ClipEngine:
    """양자화 CLIP ONNX — main 이 'fp32 로 바꿔 보니 정확도가 깨져 되돌렸다' 는 그 판."""

    def __init__(self, files):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        vision, text, tok = files
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        avail = set(ort.get_available_providers())
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        so = ort.SessionOptions()
        self.vision = ort.InferenceSession(str(vision), so, providers=prov)
        self.text = ort.InferenceSession(str(text), so, providers=prov)
        self.provider = self.vision.get_providers()[0]
        t = Tokenizer.from_file(str(tok))
        eot = t.token_to_id("<|endoftext|>")
        t.enable_padding(pad_id=eot if eot is not None else 0, length=77)
        t.enable_truncation(77)
        self.tok = t
        self._text_cache = {}

    def images(self, crops):
        if not crops:
            return np.zeros((0, 512), np.float32)
        batch = np.stack([_clip_preprocess(c) for c in crops], axis=0)
        e = self.vision.run(None, {"pixel_values": batch})[0].astype(np.float32)
        return e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-8)

    def texts(self, prompts):
        """문장 → 정규화 임베딩. 같은 문장은 한 번만 돈다(대상이 바뀌어도 다시 쓰인다)."""
        want = [p for p in prompts if p not in self._text_cache]
        if want:
            ids = np.array([e.ids for e in self.tok.encode_batch(want)], np.int64)
            e = self.text.run(None, {"input_ids": ids})[0].astype(np.float32)
            e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-8)
            for p, v in zip(want, e):
                self._text_cache[p] = v
        return np.stack([self._text_cache[p] for p in prompts], axis=0) if prompts else np.zeros((0, 512), np.float32)


# ---------------------------------------------------------------- 픽셀 근거
def median_sat(crop):
    return float(np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 1]))


def median_hue(crop):
    return float(np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 0]))


def colour_fraction(crop, names):
    """crop 에서 목표 색(이름 여럿이면 합) 픽셀 비율과 색별 비율."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    total = np.zeros(h.shape, bool)
    per = {}
    for n in names:
        c = COLOURS[n]
        m = np.ones(h.shape, bool)
        if "h" in c:
            hm = np.zeros(h.shape, bool)
            for lo, hi in c["h"]:
                hm |= (h >= lo) & (h <= hi)
            m &= hm
        if "s_min" in c:
            m &= s >= c["s_min"]
        if "s_max" in c:
            m &= s <= c["s_max"]
        if "v_min" in c:
            m &= v >= c["v_min"]
        if "v_max" in c:
            m &= v <= c["v_max"]
        per[n] = float(m.mean())
        total |= m
    return float(total.mean()), per


def _aspect_range(t):
    """형상 근거의 종횡비 범위(긴 변 / 짧은 변). 선언이 없으면 실물 크기에서."""
    sp = (t.get("shape") or {}).get("aspect")
    if sp:
        return float(sp[0]), float(sp[1])
    mm = sorted(t.get("size_mm") or [])
    if len(mm) == 3 and mm[0] > 0:
        lo = max(1.0, mm[1] / mm[2]) if mm[2] else 1.0
        return round(max(1.0, lo / 1.5), 2), round(mm[2] / mm[0] * 1.5, 2)
    return None


def shape_pct(poly, box, rng):
    """마스크(없으면 상자)의 긴 변/짧은 변 이 범위 안이면 100, 밖이면 log2 로 1 옥타브에 걸쳐 0."""
    if poly is not None and len(poly) >= 3:
        (_cx, _cy), (rw, rh), _a = cv2.minAreaRect(np.asarray(poly, np.float32).reshape(-1, 2))
    else:
        rw, rh = box[2] - box[0], box[3] - box[1]
    if min(rw, rh) < 4:
        return None, "too small"
    ar = max(rw, rh) / max(min(rw, rh), 1e-6)
    lo, hi = rng
    off = 0.0 if lo <= ar <= hi else abs(math.log2(ar / (lo if ar < lo else hi)))
    return 100.0 * max(0.0, 1.0 - off), "aspect %.2f (%.1f~%.1f)" % (ar, lo, hi)


# ---------------------------------------------------------------- 대상별 준비
class _Prepared:
    """대상 하나의 문장 임베딩을 미리 만들어 둔다 — 장마다 다시 돌지 않는다."""

    def __init__(self, t, clip):
        self.t = t
        self.name = t["name"]
        cues = t["cues"]
        self.cues = cues
        self.weights = {k: float((t.get("weights") or {}).get(k, CUES[k]["weight"]))
                        for k, r in cues.items() if r == "score"}
        self.min_score = float(t.get("min_score", 50.0))
        self.min_evidence = float(t.get("min_evidence", DEFAULT_MIN_EVIDENCE))
        self.feat = self.sem = self.colour = self.shape = self.look = None
        if clip is None:
            return
        if "dict_features" in cues and t.get("features"):
            anchored = ["a %s's %s" % (t.get("anchor") or t.get("shown", self.name), f) for f in t["features"]]
            self.feat = (anchored, clip.texts(anchored))
        if "semantic" in cues:
            pos = list(t.get("descriptions") or ["a photo of a %s" % t.get("shown", self.name)])
            neg = list(t.get("negatives") or ["a photo of an object"])
            self.sem = (pos, neg, clip.texts(pos + neg))
            if "lookalike" in cues and t.get("confusers"):
                conf = list(t["confusers"])
                self.look = (conf, clip.texts(conf))
        cc = t.get("clip_colour")
        if "clip_colour" in cues and cc:
            nm = t.get("anchor") or t.get("shown", self.name)
            prompts = ["a %s's %s color" % (nm, cc["own"])] + ["a %s's %s color" % (nm, c) for c in cc["competitors"]]
            self.colour = (prompts, clip.texts(prompts), float(cc.get("tolerance", 0.0)))
        cs = t.get("clip_shape")
        if "clip_shape" in cues and cs:
            prompts = list(cs["positive"]) + list(cs["negative"])
            self.shape = (prompts, clip.texts(prompts), len(cs["positive"]), float(cs.get("tolerance", 0.0)))


class TargetEvidence:
    """대상 목록 → 검출마다 근거를 재고 판정한다."""

    def __init__(self, targets, clip_files=None, log=print):
        self.targets = {t["name"]: t for t in targets}
        need_clip = any(CUES[k]["needs"] == ["clip"] or "clip" in CUES[k]["needs"]
                        for t in targets for k in t["cues"])
        self.clip = None
        self.clip_why = None
        if need_clip:
            try:
                self.clip = _shared_clip(clip_files)
            except Exception as exc:              # CLIP 이 없으면 CLIP 근거만 '못 잼' (AI-C-05)
                self.clip_why = "%s: %s" % (type(exc).__name__, str(exc)[:120])
                if log:
                    log("  targeted CLIP 을 못 세움 — %s (CLIP 근거는 못 잼)" % self.clip_why)
        self.prep = {n: _Prepared(t, self.clip) for n, t in self.targets.items()}

    # -- 한 장
    def __call__(self, bgr, dets, ovd=None):
        """dets 마다 `assess` 를 붙인다. ovd = {대상: [{xyxy, conf, arm}]}. 걸린 ms 를 돌려준다."""
        t0 = time.perf_counter()
        H, W = bgr.shape[:2]
        crops, owners = [], []
        for i, d in enumerate(dets):
            if d["name"] not in self.prep:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            c = bgr[y1:y2, x1:x2]
            d["_crop"] = c if (c.size and min(c.shape[:2]) >= MIN_CROP_PX) else None
            if d["_crop"] is not None and self.clip is not None:
                crops.append(d["_crop"])
                owners.append(i)
        emb = self.clip.images(crops) if crops else None
        e_of = {i: emb[k] for k, i in enumerate(owners)} if emb is not None else {}
        for i, d in enumerate(dets):
            if d["name"] in self.prep:
                d["assess"] = self._one(d, e_of.get(i), (ovd or {}).get(d["name"]) or [], W, H)
            d.pop("_crop", None)
        return round((time.perf_counter() - t0) * 1000, 1)

    def _one(self, d, emb, ovd_boxes, W, H):
        p = self.prep[d["name"]]
        t, cues = p.t, p.cues
        rows = []            # [근거, %, 설명, 역할, 통과여부(gate)]
        crop = d.get("_crop")
        small = crop is None
        box = d["xyxy"]

        def add(key, pct, text, passed=None):
            rows.append([key, None if pct is None else round(float(pct), 1), text, cues[key], passed])

        # 탐지기 — 텍스트 · prompt-free(비슷한 이름은 0.7 배) · OVD 를 noisy-OR 로
        if "class" in cues:
            pct, txt, _arm = detector_support(d, ovd_boxes)
            add("class", pct, txt)
        # 글 설명 — 대상 문장 대 일반 문장의 확률
        if "semantic" in cues:
            if p.sem is None or emb is None:
                add("semantic", None, "not measured" + (" (small crop)" if small else ""))
            else:
                pos, neg, te = p.sem
                logits = SEMANTIC_T * (te @ emb)
                e = np.exp(logits - logits.max())
                prob = float(e[:len(pos)].sum() / e.sum())
                add("semantic", 100.0 * prob, "text description")
                if p.look is not None:
                    conf_n, ce = p.look
                    l2 = SEMANTIC_T * np.concatenate([te[:len(pos)] @ emb, ce @ emb])
                    e2 = np.exp(l2 - l2.max())
                    d["_lookalike"] = float(e2[len(pos):].sum() / e2.sum())
        # 특징 사전 (main dictionary)
        if "dict_features" in cues:
            if p.feat is None or emb is None:
                add("dict_features", None, "not measured" + (" (small crop)" if small else ""))
            else:
                anchored, fe = p.feat
                sims = fe @ emb
                k = int(sims.argmax())
                ts = float(sims[k])
                thr = float(t.get("feature_sim_min", 0.25))
                pct = 0.0 if ts < thr - FEATURE_SPAN else min(100.0, 50.0 + 50.0 * (ts - thr) / FEATURE_SPAN)
                add("dict_features", max(0.0, pct), "%s %.4f (min %.2f)" % (anchored[k], ts, thr),
                    passed=ts >= thr)
        # 형상
        if "shape" in cues:
            rng = _aspect_range(t)
            if rng is None:
                add("shape", None, "no range")
            else:
                pct, txt = shape_pct(d.get("_poly"), box, rng)
                add("shape", pct, txt, passed=None if pct is None else pct >= 50)
        # 색상 (HSV 목표색 비율)
        if "colour" in cues:
            names = t.get("colours") or []
            if crop is None or not names:
                add("colour", None, "not measured")
            else:
                frac, per = colour_fraction(crop, names)
                full = float(t.get("colour_full", 0.35))
                pct = 100.0 * min(1.0, frac / full)
                add("colour", pct, " · ".join("%s %.2f" % kv for kv in per.items()), passed=pct >= 50)
        # OVD 짝 IoU (main 2026-09-15) — 동의 자체는 탐지기 근거에 들어갔다
        if "ovd_pair_iou" in cues:
            best_iou = max([_iou(box, o["xyxy"]) for o in ovd_boxes] + [0.0])
            mn = float(t.get("pair_iou_min", 0.42))
            ok = best_iou >= mn
            add("ovd_pair_iou", 100.0 if ok else 0.0, "pair IoU %.2f %s %.2f" % (best_iou, ">=" if ok else "<", mn),
                passed=ok)
        # CLIP 색 · 모양 (main 게이트)
        if "clip_colour" in cues:
            if p.colour is None or emb is None:
                add("clip_colour", None, "not measured")
            else:
                prompts, ce, tol = p.colour
                sims = ce @ emb
                gap = float(sims[0] - np.delete(sims, 0).max())
                ok = gap >= -tol
                add("clip_colour", 100.0 if ok else 0.0,
                    "%s gap %+.4f (tol %.3f) · top %s" % (t["clip_colour"]["own"], gap, tol,
                                                          prompts[int(sims.argmax())].rsplit(" ", 2)[-2]),
                    passed=ok)
        if "clip_shape" in cues:
            if p.shape is None or emb is None:
                add("clip_shape", None, "not measured")
            else:
                prompts, se, npos, tol = p.shape
                sims = se @ emb
                gap = float(sims[:npos].max() - sims[npos:].max())
                ok = gap >= -tol
                add("clip_shape", 100.0 if ok else 0.0, "%s gap %+.4f (tol %.3f)"
                    % (prompts[int(sims.argmax())], gap, tol), passed=ok)
        if "saturation" in cues:
            if crop is None:
                add("saturation", None, "not measured")
            else:
                s = median_sat(crop)
                g = t["saturation"]
                ok = (s >= g.get("min", -1)) and (s <= g.get("max", 999))
                add("saturation", 100.0 if ok else 0.0, "median S %.0f (min %s)" % (s, g.get("min")), passed=ok)
        if "hue" in cues:
            if crop is None:
                add("hue", None, "not measured")
            else:
                h = median_hue(crop)
                g = t["hue"]
                ok = g["min"] <= h <= g["max"]
                add("hue", 100.0 if ok else 0.0, "median H %.0f (%s~%s)" % (h, g["min"], g["max"]), passed=ok)
        # 종횡비 — main(h_over_w) 또는 demo_box(관문, w/h)
        if "aspect" in cues:
            g = (d.get("box_gates") or {}).get("shape_gate")
            if g is not None:
                add("aspect", 100.0 if g["passed"] else 0.0, g["words"], passed=g["passed"])
            else:
                spec = t.get("aspect") or {}
                bw, bh = box[2] - box[0], box[3] - box[1]
                if bw <= 0 or bh <= 0:
                    add("aspect", None, "no box")
                else:
                    if "h_over_w" in spec:
                        lo, hi = spec["h_over_w"]
                        v, what = bh / bw, "h/w"
                    else:
                        lo, hi = (spec.get("w_over_h") or [None, None])
                        v, what = bw / bh, "w/h"
                    ok = (lo is None or v >= lo) and (hi is None or v <= hi)
                    add("aspect", 100.0 if ok else 0.0, "%s %.2f (min %s)" % (what, v, lo), passed=ok)
        # demo_box 관문 (학습 검출기 arm 이 잰 것)
        bg = d.get("box_gates") or {}
        for key, gname in (("ground", "ground_gate"), ("height", "height_gate"), ("calibration", "calibration_range")):
            if key not in cues:
                continue
            g = bg.get(gname)
            if g is None:
                add(key, None, "not measured")
            elif g.get("applicable") is False:
                # 판정 보류 — 원본 수신기는 아랫변이 화면 끝에 닿으면 바닥 관문을 통과로 둔다.
                # Go1 카메라가 아닌 프레임도 같다(관문 상수가 그 카메라 높이에서 잰 값이라 물을 수 없다)
                add(key, None, g.get("words", ""), passed=True if g.get("passed") else None)
            else:
                add(key, 100.0 if g.get("passed") else 0.0, g.get("words", ""), passed=bool(g.get("passed")))
        # 크기 거부
        veto = None
        if "size_veto" in cues:
            real, mm = d.get("real_mm"), t.get("size_mm")
            if not real or not mm:
                add("size_veto", None, "no metric size")
            else:
                diag = math.sqrt(sum(float(x) ** 2 for x in mm))
                k = float(t.get("max_over_diagonal", 2.0))
                bad = float(real["short"]) > k * diag
                add("size_veto", 0.0 if bad else 100.0, "%.0f x %.0f cm %s %.0f cm"
                    % (real["long"] / 10, real["short"] / 10, ">" if bad else "<=", k * diag / 10), passed=not bad)
                if bad:
                    veto = "too big"

        # ---- 판정
        w = p.weights
        scored = [(r[1], w.get(r[0], 0.0)) for r in rows if r[3] == "score" and r[1] is not None]
        wsum = sum(x for _p, x in scored)
        total = sum(w.values()) or 1.0
        score = (sum(a * b for a, b in scored) / wsum) if wsum > 0 else None
        evidence = wsum / total
        gates = [r for r in rows if r[3] == "gate"]
        failed = [r[0] for r in gates if r[4] is not True]
        gate_score = score
        if veto and score is not None:
            score = gate_score = 0.0
        shown = score
        lk = d.pop("_lookalike", None)
        pen = 0.0
        if lk is not None and score is not None:
            pen = LOOKALIKE_MAX * min(1.0, max(0.0, (lk - LOOKALIKE_FROM) / (1 - LOOKALIKE_FROM)))
            if pen > 0:
                shown = max(0.0, score - pen)
                host = next((r for r in rows if r[0] == "semantic"), None)
                if host is not None:
                    host[2] = "%s · look-alike %.0f%% -%.0f" % (host[2], 100 * lk, pen)
        enough = evidence >= p.min_evidence
        ok = (gate_score is not None and enough and not failed and gate_score >= p.min_score)
        name = t["name"]
        card_rows = []
        for r in rows:
            txt = r[2]
            if r[3] == "gate":
                txt = ("pass · " if r[4] is True else ("fail · " if r[4] is False else "")) + txt
            card_rows.append([r[0], r[1], txt])
        if d.get("dist") and not any(r[0] == "distance" for r in card_rows):
            v = d["dist"]
            card_rows.append(["distance", None, "%.1f m" % v["value"] if v.get("space") == "metric"
                              else "%.3f relative" % v["value"]])
        src = ", ".join(sorted(set(d.get("arms") or [])))
        return {"scores": {name: None if shown is None else round(shown, 1)},
                "gate_scores": {name: None if gate_score is None else round(gate_score, 1)},
                "suspect": name if ok else None,
                "score": None if shown is None else round(shown, 1),
                "min_score": p.min_score, "min_scores": {name: p.min_score},
                "evidence": round(evidence, 2), "enough": bool(enough),
                "gates_failed": failed, "vetoed": veto,
                "lookalike": None if lk is None else {"p": round(lk, 4), "penalty": round(pen, 1)},
                "provisional": bool(t.get("provisional", False)),
                "cards": {name: {"score": None if shown is None else round(shown, 1), "rows": card_rows,
                                 "source": src}}}


_CLIP = {}


def _shared_clip(files):
    """CLIP 엔진은 프로세스에 하나 — 대상이 바뀌어 다시 세워도 다시 읽지 않는다."""
    key = tuple(str(f) for f in files)
    if key not in _CLIP:
        _CLIP[key] = ClipEngine([Path(f) for f in files])
    return _CLIP[key]


def resolve_same_as(dets):
    """`same_as` (demo_door ↔ door): 같은 자리를 두 대상이 함께 잡으면 판정 점수가 높은 하나만 남긴다."""
    def sc(d):
        a = d.get("assess") or {}
        return ((1 if a.get("suspect") else 0), a.get("score") or 0.0)
    gone = set()
    for i, a in enumerate(dets):
        for j, b in enumerate(dets):
            if j <= i or id(a) in gone or id(b) in gone:
                continue
            if not ((a.get("_same_as") == b["name"]) or (b.get("_same_as") == a["name"])):
                continue
            if _iou(a["xyxy"], b["xyxy"]) >= 0.5:
                gone.add(id(b) if sc(a) >= sc(b) else id(a))
    return [d for d in dets if id(d) not in gone]
