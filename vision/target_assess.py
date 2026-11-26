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
#: 기준 사진(ref_match) — 사람이 올린 사진들의 CLIP 임베딩과 후보 조각을 견준다.
#: 하한 −REF_SPAN → 0%, 하한 → 50%, 하한 +REF_SPAN → 100% (특징 사전과 같은 자 모양).
#: 0.70 은 CLIP ViT-B/32 에서 「같은 물체를 다른 각도에서 찍은 사진」 쯤이다(다른 물체는 대개 0.5 아래).
#: **문턱을 0.99 같은 값으로 두지 마라** — 양자화 ONNX 는 묶음 크기만 달라져도 코사인이 0.990 까지 흔들린다
#: (기준 임베딩은 한 번에 재고 런타임은 후보 수만큼 묶어 재므로 늘 그만큼 차이가 난다).
REF_SIM_MIN = 0.70
REF_SPAN = 0.10
#: 사진이 여럿이면 가장 닮은 것 몇 개의 평균으로 본다 — 한 장이 운 좋게 맞은 것에 끌려가지 않게
REF_TOPK = 3
LOOKALIKE_FROM, LOOKALIKE_MAX = 0.5, 15.0       # river decision.lookalike 와 같은 모양
#: 비슷한 이름(중심 명사가 같은 prompt-free 이름, 예: chair ← electric chair)의 확신에 곱하는 값. **선언이다** —
#: 같은 물건을 다르게 부른 것이지만 그 이름이 정말 그 물건인지는 확신만큼 믿지 않는다
SIMILAR_WEIGHT = 0.7
#: 설명에 적는 탐지기 이름 — YOLOE 텍스트 · prompt-free 는 이름을 적지 않는다(낱말만), OVD 만 적는다
ARM_SHOWN = {"yoloe": "", "yoloe_free": "", "yolo_world": "yolo_world", "grounding_dino": "gdino"}


#: 국소 축 — 부분이 '잡혔다' 고 보는 확신. **모든 arm 의 탐지기 하한(arm_conf)과 같은 값**이다(장면에 맞춘 값이 아니다)
PART_FLOOR = 0.05
PART_SIZE = 320          # 부분은 crop 을 이 크기로 키워 본다(사람 폭 40 px → 머리가 수 px 라 그대로는 못 본다)
#: **안 보인 것과 없는 것은 다르다.** 부분이 원본에서 이 크기(px)보다 작게 찍힐 상자는 국소를 재지 않는다(못 잼).
#: 부분의 크기는 대상 파일 parts.scale(물체 긴 변 대비, 사실). 8 px 은 탐지기가 볼 수 있는 최소 크기로 대상과 무관하다
MIN_PART_PX = 8
PART_NEGATIVES = ["wall", "floor", "jacket", "chair", "background"]   # 부분 어휘가 혼자면 아무 데나 부분이라 부른다


class PartsProbe:
    """**국소 근거** — 대상 crop 안에서 그 대상의 부분(대상 파일 parts)을 찾는다. YOLOE 텍스트(11l)를 따로 하나 올린다
    (후보 arm 의 어휘를 바꾸지 않으려고). 한 장의 후보를 한 번에 묶어 돈다."""

    def __init__(self, targets, weight, enc_dir, device):
        import contextlib as _cl
        from ultralytics import YOLOE
        self.words = []
        self.of = {}
        self.scale = {}
        for t in targets:
            spec = t.get("parts") or {}
            names = [str(n) for n in spec.get("names") or []]
            self.of[t["name"]] = (names, spec.get("where", "any"))
            self.scale[t["name"]] = float(spec.get("scale", 0.1))
            for n in names:
                if n not in self.words:
                    self.words.append(n)
        vocab = self.words + [w for w in PART_NEGATIVES if w not in self.words]
        self.vocab = vocab
        self.model = YOLOE(str(weight))
        with _cl.chdir(enc_dir):
            pe = self.model.get_text_pe(vocab)
        self.model.set_classes(vocab, pe)
        self.device = device

    def __call__(self, bgr, dets):
        H, W = bgr.shape[:2]
        crops, owners = [], []
        for d in dets:
            if d["name"] not in self.of or not self.of[d["name"]][0]:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
            c = bgr[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            scale = float((self.scale or {}).get(d["name"], 0.1))
            if c.size == 0 or max(c.shape[:2]) * scale < MIN_PART_PX:
                d["_part"] = None                       # 못 잼 — 부분이 볼 수 있을 만큼 크게 찍히지 않는다
                continue
            k = PART_SIZE / max(c.shape[:2])
            crops.append(cv2.resize(c, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC))
            owners.append(d)
        if not crops:
            return
        res = self.model.predict(crops, conf=PART_FLOOR * 0.6, imgsz=PART_SIZE, device=self.device, verbose=False)
        for d, c, r in zip(owners, crops, res):
            names, where = self.of[d["name"]]
            best, word = 0.0, None
            for b in (r.boxes or []):
                w = self.vocab[int(b.cls)]
                if w not in names:
                    continue
                if where == "top" and float(b.xyxy[0][1] + b.xyxy[0][3]) / 2 / c.shape[0] > 0.6:
                    continue                            # 머리는 위쪽에 있다
                if float(b.conf) > best:
                    best, word = float(b.conf), w
            d["_part"] = (best, word)


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
            parts.append((c, ("%s %s %.2f" % (ARM_SHOWN.get(arm, arm), w, c)).strip()))
        elif similar:
            w, c = max(similar, key=lambda t: t[1])
            parts.append((c * SIMILAR_WEIGHT, ("%s ~%s %.2fx%.1f" % (ARM_SHOWN.get(arm, arm), w, c, SIMILAR_WEIGHT)).strip()))
    if not parts and not d.get("proposed_by"):     # OVD 가 낸 후보는 그 확신이 아래 OVD 칸에서 한 번만 들어간다
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
    # 같은 낱말은 한 번만 적는다('person 0.51 · 0.49') — 계산은 위에서 둘 다 셌다
    shown, last = [], None
    for _c, t in parts:
        word, _sp, num = t.rpartition(" ")
        shown.append(num if (word and word == last) else t)
        last = word or None
    return 100.0 * (1.0 - miss), " · ".join(shown), (max(best_ovd, key=best_ovd.get) if best_ovd else None)


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
def load_reference_embeddings(t, log=None):
    """그 대상에 올려 둔 기준 사진의 CLIP 임베딩 (N, D) 또는 None.

    자리: 대상 파일 옆 `<대상>.refs.npz`, 키 `<인코더>/<대상 이름>` (cue_catalog 가 그렇게 저장한다).
    없거나 깨졌으면 None 을 돌려주고 막지 않는다 — 못 잰 것은 통과가 아니라 `None`(not measured)로 남는다."""
    info = t.get("reference_images") or {}
    if info.get("state") != "ready":
        return None
    try:
        base = Path(t.get("_file") or ".").parent
        npz = base / str(info.get("embeddings") or ("%s.refs.npz" % t["name"]))
        if not npz.exists():
            return None
        key = "%s/%s" % (info.get("encoder") or "clip_b32", t["name"])
        with np.load(str(npz)) as z:
            if key not in z:
                return None
            arr = np.asarray(z[key], dtype="float32")
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        n = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(n == 0, 1.0, n)               # 이미 정규화돼 있지만 한 번 더 (깨진 파일 대비)
    except Exception as e:  # noqa: BLE001 — 기준 사진을 못 읽어도 판정은 돌아야 한다
        if log:
            log("기준 사진 임베딩을 못 읽음 (%s): %s" % (t.get("name"), e))
        return None


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
        # 기준 사진은 엔진 없이도 읽는다 (파일에서 읽는 것뿐이다). 후보 쪽 임베딩은 clip 이 있어야 생긴다
        self.refs = load_reference_embeddings(t) if "ref_match" in cues else None
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

    def __init__(self, targets, clip_files=None, log=print, parts_model=None):
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
        self.parts = None
        pt = [t for t in targets if "parts" in t["cues"] and (t.get("parts") or {}).get("names")]
        if pt and parts_model is not None:
            try:
                self.parts = _cached_probe(pt, parts_model[0], parts_model[1], parts_model[2])
            except Exception as exc:                  # 없으면 국소 축만 못 잼
                if log:
                    log("  targeted 부분(국소) 탐지를 못 세움 — %s: %s" % (type(exc).__name__, exc))

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
        if self.parts is not None:
            self.parts(bgr, [d for d in dets if d["name"] in self.prep])
        e_of = {i: emb[k] for k, i in enumerate(owners)} if emb is not None else {}
        for i, d in enumerate(dets):
            if d["name"] in self.prep:
                d["assess"] = self._one(d, e_of.get(i), (ovd or {}).get(d["name"]) or [], W, H)
            d.pop("_crop", None)
            d.pop("_part", None)
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
        # 기준 사진 — 사람이 올린 사진들과 얼마나 닮았나 (가장 닮은 몇 장의 평균)
        if "ref_match" in cues:
            if p.refs is None or emb is None:
                why = " (small crop)" if small else ("" if p.refs is not None else " (no reference photos)")
                add("ref_match", None, "not measured" + why)
            else:
                sims = p.refs @ emb
                k = int(min(REF_TOPK, sims.shape[0]))
                ts = float(np.sort(sims)[-k:].mean())
                best = float(sims.max())
                thr = float(t.get("ref_sim_min", REF_SIM_MIN))
                pct = 0.0 if ts < thr - REF_SPAN else min(100.0, 50.0 + 50.0 * (ts - thr) / REF_SPAN)
                # 문턱 비교에 아주 작은 여유를 둔다 — 임베딩이 float32 라 「딱 하한」이 0.6999994 로 나온다
                add("ref_match", max(0.0, pct),
                    "top%d %.4f (best %.4f, min %.2f, %d photos)" % (k, ts, best, thr, sims.shape[0]),
                    passed=ts >= thr - 1e-6)
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
            # OVD 가 낸 후보(proposed_by)는 짝을 YOLOE 상자(하한 아래 것 포함)에서 찾는다 — 자기 자신과 재면 늘 1 이다
            partners = d["_pair_with"] if d.get("proposed_by") else ovd_boxes
            best_iou = max([_iou(box, o["xyxy"]) for o in partners] + [0.0])
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

        # 부분 — 국소 축(점수에 섞지 않는다). 트랙 누적은 vision_infer.AxisDecider 가 한다
        part = None
        if "parts" in cues:
            pv = d.get("_part", "na")
            if pv == "na" or pv is None:
                add("parts", None, "not measured")
            else:
                part = pv[0]
                add("parts", 100.0 * min(1.0, part / 0.5), "%s %.2f" % (pv[1] or "-", part))
        # 기하 — 실물 크기(깊이로 잰 긴 변)가 선언 크기와 같은 자릿수인가(1/3 ~ 2 배). 축으로만 센다
        geometry = None
        real, mm = d.get("real_mm"), t.get("size_mm")
        if real and mm:
            ratio = float(real["long"]) / max(float(x) for x in mm)
            geometry = (1.0 / 3.0) <= ratio <= 2.0

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
        axes = {"global": bool(ok), "local": None if part is None else round(part, 3),
                "local_declared": "parts" in cues, "geometry": geometry}
        # 트랙 누적(vision_infer.TrackEvidence)이 쓰는 장 단위 값 — 근거마다 % · 역할 · 무게 · 통과
        cue_values = {r[0]: {"pct": r[1], "role": r[3], "passed": r[4], "w": float(w.get(r[0], 0.0))} for r in rows}
        return {"axes": axes, "cue_values": cue_values, "min_evidence": p.min_evidence,
                "lookalike_pen": round(pen, 2), "veto_text": veto,
                "scores": {name: None if shown is None else round(shown, 1)},
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


def suppress_duplicates(dets, iou=OVD_ASSOC_IOU, contain=OVD_ASSOC_CONTAIN):
    """**같은 대상의 겹친 · 포개진 상자는 한 물체다**(2026-11-23). 판정된 것끼리 점수 순으로 세우고, 앞서 남긴 상자와
    IoU ≥ 0.30 이거나 작은 쪽이 0.7 이상 안에 들면 지운다 — main 의 같은 물체 연계(ASSOCIATION_IOU_THR ·
    CONTAINMENT_THR)와 같은 기준이다. 예전에는 같은 이름끼리 IoU 0.5 이상만 합쳐서 한 사람에 상체 · 다리 · 전신
    상자가 따로 판정됐다(robot1 2026-11-23: 2~3 명 장면에서 판정 평균 4.85, 정확히 맞은 장 5%).
    판정 안 된 후보는 건드리지 않는다(회색으로 남는다) — 다만 판정된 상자 안에 포개진 것은 지운다."""
    def sc(d):
        return (d.get("assess") or {}).get("score") or 0.0

    def sus(d):
        return (d.get("assess") or {}).get("suspect")
    def area(d):
        b = d["xyxy"]
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    gone = set()
    sus_d = [d for d in dets if sus(d)]
    # ① 포개진 것은 **큰 쪽(전체)** 을 남긴다 — 상체만 잡은 상자보다 전신 상자가 거리 · 피부색 근거에 낫다
    for d in sus_d:
        if any(o is not d and o["name"] == d["name"] and area(o) > area(d) and _contain(o["xyxy"], d["xyxy"]) >= contain
               and id(o) not in gone for o in sorted(sus_d, key=area, reverse=True)):
            gone.add(id(d))
    # ② 나머지 겹침은 점수 순
    kept = []
    for d in sorted((d for d in sus_d if id(d) not in gone), key=lambda d: -sc(d)):
        if any(k["name"] == d["name"] and _iou(k["xyxy"], d["xyxy"]) >= iou for k in kept):
            gone.add(id(d))
        else:
            kept.append(d)
    for d in dets:
        if not sus(d) and any(k["name"] == d["name"] and _contain(k["xyxy"], d["xyxy"]) >= contain for k in kept):
            gone.add(id(d))
    return [d for d in dets if id(d) not in gone]


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


_PROBE = {}


def _cached_probe(targets, weight, enc_dir, device):
    """부분 탐지기는 프로세스에 하나 — 어휘(부분 이름 합)가 같으면 다시 세우지 않는다."""
    key = (str(weight), tuple(sorted({n for t in targets for n in (t.get("parts") or {}).get("names") or []})))
    if key not in _PROBE:
        _PROBE.clear()                                  # 어휘가 바뀌면 옛것을 버린다(GPU 메모리)
        _PROBE[key] = PartsProbe(targets, weight, enc_dir, device)
    probe = _PROBE[key]
    probe.of = {t["name"]: ([str(n) for n in (t.get("parts") or {}).get("names") or []],
                            (t.get("parts") or {}).get("where", "any")) for t in targets}
    probe.scale = {t["name"]: float((t.get("parts") or {}).get("scale", 0.1)) for t in targets}
    return probe
