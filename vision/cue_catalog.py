#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cue_catalog.py — 지정 클래스 탐지(targeted)의 **대상 · 근거 · 오버레이 선택지**를 한 곳에서 (2026-11-23).

implements: 탐지대상_런처_계획_261123.md §2 · §3 · §4

이 파일이 가진 것
  * 근거(단서) 정의 — 런처 이름 · 기본 역할 · 허용 역할 · 무게 · 필요한 모델 (`CUES`)
  * 대상 파일 읽기 — `vision/assets/targets/<이름>.json` (`load_targets`)
  * 사용 가능 판정 — **파일이 있는가 · 대상이 그 근거를 선언했는가** 만 본다. 모델은 적재하지 않는다
  * 프로필 검증 · 병합 — 기본값(yaml + 대상 파일) 위에 런처·명령줄 프로필을 항목 단위로 덮는다
  * 모델별 오버레이 가능표 (`overlay_options`)
  * 런처가 받는 카탈로그 JSON (`catalog`, `python cue_catalog.py --json`)

**torch 를 import 하지 않는다.** 런처(WSL 호출)와 서버가 가볍게 부를 수 있어야 한다.
틀린 지정은 조용히 무시하지 않는다 — `ProfileError` 로 이유와 함께 거절한다.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TARGET_DIR = HERE / "assets" / "targets"

#: 지정 클래스 탐지의 기본 대상 (2026-11-23 사용자 지정). 향후 대상 파일만 더하면 목록에 뜬다
DEFAULT_TARGETS = ["person", "door", "chair", "desk", "demo_door", "demo_box"]

#: 검출 모델 · 깊이 모델 (vision_infer.py 와 같은 이름)
DETECT_MODELS = ("yoloe", "river", "targeted")
DEPTH_MODELS = ("unidepth", "moge2_aerial")
MODELS = ("yoloe", "river", "targeted", "unidepth", "moge2_aerial")
#: 런처에 보이는 이름 — 괄호·설명 없이
MODEL_SHOWN = {"yoloe": "yoloe", "river": "river", "targeted": "지정 클래스 탐지",
               "unidepth": "unidepth", "moge2_aerial": "moge2_aerial"}


class ProfileError(ValueError):
    """프로필이 틀렸다. 메시지는 사람이 읽는 한 줄 — 런처 상태 줄에 그대로 나간다."""


# ---------------------------------------------------------------- 근거 정의
#: key: label(런처) · role(기본) · roles(허용) · weight(기본 무게, score 일 때) · needs(모델 키) ·
#:      requires(대상 파일 칸 — 하나라도 있으면 됨) · detector(쓸 수 있는 탐지기 종류)
#: role: score 점수 · gate 필수 · veto 거부 · cond 조건(거리 신뢰) · on 켜기만(후보 넓히기·감점)
CUES = {
    # 탐지기 — YOLOE 텍스트 · prompt-free(같은 이름, 중심 명사가 같은 비슷한 이름은 0.7 배) · OVD(YOLO-World ·
    # GroundingDINO)가 같은 자리를 이 대상으로 부른 확신을 noisy-OR 로 모은다(2026-11-23: 이름과 OVD 동의를 하나로)
    "class":        {"label": "탐지기", "role": "score", "roles": ("score",), "weight": 0.50,
                     "needs": [], "requires": [], "detector": ("yoloe_text", "trained")},
    "free_arm":     {"label": "후보 넓히기", "role": "on", "roles": ("on",), "weight": 0.0,
                     "needs": ["yoloe_free"], "requires": ["free_vocab"], "detector": ("yoloe_text",)},
    "semantic":     {"label": "글 설명", "role": "score", "roles": ("score", "gate"), "weight": 0.20,
                     "needs": ["clip"], "requires": [], "detector": ("yoloe_text", "trained")},
    "dict_features": {"label": "특징 사전", "role": "score", "roles": ("score", "gate"), "weight": 0.20,
                      "needs": ["clip"], "requires": ["features"], "detector": ("yoloe_text", "trained")},
    "shape":        {"label": "형상", "role": "score", "roles": ("score", "gate"), "weight": 0.10,
                     "needs": [], "requires": ["shape", "size_mm"], "detector": ("yoloe_text",)},
    "colour":       {"label": "색상", "role": "score", "roles": ("score", "gate"), "weight": 0.15,
                     "needs": [], "requires": ["colours"], "detector": ("yoloe_text", "trained")},
    "ovd_pair_iou": {"label": "OVD 짝 IoU", "role": "gate", "roles": ("gate",), "weight": 0.0,
                     "needs": ["ovd"], "requires": ["pair_iou_min"], "detector": ("yoloe_text",)},
    "clip_colour":  {"label": "CLIP 색", "role": "gate", "roles": ("gate", "score"), "weight": 0.10,
                     "needs": ["clip"], "requires": ["clip_colour"], "detector": ("yoloe_text", "trained")},
    "clip_shape":   {"label": "CLIP 모양", "role": "gate", "roles": ("gate", "score"), "weight": 0.10,
                     "needs": ["clip"], "requires": ["clip_shape"], "detector": ("yoloe_text", "trained")},
    "saturation":   {"label": "채도", "role": "gate", "roles": ("gate",), "weight": 0.0,
                     "needs": [], "requires": ["saturation"], "detector": ("yoloe_text", "trained")},
    # 부분 — **국소 축**(2026-11-23). 그 물체에만 있는 부분(사람: 머리 · 얼굴 · 손)이 트랙의 절반 이상 장에서
    # 잡히는가. 점수에 섞지 않고 축으로 따로 센다(전역 외형 근거를 여러 번 세지 않으려고)
    "parts":        {"label": "부분", "role": "local", "roles": ("local",), "weight": 0.0,
                     "needs": ["yoloe_text"], "requires": ["parts"], "detector": ("yoloe_text",)},
    "hue":          {"label": "색상 H", "role": "gate", "roles": ("gate",), "weight": 0.0,
                     "needs": [], "requires": ["hue"], "detector": ("yoloe_text", "trained")},
    "aspect":       {"label": "종횡비", "role": "gate", "roles": ("gate", "score"), "weight": 0.10,
                     "needs": [], "requires": ["aspect", "trained"], "detector": ("yoloe_text", "trained")},
    "ground":       {"label": "바닥 위치", "role": "gate", "roles": ("gate",), "weight": 0.0,
                     "needs": [], "requires": ["trained"], "detector": ("trained",)},
    "height":       {"label": "높이 측정", "role": "cond", "roles": ("cond",), "weight": 0.0,
                     "needs": [], "requires": ["trained"], "detector": ("trained",)},
    "calibration":  {"label": "거리 검증 범위", "role": "cond", "roles": ("cond",), "weight": 0.0,
                     "needs": [], "requires": ["trained"], "detector": ("trained",)},
    "size_veto":    {"label": "크기 거부", "role": "veto", "roles": ("veto",), "weight": 0.0,
                     "needs": ["depth"], "requires": ["size_mm"], "detector": ("yoloe_text", "trained")},
    "lookalike":    {"label": "닮은꼴 감점", "role": "on", "roles": ("on",), "weight": 0.0,
                     "needs": ["clip"], "requires": ["confusers"], "detector": ("yoloe_text", "trained")},
}
ROLES = ("score", "gate", "veto", "cond", "on", "local", "off")
#: 카드(영상 · 웹)에 쓰는 근거 이름 — **카드는 영어로만**(2026-11-23). 런처 화면은 위 label(한글)
CUE_EN = {"class": "detector", "free_arm": "wider candidates", "semantic": "description",
          "dict_features": "features", "shape": "shape", "colour": "colour", "ovd_pair_iou": "OVD pair IoU",
          "clip_colour": "CLIP colour", "clip_shape": "CLIP shape", "saturation": "saturation", "hue": "hue",
          "aspect": "aspect", "ground": "ground", "height": "height", "calibration": "calibration",
          "size_veto": "size veto", "lookalike": "look-alike", "distance": "distance",
          "parts": "parts", "axes": "axes"}
#: 직접 입력(등록하지 않은 이름)에서 쓰는 근거 — 크기·색을 주면 더 붙는다
DIRECT_ALWAYS = ["class", "free_arm", "semantic"]
DIRECT_WITH_SIZE = ["shape", "size_veto"]
DIRECT_WITH_COLOUR = ["colour"]
DIRECT_MIN_SCORE = 50.0

#: 색 이름 → HSV H 범위(OpenCV 0~179) · 최소 채도 · 최소 밝기. **선언이다 — 측정해 고른 값이 아니다.**
#: 흰·검·회색은 채도로 가른다(H 무관).
COLOURS = {
    "red": {"h": [[0, 8], [170, 179]], "s_min": 90, "v_min": 50},
    "orange": {"h": [[9, 20]], "s_min": 90, "v_min": 60},
    "yellow": {"h": [[21, 34]], "s_min": 80, "v_min": 80},
    "green": {"h": [[35, 85]], "s_min": 60, "v_min": 40},
    "light_blue": {"h": [[86, 105]], "s_min": 40, "v_min": 80},
    "blue": {"h": [[100, 130]], "s_min": 80, "v_min": 40},
    "purple": {"h": [[131, 160]], "s_min": 60, "v_min": 40},
    "brown": {"h": [[5, 25]], "s_min": 40, "s_max": 200, "v_min": 30, "v_max": 170},
    "white": {"s_max": 35, "v_min": 180},
    "gray": {"s_max": 35, "v_min": 60, "v_max": 180},
    "black": {"v_max": 60},
}


# ---------------------------------------------------------------- 모델 파일 (사용 가능 판정)
def _p(path):
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else (REPO / p)


def model_files(tcfg):
    """모델 키 → 있어야 하는 파일 목록. 경로는 yaml models.targeted.files 가 이긴다."""
    f = dict(DEFAULT_FILES)
    f.update((tcfg or {}).get("files") or {})
    out = {}
    for key, paths in f.items():
        out[key] = [_p(x) for x in (paths if isinstance(paths, list) else [paths])]
    return out


DEFAULT_FILES = {
    "yoloe_text": ["models/yoloe-11l-seg.pt", "models/mobileclip_blt.ts"],
    "yoloe_free": ["models/yoloe-11s-seg-pf.pt"],
    "clip": ["models/clip-vit-base-patch32/onnx/vision_model_quantized.onnx",
             "models/clip-vit-base-patch32/onnx/text_model_quantized.onnx",
             "models/clip-vit-base-patch32/onnx/tokenizer_dictionary.json"],
    "yolo_world": ["models/yolov8s-worldv2.pt", "models/clip/ViT-B-32.pt"],
    "gdino": ["models/grounding-dino-tiny/config.json", "models/grounding-dino-tiny/model.safetensors"],
    "depth": ["~/.cache/huggingface/hub/models--lpiccinelli--unidepth-v2-vitb14",
              "external/UniDepth"],
    "trained": [],                   # 대상 파일의 trained.weight · fit 이 정한다
    "reid": ["models/lightglue_superpoint/model.safetensors",
             "models/clip-vit-base-patch32/onnx/vision_model_quantized.onnx"],
}
#: 프로세스 하나에 더해지는 GPU 메모리 추정(MB). 측정값이 아니라 **분할 계산용 어림**이다
GPU_MB = {"base": 400, "yoloe_text": 900, "yoloe_free": 500, "clip": 350, "yolo_world": 500,
          "grounding_dino": 1300, "gdino": 1300, "reid": 700, "depth": 1300, "trained": 120}


def have(key, files):
    return all(p.exists() for p in files.get(key, []))


def gdino_mode(tcfg):
    v = ((tcfg or {}).get("ovd") or {}).get("grounding_dino", "auto")
    return "auto" if v == "auto" else bool(v)


def ovd_available(tcfg, files, targets=None):
    """OVD 동의에 쓸 탐지기. YOLO-World 는 있으면 쓴다. GroundingDINO 는 무거워서(프레임당 수백 ms)
    yaml `ovd.grounding_dino` 가 true 이거나, auto(기본)이면 **그 근거를 선언한 대상**(ovd_rules.grounding_dino,
    지금은 demo_door — main 이 그 문을 GDINO 근거로 확정했다)을 고를 때만 쓴다. targets=None 이면 '쓸 수 있는가' 만 본다."""
    out = []
    if have("yolo_world", files):
        out.append("yolo_world")
    mode = gdino_mode(tcfg)
    if mode is not False and have("gdino", files):
        if mode is True or targets is None or any("grounding_dino" in (t.get("ovd_rules") or {}) for t in targets):
            out.append("grounding_dino")
    return out


# ---------------------------------------------------------------- 대상
def load_targets(target_dir=TARGET_DIR):
    """이름 → 대상 선언. 파일 이름과 name 이 다르면 거절한다(어느 쪽이 맞는지 모른다)."""
    out = {}
    for f in sorted(Path(target_dir).glob("*.json")):
        try:
            t = json.loads(f.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ProfileError("대상 파일을 못 읽음: %s (%s)" % (f.name, exc))
        if t.get("name") != f.stem:
            raise ProfileError("대상 파일 이름과 name 이 다름: %s → %r" % (f.name, t.get("name")))
        t["_file"] = str(f)
        out[f.stem] = t
    return out


_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _\-]{0,40}$")


def parse_target_arg(text):
    """명령줄 `--target` 한 개 → 프로필 대상 사전.

    `door` · `traffic cone` · `traffic cone:size=300x300x700:colour=orange` · `cone:colour=orange+white`
    """
    parts = [p.strip() for p in str(text).split(":")]
    name = parts[0]
    if not _NAME_RE.match(name or ""):
        raise ProfileError("대상 이름은 영문으로 시작하는 41자 이하 (글자·숫자·공백·_·-): %r" % name)
    out = {"name": name}
    for p in parts[1:]:
        if not p:
            continue
        if "=" not in p:
            raise ProfileError("대상 %s 의 %r — 'size=가로x세로x높이' 또는 'colour=이름' 으로 적는다" % (name, p))
        k, v = (s.strip() for s in p.split("=", 1))
        if k == "size":
            try:
                mm = [float(x) for x in v.lower().split("x")]
            except ValueError:
                raise ProfileError("대상 %s 의 크기는 mm 세 숫자: %r" % (name, v))
            if len(mm) != 3 or min(mm) <= 0:
                raise ProfileError("대상 %s 의 크기는 mm 세 숫자(가로x세로x높이): %r" % (name, v))
            out["size_mm"] = mm
        elif k in ("colour", "color"):
            out["colours"] = [c.strip() for c in v.replace(",", "+").split("+") if c.strip()]
        else:
            raise ProfileError("대상 %s 에 모르는 항목 %r (가능: size, colour)" % (name, k))
    return out


def parse_cues_arg(text):
    """`--cues default` · `default,aspect=off,colour` · `class,semantic,aspect=gate` → 프로필 cues."""
    items = [s.strip() for s in str(text).split(",") if s.strip()]
    if not items or items == ["default"]:
        return "default"
    out = {}
    for it in items:
        if it == "default":
            out["default"] = True
            continue
        k, _, v = it.partition("=")
        out[k.strip()] = v.strip() or True
    return out


def _direct_target(name, spec):
    """대상 파일 없는 이름 — 제로샷 기본값을 채운다(§3.4.8)."""
    t = {"name": name, "shown": name, "detector": "yoloe_text", "direct": True,
         "prompts": [name], "free_vocab": "auto",
         "descriptions": ["a photo of a %s" % name, "a %s in the scene" % name],
         "negatives": ["a photo of an object", "a wall", "the floor", "a person"],
         "ovd_labels": [name], "min_score": DIRECT_MIN_SCORE, "provisional": True,
         "source": "직접 입력 (제로샷 자동 값)"}
    if name == "person" or "person" in name:
        t["negatives"] = [n for n in t["negatives"] if n != "a person"]
    if spec.get("size_mm"):
        t["size_mm"] = list(spec["size_mm"])
        a, b, c = sorted(spec["size_mm"])           # 짧은 → 긴
        lo, hi = max(1.0, b / c if c else 1.0), max(1.0, c / a if a else 1.0)
        t["shape"] = {"aspect": [round(max(1.0, lo / 1.5), 2), round(hi * 1.5, 2)]}
    if spec.get("colours"):
        t["colours"] = list(spec["colours"])
    cues = {k: CUES[k]["role"] for k in DIRECT_ALWAYS}
    if spec.get("size_mm"):
        cues.update({k: CUES[k]["role"] for k in DIRECT_WITH_SIZE})
    if spec.get("colours"):
        cues.update({k: CUES[k]["role"] for k in DIRECT_WITH_COLOUR})
    t["cues"] = cues
    return t


def available_cues(t, tcfg, files, distance=True):
    """대상 하나에서 **쓸 수 있는** 근거 → (기본 역할, 쓸 수 없으면 None 과 이유)."""
    det = t.get("detector", "yoloe_text")
    out = {}
    for key, c in CUES.items():
        why = None
        if det not in c["detector"]:
            why = "탐지기와 맞지 않음"
        elif c["requires"] and not any(t.get(r) for r in c["requires"]):
            why = "대상에 선언 없음"
        else:
            for need in c["needs"]:
                if need == "ovd":
                    if not ovd_available(tcfg, files, [t]):
                        why = "OVD 가중치 없음"
                elif need == "depth":
                    if not distance:
                        why = "거리 측정 꺼짐"
                    elif not have("depth", files):
                        why = "UniDepth 없음"
                elif not have(need, files):
                    why = "%s 없음" % need
        if key == "class" and det == "trained" and not _trained_ok(t):
            why = "학습 가중치 없음"
        if key == "class" and det == "yoloe_text" and not have("yoloe_text", files):
            why = "YOLOE 텍스트 가중치 없음"
        out[key] = why
    return out


def _trained_ok(t):
    tr = t.get("trained") or {}
    return bool(tr.get("weight")) and _p(tr["weight"]).exists() and bool(tr.get("fit")) and _p(tr["fit"]).exists()


def resolve_cues(t, spec_cues, avail):
    """대상의 기본 근거(대상 파일 cues) 위에 프로필 cues 를 덮는다 → {근거: 역할}. 쓸 수 없는 근거는 빠진다."""
    base = dict(t.get("cues") or {})
    if spec_cues in (None, "default"):
        chosen = base
    elif isinstance(spec_cues, list):
        chosen = {k: CUES[k]["role"] if k in CUES else None for k in spec_cues}
    elif isinstance(spec_cues, dict):
        chosen = dict(base) if spec_cues.get("default") else {}
        for k, v in spec_cues.items():
            if k == "default":
                continue
            if v is True:
                v = (base.get(k) or CUES.get(k, {}).get("role"))
            chosen[k] = v
    else:
        raise ProfileError("대상 %s 의 cues 는 default · 목록 · 사전: %r" % (t["name"], spec_cues))
    # 사용자가 **직접 고른** 근거 — 쓸 수 없으면 거절한다. 기본값에서 온 것은 조용히 빠진다
    # (예: 이 PC 에 OVD 가중치가 없으면 문의 기본 근거에서 OVD 가 빠진다)
    if isinstance(spec_cues, list):
        explicit = set(spec_cues)
    elif isinstance(spec_cues, dict):
        explicit = {k for k, v in spec_cues.items() if k != "default" and v not in (None, False, "off")}
    else:
        explicit = set()
    out = {}
    for k, role in chosen.items():
        if k not in CUES:
            raise ProfileError("대상 %s 에 모르는 근거 %r (가능: %s)" % (t["name"], k, ", ".join(CUES)))
        if role in (None, False, "off"):
            continue
        if role not in CUES[k]["roles"]:
            raise ProfileError("대상 %s 의 근거 %s 에 쓸 수 없는 역할 %r (가능: %s)"
                               % (t["name"], k, role, ", ".join(CUES[k]["roles"])))
        if avail.get(k):
            if k in explicit:
                raise ProfileError("대상 %s 의 근거 %s 를 쓸 수 없음: %s" % (t["name"], k, avail[k]))
            continue
        out[k] = role
    if "class" not in out:
        raise ProfileError("대상 %s 에는 탐지기 이름(class) 근거가 있어야 한다 — 후보를 내는 것이 그것이다"
                           % t["name"])
    return out


# ---------------------------------------------------------------- 프로필
#: 기능 선택(모델과 따로) — 런처 '객체 재식별' 등
FEATURES = {"reid": "객체 재식별"}
OVERLAY_KEYS = ("label", "cards", "count", "roi", "show_non_suspect", "show_rejected", "title", "pose")
CARD_MODES = ("off", "hover", "always")
LABEL_TOKENS = ("name", "lead", "conf", "cls", "id", "depth", "range", "dist", "gps", "alt", "ned")


def overlay_options(models, targeted_distance=True, targets=()):
    """모델 → {항목: 쓸 수 있는가(True) 또는 못 쓰는 이유(글)}. 런처와 검증이 함께 쓴다(§4)."""
    models = list(models)
    depth_on = any(m in DEPTH_MODELS for m in models)
    out = {}
    for m in models:
        o = {}
        det = m in DETECT_MODELS
        o["name"] = True
        o["conf"] = det or "검출 모델이 아님"
        o["id"] = det or "검출 모델이 아님"
        if m == "yoloe":
            o["dist"] = True if depth_on else "깊이 모델을 함께 고를 때"
            o["gps"] = o["dist"]
        elif m == "targeted":
            o["dist"] = True if targeted_distance else "거리 측정 꺼짐"
            o["gps"] = o["dist"]
        else:
            o["dist"] = True
            o["gps"] = True
        o["count"] = det or "검출 모델이 아님"
        o["roi"] = det or "검출 모델이 아님"           # 탐지 영역 — 웹 화면에서 끌어 지정한 영역을 쓰고 그린다
        o["cards"] = True if m in ("river", "targeted") else "판정 카드가 없는 모델"
        o["show_non_suspect"] = o["cards"]
        o["show_rejected"] = (True if (m == "targeted" and "demo_box" in targets)
                              else "demo_box 를 고를 때")
        o["title"] = True
        o["pose"] = True if m in DEPTH_MODELS else "깊이 모델 전용"
        out[m] = o
    return out


def _label_tokens(value, model):
    lines = value if (value and isinstance(value[0], (list, tuple))) else [value or []]
    for line in lines:
        bad = [t for t in line if t not in LABEL_TOKENS]
        if bad:
            raise ProfileError("오버레이 %s 라벨에 모르는 항목 %s (가능: %s)" % (model, bad, ", ".join(LABEL_TOKENS)))
    return [list(l) for l in lines]


def check_overlay(model, ov, opts):
    if not isinstance(ov, dict):
        raise ProfileError("오버레이 %s 는 사전이어야 한다" % model)
    out = {}
    for k, v in ov.items():
        if k not in OVERLAY_KEYS:
            raise ProfileError("오버레이 %s 에 모르는 항목 %r (가능: %s)" % (model, k, ", ".join(OVERLAY_KEYS)))
        if k == "label":
            out[k] = _label_tokens(v, model)
            for line in out[k]:
                for tok in line:
                    key = {"dist": "dist", "depth": "dist", "range": "dist", "gps": "gps", "alt": "gps",
                           "ned": "gps", "conf": "conf", "id": "id"}.get(tok)
                    if key and opts.get(key) is not True:
                        raise ProfileError("오버레이 %s 라벨 %s 를 쓸 수 없음: %s" % (model, tok, opts.get(key)))
            continue
        if k == "cards":
            if v not in CARD_MODES:
                raise ProfileError("오버레이 %s cards 는 %s 중 하나: %r" % (model, ", ".join(CARD_MODES), v))
        elif not isinstance(v, bool):
            raise ProfileError("오버레이 %s %s 는 참/거짓: %r" % (model, k, v))
        if opts.get(k) is not True and v not in (False, "off"):
            raise ProfileError("오버레이 %s %s 를 쓸 수 없음: %s" % (model, k, opts.get(k)))
        out[k] = v
    return out


def resolve_profile(profile, cfg, models, targets_db=None):
    """프로필 + yaml → 적용할 설정 사전.

    돌려주는 것: {"version", "label", "targeted": {"targets": [대상(근거 역할 확정)], "distance"},
                 "overlay": {모델: {...}}, "devices": {...}}
    targeted 를 고르지 않았으면 targeted 칸은 None. 틀리면 ProfileError.
    """
    profile = copy.deepcopy(profile or {})
    if not isinstance(profile, dict):
        raise ProfileError("프로필은 JSON 사전이어야 한다")
    bad = set(profile) - {"profile_version", "label", "models", "devices", "features", "_doc", "saved_at"}
    if bad:
        raise ProfileError("프로필에 모르는 칸: %s" % ", ".join(sorted(bad)))
    pm = profile.get("models") or {}
    if not isinstance(pm, dict):
        raise ProfileError("프로필 models 는 모델 → 설정 사전")
    for m, part in pm.items():
        if m not in MODELS:
            raise ProfileError("프로필에 모르는 모델 %r (가능: %s)" % (m, ", ".join(MODELS)))
        if not isinstance(part, dict):
            raise ProfileError("프로필 models.%s 는 사전" % m)
        extra = set(part) - ({"targets", "distance", "overlay"} if m == "targeted" else {"overlay"})
        if extra:
            raise ProfileError("모델 %s 에는 %s 를 줄 수 없음 (대상·근거·거리는 지정 클래스 탐지에만)"
                               % (m, ", ".join(sorted(extra))))
    mcfg = (cfg or {}).get("models") or {}
    tcfg = mcfg.get("targeted") or {}
    files = model_files(tcfg)
    targets_db = load_targets() if targets_db is None else targets_db
    result = {"version": profile.get("profile_version"), "label": profile.get("label"),
              "targeted": None, "overlay": {}, "devices": {}, "features": {}}
    feats = profile.get("features") or {}
    if not isinstance(feats, dict) or set(feats) - set(FEATURES):
        raise ProfileError("features 에 모르는 기능: %s (가능: %s)" % (", ".join(sorted(set(feats) - set(FEATURES))),
                                                            ", ".join(FEATURES)))
    for k, v in feats.items():
        if not isinstance(v, bool):
            raise ProfileError("features.%s 는 참/거짓" % k)
        if k == "reid" and v and not have("reid", model_files(((cfg or {}).get("models") or {}).get("targeted") or {})):
            raise ProfileError("객체 재식별에 필요한 가중치가 없음 (models/lightglue_superpoint · CLIP)")
    result["features"] = dict(feats)
    tpart = pm.get("targeted") or {}
    distance = bool(tpart.get("distance", tcfg.get("distance", True)))
    if "targeted" in models:
        specs = tpart.get("targets")
        if specs is None:
            specs = [{"name": n} for n in (tcfg.get("targets") or DEFAULT_TARGETS)]
        if not isinstance(specs, list) or not specs:
            raise ProfileError("지정 클래스 탐지에는 대상이 하나 이상 필요합니다")
        seen, out = set(), []
        for s in specs:
            s = {"name": s} if isinstance(s, str) else dict(s)
            name = str(s.get("name") or "").strip()
            if not _NAME_RE.match(name):
                raise ProfileError("대상 이름이 틀림: %r" % name)
            if name in seen:
                raise ProfileError("대상 %s 가 두 번 있음" % name)
            seen.add(name)
            extra = set(s) - {"name", "cues", "size_mm", "colours", "min_score", "distance"}
            if extra:
                raise ProfileError("대상 %s 에 모르는 칸: %s" % (name, ", ".join(sorted(extra))))
            for c in s.get("colours") or []:
                if c not in COLOURS:
                    raise ProfileError("대상 %s 의 색 %r 을 모름 (가능: %s)" % (name, c, ", ".join(COLOURS)))
            if name in targets_db:
                t = copy.deepcopy(targets_db[name])
                if s.get("size_mm"):
                    t["size_mm"] = list(s["size_mm"])
                if s.get("colours"):
                    t["colours"] = list(s["colours"])
            else:
                t = _direct_target(name, s)
            if s.get("min_score") is not None:
                t["min_score"] = float(s["min_score"])
            t_dist = distance and bool(s.get("distance", True))
            avail = available_cues(t, tcfg, files, t_dist)
            t["cues"] = resolve_cues(t, s.get("cues"), avail)
            t["distance"] = t_dist
            t["unavailable"] = {k: v for k, v in avail.items() if v}
            out.append(t)
        result["targeted"] = {"targets": out, "distance": distance}
    names = [t["name"] for t in (result["targeted"] or {}).get("targets", [])]
    opts = overlay_options(models, distance, names)
    for m, part in pm.items():
        if "overlay" in part:
            if m not in models:
                continue                      # 이 프로세스가 돌리지 않는 모델 — 다른 프로세스 몫
            result["overlay"][m] = check_overlay(m, part["overlay"], opts[m])
    devs = profile.get("devices") or {}
    if not isinstance(devs, dict):
        raise ProfileError("프로필 devices 는 기기 → 설정 사전")
    for dv, part in devs.items():
        dm = (part or {}).get("models") or {}
        for m, mp in dm.items():
            if m not in MODELS:
                raise ProfileError("devices.%s 에 모르는 모델 %r" % (dv, m))
            if set(mp) - {"overlay"}:
                raise ProfileError("devices.%s.%s 에는 overlay 만 줄 수 있음" % (dv, m))
            if m in models:
                result["devices"].setdefault(dv, {})[m] = check_overlay(m, mp["overlay"], opts[m])
    return result


def needed_models(resolved, tcfg):
    """확정된 targeted 설정이 적재해야 할 모델 키 집합."""
    files = model_files(tcfg)
    need = set()
    for t in (resolved or {}).get("targets", []):
        if t.get("detector") == "trained":
            need.add("trained")
        else:
            need.add("yoloe_text")
            need.update(ovd_available(tcfg, files, [t]))     # 탐지기 근거에 OVD 확신이 들어간다
        for k in t["cues"]:
            for n in CUES[k]["needs"]:
                if n == "ovd":
                    need.update(ovd_available(tcfg, files, [t]))
                else:
                    need.add(n)
    return need


def gpu_mb(models, need=()):
    """비전 프로세스 하나의 GPU 메모리 어림(MB)."""
    mb = GPU_MB["base"]
    for m in models:
        if m == "yoloe":
            mb += 900
        elif m == "river":
            mb += 2400
        elif m in DEPTH_MODELS:
            mb += 1500
    for n in need:
        mb += GPU_MB.get(n, 0)
    return mb


# ---------------------------------------------------------------- 런처용 카탈로그
def catalog(cfg=None):
    """런처가 화면을 그리는 데 쓰는 것 전부. 쓸 수 없는 근거는 넣지 않는다(화면에 안 보인다)."""
    tcfg = ((cfg or {}).get("models") or {}).get("targeted") or {}
    files = model_files(tcfg)
    db = load_targets()
    order = list(tcfg.get("targets") or DEFAULT_TARGETS)
    order += [n for n in sorted(db) if n not in order]
    targets = []
    for n in order:
        if n not in db:
            continue
        t = db[n]
        avail = available_cues(t, tcfg, files, True)
        base = t.get("cues") or {}
        cues = [{"key": k, "label": CUES[k]["label"], "default": k in base and base[k] != "off",
                 "needs_distance": "depth" in CUES[k]["needs"]}
                for k in CUES if not avail.get(k)]
        targets.append({"name": n, "shown": t.get("shown", n), "default": n in (tcfg.get("targets") or DEFAULT_TARGETS),
                        "cues": cues})
    direct = {}
    for kind, keys in (("always", DIRECT_ALWAYS), ("with_size", DIRECT_WITH_SIZE), ("with_colour", DIRECT_WITH_COLOUR)):
        probe = _direct_target("x", {"size_mm": [1, 1, 1], "colours": ["red"]})
        avail = available_cues(probe, tcfg, files, True)
        direct[kind] = [{"key": k, "label": CUES[k]["label"], "needs_distance": "depth" in CUES[k]["needs"]}
                        for k in keys if not avail.get(k)]
    all_models = list(MODELS)
    return {"models": [{"key": m, "shown": MODEL_SHOWN[m]} for m in all_models],
            "overlay_defaults": overlay_defaults(cfg),
            "targets": targets, "direct": direct, "colours": list(COLOURS),
            "distance_default": bool(tcfg.get("distance", True)),
            "distance_available": have("depth", files),
            "overlay": overlay_options(all_models, True, [t["name"] for t in targets]),
            "overlay_labels": {"name": "클래스명", "conf": "신뢰도", "id": "트랙 번호", "dist": "거리",
                               "gps": "GPS", "count": "탐지 개수", "roi": "탐지 영역", "cards": "카드",
                               "show_non_suspect": "판정 아닌 검출", "show_rejected": "관문에 떨어진 검출",
                               "title": "제목 띠", "pose": "드론 위치 띠"},
            "card_modes": [{"key": "off", "shown": "끔"}, {"key": "hover", "shown": "마우스 올릴 때"},
                           {"key": "always", "shown": "항상"}],
            "features": [{"key": k, "shown": v, "available": have(k, files)} for k, v in FEATURES.items()],
            "gpu_mb": {m: gpu_mb([m]) for m in all_models if m != "targeted"},
            "gpu_mb_targeted": {k: v for k, v in GPU_MB.items()}}


def overlay_defaults(cfg):
    """yaml overlay 에서 모델별 기본값(런처가 처음 체크 상태로 쓴다). 라벨은 한 줄로 편다."""
    o = (cfg or {}).get("overlay") or {}
    out = {}
    for m in MODELS:
        kind = "depth" if m in DEPTH_MODELS else m
        blk = o.get(kind) if isinstance(o.get(kind), dict) else {}
        lab = blk.get("label")
        if lab is None and m in DETECT_MODELS:
            lab = (o.get("yoloe") or {}).get("label")
        if lab is None:
            lab = [["name", "conf"]] if m in DETECT_MODELS else [["name", "conf", "range"], ["gps"]]
        flat = [t for line in (lab if lab and isinstance(lab[0], list) else [lab]) for t in line]
        cards = blk.get("cards")
        out[m] = {"label": flat,
                  "cards": ("always" if cards is True or cards == "always" else "off" if cards == "off"
                            else "hover") if m in ("river", "targeted") else None,
                  "count": bool(blk.get("count", False)),
                  "roi": bool(blk.get("roi", True)) if m in DETECT_MODELS else None,
                  "show_non_suspect": bool(blk.get("show_non_suspect", True)),
                  "show_rejected": bool(blk.get("show_rejected", False)),
                  "title": bool(o.get("title", True)),
                  "pose": bool(blk.get("pose", True)) if m in DEPTH_MODELS else None}
    return out


def save_direct_target(name, size_mm=None, colours=None, target_dir=TARGET_DIR):
    """직접 입력 대상을 **등록 대상**으로 저장한다(런처 '등록으로 저장'). 있는 파일은 덮지 않는다."""
    spec = {"name": name}
    if size_mm:
        spec["size_mm"] = list(size_mm)
    if colours:
        for c in colours:
            if c not in COLOURS:
                raise ProfileError("색 %r 을 모름 (가능: %s)" % (c, ", ".join(COLOURS)))
        spec["colours"] = list(colours)
    parse_target_arg(name)                          # 이름 검사
    f = Path(target_dir) / ("%s.json" % name)
    if f.exists():
        raise ProfileError("이미 등록된 대상: %s" % name)
    t = _direct_target(name, spec)
    t.pop("direct", None)
    t["_doc"] = "직접 입력에서 등록(제로샷 자동 값). 특징 사전 · 게이트를 적어 다듬는다"
    rgb = [int(x) for x in __import__("hashlib").md5(name.encode()).digest()[:3]]
    t["colour_rgb"] = [max(80, v) for v in rgb]
    f.write_text(json.dumps(t, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(f)


def _load_yaml(path):
    try:
        import yaml
    except ImportError:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv=None):
    ap = argparse.ArgumentParser(description="지정 클래스 탐지 카탈로그 · 프로필 검사")
    ap.add_argument("--config", default=str(HERE / "vision_config.yaml"))
    ap.add_argument("--json", action="store_true", help="런처용 카탈로그 JSON")
    ap.add_argument("--check", metavar="프로필", help="프로필 파일을 검사하고 확정된 근거를 보여 준다")
    ap.add_argument("--models", nargs="+", default=["targeted"])
    ap.add_argument("--save-target", metavar="이름", help="직접 입력 대상을 등록 대상으로 저장")
    ap.add_argument("--size", metavar="가로x세로x높이", help="--save-target 의 크기 mm")
    ap.add_argument("--colours", metavar="색", help="--save-target 의 색 (+ 로 여럿)")
    a = ap.parse_args(argv)
    cfg = _load_yaml(a.config)
    if a.save_target:
        try:
            spec = parse_target_arg(a.save_target + (":size=" + a.size if a.size else "")
                                    + (":colour=" + a.colours if a.colours else ""))
            path = save_direct_target(spec["name"], spec.get("size_mm"), spec.get("colours"))
        except ProfileError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 1
        print(json.dumps({"ok": True, "file": path}, ensure_ascii=False))
        return 0
    if a.check:
        try:
            prof = json.loads(Path(a.check).read_text(encoding="utf-8-sig"))
            r = resolve_profile(prof, cfg, a.models)
        except (ProfileError, ValueError, OSError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 1
        brief = {"ok": True, "targets": {t["name"]: t["cues"] for t in (r["targeted"] or {}).get("targets", [])},
                 "overlay": r["overlay"]}
        print(json.dumps(brief, ensure_ascii=False))
        return 0
    print(json.dumps(catalog(cfg), ensure_ascii=False, indent=None if a.json else 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
