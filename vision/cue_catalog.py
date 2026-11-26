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
  * **사람이 직접 넣는 단서 두 가지** — 글(프롬프트)과 기준 사진 (아래)

**torch 를 import 하지 않는다.** 런처(WSL 호출)와 서버가 가볍게 부를 수 있어야 한다.
틀린 지정은 조용히 무시하지 않는다 — `ProfileError` 로 이유와 함께 거절한다.

사람이 넣는 단서 ① 글(프롬프트) — `semantic`(글 설명 · CLIP) 근거
  대상 칸 `user_descriptions`(글 목록). 프로필 · 명령줄에서는 `prompt` 로 준다.
  적으면 대상의 자동/선언 문장(`descriptions`)을 **대신한다** — 두 벌을 섞으면 확률이 희석돼
  어느 글이 점수를 끌었는지 알 수 없다. 대신한 것은 `_descriptions_default` 에 남는다.
  글을 주면 `semantic` 근거가 켜진다(기본 역할 score). 검사는 `check_prompts` — 빈 글 ·
  너무 긴 글 · 영문 밖의 글자를 사람이 읽는 오류로 거절한다(`PROMPT_GUIDE` 가 그 이유를 적는다).

사람이 넣는 단서 ② 기준 사진(zip · 이미지 파일) — `ref_match`(기준 사진) 근거
  `attach_reference_images(이름, zips=…, images=…)` · `save_direct_target(..., ref_zips=…)`.
  두 가지를 **대상 파일 옆에** 둔다 (`vision/assets/targets/`):
    `<대상>.refs/01.png …`  넣은 사진 (PNG, 긴 변 REF_STORE_PX 로 줄여 보관).
                            가중치가 뒤에 생겨도 다시 만들 수 있게 **사진을 남긴다**
    `<대상>.refs.npz`       CLIP 이미지 임베딩. 키 `clip_b32/<대상>` → (N, 512) float32,
                            L2 정규화. 줄 순서는 대상 파일 `reference_images.files` 와 같다
                            (butane_ref/prototypes.npz 와 같은 모양 — 그 규칙을 따른다)
  대상 파일에는 `reference_images: {dir, files, embeddings, encoder, state, why, made_at}`.
    state `ready` 임베딩 있음 · `pending` 사진은 있고 가중치가 없어 못 만들었다(`why` 에 어느 파일인지).
    **칸이 아예 없으면 사용자가 안 넣은 것이다** — pending(못 만든 것)과 다르다.
  임베딩 전처리는 **런타임과 같아야 한다**(`target_assess._clip_preprocess`, 224 가운데 자르기).
  hazard_verify 의 늘려 맞추기로 만들면 코사인이 다른 자가 된다.
  만들 때 필요한 것은 **비전 인코더 하나**(vision_model_quantized.onnx) — 글 인코더 · 토크나이저는
  필요 없다. 그래서 토크나이저가 없는 기계에서도 기준 사진은 지금 만들어 둘 수 있다.
  판정에서 쓰는 식은 아직 `target_assess` 에 없다 — model_registry.yaml `cues.ref_match` 주석 참고.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:                           # ds_resources.settings (경로 · 설정, 표준 라이브러리만)
    sys.path.append(str(REPO))
from ds_resources import settings  # noqa: E402
TARGET_DIR = HERE / "assets" / "targets"

#: 지정 클래스 탐지의 기본 대상 (2026-11-23 사용자 지정). 향후 대상 파일만 더하면 목록에 뜬다
DEFAULT_TARGETS = ["person", "door", "chair", "desk", "demo_door", "demo_box"]

#: 선택지 · 단서 · 모델 정의는 **vision/model_registry.yaml 한 파일**에 있다 (2026-11-24). 아래 이름들은 그 파일에서 만든다
REGISTRY = HERE / "model_registry.yaml"


def load_registry(path=REGISTRY):
    """model_registry.yaml → 사전. PyYAML 이 없거나 파일이 틀리면 바로 알린다(이 파일 없이는 선택지가 없다)."""
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML 이 필요합니다 (pip install -r requirements/core.txt): %s" % exc)
    with open(path, encoding="utf-8") as f:
        reg = yaml.safe_load(f) or {}
    for k in ("options", "cues", "models"):
        if not isinstance(reg.get(k), dict) or not reg[k]:
            raise SystemExit("%s 에 %s 구역이 없습니다" % (path, k))
    for name, c in reg["cues"].items():
        for m in c.get("needs") or []:
            if m not in reg["models"] and m not in ("ovd", "depth"):
                raise SystemExit("%s: 단서 %s 의 needs 에 모르는 모델 %r" % (path, name, m))
    for name, o in reg["options"].items():
        if isinstance(o.get("cue_models"), dict):
            for cue, ms in o["cue_models"].items():
                for m in ms or []:
                    if m not in reg["models"]:
                        raise SystemExit("%s: 선택지 %s 단서 %s 에 모르는 모델 %r" % (path, name, cue, m))
    for need, ch in (reg.get("cue_model_choices") or {}).items():
        for m in (ch or {}).get("choices") or []:
            if m not in reg["models"]:
                raise SystemExit("%s: cue_model_choices.%s 에 모르는 모델 %r" % (path, need, m))
    return reg


_REG = load_registry()
#: 근거 모델 중 바꿀 수 있는 것 — {need: {"choices": (모델 키…), "multi": bool, "model_id": {모델: 값}}}
CUE_CHOICES = {str(k): {"choices": tuple(str(m) for m in (v or {}).get("choices") or ()),
                        "multi": bool((v or {}).get("multi")),
                        "model_id": dict((v or {}).get("model_id") or {})}
               for k, v in (_REG.get("cue_model_choices") or {}).items()}
#: 검출 모델 · 깊이 모델 (vision_infer.py 와 같은 이름)
MODELS = tuple(_REG["options"])
DETECT_MODELS = tuple(m for m, o in _REG["options"].items() if o.get("kind") == "detect")
DEPTH_MODELS = tuple(m for m, o in _REG["options"].items() if o.get("kind") == "depth")
#: 런처에 보이는 이름 — 괄호·설명 없이
MODEL_SHOWN = {m: str(o.get("shown", m)) for m, o in _REG["options"].items()}
#: 선택지가 프로세스에 더하는 GPU 어림값 (targeted 는 단서 모델 합으로 따로 센다)
OPTION_GPU_MB = {m: int(o.get("gpu_mb") or 0) for m, o in _REG["options"].items()}


class ProfileError(ValueError):
    """프로필이 틀렸다. 메시지는 사람이 읽는 한 줄 — 런처 상태 줄에 그대로 나간다."""


# ---------------------------------------------------------------- 근거 정의 (model_registry.yaml cues)
#: key: label(런처) · role(기본) · roles(허용) · weight(기본 무게, score 일 때) · needs(모델 키) ·
#:      requires(대상 파일 칸 — 하나라도 있으면 됨) · detector(쓸 수 있는 탐지기 종류)
#: role: score 점수 · gate 필수 · veto 거부 · cond 조건(거리 신뢰) · on 켜기만(후보 넓히기·감점)
CUES = {k: {"label": str(c["label"]), "role": str(c["role"]), "roles": tuple(str(r) for r in c["roles"]),
            "weight": float(c.get("weight") or 0.0), "needs": list(c.get("needs") or []),
            "requires": list(c.get("requires") or []), "detector": tuple(c.get("detector") or ())}
        for k, c in _REG["cues"].items()}
ROLES = ("score", "gate", "veto", "cond", "on", "local", "off")
#: 카드(영상 · 웹)에 쓰는 근거 이름 — **카드는 영어로만**(2026-11-23). 런처 화면은 위 label(한글)
CUE_EN = {k: str(c.get("label_en", k)) for k, c in _REG["cues"].items()}
CUE_EN.update(_REG.get("card_rows_en") or {})
#: 직접 입력(등록하지 않은 이름)에서 쓰는 근거 — 크기·색을 주면 더 붙는다
DIRECT_ALWAYS = ["class", "free_arm", "semantic"]
DIRECT_WITH_SIZE = ["shape", "size_veto"]
DIRECT_WITH_COLOUR = ["colour"]
DIRECT_MIN_SCORE = 50.0

# ---------------------------------------------------------------- 사람이 넣는 단서 ① 글(프롬프트)
#: 글 하나의 길이 상한. CLIP 은 **77 낱말에서 자른다** — 120자쯤이면 영어 명사구 하나가 그 안에 든다.
#: 넘는 글은 받아 놓고 뒤를 조용히 버리는 것보다 거절하는 것이 낫다(무엇으로 쟀는지 알 수 없게 된다)
PROMPT_MAX_CHARS = 120
PROMPT_MIN_CHARS = 3
#: 글 개수 상한. 글마다 문장 임베딩 하나 — 대상 하나에 여덟이면 가르는 데 충분하고 시작이 느려지지 않는다
PROMPT_MAX_COUNT = 8
#: 쓸 수 있는 글자. 콜론 · 등호는 `--target 이름:size=…` 의 구분자라 뺀다.
#: 한글 · 한자는 **CLIP 글 인코더가 배우지 않았다** — 넣으면 뜻이 잡히지 않는다(그래서 거절한다)
PROMPT_OK_RE = re.compile(r"^[A-Za-z0-9 ,.'\-/()&+]+$")
#: 런처가 **그대로 화면에 띄우는** 가이드. 왜 그래야 하는지까지 적는다 — 규칙만 주면 지키지 않는다
PROMPT_GUIDE = {
    "title": "글 단서 적는 법",
    "lines": [
        "**영어로** 적습니다 — CLIP 글 인코더는 영어로 배웠습니다. 한글은 뜻이 잡히지 않아 거절합니다.",
        "물체 하나를 **명사구**로: `a person in a yellow safety vest`. 명령, 물음, 이야기는 사진에서 잴 수 없습니다.",
        "다른 것과 **가르는 특징**(색, 모양, 재질, 붙어 있는 것)을 한두 개만 더합니다. 넓은 글은 무엇과도 비슷해 점수가 흐려집니다.",
        "자리는 그 물체에 **늘 붙어 있을 때만**: `a fire extinguisher mounted on a wall` 은 되고, 오늘 그 자리에 있다는 말은 안 됩니다.",
        "적은 글은 대상의 자동 문장을 **대신합니다** — 두 벌을 섞으면 어느 글이 점수를 끌었는지 알 수 없습니다.",
    ],
    "good": [
        ["a person in a yellow safety vest", "색 + 입은 것 — 그냥 사람과 가른다"],
        ["a red fire extinguisher mounted on a wall", "색 + 늘 붙어 있는 자리"],
        ["a grey fabric storage box", "재질 + 색 — 종이 상자와 가른다"],
    ],
    "bad": [
        ["노란 조끼를 입은 사람", "한글. CLIP 글 인코더는 영어로 배웠다 — 영어로 바꿔 적는다"],
        ["find the person who took the box", "명령이나 이야기. 사진 한 장에서 잴 수 있는 것이 아니다"],
        ["a person", "너무 넓다. 무엇과도 비슷해 점수가 흐려진다"],
        ["a man in a yellow vest standing near the third door from the left on the second floor",
         "너무 길다. CLIP 이 77 낱말에서 자르면 뒤는 재지 않은 것이 된다"],
    ],
    "limits": {"max_chars": PROMPT_MAX_CHARS, "min_chars": PROMPT_MIN_CHARS, "max_count": PROMPT_MAX_COUNT,
               "allowed": "영문, 숫자, 공백 과 , . ' - / ( ) & +"},
}

# ---------------------------------------------------------------- 사람이 넣는 단서 ② 기준 사진
#: 기준 사진 개수 상한. CLIP 임베딩은 장마다 한 줄 — 24장이면 각도 · 조명을 덮고 npz 는 50 KB 아래다
REF_MAX_IMAGES = 24
#: zip 안 한 파일의 **푼 뒤** 크기 상한과 합계 상한. 사람이 올린 묶음이라 터지는 zip 을 그대로 읽지 않는다
REF_MAX_BYTES = 20 * 1024 * 1024
REF_TOTAL_BYTES = 120 * 1024 * 1024
REF_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
#: 보관할 때 줄이는 긴 변(px). CLIP 입력이 224 라 더 큰 원본은 쓰이지 않는다 — 저장소만 커진다
REF_STORE_PX = 512
#: npz 안의 묶음 이름. butane_ref/prototypes.npz 와 같은 규칙(`<인코더>/<묶음>`)
REF_ENCODER = "clip_b32"

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
    """모델 파일 경로 — 상대 경로는 DS_HOME(기본 저장소 루트) 기준."""
    return settings.resolve(path)


def model_files(tcfg):
    """모델 키 → 있어야 하는 파일 목록. 경로는 yaml models.targeted.files 가 이긴다."""
    f = dict(DEFAULT_FILES)
    f.update((tcfg or {}).get("files") or {})
    out = {}
    for key, paths in f.items():
        out[key] = [_p(x) for x in (paths if isinstance(paths, list) else [paths])]
    return out


#: 사용 가능 판정용 파일 목록 (model_registry.yaml models.*.files — files_key 가 있으면 그 이름으로)
DEFAULT_FILES = {str(m.get("files_key", k)): list(m.get("files") or []) for k, m in _REG["models"].items()}
#: 프로세스 하나에 더해지는 GPU 메모리 추정(MB). 측정값이 아니라 **분할 계산용 어림**이다 (model_registry.yaml)
GPU_MB = {"base": int(_REG.get("base_gpu_mb", 400))}
for _k, _m in _REG["models"].items():
    if _m.get("gpu_mb") is not None:
        for _name in [_k] + list(_m.get("gpu_mb_alias") or []):
            GPU_MB[_name] = int(_m["gpu_mb"])


def have(key, files):
    return all(p.exists() for p in files.get(key, []))


def gdino_mode(tcfg):
    v = ((tcfg or {}).get("ovd") or {}).get("grounding_dino", "auto")
    return "auto" if v == "auto" else bool(v)


def ovd_available(tcfg, files, targets=None, pick=None):
    """OVD 동의에 쓸 탐지기. YOLO-World 는 있으면 쓴다. GroundingDINO 는 무거워서(프레임당 수백 ms)
    yaml `ovd.grounding_dino` 가 true 이거나, auto(기본)이면 **그 근거를 선언한 대상**(ovd_rules.grounding_dino,
    지금은 demo_door — main 이 그 문을 GDINO 근거로 확정했다)을 고를 때만 쓴다. targets=None 이면 '쓸 수 있는가' 만 본다."""
    pick = None if pick is None else set(pick)        # 런처 근거 표에서 고른 OVD 모델 (None = 지금 규칙 그대로)
    out = []
    if have("yolo_world", files) and (pick is None or "yolo_world" in pick):
        out.append("yolo_world")
    mode = gdino_mode(tcfg)
    if pick is not None:                              # 사람이 고르면 auto 규칙보다 앞선다 (끄는 것도 · 켜는 것도)
        mode = "grounding_dino" in pick
    if mode is not False and have("gdino", files):
        if mode is True or targets is None or any("grounding_dino" in (t.get("ovd_rules") or {}) for t in targets):
            out.append("grounding_dino")
    return out


def depth_key(cue_models=None):
    """거리 근거에 쓸 모델 키 (기본 depth = UniDepth ViT-B/14)."""
    return str((cue_models or {}).get("depth") or "depth")


def depth_model_id(cue_models, files):
    """거리 모델 키 → depth_track 에 넘길 model_id. 기본(depth)이면 None(yaml 값 그대로).
    로컬 가중치(models/… · external/…)는 실제 자리(files 의 첫 경로)로, 허브 이름은 그대로 넘긴다."""
    key = depth_key(cue_models)
    if key == "depth":
        return None
    mid = str(CUE_CHOICES.get("depth", {}).get("model_id", {}).get(key) or "")
    if mid.startswith(("models/", "external/")) and files.get(key):
        return str(files[key][0])
    return mid or None


def check_cue_setup(cs):
    """프로필 models.targeted.cue_setup → {"off": [근거], "models": {need: 모델 | [모델]}}. 틀리면 ProfileError"""
    if cs in (None, {}):
        return {"off": [], "models": {}}
    if not isinstance(cs, dict) or set(cs) - {"off", "models"}:
        raise ProfileError("cue_setup 은 {off: [근거], models: {ovd: [...], depth: ...}}")
    off = [str(k) for k in (cs.get("off") or [])]
    for k in off:
        if k not in CUES:
            raise ProfileError("cue_setup.off 에 모르는 근거 %r (가능: %s)" % (k, ", ".join(CUES)))
        if k == "class":
            raise ProfileError("탐지기 이름(class) 근거는 끌 수 없다 — 후보를 내는 것이 그것이다")
    models = {}
    for need, v in (cs.get("models") or {}).items():
        ch = CUE_CHOICES.get(need)
        if ch is None:
            raise ProfileError("cue_setup.models 에 바꿀 수 없는 근거 모델 %r (가능: %s)" % (need, ", ".join(CUE_CHOICES)))
        vals = list(v) if isinstance(v, (list, tuple)) else [v]
        if not vals or (len(vals) > 1 and not ch["multi"]):
            raise ProfileError("cue_setup.models.%s 는 %s" % (need, "하나 이상" if ch["multi"] else "하나"))
        for m in vals:
            if m not in ch["choices"]:
                raise ProfileError("cue_setup.models.%s 에 고를 수 없는 모델 %r (가능: %s)"
                                   % (need, m, ", ".join(ch["choices"])))
        models[need] = [str(m) for m in vals] if ch["multi"] else str(vals[0])
    return {"off": sorted(set(off)), "models": models}


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
    `vest:prompt=a person in a yellow safety vest` (글은 콜론 · 등호를 못 쓴다 — 여기가 구분자다.
    여러 글은 `--prompt` 를 여러 번 주는 쪽이 낫다)
    """
    parts = [p.strip() for p in str(text).split(":")]
    name = parts[0]
    if not _NAME_RE.match(name or ""):
        raise ProfileError("대상 이름은 영문으로 시작하는 41자 이하 (글자, 숫자, 공백, _, -): %r" % name)
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
        elif k == "prompt":
            out["prompt"] = check_prompts(v, "대상 %s" % name)
        else:
            raise ProfileError("대상 %s 에 모르는 항목 %r (가능: size, colour, prompt)" % (name, k))
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


def check_prompts(value, where=""):
    """사람이 적은 글(프롬프트) → 검사한 글 목록. 글 하나(문자열)도 받는다.

    조용히 다듬지 않는다 — 빈 글 · 긴 글 · 못 쓰는 글자는 `ProfileError` 로 왜 안 되는지 돌려준다.
    런처는 그 한 줄을 상태 줄에 그대로 띄우고, 쓰는 법은 `PROMPT_GUIDE` 로 함께 받는다.
    """
    pre = ("%s 의 " % where) if where else ""
    if value is None:
        return []
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, (list, tuple)):
        raise ProfileError("%s글 단서는 글 하나 또는 글 목록입니다" % pre)
    if len(items) > PROMPT_MAX_COUNT:
        raise ProfileError("%s글 단서는 %d 개까지입니다 (받은 것 %d 개) — 가르는 글만 남기세요"
                           % (pre, PROMPT_MAX_COUNT, len(items)))
    out = []
    for raw in items:
        if not isinstance(raw, str):
            raise ProfileError("%s글 단서는 글자여야 합니다: %r" % (pre, raw))
        s = " ".join(raw.split())                       # 줄바꿈 · 겹친 공백만 고른다(뜻은 건드리지 않는다)
        if not s:
            raise ProfileError("%s글 단서가 비었습니다 — 무엇을 찾는지 영어 명사구로 적습니다"
                               " (예: a person in a yellow safety vest)" % pre)
        if len(s) < PROMPT_MIN_CHARS:
            raise ProfileError("%s글 단서가 너무 짧습니다 (%d자) — 물체 이름과 가르는 특징을 적습니다: %r"
                               % (pre, len(s), s))
        if len(s) > PROMPT_MAX_CHARS:
            raise ProfileError("%s글 단서가 깁니다 (%d자 > %d) — CLIP 은 77 낱말에서 자릅니다."
                               " 가르는 특징 한두 개만 남기세요: %r" % (pre, len(s), PROMPT_MAX_CHARS, s[:40] + "…"))
        if any(ord(ch) > 127 for ch in s):
            bad = "".join(sorted({ch for ch in s if ord(ch) > 127}))[:12]
            raise ProfileError("%s글 단서는 영어로 적습니다 — CLIP 글 인코더는 영어로 배웠습니다"
                               " (한글과 한자는 뜻이 잡히지 않습니다). 쓸 수 없는 글자: %s" % (pre, bad))
        if not PROMPT_OK_RE.match(s):
            bad = "".join(sorted({ch for ch in s if not PROMPT_OK_RE.match(ch)}))[:12]
            raise ProfileError("%s글 단서에 쓸 수 없는 글자 %s — 영문, 숫자, 공백 과 , . ' - / ( ) & + 만 됩니다"
                               " (콜론과 등호는 `이름:size=…` 의 구분자입니다)" % (pre, bad))
        if not any(ch.isalpha() for ch in s):
            raise ProfileError("%s글 단서에 낱말이 없습니다: %r" % (pre, s))
        if s not in out:
            out.append(s)
    return out


def apply_prompts(t, prompts):
    """검사한 글을 대상에 붙인다 — `user_descriptions` 로 담고 `descriptions` 를 **대신**하고 `semantic` 을 켠다.

    자동 반대 문장(negatives)에 사람이 적은 낱말이 들어 있으면 그 문장만 뺀다 — `a person in a vest` 를
    찾는데 반대 문장이 `a person` 이면 자기 자신과 다툰다(`_direct_target` 이 person 에 하던 것과 같은 손질).
    대상 파일에 손으로 적은 negatives 는 건드리지 않는다 — 그것은 고른 반대다.
    """
    if not prompts:
        return t
    t["user_descriptions"] = list(prompts)
    if t.get("descriptions") and t["descriptions"] != list(prompts):
        t["_descriptions_default"] = list(t["descriptions"])
    t["descriptions"] = list(prompts)
    if t.get("_negatives_auto"):
        words = {w.strip(".,'()&+/-").lower() for p in prompts for w in p.split()}
        t["negatives"] = [n for n in t.get("negatives") or [] if n.split()[-1].lower() not in words]
    cues = dict(t.get("cues") or {})
    if cues.get("semantic") in (None, "off", False):     # 글을 줬으면 글 설명 근거를 켠다
        cues["semantic"] = CUES["semantic"]["role"]
        t["cues"] = cues
    return t


def _direct_target(name, spec):
    """대상 파일 없는 이름 — 제로샷 기본값을 채운다(§3.4.8)."""
    t = {"name": name, "shown": name, "detector": "yoloe_text", "direct": True,
         "prompts": [name], "free_vocab": "auto",
         "descriptions": ["a photo of a %s" % name, "a %s in the scene" % name],
         "negatives": ["a photo of an object", "a wall", "the floor", "a person"],
         "_negatives_auto": True,                   # 자동 반대 문장 — 사람이 적은 글과 부딪히면 뺀다(apply_prompts)
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
    return apply_prompts(t, check_prompts(spec.get("prompt"), "대상 %s" % name))


def available_cues(t, tcfg, files, distance=True, cue_models=None):
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
                    if not ovd_available(tcfg, files, [t], (cue_models or {}).get("ovd")):
                        why = "OVD 가중치 없음"
                elif need == "depth":
                    if not distance:
                        why = "거리 측정 꺼짐"
                    elif not have(depth_key(cue_models), files):
                        why = "%s 없음" % depth_key(cue_models)
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
        raise ProfileError("대상 %s 의 cues 는 default, 목록, 사전 중 하나: %r" % (t["name"], spec_cues))
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
#: 모델과 따로 켜는 기능. track 은 가중치가 없다 — 트래커는 ultralytics 에 들어 있다(AI-C-16: 내려받지 않는다).
#: 탐지가 장마다 들락거리는 것을 막는다 — 기성 트래커는 잘 잡힌 검출로만 track 을 시작하고 약한 검출은
#: 이미 있는 track 을 되찾는 데만 쓴다(ByteTrack). 그래서 얕은 검출이 새 물체로 튀어나오지 않는다.
FEATURES = {"reid": "객체 재식별", "track": "추적"}
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
        extra = set(part) - ({"targets", "distance", "overlay", "cue_setup"} if m == "targeted" else {"overlay"})
        if extra:
            raise ProfileError("모델 %s 에는 %s 를 줄 수 없음 (대상, 근거, 거리는 지정 클래스 탐지에만)"
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
            raise ProfileError("객체 재식별에 필요한 가중치가 없음 (models/lightglue_superpoint 과 CLIP)")
    result["features"] = dict(feats)
    tpart = pm.get("targeted") or {}
    distance = bool(tpart.get("distance", tcfg.get("distance", True)))
    setup = check_cue_setup(tpart.get("cue_setup"))
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
            extra = set(s) - {"name", "cues", "size_mm", "colours", "min_score", "distance", "prompt"}
            if extra:
                raise ProfileError("대상 %s 에 모르는 칸: %s" % (name, ", ".join(sorted(extra))))
            for c in s.get("colours") or []:
                if c not in COLOURS:
                    raise ProfileError("대상 %s 의 색 %r 을 모름 (가능: %s)" % (name, c, ", ".join(COLOURS)))
            prompts = check_prompts(s.get("prompt"), "대상 %s" % name)
            if name in targets_db:
                t = copy.deepcopy(targets_db[name])
                if s.get("size_mm"):
                    t["size_mm"] = list(s["size_mm"])
                if s.get("colours"):
                    t["colours"] = list(s["colours"])
                # 프로필의 글이 대상 파일의 글(user_descriptions)을 덮는다 — 지금 사람이 적은 것이 이긴다
                apply_prompts(t, prompts or check_prompts(t.get("user_descriptions"), "대상 %s 파일" % name))
            else:
                t = _direct_target(name, s)
            if s.get("min_score") is not None:
                t["min_score"] = float(s["min_score"])
            t_dist = distance and bool(s.get("distance", True))
            avail = available_cues(t, tcfg, files, t_dist, setup["models"])
            # 근거 표에서 끈 근거는 모든 대상에서 뺀다 — 대상별로 직접 고른 것이어도 (근거 표가 위다)
            for k in setup["off"]:
                avail[k] = avail.get(k) or "근거 표에서 끔"
            spec_cues = s.get("cues")
            if isinstance(spec_cues, list):
                spec_cues = [k for k in spec_cues if k not in setup["off"]]
            elif isinstance(spec_cues, dict):
                spec_cues = {k: v for k, v in spec_cues.items() if k not in setup["off"]}
            t["cues"] = resolve_cues(t, spec_cues, avail)
            t["distance"] = t_dist
            t["unavailable"] = {k: v for k, v in avail.items() if v}
            out.append(t)
        result["targeted"] = {"targets": out, "distance": distance, "cue_models": setup["models"],
                              "cues_off": setup["off"]}
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
    cm = (resolved or {}).get("cue_models") or {}
    need = set()
    for t in (resolved or {}).get("targets", []):
        if t.get("detector") == "trained":
            need.add("trained")
        else:
            need.add("yoloe_text")
            need.update(ovd_available(tcfg, files, [t], cm.get("ovd")))     # 탐지기 근거에 OVD 확신이 들어간다
        for k in t["cues"]:
            for n in CUES[k]["needs"]:
                if n == "ovd":
                    need.update(ovd_available(tcfg, files, [t], cm.get("ovd")))
                elif n == "depth":
                    need.add(depth_key(cm))
                else:
                    need.add(n)
    return need


def gpu_mb(models, need=()):
    """비전 프로세스 하나의 GPU 메모리 어림(MB)."""
    mb = GPU_MB["base"]
    for m in models:
        mb += OPTION_GPU_MB.get(m, 0)
    for n in need:
        mb += GPU_MB.get(n, 0)
    return mb


def option_cues(files=None, tcfg=None):
    """선택지 → [{key, label, models: [{key, note, available}]}] — 런처가 「이 선택지는 어떤 단서를 어떤 모델로 쓰나」를
    보여 준다. targeted 는 cues 구역 전체(needs 의 ovd · depth 를 실제 모델 키로 편다)."""
    files = model_files(tcfg) if files is None else files
    labels = dict((k, c["label"]) for k, c in CUES.items())
    labels.update(_REG.get("cue_type_labels") or {})

    def model_row(m):
        spec = _REG["models"].get(m, {})
        return {"key": m, "note": str(spec.get("note", "")), "available": have(str(spec.get("files_key", m)), files)}

    out = {}
    for opt, o in _REG["options"].items():
        cm = o.get("cue_models")
        rows = []
        if cm == "targeted":
            for k, c in CUES.items():
                ms = []
                for n in c["needs"]:
                    ms += ["yolo_world", "grounding_dino"] if n == "ovd" else [n]
                if k == "class":
                    ms = ["yoloe_text", "trained"]
                # needs: 바꿀 수 있는 근거 모델(cue_model_choices)과 잇는 열쇠 — 런처 근거 표가 고르는 칸을 단다
                rows.append({"key": k, "label": c["label"], "models": [model_row(m) for m in ms],
                             "needs": list(c["needs"])})
        else:
            for k, ms in (cm or {}).items():
                rows.append({"key": k, "label": labels.get(k, k), "models": [model_row(m) for m in ms or []]})
        out[opt] = rows
    return out


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
                        "cues": cues,
                        # 사람이 넣은 단서 — 글은 그대로, 사진은 상태만(없으면 null = 안 넣은 것)
                        "prompt": list(t.get("user_descriptions") or []),
                        "refs": ref_status(n, TARGET_DIR, cfg, t)})
    direct = {}
    for kind, keys in (("always", DIRECT_ALWAYS), ("with_size", DIRECT_WITH_SIZE), ("with_colour", DIRECT_WITH_COLOUR)):
        probe = _direct_target("x", {"size_mm": [1, 1, 1], "colours": ["red"]})
        avail = available_cues(probe, tcfg, files, True)
        direct[kind] = [{"key": k, "label": CUES[k]["label"], "needs_distance": "depth" in CUES[k]["needs"]}
                        for k in keys if not avail.get(k)]
    all_models = list(MODELS)
    enc = ref_encoder_file(tcfg)
    return {"models": [{"key": m, "shown": MODEL_SHOWN[m]} for m in all_models],
            "overlay_defaults": overlay_defaults(cfg),
            "targets": targets, "direct": direct, "colours": list(COLOURS),
            # 글 단서 — 런처가 PROMPT_GUIDE 를 그대로 띄운다. semantic 은 CLIP 이 있어야 재진다
            "prompt_guide": PROMPT_GUIDE,
            "prompt_available": have("clip", files) or "CLIP 가중치 없음 (글은 저장되고, 가중치가 생기면 재진다)",
            # 기준 사진 — 만들 때 필요한 것은 비전 인코더 하나다(글 인코더 · 토크나이저는 아니다)
            "ref_images": {"available": True if (enc and enc.exists()) else "CLIP 비전 인코더 없음: %s" % enc,
                           "max_images": REF_MAX_IMAGES, "suffixes": list(REF_SUFFIXES),
                           "max_mb": REF_MAX_BYTES // (1024 * 1024), "store_px": REF_STORE_PX,
                           "dir": str(TARGET_DIR), "encoder": REF_ENCODER,
                           "guide": ["그 물체만 꽉 차게 **자른** 사진이 좋습니다 — CLIP 은 가운데 224 만 봅니다.",
                                     "각도 · 조명이 다른 3~10장. 같은 장면만 여러 장이면 그 장면과만 닮습니다.",
                                     "운영 영상은 넣지 마세요 — 그것은 시험 자료입니다(기준 사진에서만 만듭니다)."]},
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
            "gpu_mb_targeted": {k: v for k, v in GPU_MB.items()},
            "gpu_mb_base": GPU_MB["base"],
            "gpu_mb_option": dict(OPTION_GPU_MB),
            "options": option_cues(files, tcfg),
            # 근거 표에서 바꿀 수 있는 근거 모델 — {need: {choices: [{key, note, available}], multi, default}}
            "cue_choices": {need: {"choices": [{"key": m, "note": str(_REG["models"].get(m, {}).get("note", "")),
                                                "available": have(str(_REG["models"].get(m, {}).get("files_key", m)),
                                                                  files)} for m in ch["choices"]],
                                   "multi": ch["multi"],
                                   "default": (ovd_available(tcfg, files) if need == "ovd" else ["depth"])}
                            for need, ch in CUE_CHOICES.items()}}


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


def save_direct_target(name, size_mm=None, colours=None, target_dir=TARGET_DIR,
                       prompts=None, ref_zips=(), ref_images=(), cfg=None):
    """직접 입력 대상을 **등록 대상**으로 저장한다(런처 '등록으로 저장'). 있는 파일은 덮지 않는다.

    `prompts` 사람이 적은 글 — `semantic` 근거가 켜진다 (`check_prompts` 가 먼저 검사한다)
    `ref_zips` · `ref_images` 기준 사진 — 저장한 뒤 `attach_reference_images` 로 붙인다.
    사진 쪽이 실패해도 **대상 파일은 남는다** — 글만으로도 쓸 수 있고, 사진은 뒤에 다시 붙이면 된다.
    """
    spec = {"name": name}
    if prompts:
        spec["prompt"] = check_prompts(prompts, "대상 %s" % name)
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
    if ref_zips or ref_images:
        attach_reference_images(name, ref_zips, ref_images, target_dir, cfg)
    return str(f)


# ---------------------------------------------------------------- 기준 사진 (zip → CLIP 이미지 임베딩)
def ref_encoder_file(tcfg=None):
    """기준 사진 임베딩에 쓰는 파일 — CLIP **비전 인코더 하나**.

    글 인코더 · 토크나이저는 필요 없다(이미지끼리 견주는 근거다). 그래서 토크나이저가 아직 없는
    기계에서도 기준 사진은 지금 만들어 둘 수 있다 — 쓰는 쪽(`ref_match`)은 clip 전체를 본다.
    """
    for p in model_files(tcfg).get("clip", []):
        if p.name.startswith("vision_model"):
            return p
    return None


def ref_paths(name, target_dir=TARGET_DIR):
    """(대상 파일, 사진 폴더, 임베딩 파일) — 셋은 늘 **대상 파일 옆**에 있다."""
    d = Path(target_dir)
    return d / ("%s.json" % name), d / ("%s.refs" % name), d / ("%s.refs.npz" % name)


def _read_ref_sources(zips=(), images=()):
    """zip · 이미지 파일 → ([(이름, 바이트)], 건너뛴 것).

    **zip 을 디스크로 풀지 않는다.** 멤버 이름을 그대로 쓰면 `../` 로 폴더 밖에 쓸 수 있고
    링크가 섞일 수도 있다 — 바이트만 읽고 이름은 우리가 새로 붙인다. 푼 뒤 크기를 먼저 보고
    한 파일 · 합계 상한을 넘으면 읽지 않는다(터지는 zip).
    """
    got, skipped, total = [], [], 0

    def take(label, data):
        nonlocal total
        if len(got) >= REF_MAX_IMAGES:
            skipped.append([label, "기준 사진은 %d장까지" % REF_MAX_IMAGES])
            return
        total += len(data)
        if total > REF_TOTAL_BYTES:
            skipped.append([label, "합계가 %d MB 를 넘음" % (REF_TOTAL_BYTES // (1024 * 1024))])
            return
        got.append((label, data))

    for z in zips or ():
        p = Path(z).expanduser()
        if not p.exists():
            raise ProfileError("zip 을 못 찾음: %s" % p)
        try:
            with zipfile.ZipFile(p) as zf:
                for info in zf.infolist():
                    label = Path(info.filename).name
                    if info.is_dir():
                        continue
                    if Path(info.filename).suffix.lower() not in REF_SUFFIXES:
                        skipped.append([info.filename, "그림 파일이 아님 (%s)" % ", ".join(REF_SUFFIXES)])
                        continue
                    if info.file_size > REF_MAX_BYTES:
                        skipped.append([label, "한 장이 %d MB 를 넘음" % (REF_MAX_BYTES // (1024 * 1024))])
                        continue
                    with zf.open(info) as f:
                        take(label, f.read(REF_MAX_BYTES + 1))
        except zipfile.BadZipFile as exc:
            raise ProfileError("zip 을 못 읽음: %s (%s)" % (p.name, exc))
    for im in images or ():
        p = Path(im).expanduser()
        if not p.exists():
            raise ProfileError("사진을 못 찾음: %s" % p)
        if p.suffix.lower() not in REF_SUFFIXES:
            skipped.append([p.name, "그림 파일이 아님 (%s)" % ", ".join(REF_SUFFIXES)])
            continue
        if p.stat().st_size > REF_MAX_BYTES:
            skipped.append([p.name, "한 장이 %d MB 를 넘음" % (REF_MAX_BYTES // (1024 * 1024))])
            continue
        take(p.name, p.read_bytes())
    return got, skipped


def _store_ref_images(got, ref_dir):
    """읽은 바이트 → `<대상>.refs/NN.png`. 긴 변을 REF_STORE_PX 로 줄여 PNG 하나로 통일한다
    (CLIP 입력이 224 라 더 큰 원본은 쓰이지 않는다). 못 푸는 파일은 건너뛴 것으로 돌려준다."""
    import cv2
    import numpy as np
    ref_dir = Path(ref_dir)
    for old in sorted(ref_dir.glob("*.png")):           # 다시 올리면 **갈아 끼운다** (섞이면 무엇으로 쟀는지 모른다)
        old.unlink()
    ref_dir.mkdir(parents=True, exist_ok=True)
    kept, skipped = [], []
    for label, data in got:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None or min(img.shape[:2]) < 8:
            skipped.append([label, "그림으로 풀리지 않음 (또는 8 px 아래)"])
            continue
        k = REF_STORE_PX / max(img.shape[:2])
        if k < 1.0:
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        fn = "%02d.png" % (len(kept) + 1)
        cv2.imwrite(str(ref_dir / fn), img)
        kept.append([fn, label])
    return kept, skipped


def _embed_refs(paths, enc_path):
    """사진 경로 목록 → (N, 512) L2 정규화 임베딩.

    전처리는 **런타임과 같은 것**을 쓴다(`target_assess._clip_preprocess`, 224 가운데 자르기).
    hazard_verify 의 늘려 맞추기로 만들면 코사인이 다른 자가 되어 견줄 수 없다.

    **잰 것(2026-11-26, RTX 4050 · CUDAExecutionProvider):** 양자화 CLIP ONNX 는 **묶음 크기에
    따라 값이 조금 달라진다** — 같은 사진을 1장으로 돌린 것과 3장 묶음의 한 줄은 코사인 0.990
    (같은 묶음 크기끼리는 1.000). 런타임은 한 장에 보이는 후보 수만큼 묶어 돌리므로 기준 임베딩과
    런타임 임베딩 사이에는 이만큼의 흔들림이 늘 있다. `ref_match` 문턱을 0.99 같은 값으로 두면
    묶음 크기가 판정을 가른다.
    """
    import cv2
    import numpy as np
    import onnxruntime as ort

    import target_assess as ta
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass
    avail = set(ort.get_available_providers())
    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    sess = ort.InferenceSession(str(enc_path), ort.SessionOptions(), providers=prov)
    batch = np.stack([ta._clip_preprocess(cv2.imread(str(p))) for p in paths], axis=0)
    e = sess.run(None, {sess.get_inputs()[0].name: batch})[0].astype(np.float32)
    return e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-8), sess.get_providers()[0]


def build_ref_embeddings(name, target_dir=TARGET_DIR, cfg=None):
    """보관한 기준 사진 → `<대상>.refs.npz`. **가중치가 없으면 왜 못 만드는지 돌려준다**(조용히 비우지 않는다).

    돌려주는 것: {"ok", "name", "state": ready|pending, "images", "embeddings", "why", "missing", "provider"}
    가중치가 뒤에 생기면 이것만 다시 부르면 된다 — 사진은 남아 있다 (`--build-refs`).
    """
    tf, ref_dir, npz = ref_paths(name, target_dir)
    if not tf.exists():
        raise ProfileError("등록된 대상이 아님: %s" % name)
    t = json.loads(tf.read_text(encoding="utf-8"))
    info = dict(t.get("reference_images") or {})
    files = [f for f in (info.get("files") or []) if (ref_dir / f).exists()]
    if not files:
        raise ProfileError("대상 %s 에 보관된 기준 사진이 없습니다 — 먼저 zip 을 붙이세요" % name)
    enc = ref_encoder_file(((cfg or {}).get("models") or {}).get("targeted") or {})
    out = {"ok": True, "name": name, "images": len(files), "provider": None,
           "dir": str(ref_dir), "embeddings": None, "why": None, "missing": []}
    if enc is None or not enc.exists():
        info.update({"state": "pending", "why": "CLIP 비전 인코더가 없습니다: %s" % (enc or "models.clip 선언 없음"),
                     "encoder": REF_ENCODER, "embeddings": None})
        out.update({"state": "pending", "why": info["why"], "missing": [str(enc)] if enc else []})
    else:
        import numpy as np
        emb, provider = _embed_refs([ref_dir / f for f in files], enc)
        np.savez(npz, **{"%s/%s" % (REF_ENCODER, name): emb.astype(np.float32)})
        info.update({"state": "ready", "why": None, "encoder": REF_ENCODER, "embeddings": npz.name,
                     "dim": int(emb.shape[1]), "made_at": _now()})
        out.update({"state": "ready", "embeddings": str(npz), "provider": provider})
    info["files"] = files
    info["count"] = len(files)
    info["dir"] = ref_dir.name
    t["reference_images"] = info
    cues = dict(t.get("cues") or {})
    # 기준 사진이 준비되면 근거를 켠다. 쓸 수 없는 기계에서는 available_cues 가 알아서 뺀다
    if info["state"] == "ready":
        cues.setdefault("ref_match", CUES["ref_match"]["role"])
    t["cues"] = cues
    tf.write_text(json.dumps(t, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def attach_reference_images(name, zips=(), images=(), target_dir=TARGET_DIR, cfg=None):
    """등록된 대상에 기준 사진(zip 또는 이미지 파일 여럿)을 붙인다.

    사진은 `<대상>.refs/` 에 **남긴다** — 가중치가 뒤에 생겨도 임베딩을 다시 만들 수 있어야 한다.
    임베딩을 못 만들어도 붙이기는 성공이다(state: pending, why 에 무엇이 없는지).
    """
    tf, ref_dir, _npz = ref_paths(name, target_dir)
    if not tf.exists():
        raise ProfileError("등록된 대상이 아님: %s (먼저 --save-target 으로 등록하세요)" % name)
    got, skipped = _read_ref_sources(zips, images)
    if not got:
        raise ProfileError("기준 사진을 하나도 못 읽었습니다 — %s"
                           % ("; ".join("%s: %s" % (a, b) for a, b in skipped[:3]) or "묶음이 비었습니다"))
    kept, bad = _store_ref_images(got, ref_dir)
    skipped += bad
    if not kept:
        raise ProfileError("기준 사진을 하나도 못 풀었습니다 — %s"
                           % "; ".join("%s: %s" % (a, b) for a, b in skipped[:3]))
    t = json.loads(tf.read_text(encoding="utf-8"))
    t["reference_images"] = {"dir": ref_dir.name, "files": [f for f, _src in kept], "count": len(kept),
                             "sources": [src for _f, src in kept], "encoder": REF_ENCODER,
                             "state": "pending", "why": "아직 만들지 않음", "embeddings": None,
                             "_doc": "사람이 올린 기준 사진. 임베딩은 <대상>.refs.npz 의 clip_b32/<대상> (N, 512)"}
    tf.write_text(json.dumps(t, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out = build_ref_embeddings(name, target_dir, cfg)
    out["skipped"] = skipped
    out["files"] = [f for f, _src in kept]
    return out


def _now():
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def ref_status(name, target_dir=TARGET_DIR, cfg=None, t=None):
    """대상 하나의 기준 사진 상태 — 런처가 화면에 쓴다. **없음과 모름을 가른다.**

    None  사용자가 넣지 않았다 (칸이 아예 없다)
    dict  {state: ready|pending, count, why} — pending 은 사진은 있고 가중치가 없어 못 만든 것
    """
    if t is None:
        tf, _d, _n = ref_paths(name, target_dir)
        if not tf.exists():
            return None
        t = json.loads(tf.read_text(encoding="utf-8"))
    info = t.get("reference_images")
    if not info:
        return None
    return {"state": info.get("state"), "count": int(info.get("count") or 0),
            "why": info.get("why"), "embeddings": info.get("embeddings"), "dir": info.get("dir")}


def _load_yaml(path):
    """--config 읽기. 기본 vision_config.yaml 이면 config/vision_config.local.yaml 덮어쓰기도 (vision_infer 와 같게)."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        return {}
    if Path(path).resolve() == (HERE / "vision_config.yaml").resolve():
        return settings.load_vision_config(path)
    return settings.load_yaml(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="지정 클래스 탐지 카탈로그 · 프로필 검사")
    ap.add_argument("--config", default=str(HERE / "vision_config.yaml"))
    ap.add_argument("--json", action="store_true", help="런처용 카탈로그 JSON")
    ap.add_argument("--check", metavar="프로필", help="프로필 파일을 검사하고 확정된 근거를 보여 준다")
    ap.add_argument("--models", nargs="+", default=["targeted"])
    ap.add_argument("--save-target", metavar="이름", help="직접 입력 대상을 등록 대상으로 저장")
    ap.add_argument("--size", metavar="가로x세로x높이", help="--save-target 의 크기 mm")
    ap.add_argument("--colours", metavar="색", help="--save-target 의 색 (+ 로 여럿)")
    ap.add_argument("--prompt", action="append", metavar="글", default=[],
                    help="사람이 적는 글 단서(영어 명사구). 여러 번 줄 수 있다 — 글 설명(semantic) 근거가 된다")
    ap.add_argument("--check-prompt", action="store_true", help="글 단서만 검사하고 끝낸다(저장하지 않는다)")
    ap.add_argument("--attach-refs", metavar="이름", help="등록된 대상에 기준 사진을 붙인다")
    ap.add_argument("--ref-zip", action="append", metavar="zip", default=[], help="기준 사진 묶음(zip). 여러 번 가능")
    ap.add_argument("--ref-image", action="append", metavar="사진", default=[], help="기준 사진 한 장. 여러 번 가능")
    ap.add_argument("--build-refs", metavar="이름|all", help="보관한 기준 사진으로 임베딩을 (다시) 만든다")
    a = ap.parse_args(argv)
    cfg = _load_yaml(a.config)
    if a.check_prompt:
        try:
            prompts = check_prompts(a.prompt, "")
        except ProfileError as exc:
            print(json.dumps({"ok": False, "error": str(exc), "guide": PROMPT_GUIDE}, ensure_ascii=False))
            return 1
        print(json.dumps({"ok": True, "prompts": prompts, "guide": PROMPT_GUIDE}, ensure_ascii=False))
        return 0
    if a.save_target:
        try:
            spec = parse_target_arg(a.save_target + (":size=" + a.size if a.size else "")
                                    + (":colour=" + a.colours if a.colours else ""))
            prompts = check_prompts(a.prompt or spec.get("prompt"), "대상 %s" % spec["name"])
            path = save_direct_target(spec["name"], spec.get("size_mm"), spec.get("colours"), prompts=prompts)
        except ProfileError as exc:
            bad = {"ok": False, "error": str(exc)}
            if "글 단서" in str(exc):                   # 글이 틀렸을 때만 쓰는 법을 함께 — 다른 오류에는 군더더기다
                bad["guide"] = PROMPT_GUIDE
            print(json.dumps(bad, ensure_ascii=False))
            return 1
        refs = None
        if a.ref_zip or a.ref_image:
            # **사진이 실패해도 대상은 남는다** — 글만으로도 쓸 수 있고 사진은 뒤에 다시 붙이면 된다
            try:
                refs = attach_reference_images(spec["name"], a.ref_zip, a.ref_image, cfg=cfg)
            except (ProfileError, OSError, ValueError) as exc:
                refs = {"ok": False, "state": None, "error": str(exc)}
        out = {"ok": True, "file": path, "name": spec["name"], "prompts": prompts, "refs": refs}
        print(json.dumps(out, ensure_ascii=False))
        return 0
    if a.attach_refs:
        try:
            out = attach_reference_images(a.attach_refs, a.ref_zip, a.ref_image, cfg=cfg)
        except (ProfileError, OSError, ValueError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 1
        print(json.dumps(out, ensure_ascii=False))
        return 0
    if a.build_refs:
        names = sorted(load_targets()) if a.build_refs == "all" else [a.build_refs]
        rows, bad = [], False
        for n in names:
            if a.build_refs == "all" and not (ref_paths(n)[1]).exists():
                continue                                # 기준 사진을 안 넣은 대상 — 못 만든 것이 아니다
            try:
                rows.append(build_ref_embeddings(n, cfg=cfg))
            except (ProfileError, OSError, ValueError) as exc:
                bad = True
                rows.append({"ok": False, "name": n, "error": str(exc)})
        print(json.dumps({"ok": not bad, "targets": rows}, ensure_ascii=False))
        return 1 if bad else 0
    if a.check:
        try:
            prof = json.loads(Path(a.check).read_text(encoding="utf-8-sig"))
            r = resolve_profile(prof, cfg, a.models)
        except (ProfileError, ValueError, OSError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 1
        tt = (r["targeted"] or {}).get("targets", [])
        brief = {"ok": True, "targets": {t["name"]: t["cues"] for t in tt}, "overlay": r["overlay"],
                 # 사람이 넣은 단서가 실제로 쓰이는지 — 못 쓰는 근거는 unavailable 에 까닭이 적힌다
                 "prompts": {t["name"]: t.get("user_descriptions") or [] for t in tt},
                 "unavailable": {t["name"]: t.get("unavailable") or {} for t in tt}}
        print(json.dumps(brief, ensure_ascii=False))
        return 0
    print(json.dumps(catalog(cfg), ensure_ascii=False, indent=None if a.json else 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
