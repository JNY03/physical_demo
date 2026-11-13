"""위험물 근거 카드 — **그리는 쪽의 주인.**

implements: AI-S-03, AI-S-04, AI-L-03 · 관련: AI-C-19, AI-C-02, AI-C-04
tests: test_hazard_overlay.py · mission_runtime/tests/test_hazard_card.py (CLI)

## 왜 vision 안에 있나

river 가 돌면서 바로 이 카드를 그린다(`overlay.river.cards`). 전에는 그리는 코드가
백업 저장소의 오프라인 도구에만 있어서 **운영 화면과 검수 화면이 달랐다** — 같은
검출을 두 곳이 각자 조립하면 어느 날 둘이 다른 말을 한다.

그래서 **주인을 여기 하나로** 두고 `mission_runtime/tools/hazard_card.py` 는 이
모듈을 불러 쓰는 얇은 CLI 가 된다(`draft_vision.py` 가 `vision_infer.py` 를 불러
쓰는 것과 같은 방식).

## 실시간

카드는 **위험물 후보가 있는 장에서만** 그려진다. 없으면 비용이 0 이다. 켜고 끄는
것은 선언이다(`overlay.river.cards`) — 끄면 예전의 가벼운 박스 라벨만 남는다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

#: Hazards. They call for a different action, so they must not pass unnoticed.
#: **표적은 둘뿐이다** (2026-11-13). 스티로폼·물통·캔은 표적이 아니다 —
#: 어휘에는 미끼로 남아 있지만 출력에 오지 않는다.
HAZARD = ("wooden_box_mine", "butane_canister")
#: 단서 선언의 기본 자리. `--cues` 로 바꾼다. **식의 모양은 코드가, 상수는 설정이** 가진다.
CUES_DEFAULT = Path(__file__).resolve().parent / "assets/hazard_cues.json"
CFG = {}            # load_cues() 가 채운다


def load_cues(path=None):
    """단서 선언을 읽는다. 없으면 코드의 기본값으로 돈다(설정 없이도 선다)."""
    global CFG, SHOWN, REAL_MM, HAZARD
    p = Path(path) if path else CUES_DEFAULT
    CFG = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    objs = CFG.get("objects") or {}
    if objs:
        SHOWN = {k: v.get("shown", k) for k, v in objs.items()}
        REAL_MM = {k: tuple(v["size_mm"][:2]) for k, v in objs.items() if v.get("size_mm")}
        HAZARD = tuple(objs)
    return CFG


def box_colour(det, bgr=False):
    """상자·카드 테두리 색. **위험물만 노랑, 나머지는 회색.**"""
    name = det.get("name") or det.get("cls")
    c = HAZARD_RGB if name in HAZARD else OTHER_RGB
    return tuple(reversed(c)) if bgr else c


def card_opt(key, default):
    return (CFG.get("card") or {}).get(key, default)


def cue(name, key, default):
    return ((CFG.get("cues") or {}).get(name) or {}).get(key, default)


def cue_on(name):
    return bool(cue(name, "enabled", True))


#: 보이는 이름 — **소문자, 기호 없음**. 설정이 있으면 그것이 이긴다.
SHOWN = {"wooden_box_mine": "wooden box mine", "butane_canister": "butane canister"}
#: Material needs pixels. Andong river GT: 40 px median short side — no material there
#: (naming by material words scored wooden-box F1 0.000). Indoor close range: 167 px.
#: This value sits **between those two measurements**; it is not tuned. The accuracy
#: curve by crop size has to set it (design note 2026-11-13 §3).
MATERIAL_PX = 64
#: Butane livery (Korean portable cartridge): **saturated red and saturated blue
#: together**. Deployment data, not a law — a different market's cartridge is a
#: different entry. Measured 2026-11-13 from one reference photo:
#:   reference, down to 11 px:  both = 0.099  (scale-stable: 0.185-0.193 for the
#:     vertical red-over-blue form, which is why colour is usable where texture is not)
#:   151 current butane false positives (indoor + river):  both = 0.0000, every one
#: **Order is deliberately not required.** Red-over-blue scores 0.187 upright but
#: 0.011 upside down and 0.024 tilted — a cartridge afloat is rarely upright. The
#: order-free form holds across sideways/dark/blurred/tinted (0.091-0.127) and still
#: rejects all 151, because those have **no red at all**.
#: This is one exemplar. It is **reported, not enforced** — see `colour` in grounds().
BUTANE_RED, BUTANE_BLUE = [(0, 8), (170, 179)], [(100, 130)]
COLOUR_SAT, COLOUR_VAL, COLOUR_REF = 90, 50, 0.099
#: Real-world proportions, from published specs. Used to score how box-like the box is.
#:   butane cartridge  EN 417 class 220 g: 200 x 66 mm  -> 3.0 : 1
#:   wooden box mine   PMD-6 / KPA type:   200 x 90 mm  -> 2.2 : 1
#: A lying or foreshortened object reads lower; that is correct, not a bug.
ASPECT = {"butane_canister": 3.0, "wooden_box_mine": 2.2}
#: 실물 치수 (long, short) mm — fx 가 있을 때 픽셀을 이것과 견준다.
#:   목함지뢰 PMD-6 / KPA 형: 200 x 90 mm · 부탄 EN 417 급 220 g: 200 x 66 mm
REAL_MM = {"wooden_box_mine": (200.0, 90.0), "butane_canister": (200.0, 66.0)}
#: **These percentages are a declared mapping, not a calibrated probability.**
#: Nothing here was fitted to labelled outcomes - there are almost none. They are a
#: readable common scale so a person can see which cue is weak, and they must not be
#: read as "62% chance this is a mine".
MARGIN_SPAN = 0.20                      # match%: margin -0.20 -> 0, 0 -> 50, +0.20 -> 100
#: Fusion rule, same as vision_config.yaml `fusion:` — one object distance, one meaning.
DEPTH_PCT, DEPTH_MIN_VALID = 40, 0.2
FONTS = [Path.home() / ".local/share/fonts/NotoSansKR-Regular.otf",
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
#: **색의 주인은 여기 하나다.** 위험물은 노랑, 나머지는 회색 — 점수 머리가 빨강·
#: 주황·파랑을 쓰므로 상자까지 색을 쓰면 화면이 색으로만 떠든다.
HAZARD_RGB, OTHER_RGB = (255, 200, 60), (150, 155, 165)
#: 점수 머리: 높으면 파랑, 중간 주황, 낮으면 빨강. **문턱은 자료로 고른 값이 아니다** —
#: 어느 카드를 먼저 볼지 고르는 데 쓰고 판정으로 읽으면 안 된다.
SCORE_HIGH, SCORE_MID, SCORE_LOW = (90, 160, 255), (255, 150, 60), (255, 110, 100)
RGB = {"wooden_box_mine": (255, 190, 60), "butane_canister": (255, 70, 70),
       "styrofoam_box": (120, 200, 255), "water_jug": (110, 230, 140),
       "beverage_can": (255, 120, 220), "unconfirmed": (190, 190, 190)}
CARD_W, THUMB_W, PAD, LINE = 680, 170, 12, 25
#: Inline cards need room. Five of them do not fit a 720p frame without covering the
#: scene, and **dropping one would hide a hazard** — so the canvas grows instead.
SCALE_FOR = ((2, 1.0), (4, 1.5), (99, 2.0))


def find_font():
    for p in FONTS:
        if p.exists():
            return p
    raise SystemExit("no usable font; pass --font")


class Pen:
    def __init__(self, path):
        self.f = {s: ImageFont.truetype(str(path), s) for s in (17, 18, 21, 24)}

    def w(self, t, s):
        return self.f[s].getbbox(t)[2]

    def t(self, d, xy, text, s, fill):
        d.text(xy, text, font=self.f[s], fill=fill)


# ---------------------------------------------------------------- depth
def zipdepth_session(weights, threads=8):
    """ZipDepth ONNX — **affine-invariant inverse depth** 다. 미터가 아니다.

    보정 프로파일(`zipdepth_calibration_profile.json`)은 pi6 의 464x400 crop 에 묶여
    있어 우리 프레임에는 적용되지 않는다. 그래서 cm 로 바꾸지 않는다 — 없는 기준으로
    만든 거리는 거리가 아니다(AI-C-02). 쓸 수 있는 것은 **상대 순서** 뿐이다.
    """
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(weights), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = [d for d in inp.shape if isinstance(d, int) and d > 4]
    return sess, inp.name, sess.get_outputs()[0].name, (size[-1] if size else 256)


def zipdepth_frame(sess_tuple, bgr):
    sess, iname, oname, size = sess_tuple
    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = sess.run([oname], {iname: rgb.transpose(2, 0, 1)[None]})[0]
    return np.squeeze(out).astype(np.float32)


def relative_depth(dmap, box, shape):
    """상자 안 역깊이의 p60 과 **화면 전체에서의 백분위**.

    역깊이는 가까울수록 크다. 숫자 하나로는 읽히지 않으므로 "이 장면에서 몇 %보다
    가까운가" 로 바꿔 돌려준다 — 기준 없는 스칼라보다 사람이 읽을 수 있다.
    """
    if dmap is None:
        return None
    H, W = shape
    dh, dw = dmap.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    sx, sy = dw / float(W), dh / float(H)
    a, b = max(0, int(x1 * sx)), max(0, int(y1 * sy))
    c, d = min(dw, int(x2 * sx)), min(dh, int(y2 * sy))
    patch = dmap[b:d, a:c]
    if patch.size == 0:
        return None
    v = float(np.percentile(patch, 60))          # 가까운 쪽 (provider 의 roi_rule 과 같다)
    pct = float((dmap < v).mean())
    return {"inv": v, "nearer_than": pct}


def depth_map_for(image_path, depth_dir):
    """`<depth_dir>/<stem>.npy` — metres, 0 = invalid (vision_infer offline format)."""
    if not depth_dir:
        return None
    p = Path(depth_dir) / (Path(image_path).stem + ".npy")
    return np.load(p).astype("float32") if p.exists() else None


def distance_of(det, dmap, shape):
    """Metres, or None. **None prints as `no depth`** — never a filled-in number."""
    if det.get("depth_m") is not None:          # already fused upstream; trust it
        return float(det["depth_m"])
    if dmap is None:
        return None
    H, W = shape
    if dmap.shape[:2] != (H, W):
        dmap = cv2.resize(dmap, (W, H), interpolation=cv2.INTER_NEAREST)
    x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
    ex = max(1, int(0.08 * max(x2 - x1, y2 - y1)))        # mask_erode_px, box version
    patch = dmap[max(0, y1 + ex):max(1, y2 - ex), max(0, x1 + ex):max(1, x2 - ex)]
    if patch.size == 0:
        return None
    good = patch[patch > 0]
    if good.size / patch.size < DEPTH_MIN_VALID:
        return None                              # too few valid pixels to mean anything
    return float(np.percentile(good, DEPTH_PCT))


def fmt_distance(m):
    """미터이거나 상대 깊이. 없으면 호출하는 쪽이 줄을 내지 않는다."""
    if m is None:
        return ""
    if isinstance(m, dict) and "space" in m:
        # **값은 기준을 들고 다닌다.** 상대 깊이를 미터처럼 쓰지 않는다(AI-C-02).
        # **미터만 적는다.** measured/carried 와 나이는 기록에는 남지만 화면에는 안 쓴다
        if m["space"] == "metric":
            return "%.1f m" % m["value"]
        return "%.3f relative, not metres" % m["value"]
    if isinstance(m, dict):
        return "rel %.3f · nearer than %.0f%% of frame (not metres)" % (m["inv"], 100 * m["nearer_than"])
    return "%.1f m" % m if m < 100 else "%.0f m" % m


def colour_cue(crop):
    """Saturated red and blue coverage. `None` if the crop is too small to mean anything."""
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 6:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (sat >= COLOUR_SAT) & (val >= COLOUR_VAL)

    def frac(bands):
        m = np.zeros_like(lit)
        for lo, hi in bands:
            m |= (h >= lo) & (h <= hi)
        return float((m & lit).sum()) / max(1, lit.size)

    red, blue = frac(BUTANE_RED), frac(BUTANE_BLUE)
    wood = float(((h >= 10) & (h <= 30) & (sat >= 25) & (sat < 130) & (val >= 110)).mean())
    chrome = float(((sat < 60) & (val >= 120)).mean())      # 도색 안 된 금속
    return {"red": red, "blue": blue, "both": min(red, blue),
            "wood": wood, "chrome": chrome, "lit": float(lit.mean())}


# ---------------------------------------------------------------- text
def shown_name(det):
    cls = det["cls"]
    lead = det.get("lead") or (det.get("prompts") or [None])[0]
    if cls == "unconfirmed":
        return lead or "unknown", True
    return SHOWN.get(cls, cls), False


def band_of(ratio, bands):
    for i, b in enumerate(bands or []):
        if ratio <= float(b["upto"]):
            return ("far" if i == 0 else "near"), float(b["upto"]), float(b["conf"])
    return None, None, None


def grounds(det, W, H, bands, metres, colour=None):
    """Cues as percentages on one scale, plus the models that produced them.

    Each row is `(key, pct|None, text, warn)`. `pct is None` means **not measurable**,
    which is different from 0 - a crop too small to judge is not a crop that failed.
    Unmeasurable rows are left out of the average instead of dragging it down.
    """
    x1, y1, x2, y2 = det["xyxy"]
    w, h, short, long = x2 - x1, y2 - y1, min(x2 - x1, y2 - y1), max(x2 - x1, y2 - y1)
    ratio = (w * h) / float(W * H)
    cls = det["cls"]
    arms = det.get("arms") or []
    lead = det.get("lead") or (det.get("prompts") or ["?"])[0]
    rows, src = [], {}

    # class - the detector's own confidence
    # **후보와 각 신뢰도.** 한 낱말만 보이면 다른 후보가 얼마나 가까웠는지 알 수 없다
    if cue_on("class"):
        votes = det.get("votes") or ({lead: det["conf"]} if lead else {})
        n = int(cue("class", "max_candidates", 4))
        cand = ", ".join("%s %.2f" % (k, v) for k, v in list(votes.items())[:n]) or "?"
        rows.append(("class", 100.0 * det["conf"], cand, False, 1.0))
        src["class"] = (arms[0] if arms else "detector").replace("_free", "")

    # size - how close the box is to the object's real proportions
    real = det.get("real_mm")
    spec_mm = REAL_MM.get(cls) if cue_on("size") else None
    tol = float(cue("size", "tolerance_octaves", 1.0))
    if real and spec_mm:
        # **드디어 치수를 묻는다.** fx 와 거리가 있어야 가능하다(UniDepth).
        # 비스듬히 보면 짧게 나오므로 **하한**으로 읽는다 — 그래서 큰 쪽만 벌한다.
        # **두 축을 다 본다.** 긴 변만 보면 225x194 mm 짜리 덩어리가 통과한다
        # (실측에서 실제로 그랬다 — 긴 변 225 는 규격 200 에 가깝지만 짧은 변이 194 다).
        offs = [abs(np.log(max(real[k], 1.0) / spec_mm[i])) / np.log(2.0)
                for i, k in enumerate(("long", "short"))]
        off = max(offs) / max(tol, 1e-6)                 # 나쁜 축이 점수를 정한다
        pct = 100.0 * max(0.0, 1.0 - off)
        rows.append(("size", pct, "%.1f x %.1f (cm)"
                     % (real["long"] / 10.0, real["short"] / 10.0), False, 1.0))
        src["size"] = "unidepth+rule"
        want = None
    else:
        want = ASPECT.get(cls) if cue_on("size") else None
    if want and short > 0:
        got = long / short
        off = abs(np.log(got / want)) / np.log(2.0) / max(tol, 1e-6)
        pct = 100.0 * max(0.0, 1.0 - off)
        rows.append(("size", pct, "%.0f x %.0f (px)" % (w, h), False, 1.0))
        src["size"] = "rule"

    # material - **재질이 일치하는가, 그리고 그것을 얼마나 확신하는가.**
    # 둘은 다른 질문이라 숫자도 둘이다. 앞은 물체에 대한 판단이고, 뒤는 그 판단을
    # 이 crop 이 담을 수 있느냐다. 픽셀 예산은 **증거가 아니다** — 30 px 짜리는
    # 틀린 crop 이 아니라 **물을 수 없는** crop 이다. 그래서 옆에서 같이 점수가 되지
    # 않고 이 줄의 **무게**가 된다.
    if colour is not None and cue_on("material"):
        full = float(cue("material", "coverage_full", 0.35))
        px = float(cue("material", "confidence_px", MATERIAL_PX))
        if cls == "butane_canister":
            # 인쇄된 금속: 채도 높은 도색 + 도색 안 된 돔/바닥
            lw = float(cue("material", "butane_livery_weight", 0.6))
            mf = float(cue("material", "butane_metal_full", 0.30))
            cov = lw * max(colour["red"], colour["blue"]) / full + (1 - lw) * colour["chrome"] / mf
            txt = "livery %.2f · bare metal %.2f" % (max(colour["red"], colour["blue"]),
                                                     colour["chrome"])
        else:
            # 나무: 따뜻한 색조이면서 **선명하지 않다**
            wp = float(cue("material", "wood_saturation_penalty", 0.5))
            cov = (colour["wood"] - wp * colour["lit"]) / full
            txt = "wood tone %.2f · saturated %.2f" % (colour["wood"], colour["lit"])
        pct = 100.0 * min(1.0, max(0.0, cov))
        conf = min(1.0, short / max(px, 1e-6))
        stuff = "can" if cls == "butane_canister" else "wood"
        rows.append(("material", pct, stuff, False, conf))
        src["material"] = "rule"

    # match / structure - from the unsupervised verifier
    v = det.get("verify")
    if v:
        per = v.get("per_model") or {}
        learned = {k: r for k, r in per.items() if r.get("learned", True)}
        if learned and cue_on("match"):
            m = float(np.mean([r["margin"] for r in learned.values()]))
            span = float(cue("match", "margin_span", MARGIN_SPAN))
            pct = 100.0 * min(1.0, max(0.0, 0.5 + m / (2 * span)))
            # 숫자는 빼고 **무엇과 견줬는지**만 적는다 — margin 은 사람이 읽는 수가 아니다
            rows.append(("match", pct, "against reference image", False, 1.0))
            src["match"] = ",".join(sorted(learned))
        st = per.get("structure")
        if st is not None and cue_on("structure"):
            pct = 100.0 * min(1.0, max(0.0, st["pos"]))
            # 물건마다 **무엇을 본 것인지**가 다르다. 상자는 평평한 면(직육면체),
            # 부탄은 구성 순서다 — 같은 이름이라고 같은 것을 재지 않는다
            rows.append(("structure", pct,
                         "cap, shoulder, body" if cls == "butane_canister"
                         else "cuboid, flat face", False, 1.0))
            src["structure"] = "rule"

    # distance is reported, never scored - a missing depth map is not evidence
    # **없으면 적지 않는다.** 'no depth' 를 쓰면 없다는 사실이 한 줄을 차지한다
    if metres is not None and cue_on("distance"):
        rows.append(("distance", None, fmt_distance(metres), False, 1.0))

    # **확신으로 가중한 평균.** 못 잰 줄은 빠지고, 덜 믿을 줄은 덜 센다.
    # 단서끼리의 중요도는 가중하지 않는다 — 어느 쪽이 더 중한지 정할 근거가 없다.
    # **중요도 순으로 세운다.** match 와 size 는 선언(기준 사진·규격)에 기대는 단서라
    # 뒤에 둔다 — 선언이 현장과 어긋나면 먼저 흔들리는 쪽이다.
    order = {"class": 0, "material": 1, "structure": 2, "match": 3, "size": 4,
             "distance": 8, "source": 9}
    rows.sort(key=lambda r: order.get(r[0], 5))
    scored = [(r[1], r[4]) for r in rows if r[1] is not None]
    wsum = sum(c for _p, c in scored)
    score = (sum(p * c for p, c in scored) / wsum) if wsum else None
    return score, rows, src


# ---------------------------------------------------------------- draw
def place(box, card_wh, W, H, taken):
    """Right of the box if it fits, else left, else inside. Nudge down off neighbours."""
    x1, y1, x2, y2 = box
    cw, chh = card_wh
    x = x2 + 12 if x2 + 12 + cw <= W else (x1 - 12 - cw if x1 - 12 - cw >= 0 else W - cw - 4)
    y = min(max(0, y1), H - chh)
    for _ in range(40):
        if not any(not (y + chh <= ty or ty + th <= y or x + cw <= tx or tx + tw <= x)
                   for tx, ty, tw, th in taken):
            break
        y += 24
        if y + chh > H:
            y = 0
            x = max(0, x - cw - 12)
    return int(max(0, min(x, W - cw))), int(max(0, min(y, H - chh)))


def render(image_path, dets, pen, bands, depth_dir, zip_sess=None):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return None
    n_haz = sum(d["cls"] in HAZARD for d in dets)
    scale = next(s for lim, s in SCALE_FOR if n_haz <= lim)
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        dets = [dict(d, xyxy=[v * scale for v in d["xyxy"]]) for d in dets]
    H, W = bgr.shape[:2]
    clean = bgr.copy()                                   # crops come from the clean frame
    dmap = depth_map_for(image_path, depth_dir)
    # 측정 depth 가 있으면 그것이 먼저다. 없고 ZipDepth 가 있으면 **상대 깊이**를 쓴다 —
    # 미터로 꾸미지 않는다(AI-C-02).
    zmap = zipdepth_frame(zip_sess, clean) if (zip_sess is not None and dmap is None) else None
    for d in dets:
        m = d.get("dist") or distance_of(d, dmap, (H, W))
        if m is None and zmap is not None:
            m = relative_depth(zmap, [v / scale for v in d["xyxy"]],
                               (H / scale, W / scale))
        d["_m"] = m
    haz = _pick_cards([d for d in dets if d["cls"] in HAZARD], clean, (W, H), scale, bands)

    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    big_first = sorted(dets, key=lambda d: -(d["xyxy"][2]-d["xyxy"][0])*(d["xyxy"][3]-d["xyxy"][1]))
    for d in big_first:
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        c = box_colour(d)
        dr.rectangle([x1, y1, x2, y2], outline=c, width=3 if d["cls"] in HAZARD else 1)
    for d in big_first:
        x1, y1 = int(d["xyxy"][0]), int(d["xyxy"][1])
        c = box_colour(d)
        name, faint = shown_name(d)
        # **distance, not confidence.** Confidence lives on the card.
        text = "%s  %s" % (name, fmt_distance(d["_m"]))
        tw = pen.w(text, 18)
        ly = y1 - 26 if y1 > 28 else y1 + 2
        dr.rectangle([x1, ly, x1 + tw + 11, ly + 25], fill=(55, 55, 55) if faint else c)
        pen.t(dr, (x1 + 5, ly + 2), text, 18, (232, 232, 232) if faint else (10, 12, 16))

    taken = []
    for rank, d in haz:
        _draw_one_card(pil, dr, pen, d, clean, (W, H), rank, taken, scale, bands)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _card_score(d, clean, wh, scale, bands):
    """그릴지 말지, 누가 위로 갈지를 정하려면 **먼저 점수를 알아야 한다.**"""
    W, H = wh
    x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
    crop = clean[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    sc, _rows, _src = grounds(dict(d, xyxy=[v / scale for v in d["xyxy"]]),
                              W / scale, H / scale, bands, d.get("_m"),
                              colour=colour_cue(crop))
    return sc if sc is not None else -1.0


def _pick_cards(haz, clean, wh, scale, bands):
    """한 장에 **점수 높은 것부터 최대 N 개**. 카드가 많으면 장면이 사라진다.

    돌려주는 차례는 **낮은 점수 → 높은 점수** 다. 겹칠 때 나중에 그린 것이 위로
    가므로 가장 높은 카드가 맨 위에 온다.
    """
    cap = int(card_opt("max_cards", 5))
    ranked = sorted(((_card_score(d, clean, wh, scale, bands), i, d)
                     for i, d in enumerate(haz)), key=lambda t: (-t[0], t[1]))
    top = ranked[:max(0, cap)]
    # 번호는 **점수 차례**(1 등이 가장 높다), 그리는 차례는 **낮은 점수부터**
    # (나중에 그린 것이 위로 올라오므로 1 등이 맨 위에 남는다)
    return [(rank, d) for rank, (_s, _i, d) in reversed(list(enumerate(top, 1)))]


def _draw_one_card(pil, dr, pen, d, clean, wh, n, taken, scale=1.0, bands=None,
                   colour_of=None):
    """카드 하나. `render()` 와 `add_cards()` 가 **같은 코드로** 그린다 —
    운영 화면과 검수 화면이 달라지지 않게 하는 유일한 방법이다.
    """
    W, H = wh
    x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
    # **테두리는 박스와 같은 색.** 두 색이 다르면 어느 카드가 어느 상자인지
    # 지시선을 따라가야 알 수 있다
    c = colour_of(d) if colour_of else box_colour(d)
    pad = max(5, int(0.08 * max(x2 - x1, y2 - y1)))
    crop = clean[max(0, y1-pad):min(H, y2+pad), max(0, x1-pad):min(W, x2+pad)]
    if crop.size == 0:
        return
    # **작아서 눈으로 못 보는 것만** 확대해 붙인다. 원본에서 이미 보이는 물체에
    # 썸네일을 또 넣으면 카드가 자리만 먹고, 카드가 커지면 가리는 장면도 넓어진다.
    seen_ratio = ((x2 - x1) * (y2 - y1)) / float(W * H)
    want_thumb = seen_ratio < float(card_opt("thumbnail_below_frame_ratio", 0.02))
    if want_thumb:
        tw_ = THUMB_W
        th_ = max(28, min(int(crop.shape[0] * tw_ / crop.shape[1]), 140))
        crop = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (tw_, th_),
                          interpolation=cv2.INTER_CUBIC)
        card_w, cw = CARD_W, CARD_W - THUMB_W - 3 * PAD
    else:
        crop, th_ = None, 0
        card_w = cw = CARD_W - THUMB_W - PAD
    # grounds must read the **original** pixels. Measuring on an enlarged canvas
    # would print a size the camera never saw.
    score, rows, src = grounds(dict(d, xyxy=[v / scale for v in d["xyxy"]]),
                               W / scale, H / scale, bands, d["_m"],
                               colour=colour_cue(clean[max(0, y1):y2, max(0, x1):x2]))
    # 출처는 **모델별로 묶는다** — 같은 모델이 낸 단서를 한 묶음으로 읽는다
    # **단서(출처모델)** 순서. 읽는 사람은 단서를 먼저 찾는다
    rows = rows + [("source", None,
                    ", ".join("%s(%s)" % (cue, who) for cue, who in src.items()),
                    False, 1.0)]
    lines = []
    BODY_X, BODY_X_BARE = 248, 100      # 막대가 있는 줄 / 없는 줄의 글자 시작
    for key, pct, body, warn, _conf in rows:
        room = cw - (BODY_X if pct is not None else BODY_X_BARE) - 14
        words, line = body.split(" "), ""
        wrapped = []
        for word in words:
            trial = (line + " " + word).strip()
            if line and pen.w(trial, 17) > room:
                wrapped.append(line); line = word
            else:
                line = trial
        if line:
            wrapped.append(line)
        lines.append((key, pct, wrapped[0], warn))
        # 이어지는 줄은 **첫 줄과 같은 자리**에서 시작한다
        lines += [(None, pct, extra, warn) for extra in wrapped[1:]]
    chh = max(PAD + 30 + th_ + PAD, PAD + 30 + len(lines) * LINE + PAD)
    cx, cy = place((x1, y1, x2, y2), (card_w, chh), W, H, taken)
    taken.append((cx, cy, card_w, chh))
    dr.line([(x1 + (x2-x1)//2, y1 + (y2-y1)//2), (cx + card_w//2, cy + chh//2)],
            fill=c, width=1)
    dr.rectangle([cx, cy, cx + card_w, cy + chh], fill=(18, 20, 24), outline=c, width=2)
    head = "%d  %s" % (n, SHOWN.get(d["cls"], d["cls"]))
    pen.t(dr, (cx + PAD, cy + PAD - 3), head, 21, c)
    if score is not None:
        # **맨 위에 평균부터.** 아래 줄들이 그 평균이 어디서 왔는지를 말한다
        pen.t(dr, (cx + PAD + pen.w(head, 21) + 10, cy + PAD - 3),
              "score %d/100" % round(score), 21,
              SCORE_HIGH if score >= 60 else
              (SCORE_MID if score >= 40 else SCORE_LOW))
    if crop is not None:
        pil.paste(Image.fromarray(crop), (cx + PAD, cy + PAD + 28))
    tx = cx + (THUMB_W + 2 * PAD if crop is not None else PAD)
    y = cy + PAD + 28
    for key, pct, body, warn in lines:
        # **세부 단서는 모두 흰색.** 색으로 미리 판정해 주지 않는다 — 어느 줄이
        # 약한지는 퍼센트가 말하고, 색이 또 말하면 두 번 말하는 것이다
        if key:
            pen.t(dr, (tx, y), key, 17, (255, 255, 255))
        if pct is not None:
            # **막대로 보여 준다.** 숫자는 정확하고 막대는 한눈에 들어온다 —
            # 카드가 여럿일 때 어느 단서가 약한지는 눈이 먼저 찾는다
            bx, by, bw, bh = tx + 100, y + 6, 78, 11
            dr.rectangle([bx, by, bx + bw, by + bh], fill=(58, 62, 70))
            fill = int(bw * min(1.0, max(0.0, pct / 100.0)))
            if fill > 0:
                dr.rectangle([bx, by, bx + fill, by + bh], fill=(255, 255, 255))
            if key:
                pen.t(dr, (bx + bw + 10, y), "%3d%%" % round(pct), 17, (255, 255, 255))
        # 부가 설명은 **회색** — 값이 아니라 그 값이 어디서 왔는지의 설명이다
        pen.t(dr, (tx + (BODY_X if pct is not None else BODY_X_BARE), y),
              body, 17, (160, 168, 180))
        y += LINE





# ---------------------------------------------------------------- vision 진입점
def add_cards(canvas, clean_bgr, dets, cues=None, font=None, colour_of=None, _state={}):
    """이미 박스가 그려진 `canvas` 위에 **카드만** 올린다.

    `canvas` 는 머리 띠가 붙기 전이어야 좌표가 맞는다. 위험물 후보가 없으면 아무것도
    하지 않고 그대로 돌려준다 — 비용 0.

    펜과 선언은 **한 번만** 세운다: 장마다 글꼴을 다시 읽으면 그것이 카드를 그리는
    비용보다 커진다.
    """
    # 카드 안에서 `cls` 는 **이름**이고 vision 의 검출에서는 **정수**다. 색을 고르는
    # 쪽은 정수를 쓰므로 원래 값을 `_cls_id` 로 같이 들고 간다 — 한쪽에 맞추려고
    # 다른 쪽을 바꾸면 두 자리에서 같은 칸이 다른 뜻이 된다.
    haz = [dict(d, cls=(d.get("name") or d.get("cls")), _cls_id=d.get("cls"),
                _m=d.get("dist"))
           for d in (dets or []) if (d.get("name") or d.get("cls")) in HAZARD]
    if not haz:
        return canvas
    if "pen" not in _state:
        load_cues(cues)
        _state["pen"] = Pen(Path(font) if font else find_font())
    pen = _state["pen"]
    H, W = clean_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    taken = []
    for rank, det in _pick_cards(haz, clean_bgr, (W, H), 1.0, None):
        _draw_one_card(pil, dr, pen, det, clean_bgr, (W, H), rank, taken,
                       colour_of=colour_of)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
