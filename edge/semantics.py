"""엣지 의미 계층 — 후처리·합의·지속 객체 레코드·특징 사전.

implements: AI-S-03, AI-S-04, AI-S-06, AI-E-04, AI-C-04, AI-C-15, AI-L-01, AI-L-02

**한 파일에 모은 이유**: 넷 다 "검출 결과를 어떻게 믿을 것인가"라는 같은 질문을
다른 각도에서 다룬다 — 기하로(postprocess), 모델 간 합의로(consensus), 시간
누적으로(object_tracks), 특징 사전으로(FeatureDictionary). 따로 두면 임계값이
어느 축의 것인지 추적하기 어려워진다.

여기에 없는 것: 모델 실행(models.py), 저장(store.py), 통신(server.py).
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests


# ─────────────────────────────────────────────────────────────────────────────
# 기하 — IoU, 박스 확장, known/unknown 1차 분리
# ─────────────────────────────────────────────────────────────────────────────
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def expand_box(box, context: float, w: int, h: int):
    """crop용 컨텍스트 확대. 1.5배는 zeroshot_crops.py 실측과 같은 값이다."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw = max(1.0, x2 - x1) * context
    bh = max(1.0, y2 - y1) * context
    return (max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2)),
            min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2)))


def split_known_unknown(rpn_items, ovd_dets, match_iou: float):
    """RPN 후보를 OVD가 설명한 것 / 못 한 것으로 가른다.

    `rpn_items`는 {box, objectness, track_id?} dict 목록이다. box/score만 받으면
    **track_id가 유실된다** — 2026-09-21에 실제로 그 버그로 지연 반영이 0건이었다.
    말단이 판정을 현재 트랙에 붙이려면 이 id가 끝까지 살아 있어야 한다(records.py).

    "OVD에 안 잡힘"은 **미지 객체라는 증거가 아니라 미지 후보**다(AI-S-04).
    OVD 어휘에 없을 뿐인 흔한 물체도, 그냥 배경 오탐도 전부 여기 들어온다 —
    실제로 RPN precision@50은 0.19~0.25로 측정돼 있어 이 집합의 다수는 오탐이다.
    """
    ovd_boxes = [d["box"] for d in ovd_dets]
    known, unknown = [], []
    for item in rpn_items:
        b, s = item["box"], item.get("objectness", 1.0)
        tid = item.get("track_id")
        best_i, best_v = -1, 0.0
        for i, ob in enumerate(ovd_boxes):
            v = iou(b, ob)
            if v > best_v:
                best_i, best_v = i, v
        if best_v >= match_iou:
            known.append({"box": b, "objectness": s, "track_id": tid,
                          "ovd_index": best_i, "iou": best_v})
        else:
            unknown.append({"box": b, "objectness": s, "track_id": tid,
                            "best_iou": best_v})
    return known, unknown


# ─────────────────────────────────────────────────────────────────────────────
# 후처리 — 조각 흡수·병합 판정. IoU만으로는 두 경우가 미지로 샌다.
# ─────────────────────────────────────────────────────────────────────────────
def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def containment(inner, outer) -> float:
    """inner 면적 중 outer 안에 들어간 비율. 조각 판정의 축이다."""
    a = _area(inner)
    return (_inter(inner, outer) / a) if a > 0 else 0.0


def union_box(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def absorb(rpn_items, ovd_dets, cfg_unknown: dict):
    """RPN 후보를 `known`(OVD가 **확신을 갖고** 설명함)과 `unknown`으로 가른다.

    **확신 OVD만 흡수한다**(사용자 2026-09-21). OVD가 낮은 conf로 이름을 붙인 것은
    "배경이 아닌데 뭔지 모르는 것"이지 설명된 것이 아니다 — 미식별과 같은 범주이고,
    다만 OVD 쪽 근거가 하나 더 붙어 있을 뿐이다. 그래서 잠정 OVD와 겹친 후보는
    unknown으로 보내되 그 라벨을 `ovd_hint`로 실어 보낸다. 그러면 CLIP 근거까지
    붙어 **두 근거를 나란히** 관리자에게 보여줄 수 있다(AI-S-03, AI-L-03).

    반환: (known, unknown, stats). stats는 어느 규칙이 몇 개를 흡수했는지 — 규칙을
    켜고 끌 때 효과를 바로 볼 수 있어야 한다(AI-O-01).
    """
    match_iou = float(cfg_unknown.get("match_iou", 0.3))
    frag_c = float(cfg_unknown.get("fragment_containment", 0.70))
    # **큰 검출은 조각을 삼키지 못한다.** 2026-09-21 실측: 화면 88%를 덮는
    # `water surface` 검출이 후보 298개를 fragment로 흡수했고, 다음 프레임에서 그
    # conf가 0.35를 0.019 차이로 못 넘자 흡수가 102개로 무너지며 미지가 3배로 튀었다
    # (frame 002→003). 화면의 상당 부분을 덮는 박스는 '객체'가 아니라 장면이며,
    # 그 안에 든 것을 그 객체의 조각으로 볼 근거가 없다.
    # 부유목 더미처럼 **검출 대상이면서 큰** 객체에도 같은 보호가 필요하다 —
    # 더미 위에 얹힌 페트병이 더미의 조각으로 흡수되면 안 된다(사용자 2026-09-21).
    max_abs = float(cfg_unknown.get("max_absorber_area_frac", 0.80))
    min_ratio = float(cfg_unknown.get("min_fragment_ratio", 0.05))
    frame_area = float(cfg_unknown.get("_frame_area", 0.0))
    merge_cov = float(cfg_unknown.get("merged_coverage", 0.55))
    merge_min = int(cfg_unknown.get("merged_min_dets", 2))

    # 흡수 대상은 확신 검출뿐이다. verdict가 없으면(구버전 호출) 전부 대상으로 본다.
    def _can_absorb(d) -> bool:
        """흡수 자격: 확신이고, 프레임을 통째로 덮지 않는 검출.

        `max_absorber_area_frac`는 '장면을 덮는 박스'만 걸러낸다(수면 88%). 부유목
        더미(16~45%)처럼 **검출 대상이면서 큰** 객체는 흡수 자격을 유지해야 한다 —
        그 조각까지 미식별로 보내면 관리자가 볼 것이 조각으로 뒤덮인다.
        더미 안의 작은 쓰레기를 지키는 일은 아래 `min_fragment_ratio`가 맡는다.
        """
        if d.get("verdict", "confident") != "confident":
            return False
        if frame_area <= 0 or max_abs >= 1.0:
            return True
        return (_area(d["box"]) / frame_area) <= max_abs

    conf_idx = [i for i, d in enumerate(ovd_dets) if _can_absorb(d)]
    ovd_boxes = [ovd_dets[i]["box"] for i in conf_idx]
    tent_idx = [i for i, d in enumerate(ovd_dets)
                if d.get("verdict", "confident") == "tentative"]

    known, unknown = [], []
    stats = {"match": 0, "fragment": 0, "merged": 0, "unknown": 0, "with_ovd_hint": 0,
             "absorbers": len(conf_idx),
             "too_large_to_absorb": sum(
                 1 for d in ovd_dets
                 if d.get("verdict", "confident") == "confident" and not _can_absorb(d))}

    for item in rpn_items:
        b = item["box"]
        s = item.get("objectness", 1.0)
        tid = item.get("track_id")
        base = {"box": b, "objectness": s, "track_id": tid}

        # 1) 기존 매칭 — 같은 것을 봤다
        best_i, best_v = -1, 0.0
        for i, ob in enumerate(ovd_boxes):
            v = iou(b, ob)
            if v > best_v:
                best_i, best_v = i, v
        if best_v >= match_iou:
            known.append({**base, "ovd_index": conf_idx[best_i], "iou": best_v,
                          "absorbed_by": "match"})
            stats["match"] += 1
            continue

        # 2) 조각 — 후보 면적의 대부분이 한 OVD 박스 안에 있다
        c_i, c_v = -1, 0.0
        for i, ob in enumerate(ovd_boxes):
            v = containment(b, ob)
            if v > c_v:
                c_i, c_v = i, v
        # **조각이려면 그 객체의 상당 부분이어야 한다.** 2026-09-21 실측: 하천에서
        # 미식별 후보의 면적 중앙값은 프레임의 0.347%인데 확신 박스 중앙값은 16.4%다
        # — 작은 쓰레기는 큰 박스의 **2.1%** 크기다. containment만 보면 부유목 더미
        # 위에 얹힌 페트병도 "더미 안에 있으니 더미의 조각"이 되어 삼켜진다.
        # 상대 크기 하한을 두면 조각(더미의 10~30%)은 흡수되고 쓰레기(2%)는 살아남는다
        # (사용자 2026-09-21: "부유목 안으로 다른 클래스가 흡수되지 않도록").
        if c_v >= frag_c and c_i >= 0:
            ratio = _area(b) / max(1e-9, _area(ovd_boxes[c_i]))
            if ratio < min_ratio:
                c_v = 0.0                      # 조각으로 보지 않는다
                stats["kept_small_inside"] = stats.get("kept_small_inside", 0) + 1
        if c_v >= frag_c:
            # **iou는 기록된 짝의 IoU여야 한다.** 여기서 고르는 것은 containment
            # 승자 c_i인데 예전에는 best_v(다른 검출의 IoU)를 적었다. 그 값이
            # 0.849처럼 높게 찍혀 있어 짝이 맞는 것처럼 보였지만 실제로는 두 박스가
            # 거의 겹치지 않았다 — 조용한 거짓말이었다(2026-09-21).
            known.append({**base, "ovd_index": conf_idx[c_i],
                          "iou": round(iou(b, ovd_boxes[c_i]), 3),
                          "containment": round(c_v, 3), "absorbed_by": "fragment"})
            stats["fragment"] += 1
            continue

        # 3) 병합 — 후보가 OVD 박스 여러 개를 상당 부분 덮는다.
        #    각 OVD 박스가 후보 안에 얼마나 들어왔는지로 센다(후보가 크므로
        #    반대 방향 containment를 본다).
        inside = [i for i, ob in enumerate(ovd_boxes)
                  if containment(ob, b) >= merge_cov]
        if len(inside) >= merge_min:
            known.append({**base, "ovd_index": conf_idx[inside[0]],
                          "iou": round(iou(b, ovd_boxes[inside[0]]), 3),
                          "covers_ovd": [conf_idx[i] for i in inside],
                          "absorbed_by": "merged"})
            stats["merged"] += 1
            continue

        # 잠정 OVD와 겹치면 그 라벨을 근거로 싣는다 — 버리지 않는다.
        hint, hv = None, 0.0
        for i in tent_idx:
            v = max(iou(b, ovd_dets[i]["box"]),
                    containment(b, ovd_dets[i]["box"]))
            if v > hv:
                hint, hv = ovd_dets[i], v
        item_out = {**base, "best_iou": best_v, "best_containment": round(c_v, 3)}
        if hint is not None and hv >= match_iou:
            item_out["ovd_hint"] = {"label": hint["label"],
                                    "conf": round(hint["conf"], 3),
                                    "overlap": round(hv, 3)}
            stats["with_ovd_hint"] += 1
        unknown.append(item_out)

    stats["unknown"] = len(unknown)
    return known, unknown, stats


def merge_overlapping(unknown, iou_thresh: float):
    """남은 미지 후보끼리 겹치는 것을 합친다.

    RPN은 같은 대상에 박스를 여러 개 낸다(NMS를 통과해도 남는다). 안 합치면 같은
    것이 crop 여러 장으로 저장되고 오버레이가 박스로 덮인다.

    합칠 때 박스는 **합집합**을 쓰고 objectness는 **최대값**을 쓴다. track_id는
    가장 objectness가 높은 구성원 것을 남긴다 — 말단이 붙일 앵커는 하나여야 한다.
    `merged_from`에 몇 개가 합쳐졌는지 남긴다.
    """
    if iou_thresh <= 0 or len(unknown) < 2:
        return unknown, 0

    order = sorted(range(len(unknown)),
                   key=lambda i: -unknown[i].get("objectness", 0.0))
    used = [False] * len(unknown)
    out = []
    merged_away = 0

    for oi in order:
        if used[oi]:
            continue
        group = [oi]
        used[oi] = True
        for oj in order:
            if used[oj]:
                continue
            if iou(unknown[oi]["box"], unknown[oj]["box"]) >= iou_thresh:
                group.append(oj)
                used[oj] = True
        if len(group) == 1:
            out.append(unknown[oi])
            continue
        members = [unknown[i] for i in group]
        lead = members[0]                      # objectness 최대 (order가 정렬돼 있다)
        out.append({**lead,
                    "box": union_box([m["box"] for m in members]),
                    "objectness": max(m.get("objectness", 0.0) for m in members),
                    "merged_from": len(members)})
        merged_away += len(members) - 1
    return out, merged_away


# ─────────────────────────────────────────────────────────────────────────────
# 합의 — 두 OVD가 같은 자리에서 같은 것을 말하면 단일 conf와 질이 다른 근거다.
# ─────────────────────────────────────────────────────────────────────────────
# 라벨 토큰 비교에서 뺄 말들. 모델마다 붙이는 수식어가 달라서("computer chair"
# vs "office chair") 이걸 안 빼면 같은 것을 말해도 합의로 안 잡힌다.
STOPWORDS = {"a", "an", "the", "of", "with", "and", "computer", "office",
             "electronic", "small", "large", "indoor", "outdoor"}


def _tokens(label: str) -> set[str]:
    return {t for t in str(label).lower().replace("-", " ").replace("_", " ").split()
            if t and t not in STOPWORDS}


def concept_key(label: str, synonyms: dict | None = None) -> str:
    """라벨을 **비교 가능한 개념 키**로 정규화한다. 합의 판정의 유일한 규칙이다.

    합의를 말단에서 판정하려면 말단이 라벨 비교 규칙을 알아야 하는데, 같은 규칙을
    두 곳에 두면 반드시 어긋난다(이 저장소에서 `background_classes`가 실제로 그렇게
    어긋난 적이 있다). 그래서 **규칙은 여기 한 곳에만** 두고 말단에는 정규화된 키만
    보낸다. 말단은 문자열이 같은지만 본다.

    키는 **머리 명사**다. 영어 복합명사는 뒤가 머리이므로 마지막 내용 토큰을 쓴다:

        computer chair / office chair      → chair    (같다)
        storage box / cardboard box        → box      (같다)
        computer monitor                   → monitor  (chair와 다르다)
        window blind                       → blind

    수식어(`computer`, `office`...)는 STOPWORDS로 빠지므로 `computer chair`와
    `computer monitor`가 수식어만으로 묶이는 일이 없다. 머리 명사로 못 잡는
    짝(`couch`/`sofa`)은 설정의 `synonyms`에 명시한다 — 규칙을 코드에 늘리지 않는다.
    """
    la = str(label).lower().strip()
    for name, group in (synonyms or {}).items():
        if isinstance(group, list) and la in {str(x).lower() for x in group}:
            return f"syn:{name}"
    toks = [t for t in la.replace("-", " ").replace("_", " ").split()
            if t and t not in STOPWORDS]
    return toks[-1] if toks else la


def labels_agree(a: str, b: str, synonyms: dict | None = None) -> bool:
    """두 라벨이 같은 것을 가리키는가 — `concept_key`가 같은가로 정의한다.

    규칙이 하나뿐이어야 엣지의 판정과 말단의 판정이 갈라지지 않는다.
    """
    return concept_key(a, synonyms) == concept_key(b, synonyms)


def merge_by_consensus(primary, secondary, cfg_ovd: dict):
    """두 검출 목록을 합쳐 `verdict`를 정한다.

    primary   = prompt-free(도메인 무관). secondary = 텍스트 프롬프트(도메인 어휘).
    반환: (검출 목록, 통계). 각 검출은
        verdict   "confident"(합의) | "tentative"(단독)
        sources   기여한 모델 이름들
        agree_iou 합의한 두 박스의 IoU (합의일 때만)
        alt       상대 모델이 말한 라벨·conf (합의일 때만)

    박스는 **conf가 높은 쪽**을 쓴다. 두 기하를 평균하면 어느 모델도 내놓지 않은
    좌표가 만들어지고, 그것은 관측이 아니다(AI-S-03).
    """
    thr = float(cfg_ovd.get("agree_iou", 0.5))
    syn = cfg_ovd.get("synonyms") or {}
    floor = float(cfg_ovd.get("agree_min_conf", 0.0))

    used_b: set[int] = set()
    out = []
    n_agree = 0
    for da in primary:
        best_j, best_v = -1, thr
        for j, db in enumerate(secondary):
            if j in used_b:
                continue
            v = iou(da["box"], db["box"])
            if v >= best_v and labels_agree(da["label"], db["label"], syn):
                best_j, best_v = j, v
        if best_j >= 0:
            db = secondary[best_j]
            used_b.add(best_j)
            win, lose = (da, db) if da["conf"] >= db["conf"] else (db, da)
            if min(da["conf"], db["conf"]) >= floor:
                n_agree += 1
                out.append({**win, "verdict": "confident",
                            "sources": ["prompt_free", "text_prompt"],
                            "agree_iou": round(best_v, 3),
                            "alt": {"label": lose["label"],
                                    "conf": round(lose["conf"], 3)}})
                continue
        out.append({**da, "verdict": "tentative", "sources": ["prompt_free"]})

    for j, db in enumerate(secondary):
        if j not in used_b:
            out.append({**db, "verdict": "tentative", "sources": ["text_prompt"]})

    stats = {"primary": len(primary), "secondary": len(secondary),
             "agreed": n_agree,
             "tentative": sum(1 for d in out if d["verdict"] == "tentative")}
    return out, stats


# ─────────────────────────────────────────────────────────────────────────────
# 지속 객체 레코드 — 근거를 시간에 걸쳐 누적해 상태를 해소한다.
# ─────────────────────────────────────────────────────────────────────────────
# progressive-object-record 논문 §3의 기본값. **config가 있으면 config가 이긴다** —
# 이 값들은 그쪽 검출기 분포 기준이라 이 구성에 그대로 맞지 않는다(예: INIT_CONF
# 0.30은 여기 실측 p50 0.156을 74% 잘라내 클러스터 11개 중 8개가 객체가 못 됐다,
# 2026-09-21). 남겨 둔 이유는 config 키가 빠졌을 때의 바닥값이 필요해서다.
ASSOC_IOU = 0.30          # 관측 내 근거 클러스터링
TRACK_IOU = 0.20          # 관측 간 지속 객체 매칭
INIT_CONF = 0.30          # 새 객체 초기화 최소 confidence
STALE_MISSES = 3
EXPIRE_MISSES = 5
CONFIRM_GROUPS = 2
# **이 값은 모델 수에 묶여 있다.** "두 소스 그룹의 지지"는 OVD가 둘일 때의 조건
# 이고, 하나만 켜면 어떤 객체도 영원히 확정되지 않는다 — 2026-09-21에 ovd2를 끄자
# 실제로 장애물이 0개가 됐고 화면이 비었다. 그래서 config(object_tracks.
# confirm_groups)로 뺐고, 없으면 켜진 OVD 수를 따른다(ObjectTracks.__init__).


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


@dataclass
class Evidence:
    """근거 사건 e = (id, o, s, t, k, c, p) — 논문 식 1."""

    obs: int                 # 관측 식별자 o
    source: str              # 소스 그룹 s
    at: float                # 완료 시각 t
    kind: str                # 근거 종류 k ("box")
    conf: float              # confidence c
    box: tuple               # 페이로드 p
    label: str = ""
    concept: str = ""


@dataclass
class ObjectRecord:
    """지속 객체 레코드 R_j — 논문 식 12."""

    obj_id: int
    box: tuple
    first_obs: int
    last_obs: int
    misses: int = 0
    lifecycle: str = "provisional"
    concept: str = ""            # 확립된 의미 클래스(개념 키)
    label: str = ""              # 그 클래스의 대표 라벨
    conf: float = 0.0
    class_groups: int = 0        # 현재 클래스를 지지하는 소스 그룹 수 |S(c)|
    groups_seen: set = field(default_factory=set)
    alt: dict | None = None      # 상대 소스가 말한 것(관리자 판단용)
    revision: int = 0            # 노출 갱신 번호 r
    # **노출 기하.** 해소된 박스(`box`)와 따로 둔다 — V가 바뀌지 않으면 노출은
    # 그대로여야 하기 때문이다. 이것을 분리하지 않으면 갱신 번호만 안 올라가고
    # 화면의 박스는 매 프레임 떨린다(양자화가 무의미해진다).
    exposed_box: tuple = (0.0, 0.0, 0.0, 0.0)
    # 확정에 필요한 소스 그룹 수. ObjectTracks가 켜진 OVD 수에 맞춰 넣는다.
    confirm_groups: int = CONFIRM_GROUPS

    def visible_semantic(self) -> tuple:
        """노출 상태 V(R)의 비기하 부분 — 논문 식 13. confidence는 **들어가지 않는다.**"""
        return (self.lifecycle, self.concept)

    def geometry_moved(self, resolved, quantum: float) -> bool:
        """노출 기하를 갱신할 만큼 움직였는가 — **불감대**로 판정한다.

        논문은 월드 좌표를 0.02 m로 양자화해 V를 만든다. 그 방식을 영상 좌표에
        그대로 쓰면 값이 **양자 경계에 걸쳤을 때** 1~2 px 흔들림에도 매 프레임
        반올림이 뒤집힌다(2026-09-21 실측: 1 px 지터에 갱신 3회). 경계는 고정된
        격자가 아니라 **현재 노출값**을 기준으로 잡아야 한다. 노출값에서 quantum
        이상 벗어날 때만 갱신하면 경계 문제가 원천적으로 없어진다.
        """
        return max(abs(resolved[i] - self.exposed_box[i]) for i in range(4)) > quantum

    def confirmed_ready(self) -> bool:
        """화면에 확정으로 올릴 수 있는가 — lifecycle과 **클래스 지지**를 모두 본다.

        논문의 lifecycle confirmed는 '객체가 두 소스 그룹의 지지를 받았다'이고
        클래스 일치까지 요구하지는 않는다. 이 배치의 요구는 "두 모델이 합의할 때만
        확정"이므로(사용자 2026-09-21) 클래스 지지 조건을 덧붙인다.
        """
        return (self.lifecycle == "confirmed"
                and self.class_groups >= self.confirm_groups)

    def to_dict(self) -> dict:
        return {"obj_id": self.obj_id, "box": [round(v, 1) for v in self.exposed_box],
                "label": self.label, "conf": round(self.conf, 3),
                "concept": self.concept, "alt": self.alt,
                "lifecycle": self.lifecycle,
                "confirmed": self.confirmed_ready(),
                "groups": self.class_groups, "revision": self.revision,
                "seen_obs": self.last_obs - self.first_obs + 1}


def _smooth_box(prev, cur, *, alpha: float, snap_px: float):
    """박스 떨림을 줄이되 **실제 이동은 그대로 따라간다.**

    지수 평활만 쓰면 빠르게 움직이는 대상에서 박스가 뒤처진다. 변화가
    snap_px를 넘으면 계수를 1로 올려 그 프레임은 새 값을 그대로 쓴다 —
    "떨림은 억제, 이동은 추종"이 한 줄로 갈린다.
    """
    if not prev or alpha >= 1.0:
        return tuple(cur)
    d = max(abs(cur[i] - prev[i]) for i in range(4))
    a = 1.0 if d >= snap_px else alpha
    return tuple(prev[i] * (1 - a) + cur[i] * a for i in range(4))


class ObjectTracks:
    """카메라 하나의 지속 객체 레코드 집합."""

    def __init__(self, cfg: dict):
        c = cfg.get("object_tracks", {})
        self.assoc_iou = float(c.get("assoc_iou", ASSOC_IOU))
        self.track_iou = float(c.get("track_iou", TRACK_IOU))
        self.init_conf = float(c.get("init_conf", INIT_CONF))
        self.stale_misses = int(c.get("stale_misses", STALE_MISSES))
        self.expire_misses = int(c.get("expire_misses", EXPIRE_MISSES))
        self.quantum = float(c.get("geometry_quantum_px", 8.0))
        # 해소된 기하를 레코드에 반영할 때의 평활 계수. 1이면 평활 없음.
        # **왜 필요한가**: 기하는 가용 박스의 confidence 가중 평균이라, 한 모델이
        # 한 관측에서 빠지면 '두 박스의 중간'에서 '한 박스'로 확 옮겨간다. 그러면
        # 관측 간 매칭(IoU 0.20)이 끊겨 새 객체가 생기고 누적 지지가 리셋된다 —
        # 실측에서 확정 객체 수가 중앙 0으로 떨어진 원인이 이것이었다.
        self.geom_alpha = float(c.get("geometry_alpha", 0.5))
        # **확정에 필요한 소스 그룹 수.** 기본 2는 OVD가 둘일 때의 조건이고,
        # 하나만 켜면 어떤 객체도 영원히 확정되지 않는다(2026-09-21에 ovd2를 끄자
        # 장애물이 0개가 되어 화면이 비었다). config에 없으면 호출부가 켜진 OVD
        # 수를 넣어 주고, 그것도 없으면 1로 내려간다 — 결손은 실패가 아니라
        # 축소다(AI-C-05).
        self.confirm_groups = max(1, int(c.get("confirm_groups", CONFIRM_GROUPS)))
        # 노출 박스 평활화. alpha=1이면 평활 없음.
        self.expose_alpha = float(c.get("expose_alpha", 0.4))
        self.expose_snap_px = float(c.get("expose_snap_px", 24.0))
        self.objects: dict[int, ObjectRecord] = {}
        self.last_stats: dict = {}
        self._next = 1
        self._obs = 0

    # -- 관측 내 근거 클러스터링 -------------------------------------------
    def _cluster(self, evs: list[Evidence]) -> list[list[Evidence]]:
        """같은 물리 객체를 가리키는 근거를 모은다(논문 §3, 임계 0.30).

        confidence가 높은 근거부터 클러스터의 **씨앗**이 되고, 이후 근거는 씨앗과
        직접 겹칠 때만 그 클러스터에 들어간다.

        **멤버 아무거나와 비교하면 안 된다**(single-link). 그러면 A-B가 겹치고
        B-C가 겹칠 때 A와 C가 전혀 겹치지 않아도 한 덩어리가 된다. 2026-09-21
        실측에서 사무실 장면의 검출 56개가 이 연쇄로 **객체 3개**가 됐고, 기하는
        전부의 평균이라 화면을 덮는 박스가 나왔으며 소스 그룹 지지도 뭉개졌다.
        씨앗 기준으로 묶으면 각 클러스터가 하나의 물체 주변에 머문다.
        """
        clusters: list[list[Evidence]] = []
        for e in sorted(evs, key=lambda x: -x.conf):
            for cl in clusters:
                if _iou(e.box, cl[0].box) >= self.assoc_iou:
                    cl.append(e)
                    break
            else:
                clusters.append([e])
        return clusters

    # -- 상태 해소 ---------------------------------------------------------
    @staticmethod
    def _representatives(cl: list[Evidence]) -> dict[str, Evidence]:
        """소스 그룹별 대표 가설 h_{j,g} — 그룹 내 최고 confidence(논문 §3.1)."""
        rep: dict[str, Evidence] = {}
        for e in cl:
            if e.source not in rep or e.conf > rep[e.source].conf:
                rep[e.source] = e
        return rep

    def _resolve_semantics(self, rec: ObjectRecord, rep: dict[str, Evidence]) -> None:
        """의미 상태를 해소한다 — 논문 식 7~8 + 클래스 교체 이력 규칙.

        **핵심**: 이미 클래스가 확립돼 있으면 새 클래스는 **엄격히 더 많은** 소스
        그룹의 지지를 받을 때만 대체한다. 동수면 기존 클래스를 유지한다. 이 한 줄이
        "장면이 그대로인데 이름이 프레임마다 뒤집히는" 현상을 막는다.
        """
        support: dict[str, list[Evidence]] = {}
        for e in rep.values():
            if e.concept:
                support.setdefault(e.concept, []).append(e)
        if not support:
            return                      # 이름 붙은 근거가 없으면 unknown 유지

        def rank(item):
            c, evs = item
            return (len(evs), sum(x.conf for x in evs), c)   # 그룹 수 → conf 합 → 결정적 순서

        best_c, best_evs = max(support.items(), key=rank)
        n_best = len(best_evs)

        if rec.concept and best_c != rec.concept:
            # 비교 대상은 **현재 관측의 지지**가 아니라 그 클래스가 **확립된 지지
            # 수준**이다. 이번 프레임에 기존 클래스를 아무도 말하지 않았다고 해서
            # 단일 소스 주장에 자리를 내주면 이력이 성립하지 않는다 — 실측에서
            # 2그룹으로 확립된 `chair`가 한 모델의 `person` 한 번에 뒤집혔다.
            cur = max(len(support.get(rec.concept, [])), rec.class_groups)
            if n_best <= cur:
                keep = support.get(rec.concept)
                if keep:
                    best_c, best_evs, n_best = rec.concept, keep, len(keep)
                else:
                    return          # 기존 클래스를 그대로 둔다(이번 관측에 근거 없음)

        top = max(best_evs, key=lambda x: x.conf)
        same_class = (rec.concept == best_c)
        rec.concept, rec.label, rec.conf = best_c, top.label, top.conf
        # **확립된 지지 수준을 유지한다.** 순간값으로 덮어쓰면 한 모델이 한 프레임
        # 놓칠 때마다 class_groups가 1로 떨어져 확정이 풀린다 — 실측에서 확정
        # 집합이 정지 장면인데도 2↔3으로 깜빡였다(2026-09-21). 바로 위 비교문이
        # 이미 `max(현재, 확립)`을 쓰고 있는데 저장만 순간값이라 앞뒤가 안 맞았다.
        # 클래스가 바뀌면 새로 시작한다 — 그때는 이력을 물려받으면 안 된다.
        rec.class_groups = max(rec.class_groups, n_best) if same_class else n_best
        others = [e for e in rep.values() if e is not top and e.concept == best_c]
        rec.alt = ({"label": others[0].label, "conf": round(others[0].conf, 3)}
                   if others else None)

    @staticmethod
    def _resolve_geometry(cl: list[Evidence]) -> tuple:
        """기하 — 마스크가 없으므로 **가용 박스의 confidence 가중 평균**(논문 §3.1).

        최고 점수 박스 하나만 쓰면 어느 모델이 이겼는지에 따라 박스가 튄다.
        가중 평균은 두 모델이 같은 것을 볼 때 그 사이에 자리잡아 훨씬 덜 움직인다.
        """
        w = sum(max(1e-6, e.conf) for e in cl)
        return tuple(sum(max(1e-6, e.conf) * e.box[i] for e in cl) / w
                     for i in range(4))

    # -- 점진 구성 ---------------------------------------------------------
    def update(self, dets: list[dict], now: float) -> list[ObjectRecord]:
        """한 관측의 검출들을 받아 레코드를 갱신한다(논문 §3.2)."""
        self._obs += 1
        obs = self._obs

        evs = [Evidence(obs=obs, source=("text_prompt"
                                         if "text_prompt" in (d.get("sources") or [])
                                         and "prompt_free" not in (d.get("sources") or [])
                                         else "prompt_free"),
                        at=now, kind="box", conf=float(d["conf"]),
                        box=tuple(d["box"]), label=d.get("label", ""),
                        concept=d.get("concept", ""))
               for d in dets]
        # 합의로 묶인 검출은 두 소스가 같은 자리에서 말한 것이다 — 두 근거로 편다.
        for d in dets:
            if len(d.get("sources") or []) == 2 and d.get("alt"):
                evs.append(Evidence(obs=obs, source="text_prompt", at=now, kind="box",
                                    conf=float(d["alt"].get("conf", 0.0)),
                                    box=tuple(d["box"]),
                                    label=d["alt"].get("label", ""),
                                    concept=d.get("concept", "")))

        clusters = self._cluster(evs)

        # 관측 간 매칭 — 기존 레코드에 붙이거나 새로 만든다.
        alive = [r for r in self.objects.values() if r.lifecycle != "expired"]
        used = set()
        matched: list[tuple[ObjectRecord, list[Evidence]]] = []
        # 매칭 기준은 **노출 기하**다 — 해소된 기하는 근거 구성에 따라 튀므로
        # 그것으로 매칭하면 흔들리는 값끼리 비교하게 된다.
        pairs = sorted(((_iou(r.exposed_box, self._resolve_geometry(cl)), ri, ci)
                        for ri, r in enumerate(alive)
                        for ci, cl in enumerate(clusters)),
                       key=lambda p: -p[0])
        taken_r, taken_c = set(), set()
        for v, ri, ci in pairs:
            if v < self.track_iou or ri in taken_r or ci in taken_c:
                continue
            taken_r.add(ri)
            taken_c.add(ci)
            matched.append((alive[ri], clusters[ci]))
            used.add(ci)

        for ci, cl in enumerate(clusters):
            if ci in used:
                continue
            if max(e.conf for e in cl) < self.init_conf:
                continue                    # 논문: conf > 0.30에서만 새 객체 생성
            g = self._resolve_geometry(cl)
            rec = ObjectRecord(obj_id=self._next, box=g, exposed_box=g,
                               confirm_groups=self.confirm_groups,
                               first_obs=obs, last_obs=obs)
            self._next += 1
            self.objects[rec.obj_id] = rec
            matched.append((rec, cl))

        for rec, cl in matched:
            rep = self._representatives(cl)
            before = rec.visible_semantic()
            resolved = self._resolve_geometry(cl)
            a = self.geom_alpha
            rec.box = tuple(rec.box[i] * (1 - a) + resolved[i] * a for i in range(4))
            self._resolve_semantics(rec, rep)
            rec.groups_seen |= set(rep)
            rec.last_obs = obs
            rec.misses = 0
            if rec.lifecycle in ("provisional", "stale"):
                rec.lifecycle = ("confirmed" if len(rec.groups_seen) >= self.confirm_groups
                                 else "provisional")
            # **흔들림 보정.** 노출 박스를 새 값으로 확 바꾸지 않고 따라가게 한다.
            # 실측 떨림이 중앙 0~5px, 최대 10.8px인데 그대로 그리면 정지 장면에서도
            # 박스가 떤다. 변화가 snap_px를 넘으면 계수가 1이 되어 **실제 이동은
            # 지연 없이** 따라간다 — 움직일 때 박스가 뒤처지면 목적이 무너진다.
            moved = rec.geometry_moved(rec.box, self.quantum)
            if rec.visible_semantic() != before or moved:
                # 노출 갱신은 V가 바뀔 때만이고, 그때만 노출 기하도 새 값이 된다.
                # confidence만 달라진 것은 갱신이 아니다(논문 식 13).
                rec.revision += 1
                if moved:
                    rec.exposed_box = _smooth_box(rec.exposed_box, rec.box,
                                                  alpha=self.expose_alpha,
                                                  snap_px=self.expose_snap_px)

        seen_ids = {r.obj_id for r, _ in matched}
        for rec in list(self.objects.values()):
            if rec.obj_id in seen_ids or rec.lifecycle == "expired":
                continue
            rec.misses += 1
            # 오래 안 보이면 확립 수준도 내려간다. 올릴 때는 즉시, 내릴 때는
            # 천천히 — 한 프레임 놓침에는 버티고 정말 사라진 것은 놓아준다.
            if rec.misses >= self.stale_misses and rec.class_groups > 1:
                rec.class_groups -= 1
            if rec.misses >= self.expire_misses:
                rec.lifecycle = "expired"
                self.objects.pop(rec.obj_id, None)
            elif rec.misses >= self.stale_misses:
                rec.lifecycle = "stale"

        self.last_stats = {"evidence": len(evs), "clusters": len(clusters),
                           "matched": len(taken_c),
                           "new": len(matched) - len(taken_c),
                           "below_init_conf": sum(
                               1 for ci, cl in enumerate(clusters)
                               if ci not in used
                               and max(e.conf for e in cl) < self.init_conf),
                           "objects": len(self.objects)}
        return [r for r in self.objects.values() if r.lifecycle != "expired"]


# ─────────────────────────────────────────────────────────────────────────────
# 특징 사전 + LLM 생성 + VLM 서술
# ─────────────────────────────────────────────────────────────────────────────
class Ollama:
    def __init__(self, host: str, timeout_s: float = 120.0):
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, model: str, prompt: str, *, system: str | None = None,
                 images_b64: list[str] | None = None,
                 temperature: float = 0.0, seed: int = 42) -> str:
        import requests

        payload: dict[str, Any] = {
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "seed": seed},
        }
        if system:
            payload["system"] = system
        if images_b64:
            payload["images"] = images_b64
        r = requests.post(f"{self.host}/api/generate", json=payload,
                          timeout=self.timeout_s)
        r.raise_for_status()
        return r.json().get("response", "")

    def available_models(self) -> list[str]:
        import requests

        r = requests.get(f"{self.host}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]


# --------------------------------------------------------------------------
# 특징 사전
# --------------------------------------------------------------------------
class FeatureDictionary:
    """클래스 → 특징 문구. CLIP 텍스트 앵커로 쓴다.

    `class_features.json`과 같은 모양이며 `_meta`는 무시한다. 클래스별 점수는
    그 클래스 특징 문구들의 상위 `top_k` 유사도 평균이다 — 문구 하나가 우연히
    튀는 것을 완화하려는 것이고, 원 PoC의 AND 게이트(색·모양·채도)는 여기 없다.
    그 게이트들은 문(door)처럼 **알려진 클래스**에 튜닝된 값이라 미지 객체 서술에는
    쓸 수 없다. 즉 이 클래스는 PoC보다 **약한** 판정기다 — 그 대신 어떤 클래스에도
    적용된다.
    """

    def __init__(self, path: Path | None, top_k: int = 3):
        self.path = path
        self.top_k = top_k
        self.entries: dict[str, dict] = {}
        if path and path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.entries = {k: v for k, v in raw.items() if not k.startswith("_")}

    def classes(self) -> list[str]:
        return list(self.entries)

    def background_classes(self) -> set[str]:
        """사전이 스스로 선언한 배경 클래스.

        **단일 출처다.** config에 같은 목록을 두면 반드시 어긋난다 — 2026-09-21에
        사전에 배경 5개를 추가하고 config를 안 고쳐서 crop의 64%(vehicle_part/
        vehicle_group)가 객체로 취급됐고, 배경 버림이 30개에서 4개로 붕괴했다.
        """
        return {k for k, v in self.entries.items() if v.get("background")}

    def prompt_plan(self) -> tuple[list[str], dict[str, list[int]]]:
        """(전체 프롬프트 목록, 클래스 → 그 프롬프트들의 인덱스)."""
        prompts: list[str] = []
        index: dict[str, list[int]] = {}
        for cls, entry in self.entries.items():
            feats = [f for f in entry.get("features", []) if isinstance(f, str)]
            if not feats:
                feats = [cls]
            idx = []
            for f in feats:
                # CLIP은 문장형 프롬프트에서 더 안정적이다(PoC 관례).
                prompts.append(f"a photo of {cls}, {f}")
                idx.append(len(prompts) - 1)
            index[cls] = idx
        return prompts, index

    def score_row(self, sims: list[float], index: dict[str, list[int]]) -> dict[str, float]:
        out = {}
        for cls, idx in index.items():
            vals = sorted((sims[i] for i in idx), reverse=True)[: self.top_k]
            out[cls] = float(np.mean(vals)) if vals else float("-inf")
        return out

    def add(self, cls: str, entry: dict, *, provenance: dict) -> None:
        """생성 항목을 **메모리에만** 올린다. 디스크 사전은 사람 검수 뒤에 쓴다."""
        e = dict(entry)
        e["_provenance"] = provenance
        e["_verified"] = False
        self.entries[cls] = e


# --------------------------------------------------------------------------
# LLM — 사전에 없는 클래스의 항목 생성
# --------------------------------------------------------------------------
SYSTEM = ("너는 로봇 인지 파이프라인의 클래스 특징 사전을 작성한다. "
          "CLIP(clip-vit-base-patch32) 텍스트 앵커를 쓰는 실제 소비자 코드가 "
          "이 출력을 그대로 읽는다.")

SCHEMA = """\
{
  "features": ["영어 특징 문구 4~8개"],
  "min_aspect_ratio_h_over_w": 숫자 또는 null,
  "target_sim_threshold": 0.2~0.3 사이 숫자,
  "rationale": "왜 이 특징·게이트를 골랐는지 한국어 한두 문장"
}"""


class LlmFeatureGenerator:
    """`research/class_feature_llm`의 reason_json 변형을 그대로 쓴다.

    거기 하니스가 프롬프트 4종을 격자로 비교한 결과물이므로, 여기서 프롬프트를
    새로 지어내면 그 비교 결과를 버리는 셈이 된다.
    """

    def __init__(self, ollama: Ollama, model: str):
        self.ollama = ollama
        self.model = model

    def generate(self, korean: str, english: str) -> dict:
        prompt = (
            f'로봇이 새 탐색 명령 "{korean}"을 받았다. 이 클래스는 아직 특징 사전에 없다.\n'
            f'영어 클래스 키는 "{english}"다.\n\n'
            "이 클래스를 CLIP 기반 후보 게이트로 찾기 위한 특징 사전 항목을 만들어라.\n\n"
            f"## 출력 스키마\n{SCHEMA}\n\n"
            "## 출력 순서 (반드시 지킨다)\n"
            "1) \"## 판단\" 아래에서 한국어로 짧게 따져본다: 세로로 긴가 가로로 넓은가, "
            "색이 클래스 정의의 일부인가, 어떤 문구가 비특이적인가.\n"
            "2) 그 다음 \"## JSON\" 아래에 JSON 객체 하나만 코드블록으로 출력한다.\n"
            "**특징 문구는 반드시 영어로 쓴다** — CLIP이 한국어를 거의 구분하지 못한다는 "
            "실측 결과가 있다."
        )
        t0 = time.perf_counter()
        text = self.ollama.generate(self.model, prompt, system=SYSTEM)
        entry = _extract_json(text)
        return {"entry": entry, "raw": text[:4000], "model": self.model,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1)}


def _extract_json(text: str) -> dict | None:
    """코드블록 우선, 없으면 마지막 중괄호 덩어리. 실패는 None이지 예외가 아니다."""
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S):
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    start = None
    return None


def validate_entry(entry: dict | None) -> tuple[bool, str]:
    """`research/class_feature_llm/validator.py`의 LOADABLE/USABLE 2단계를 줄인 것."""
    if not isinstance(entry, dict):
        return False, "JSON 파싱 실패"
    feats = entry.get("features")
    if not isinstance(feats, list) or not feats:
        return False, "features 없음"
    if not all(isinstance(f, str) and f.strip() for f in feats):
        return False, "features에 비문자열"
    if any(re.search(r"[가-힣]", f) for f in feats):
        return False, "features에 한국어 — CLIP이 구분 못 한다"
    thr = entry.get("target_sim_threshold")
    if thr is not None and not (isinstance(thr, (int, float)) and 0.0 < thr < 1.0):
        return False, "target_sim_threshold 범위 밖"
    ar = entry.get("min_aspect_ratio_h_over_w")
    if ar is not None and not (isinstance(ar, (int, float)) and ar > 0):
        return False, "종횡비 값 이상"
    return True, "ok"


# --------------------------------------------------------------------------
# VLM — 미지 crop 서술
# --------------------------------------------------------------------------
class VlmDescriber:
    """**현재 환경 서술**. 객체 확정이 아니다(사용자 2026-09-21).

    이전 판은 미지 crop마다 "이게 뭐냐"를 물었다. 그 쓰임에서는 RPN 후보 다수가
    오탐이라 VLM이 배경에도 그럴듯한 이름을 붙이는 문제가 있었다. 지금 역할은
    프레임 한 장을 통째로 보고 **장면이 어떤 상황인가**를 서술하는 것이며, 그래서
    객체 판정 경로(stage2)에서 분리돼 저빈도·비동기로 돈다.

    프롬프트는 도메인마다 다르다 — 하천 감시와 도시 안전은 같은 그림에서 서로 다른
    것을 봐야 한다. 그래서 프롬프트가 생성자 인자다(AI-C-15: 도메인 차이는 핵심 코드
    분기가 아니라 구성으로 표현한다).
    """

    SCENE_PROMPT = ("Describe this aerial scene in one sentence: what kind of place "
                    "is it, and what is happening?")

    def __init__(self, ollama: Ollama, model: str, scene_prompt: str | None = None):
        self.ollama = ollama
        self.model = model
        self.scene_prompt = scene_prompt or self.SCENE_PROMPT

    def describe_scene(self, jpeg_bytes: bytes) -> dict:
        """프레임 한 장의 환경 서술. 실패는 예외가 아니라 사유로 돌려준다(AI-C-05)."""
        b64 = base64.b64encode(jpeg_bytes).decode()
        try:
            t0 = time.perf_counter()
            text = self.ollama.generate(self.model, self.scene_prompt, images_b64=[b64])
            return {"text": text.strip()[:600], "model": self.model,
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                    "status": "description"}   # 확정 label이 아니다
        except Exception as exc:
            return {"error": repr(exc)[:160], "status": "unavailable"}
