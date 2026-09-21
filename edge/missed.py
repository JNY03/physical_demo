"""미검출 분석 — **정말 안 잡힌 것만** 남긴다.

implements: AI-S-04, AI-S-03, AI-L-01, AI-E-04, AI-C-05

── 무엇을 하는가 ───────────────────────────────────────────────────────────
OVD(YOLOE + YOLO-World 합의)가 확정한 것은 화면 위쪽 패널이 맡는다. 이 모듈은
그 **반대편**을 본다: RPN이 "무언가 있다"고 했는데 OVD가 이름을 못 붙인 것.

세 겹으로 거른다. 순서가 중요하다 — 싼 것부터, 그리고 각 겹이 거르는 대상이
서로 다르다.

  1. **설명된 것 제거**(postprocess.absorb) — IoU만 보면 두 경우가 미지로 샌다.
     조각: 차 문짝만 딴 박스는 car 안에 있지만 합집합이 커서 IoU가 낮다.
     병합: 차 3대를 감싼 박스는 어느 car와도 IoU가 낮다.
     그래서 축을 셋으로 나눈다 — match(IoU) / fragment(포함) / merged(덮음).

  2. **같은 것끼리 합치기**(merge_overlapping) — 안 합치면 한 대상이 crop
     여러 장이 되고, 패널이 같은 물체로 덮인다.

  3. **배경 버리기**(CLIP + 특징 사전) — RPN precision이 낮아 이 겹이 없으면
     남는 것의 대부분이 도로·수면·벽이다. 사전 항목이 `background: true`면
     그 crop은 **저장하지도 그리지도 않는다.**

남은 것이 "미검출 후보"다. 확정 라벨을 붙이지 않는다 — CLIP은 닫힌 사전의
argmax라 '해당 없음'이 없고, 정답이 사전에 없으면 가장 덜 틀린 것을 고른다
(하천 사전을 도시에 쓰자 71.7%가 tire/plastic_bag으로 몰린 실측이 있다).
그래서 **후보와 근거만** 낸다 — 보는 사람이 판단한다(AI-S-03, AI-S-04).
"""

from __future__ import annotations

import numpy as np

import numpy as np

from semantics import absorb, merge_overlapping


def find_missed(cfg: dict, rpn_items: list[dict], ovd_dets: list[dict],
                frame_wh: tuple[int, int]) -> tuple[list[dict], dict]:
    """OVD가 설명 못 한 후보를 골라낸다. (미지 목록, 통계)

    `rpn_items`는 [{"box": [...], "objectness": float}, ...] 형태다.
    """
    w, h = frame_wh
    u = dict(cfg.get("unknown", {}))
    u["_frame_area"] = float(w * h)
    known, unknown, stats = absorb(rpn_items, ovd_dets, u)

    # 너무 작은 것은 crop해도 볼 수 없다. 사람이 판단할 수 없는 것을 패널에
    # 올리는 것은 화면만 채우는 일이다.
    min_side = float(u.get("min_short_side_px", 8))
    unknown = [d for d in unknown
               if min(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) >= min_side]

    unknown, merged_away = merge_overlapping(
        unknown, float(u.get("merge_unknown_iou", 0.55)))
    stats["merged_away"] = merged_away
    stats["after_filters"] = len(unknown)

    # objectness 높은 것부터 — 상한에 걸려도 "있을 법한 것"이 먼저 남는다.
    unknown.sort(key=lambda d: -float(d.get("objectness", 0.0)))
    unknown = unknown[: int(u.get("max_per_frame", 60))]
    stats["known"] = len(known)
    stats["unknown"] = len(unknown)
    return unknown, stats


def crop_of(bgr, box, *, context: float = 1.5, min_px: int = 16):
    """박스 주변까지 조금 넓게 자른다.

    맥락 없이 딱 맞게 자르면 CLIP도 사람도 무엇인지 알기 어렵다 — 페트병만
    꽉 찬 crop과 물 위에 뜬 페트병 crop은 판단 난이도가 다르다.
    """
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(min_px, (x2 - x1) * context), max(min_px, (y2 - y1) * context)
    X1, Y1 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
    X2, Y2 = int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2))
    if X2 - X1 < 4 or Y2 - Y1 < 4:
        return None
    return bgr[Y1:Y2, X1:X2].copy()


def classify(cfg: dict, crops: list, scorer, dictionary,
             background_classes: set) -> list[dict]:
    """CLIP으로 crop마다 **후보와 근거**를 낸다. 확정 라벨은 만들지 않는다.

    돌려주는 항목:
        verdict   "background"(버림) | "unidentified"(남김)
        top       [(클래스, 점수), ...] 상위 몇 개 — 이것이 '근거'다
        gap       1등과 2등의 차이. 작으면 사전이 갈라내지 못한 것이다.
    """
    if scorer is None or dictionary is None or not crops:
        return [{"verdict": "unidentified", "top": [], "gap": None,
                 "reason": "clip_unavailable" if scorer is None else "no_dictionary"}
                for _ in crops]

    cc = cfg.get("clip", {})
    top_k = int(cc.get("dictionary_top_k", 3))
    # **사전의 프롬프트를 먼저 CLIP에 넣어야 한다.** `score_row`는 그 프롬프트
    # 순서로 만든 index를 요구한다 — 둘이 어긋나면 엉뚱한 클래스에 점수가
    # 붙는다. prompt_plan()이 (프롬프트 목록, 클래스→인덱스)를 함께 낸다.
    prompts, index = dictionary.prompt_plan()
    if not prompts:
        return [{"verdict": "unidentified", "top": [], "gap": None,
                 "reason": "empty_dictionary"} for _ in crops]
    scorer.set_prompts(prompts)
    sims = scorer.score(crops, int(cc.get("batch_size", 16)))

    out = []
    for row in sims:
        per_class = dictionary.score_row(row, index)
        if not per_class:
            out.append({"verdict": "unidentified", "top": [], "gap": None,
                        "reason": "no_dictionary"})
            continue
        ranked = sorted(per_class.items(), key=lambda kv: -kv[1])[:max(top_k, 3)]
        top1 = ranked[0][0]
        gap = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 1.0
        # **1등이 배경 클래스면 배경이다.** gap 조건을 걸지 않는 이유는 실측이다:
        # 배경 1등 gap의 p95가 0.0018이라 어떤 임계를 걸어도 거의 못 거른다.
        # 배경끼리는 원래 비슷하다 — 'gap이 작으니 배경이 아니다'가 성립하지 않는다.
        bg = top1 in background_classes
        out.append({
            "verdict": "background" if bg else "unidentified",
            "top": [(k, round(float(v), 4)) for k, v in ranked],
            "gap": round(float(gap), 4),
        })
    return out


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


class MissedTracker:
    """미검출 후보에 **프레임 간 정체성**을 준다.

    ── 왜 필요했나 ─────────────────────────────────────────────────────────
    RPN 후보 박스는 매 프레임 조금씩 다르다. 그래서 패널을 프레임마다 새로
    그리면 같은 물체가 다른 칸으로 옮겨 다니고, 화면이 계속 바뀌어 **읽을 수가
    없다**(사용자 2026-09-21). 검출된 객체는 OVD가 ObjectTracks로 정체성을
    받지만, 미검출 후보는 정의상 OVD가 못 본 것이라 그 경로가 없다.

    ── 무엇을 하나 ─────────────────────────────────────────────────────────
    이미 본 자리와 IoU로 맞춰 **슬롯**을 유지한다. 슬롯은 자리를 지키므로
    칸이 튀지 않고, 덤으로 프레임을 건너 정보가 쌓인다:

      seen      몇 프레임에서 봤나 — 1~2회는 RPN 잡음일 가능성이 높다
      votes     CLIP 1등이 프레임마다 무엇이었나. 흔들리면 사전이 못 가른 것이다
      best_obj  가장 높았던 objectness

    `min_seen` 이상 본 것만 그린다. 이것이 잡음 필터이면서 동시에 화면을
    안정시킨다 — 한 프레임만 반짝한 후보는 애초에 칸을 차지하지 않는다.

    ── 한계 ────────────────────────────────────────────────────────────────
    IoU 매칭이라 **시야가 빠르게 움직이면 끊긴다.** 말단은 KLT 플로우로 그걸
    풀지만 엣지는 프레임을 띄엄띄엄 받아 같은 수를 쓸 수 없다. 끊기면 새 슬롯이
    생기고 seen이 리셋되므로, 움직이는 동안에는 패널이 비는 쪽으로 틀린다 —
    엉뚱한 것을 오래 본 것처럼 보이는 것보다 낫다.
    """

    def __init__(self, cfg: dict):
        u = cfg.get("unknown", {})
        self.match_iou = float(u.get("track_iou", 0.35))
        self.min_seen = int(u.get("min_seen", 3))
        self.expire_misses = int(u.get("expire_misses", 6))
        self.max_slots = int(u.get("max_slots", 60))
        self.slots: dict[int, dict] = {}
        self._next = 1
        self._frame = 0

    def update(self, items: list[dict], crops: list, verdicts: list[dict]) -> list[dict]:
        """이번 프레임 후보를 슬롯에 흡수하고, 보여줄 슬롯을 돌려준다."""
        self._frame += 1
        used = set()
        for d, cr, v in zip(items, crops, verdicts):
            box = d["box"]
            # 이미 있는 슬롯 중 가장 잘 맞는 것 하나. 양방향 1:1이라
            # 후보 하나가 슬롯 여럿을 살찌우지 않는다.
            best, best_v = None, self.match_iou
            for sid, sl in self.slots.items():
                if sid in used:
                    continue
                iv = _iou(sl["box"], box)
                if iv >= best_v:
                    best, best_v = sid, iv
            if best is None:
                best = self._next
                self._next += 1
                self.slots[best] = {"id": best, "first": self._frame, "seen": 0,
                                    "votes": {}, "best_obj": 0.0, "crop": None}
            used.add(best)
            sl = self.slots[best]
            sl["box"] = box
            sl["last"] = self._frame
            sl["seen"] += 1
            sl["verdict"] = v.get("verdict")
            sl["reason"] = v.get("reason")
            # 기하 판정이 이미 계산한 값을 슬롯에 남긴다 — 패널이 다시 재지
            # 않도록. depth 맵은 async라 매 프레임 있는 것도 아니다.
            for k in ("dist_m", "w_m", "h_m", "protrusion_m"):
                if v.get(k) is not None:
                    sl[k] = v[k]
            obj = float(d.get("objectness", 0.0))
            # **crop은 objectness가 가장 높았던 프레임의 것을 남긴다.** 매번
            # 갈아치우면 썸네일이 깜빡이고, 흐릿한 프레임이 마지막이면 그게 남는다.
            if cr is not None and obj >= sl["best_obj"]:
                sl["best_obj"] = obj
                sl["crop"] = cr
            # 판정 이유가 프레임마다 바뀌면 경계에 걸친 것이다 — 그 사실을 센다.
            if sl.get("reason"):
                sl["votes"][sl["reason"]] = sl["votes"].get(sl["reason"], 0) + 1

        # 만료 — 한동안 안 보인 슬롯은 접는다.
        for sid in [s for s, sl in self.slots.items()
                    if self._frame - sl.get("last", 0) > self.expire_misses]:
            self.slots.pop(sid, None)
        if len(self.slots) > self.max_slots:
            for sid in sorted(self.slots, key=lambda s: self.slots[s]["seen"]
                              )[:len(self.slots) - self.max_slots]:
                self.slots.pop(sid, None)

        # 보여줄 것: 충분히 본 것 + 배경이 아닌 것. **id 순으로 정렬한다** —
        # 점수 순으로 두면 순위가 바뀔 때마다 칸이 뒤섞여 다시 읽을 수 없게 된다.
        show = [sl for sl in self.slots.values()
                if sl["seen"] >= self.min_seen
                and sl.get("verdict") != "background"
                and sl.get("crop") is not None]
        show.sort(key=lambda sl: sl["id"])
        return show

    def stats(self) -> dict:
        return {"slots": len(self.slots),
                "ripe": sum(1 for s in self.slots.values()
                            if s["seen"] >= self.min_seen)}


def suspect_classes(box, ovd_dets: list[dict], *, min_iou: float = 0.25,
                    top: int = 3) -> list[tuple[str, float, float]]:
    """이 후보와 겹치는 **낮은 conf OVD 검출**을 의심 클래스로 낸다.

    ── 왜 CLIP 대신 이것인가 ───────────────────────────────────────────────
    CLIP + 특징 사전이 내던 것은 `rigid_manmade_object 0.247` 같은 추상 속성
    이었고, 실측에서 1등-2등 gap이 0.002~0.009라 **사실상 구분을 못 했다**
    (사용자 2026-09-21: "clip 정보도 쓸모있지 않음"). 닫힌 사전의 argmax에는
    '해당 없음'이 없어, 정답이 사전에 없으면 가장 덜 틀린 것을 고른다.

    그런데 **더 나은 출처가 이미 있다.** OVD는 conf 0.10으로 돌아 후보를 넓게
    내는데, 그중 확정 문턱(confident_conf)을 못 넘은 것들이 버려진다. 미검출
    후보와 겹치는 그 검출이 곧 "모델이 낮은 확신으로 무엇이라 봤나"이고,
    그것이 사람이 판단할 때 실제로 쓰는 정보다. 추가 연산도 0이다 —
    이미 돌린 결과를 다시 보는 것뿐이다.

    돌려주는 것: [(라벨, conf, iou), ...] conf 높은 순.
    """
    out = []
    for d in ovd_dets:
        v = _iou(box, d["box"])
        if v >= min_iou and d.get("label"):
            out.append((d["label"], float(d.get("conf", 0.0)), round(v, 2)))
    out.sort(key=lambda t: -t[1])
    return out[:top]


def physical_size(box, depth_map, focal_px: float | None,
                  *, percentile: float = 25.0):
    """박스의 **실제 크기(m)와 거리(m)**. 판단에 가장 직접 쓰이는 값이다.

    거리를 알면 겉보기 픽셀이 실제 몇 미터인지 나온다(h_m = h_px · Z / f).
    "5m 앞의 높이 0.3m짜리"와 "5m 앞의 1.8m짜리"는 후보 클래스가 전혀 다르다 —
    추상 속성 점수보다 이쪽이 훨씬 잘 가른다.

    depth가 없으면 (None, None, None). 없는 것을 있는 척하지 않는다.
    """
    if depth_map is None or not focal_px:
        return None, None, None
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    H, W = depth_map.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None, None, None
    patch = depth_map[y1:y2, x1:x2]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0:
        return None, None, None
    # 하위 백분위수 — 박스에는 대상 뒤 배경이 섞이고, 위험 판단에서는
    # 가까운 쪽으로 틀리는 편이 안전하다.
    z = float(np.percentile(patch, percentile))
    w_m = (x2 - x1) * z / float(focal_px)
    h_m = (y2 - y1) * z / float(focal_px)
    return z, w_m, h_m


def depth_contrast(box, depth_map, *, ring: float = 0.6,
                   inner_pct: float = 25.0, ring_pct: float = 50.0):
    """박스 안이 **주변보다 얼마나 앞에 있나**(m). 클수록 튀어나와 있다.

    ── 왜 이것이 배경 판정인가 ─────────────────────────────────────────────
    **객체는 주변에서 튀어나온다.** 벽·바닥·책상 상판 같은 배경면은 주변과
    깊이가 이어져 있고, 그 위에 놓인 물체는 앞으로 돌출한다. 그래서 박스 안쪽
    깊이와 그 둘레 고리의 깊이를 비교하면 둘이 갈린다 — 추가 모델 없이,
    이미 낸 depth 맵 하나로.

    CLIP + 특징 사전이 하던 일을 대신한다. 그쪽은 1등-2등 gap이 0.002~0.009라
    사실상 구분을 못 했고(닫힌 사전 argmax에 '해당 없음'이 없다) crop 수에
    선형이라 비쌌다. 이 규칙은 **물리적으로 말이 되는 근거**를 쓰고 공짜다.

    안쪽은 하위 백분위수(가까운 쪽), 고리는 중앙값을 쓴다 — 안쪽에는 대상 뒤
    배경이 섞이므로 가까운 쪽을 대표로 잡아야 실제 돌출을 잰다.

    Returns: (돌출 m, 안쪽 거리 m, 고리 거리 m). 잴 수 없으면 (None, None, None).
    """
    if depth_map is None:
        return None, None, None
    H, W = depth_map.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None, None, None

    inner = depth_map[y1:y2, x1:x2]
    inner = inner[np.isfinite(inner) & (inner > 0)]
    if inner.size < 8:
        return None, None, None

    bw, bh = x2 - x1, y2 - y1
    rx, ry = int(bw * ring), int(bh * ring)
    X1, Y1 = max(0, x1 - rx), max(0, y1 - ry)
    X2, Y2 = min(W, x2 + rx), min(H, y2 + ry)
    outer = depth_map[Y1:Y2, X1:X2].copy()
    # 고리 = 확장 영역에서 박스를 도려낸 것. 박스를 안 빼면 자기 자신과 비교하게 된다.
    outer[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = np.nan
    outer = outer[np.isfinite(outer) & (outer > 0)]
    if outer.size < 8:
        return None, None, None

    z_in = float(np.percentile(inner, inner_pct))
    z_ring = float(np.percentile(outer, ring_pct))
    return z_ring - z_in, z_in, z_ring


def geometric_verdict(box, depth_map, focal_px, cfg: dict) -> dict:
    """**기하만으로** 배경/후보를 가른다. CLIP을 대신한다.

    셋을 본다. 각 규칙이 거르는 대상이 서로 다르다.

      1. 돌출  — 주변보다 앞에 있지 않으면 면(벽·바닥·상판)이다
      2. 크기  — 실제 크기가 말이 안 되면(수 미터짜리 '물체') 면이거나 잡음
      3. 형상  — 극단적으로 가늘고 긴 것은 대개 모서리·경계선이다

    **depth가 없으면 아무것도 버리지 않는다.** 판단 근거가 없는데 버리면
    그건 필터가 아니라 손실이다(AI-S-03). reason으로 그 사실을 드러낸다.
    """
    g = cfg.get("unknown", {}).get("geometry", {})
    prot, z_in, z_ring = depth_contrast(box, depth_map)
    w_px, h_px = float(box[2] - box[0]), float(box[3] - box[1])

    if prot is None or not focal_px:
        return {"verdict": "candidate", "reason": "no_depth",
                "protrusion_m": None, "dist_m": None}

    w_m, h_m = w_px * z_in / focal_px, h_px * z_in / focal_px
    out = {"protrusion_m": round(prot, 3), "dist_m": round(z_in, 2),
           "ring_m": round(z_ring, 2), "w_m": w_m, "h_m": h_m}

    if prot < float(g.get("min_protrusion_m", 0.03)):
        return {**out, "verdict": "background", "reason": "flat"}

    big = max(w_m, h_m)
    if big > float(g.get("max_size_m", 3.0)):
        return {**out, "verdict": "background", "reason": "too_large"}
    if big < float(g.get("min_size_m", 0.02)):
        return {**out, "verdict": "background", "reason": "too_small"}

    ar = max(w_px / max(1.0, h_px), h_px / max(1.0, w_px))
    if ar > float(g.get("max_aspect", 8.0)):
        return {**out, "verdict": "background", "reason": "sliver"}

    return {**out, "verdict": "candidate", "reason": "protrudes"}
