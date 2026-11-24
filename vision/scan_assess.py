"""스캔 탐지 창구의 산출 — 각도마다 잡은 targeted 결과 한 장으로 판정 · 방위 · 거리를 낸다(2026-11-24).

스캔탐지_창구_계획_261124.md §4. 서버(scan_desk.py)가 부르고 시험(test_scan_assess.py)이 부른다.
**torch · cv2 · numpy 를 쓰지 않는다** — 서버가 가볍게 import 한다.

입력은 Redis vision_stream:<소스>:targeted 의 result JSON 그대로(v2 모양: detections[].label · bbox · assess · dist ·
bearing · box_distance). 판정은 **그 장의 값**(score_basis "frame")이다 — TrackEvidence 가 트랙 평균으로 덮기 전
값(assess.frame · cue_values · axes 의 장 단위 칸)으로 TrackEvidence 와 같은 축 규칙을 한 장에 적용한다.
"""
from __future__ import annotations

import re

#: 국소 축의 부분 하한 — target_assess.PART_FLOOR 와 같아야 한다(시험이 묶는다)
PART_FLOOR = 0.05
#: 박스 높이 거리식의 검증 범위(cm) — Box_Detection calibration_range
BOX_RANGE_CM = (50.0, 300.0)

#: 가시화의 target → targeted 대상 이름. object 는 query 그대로
TARGET_NAMES = {"door": "demo_door", "box": "demo_box"}

#: 스캔 전용 점수 하한(2026-11-24). 대상 파일의 min_score 는 여러 장을 쌓는 실시간 판정의 기준이다(demo_box 25 =
#: 원본 conf 0.25). 스캔은 **한 장**으로 판정하므로 더 엄격하게 본다. 오늘 박스 세 판: 진짜 89~95 · 오탐 42~52
#: (선반 아래 골판지 상자 41.7 · 51.5, 가장자리 검은 물체 48.3). 본체(78.7)는 이 하한이 아니라 방위 일치가 거른다
SCAN_MIN_SCORE = {"demo_box": 65.0}
#: 같은 물체로 볼 방위 차(도). 제자리 회전이라 같은 물체는 이웃 각도에서 같은 방위로 다시 보인다
#: (오늘 진짜 박스 쌍의 차 0.5 · 2.3 · 6.2 · 7.1°, 소화기는 가장자리 225° 가 어안 보정 없이 15~16° 어긋났다).
#: 한 각도에서만 나온 검출은 isolated — 지우지 않고 순위만 내린다
BEARING_TOL_DEG = 20.0


#: 발화 원문(query_ko) → 대상 이름(2026-11-24). **번역보다 먼저 본다** — 가시화의 LLM 번역이 실패 · 시간 초과하거나
#: 깨진 영어('사람' → 'celedar or person orhumanbeingorhumanh')를 냈다. 등록 대상(person · door · chair · desk)은 그 근거로
#: 판정되고, 나머지는 직접 입력(제로샷) 이름이다
KO_NAMES = {"사람": "person", "사람들": "person", "인간": "person", "문": "door", "출입문": "door",
            "의자": "chair", "책상": "desk", "소화기": "fire extinguisher", "상자": "box", "박스": "box",
            "가방": "bag", "배낭": "backpack", "노트북": "laptop", "모니터": "monitor", "키보드": "keyboard",
            "쓰레기통": "trash can", "휴지통": "trash can", "컵": "cup", "병": "bottle", "물병": "bottle",
            "화분": "potted plant", "자전거": "bicycle", "프린터": "printer", "선반": "shelf", "책장": "bookshelf",
            "카트": "cart", "콘": "traffic cone", "라바콘": "traffic cone", "우산": "umbrella", "공": "ball"}
#: 영어 질의의 같은 말 → 등록 대상 이름
EN_SYNONYMS = {"human": "person", "humans": "person", "people": "person", "human being": "person", "man": "person",
               "woman": "person", "pedestrian": "person", "persons": "person", "doorway": "door", "seat": "chair",
               "table": "desk"}
_QUERY_RE = re.compile(r"^[a-z][a-z \-]{0,40}$")


def resolve_query(query, query_ko=None):
    """object 판의 대상 이름 → (이름, 출처) 또는 (None, 이유). 출처: query_ko(표) · query · query_synonym."""
    ko = str(query_ko or "").strip()
    if ko in KO_NAMES:
        return KO_NAMES[ko], "query_ko"
    q = " ".join(str(query or "").strip().lower().replace("_", " ").split())
    if q in EN_SYNONYMS:
        return EN_SYNONYMS[q], "query_synonym"
    if not q:
        return None, "no query"
    if not _QUERY_RE.match(q) or " or " in " %s " % q:
        return None, "query is not a plain English name: %r" % query
    return q, "query"


def target_name(target, query=None):
    if target == "object":
        return str(query or "").strip()
    return TARGET_NAMES[target]


def wrap180(deg):
    """각을 −180 이상 180 미만으로."""
    d = (float(deg) + 180.0) % 360.0 - 180.0
    return round(d, 1) + 0.0


def _yn(v):
    return "-" if v is None else ("yes" if v else "no")


def axes_of(det):
    """그 장의 (전역, 국소, 기하) — 잰 것만 참/거짓, 안 잰 것은 None."""
    a = det.get("assess") or {}
    fr = a.get("frame") or a
    ax = a.get("axes") or {}
    glob = ax.get("global")
    if glob is None:
        glob = fr.get("suspect") is not None
    local = None
    if ax.get("local_declared") and ax.get("local") is not None:
        local = float(ax["local"]) >= PART_FLOOR
    geom = ax.get("geometry")
    return bool(glob), local, (None if geom is None else bool(geom))


def frame_verdict(det, floor=None):
    """검출 하나의 장 단위 판정 → (found, score, reason). TrackEvidence 의 축 규칙(길이 1 트랙)과 같다:
    전역이 통과하고, 잰 축 가운데 반대가 없고, 동의가 둘 이상(잰 축이 하나면 하나).
    floor = 스캔 전용 점수 하한(SCAN_MIN_SCORE) — 축 규칙을 통과해도 이보다 낮으면 found 가 아니다."""
    a = det.get("assess") or {}
    fr = a.get("frame") or a
    score = fr.get("score")
    glob, local, geom = axes_of(det)
    votes = [v for v in (glob, local, geom) if v is not None]
    ok = glob and False not in votes and votes.count(True) >= (2 if len(votes) >= 2 else 1)
    if ok and floor is not None and score is not None and float(score) < float(floor):
        return False, score, "score %.0f < %.0f (scan floor)" % (score, float(floor))
    if ok:
        return True, score, None
    mn = a.get("min_score", 50.0)
    if score is None:
        reason = "no evidence"
    elif fr.get("gates_failed"):
        reason = "gate failed: %s" % ", ".join(fr["gates_failed"])
    elif a.get("veto_text"):
        reason = "vetoed: %s" % a["veto_text"]
    elif mn is not None and score < float(mn):
        reason = "score %.0f < %.0f" % (score, float(mn))
    elif not glob:
        reason = "not enough evidence"
    elif local is False:
        reason = "local axis disagrees"
    elif geom is False:
        reason = "size disagrees"
    else:
        reason = "axes disagree"
    return False, score, reason


def pick(rec, name):
    """그 장에서 대상 이름의 대표 검출 → (det, found, score, reason). 판정된 것 · 점수 높은 것이 앞."""
    best = None
    floor = SCAN_MIN_SCORE.get(name)
    for d in rec.get("detections") or []:
        if d.get("label") != name:
            continue
        found, score, reason = frame_verdict(d, floor)
        key = (found, -1.0 if score is None else float(score))
        if best is None or key > best[0]:
            best = (key, d, found, score, reason)
    if best is None:
        rej = [r for r in (rec.get("rejected_by_gates") or []) if r.get("target") == name]
        return None, False, None, ("detector box rejected by gates" if rej else "no candidate")
    return best[1], best[2], best[3], best[4]


def bearing(rotation_deg, det):
    """(bearing_deg, note). 스캔 시작 방향 기준 · 시계 방향 + · −180~180. 화면 중심 대비 각은 비전이 낸다."""
    b = det.get("bearing") or {}
    off = b.get("offset_deg")
    if off is None:
        off = (det.get("box_bearing") or {}).get("offset_deg")
    if off is None:
        return wrap180(rotation_deg), "no focal length - heading of the scan step"
    note = None
    if b.get("f_source") == "go1_box":
        note = "reference - no fisheye correction"
    return wrap180(float(rotation_deg) + float(off)), note


#: 바닥에 서 있는 대상 — 바닥선 거리(ground_distance)를 이것들에만 쓴다(2026-11-24). 책상 위 물체(컵 · 노트북 ·
#: 모니터 · 키보드 …)는 아래 변이 바닥보다 높아 **더 멀게** 나오므로 쓰지 않는다 — 멀게 틀리면 로봇이 부딪힌다
FLOOR_OBJECTS = {"demo_box", "demo_door", "person", "door", "chair", "fire extinguisher", "box", "bag", "backpack",
                 "trash can", "potted plant", "bicycle", "shelf", "bookshelf", "cart", "traffic cone", "umbrella", "ball"}
#: Go1 어안 화면의 단안 깊이(UniDepth)는 거리를 1.75~2.5 배로 부풀렸다(같은 장의 시연 박스 높이식과 대조) — valid 아님
GO1_DEPTH_NOTE = "monocular depth on the Go1 fisheye camera is not calibrated"


def _go1(det):
    return det.get("ground_distance") is not None or (det.get("bearing") or {}).get("f_source") == "go1_box"


def distance(det, name=None):
    """{distance_m, distance_valid, distance_source, distance_note} — §4.3. 하나라도 어긋나면 valid 가 아니다.
    Go1 프레임: 박스 높이식(demo_box) → 바닥선(FLOOR_OBJECTS) → 둘 다 안 되면 UniDepth 값을 valid 아님으로."""
    v = det.get("dist")
    name = name or det.get("label")
    out = {"distance_m": None, "distance_valid": False, "distance_source": None, "distance_note": None}
    gd = det.get("ground_distance") or {}
    box_ok = bool(v) and v.get("source") == "box_height"
    if not box_ok and gd.get("distance_cm") is not None and not gd.get("reason") and name in FLOOR_OBJECTS:
        out.update(distance_m=round(gd["distance_cm"] / 100.0, 3), distance_valid=True, distance_source="ground_line")
        return out
    if not v:
        bd = det.get("box_distance") or {}
        out["distance_note"] = "box touches the edge" if bd.get("reason") else "no distance"
        return out
    out["distance_m"] = v.get("value")
    out["distance_source"] = v.get("source")
    if v.get("space") != "metric":
        out["distance_note"] = "relative depth only"
    elif v.get("source") == "carried" or (v.get("age") or 0) > 0:
        out["distance_note"] = "carried from an earlier frame"
    elif v.get("source") == "box_height":
        bd = det.get("box_distance") or {}
        cm = bd.get("distance_cm")
        if bd.get("reason"):
            out["distance_note"] = "box touches the edge"
        elif cm is None or not (BOX_RANGE_CM[0] <= float(cm) <= BOX_RANGE_CM[1]):
            out["distance_note"] = "outside %d-%d cm" % BOX_RANGE_CM
        else:
            out["distance_valid"] = True
    elif v.get("value") is None:
        out["distance_note"] = "no distance"
    elif _go1(det):
        why = ("not standing on the floor" if name not in FLOOR_OBJECTS else
               gd.get("reason") or "no ground line")
        out["distance_note"] = "%s - ground line: %s" % (GO1_DEPTH_NOTE, why)
    else:
        out["distance_valid"] = True
    return out


def evidence(det):
    """best 장의 근거 → [[이름, 0~1, 설명], ...] + 축 한 줄. 설명은 그 장 카드 줄(assess.frame.cards)."""
    a = det.get("assess") or {}
    name = det.get("label")
    fcards = (a.get("frame") or {}).get("cards") or {}
    card = fcards.get(name) or next(iter(fcards.values()), None)
    texts = {}
    if card is None:                         # 옛 결과 — 트랙 카드의 평균 표시를 걷어 낸다
        card = (a.get("cards") or {}).get(name) or {}
        for r in card.get("rows") or []:
            texts[r[0]] = str(r[2]).split(" · avg of ")[0].split(" · passed ")[0]
    else:
        texts = {r[0]: str(r[2]) for r in card.get("rows") or []}
    cues = []
    for c, v in (a.get("cue_values") or {}).items():
        pct = v.get("pct")
        cues.append([c, None if pct is None else round(float(pct) / 100.0, 3), texts.get(c, v.get("role", ""))])
    glob, local, geom = axes_of(det)
    cues.append(["axes", None, "global %s · local %s · geometry %s" % (_yn(glob), _yn(local), _yn(geom))])
    return cues


def frame_entry(rotation_deg, frame_n, rec, name, frozen_of=None):
    """frames[] 한 칸. frozen_of = 같은 그림이었던 앞 각도(있으면 판정에서 뺀다)."""
    det, found, score, reason = pick(rec, name)
    e = {"rotation_deg": rotation_deg, "frame_n": frame_n, "found": bool(found),
         "score": None if score is None else round(float(score), 1),
         "box_xyxy": None, "frozen": frozen_of is not None}
    if det is not None and found:
        e["box_xyxy"] = [round(float(x), 1) for x in det.get("bbox") or []] or None
        e["bearing_deg"], e["bearing_note"] = bearing(rotation_deg, det)
        e.update(distance(det, name))
    if frozen_of is not None:
        e["found"] = False
        e["reason"] = "frozen frame (same as %s°)" % frozen_of
    elif not found:
        e["reason"] = reason
    return e, det


def assemble(scan, captured):
    """scan = {scan_id, target, query, targets, profile_version}, captured = [{rotation_deg, frame_n, rec, frozen_of}]
    (각도 순) → /scan/result 의 몸(state 제외)."""
    name = scan["targets"][0]
    frames, dets = [], {}
    for c in captured:
        e, det = frame_entry(c["rotation_deg"], c.get("frame_n"), c["rec"], name, c.get("frozen_of"))
        frames.append(e)
        dets[c["rotation_deg"]] = (det, c["rec"])
    cand = [e for e in frames if e["found"] and not e["frozen"]]
    for e in cand:                           # 같은 물체가 다른 각도에서도 같은 방위로 보였나
        e["seen_at"] = sorted([e["rotation_deg"]] + [o["rotation_deg"] for o in cand if o is not e and
                                                     abs(wrap180(e["bearing_deg"] - o["bearing_deg"])) <= BEARING_TOL_DEG])
        e["isolated"] = len(e["seen_at"]) < 2
    best = ev = None
    if cand:
        top = max(cand, key=lambda e: (not e["isolated"], e["score"] or 0.0, e.get("distance_valid", False)))
        det, rec = dets[top["rotation_deg"]]
        best = {k: top.get(k) for k in ("rotation_deg", "bearing_deg", "bearing_note", "distance_m", "distance_valid",
                                        "distance_source", "distance_note")}
        best["score"] = top["score"]
        best["box_xyxy"] = top["box_xyxy"]
        best["provisional"] = name in (rec.get("provisional") or [])
        best["isolated"] = top["isolated"]
        best["seen_at"] = top["seen_at"]
        ev = {"cues": evidence(det), "enough": bool((det.get("assess") or {}).get("enough", True))}
        reason = "found %s at %s° - score %.0f" % (name, top["rotation_deg"], top["score"] or 0.0)
        if top["isolated"]:
            reason += " - seen at one angle only"
    else:
        frozen = sum(1 for e in frames if e["frozen"])
        reason = "%s not found in %d angle(s)%s" % (name, len(frames),
                                                    (" (%d frozen)" % frozen) if frozen else "")
    return {"scan_id": scan["scan_id"], "target": scan["target"], "query": scan.get("query"),
            "targets": list(scan["targets"]), "frames": frames, "best": best,
            "evidence": ev or {"cues": [], "enough": False}, "reason": reason, "model": "targeted",
            "profile_version": scan.get("profile_version"), "score_basis": "frame"}
