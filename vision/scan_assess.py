"""스캔 탐지 창구의 산출 — 각도마다 잡은 targeted 결과 한 장으로 판정 · 방위 · 거리를 낸다(2026-11-24).

스캔탐지_창구_계획_261124.md §4. 서버(scan_desk.py)가 부르고 시험(test_scan_assess.py)이 부른다.
**torch · cv2 · numpy 를 쓰지 않는다** — 서버가 가볍게 import 한다.

입력은 Redis vision_stream:<소스>:targeted 의 result JSON 그대로(v2 모양: detections[].label · bbox · assess · dist ·
bearing · box_distance). 판정은 **그 장의 값**(score_basis "frame")이다 — TrackEvidence 가 트랙 평균으로 덮기 전
값(assess.frame · cue_values · axes 의 장 단위 칸)으로 TrackEvidence 와 같은 축 규칙을 한 장에 적용한다.
"""
from __future__ import annotations

#: 국소 축의 부분 하한 — target_assess.PART_FLOOR 와 같아야 한다(시험이 묶는다)
PART_FLOOR = 0.05
#: 박스 높이 거리식의 검증 범위(cm) — Box_Detection calibration_range
BOX_RANGE_CM = (50.0, 300.0)

#: 가시화의 target → targeted 대상 이름. object 는 query 그대로
TARGET_NAMES = {"door": "demo_door", "box": "demo_box"}


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


def frame_verdict(det):
    """검출 하나의 장 단위 판정 → (found, score, reason). TrackEvidence 의 축 규칙(길이 1 트랙)과 같다:
    전역이 통과하고, 잰 축 가운데 반대가 없고, 동의가 둘 이상(잰 축이 하나면 하나)."""
    a = det.get("assess") or {}
    fr = a.get("frame") or a
    score = fr.get("score")
    glob, local, geom = axes_of(det)
    votes = [v for v in (glob, local, geom) if v is not None]
    ok = glob and False not in votes and votes.count(True) >= (2 if len(votes) >= 2 else 1)
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
    for d in rec.get("detections") or []:
        if d.get("label") != name:
            continue
        found, score, reason = frame_verdict(d)
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


def distance(det):
    """{distance_m, distance_valid, distance_source, distance_note} — §4.3. 하나라도 어긋나면 valid 가 아니다."""
    v = det.get("dist")
    out = {"distance_m": None, "distance_valid": False, "distance_source": None, "distance_note": None}
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
        e.update(distance(det))
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
    best = ev = None
    if cand:
        top = max(cand, key=lambda e: (e["score"] or 0.0, e.get("distance_valid", False)))
        det, rec = dets[top["rotation_deg"]]
        best = {k: top.get(k) for k in ("rotation_deg", "bearing_deg", "bearing_note", "distance_m", "distance_valid",
                                        "distance_source", "distance_note")}
        best["score"] = top["score"]
        best["box_xyxy"] = top["box_xyxy"]
        best["provisional"] = name in (rec.get("provisional") or [])
        ev = {"cues": evidence(det), "enough": bool((det.get("assess") or {}).get("enough", True))}
        reason = "found %s at %s° - score %.0f" % (name, top["rotation_deg"], top["score"] or 0.0)
    else:
        frozen = sum(1 for e in frames if e["frozen"])
        reason = "%s not found in %d angle(s)%s" % (name, len(frames),
                                                    (" (%d frozen)" % frozen) if frozen else "")
    return {"scan_id": scan["scan_id"], "target": scan["target"], "query": scan.get("query"),
            "targets": list(scan["targets"]), "frames": frames, "best": best,
            "evidence": ev or {"cues": [], "enough": False}, "reason": reason, "model": "targeted",
            "profile_version": scan.get("profile_version"), "score_basis": "frame"}
