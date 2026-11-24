"""스캔 탐지 창구의 산출(scan_assess) 시험 — 모델 없이 결과 JSON 만으로.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../venv/bin/python -m pytest test_scan_assess.py -q
"""
import scan_assess as sa
import target_assess as ta


def det(label="demo_door", score=70.0, suspect=True, glob=None, local=None, local_declared=False, geometry=None,
        dist=None, offset=None, f_source="fov", gates_failed=(), bbox=(10, 20, 110, 220), track_score=None,
        cue_values=None, frame_cards=None, **extra):
    glob = suspect if glob is None else glob
    a = {"frame": {"score": score, "suspect": label if suspect else None, "gates_failed": list(gates_failed)},
         "axes": {"global": glob, "local": local, "local_declared": local_declared, "geometry": geometry},
         "cue_values": cue_values or {"class": {"pct": 80.0, "role": "score", "passed": None, "w": 0.5}},
         "score": track_score if track_score is not None else score, "suspect": label if suspect else None,
         "min_score": 50.0, "enough": True,
         "cards": {label: {"rows": [["class", 81.0, "door 0.6 · avg of 30"]]}}}
    if frame_cards is not None:
        a["frame"]["cards"] = frame_cards
    d = {"label": label, "bbox": list(bbox), "assess": a}
    if dist is not None:
        d["dist"] = dist
    if offset is not None:
        d["bearing"] = {"offset_deg": offset, "f_px": 300.0, "f_source": f_source}
    d.update(extra)
    return d


def rec(*dets, provisional=("demo_door",), **extra):
    return dict({"detections": list(dets), "provisional": list(provisional)}, **extra)


METRIC = {"value": 2.35, "space": "metric", "source": "unidepth", "age": 0}


# ---------------------------------------------------------------- 상수
def test_part_floor_matches_target_assess():
    assert sa.PART_FLOOR == ta.PART_FLOOR


def test_target_names():
    assert sa.target_name("door") == "demo_door"
    assert sa.target_name("box") == "demo_box"
    assert sa.target_name("object", " fire extinguisher ") == "fire extinguisher"


# ---------------------------------------------------------------- 방위
def test_wrap180():
    assert sa.wrap180(315 + 60) == 15.0
    assert sa.wrap180(270) == -90.0
    assert sa.wrap180(180) == -180.0
    assert sa.wrap180(-190) == 170.0
    assert sa.wrap180(0) == 0.0


def test_bearing_adds_rotation_and_offset():
    assert sa.bearing(90, det(offset=6.4)) == (96.4, None)
    assert sa.bearing(315, det(offset=-50.0))[0] == -95.0


def test_bearing_without_focal_is_the_step_heading():
    b, note = sa.bearing(225, det())
    assert b == -135.0 and "no focal" in note


def test_bearing_falls_back_to_box_bearing():
    d = det(label="demo_box", box_bearing={"bearing_deg": 357.0, "offset_deg": -3.0})
    assert sa.bearing(45, d)[0] == 42.0


# ---------------------------------------------------------------- 거리
def test_distance_metric_fresh_is_valid():
    d = sa.distance(det(dist=METRIC))
    assert d["distance_valid"] and d["distance_m"] == 2.35 and d["distance_source"] == "unidepth"


def test_distance_carried_is_not_valid():
    d = sa.distance(det(dist={"value": 2.5, "space": "metric", "source": "carried", "age": 4}))
    assert not d["distance_valid"] and "carried" in d["distance_note"]


def test_distance_relative_is_not_valid():
    d = sa.distance(det(dist={"value": 0.4, "space": "relative", "source": "unidepth", "age": 0}))
    assert not d["distance_valid"] and "relative" in d["distance_note"]


def test_distance_missing():
    assert sa.distance(det())["distance_note"] == "no distance"
    assert sa.distance(det(box_distance={"distance_cm": None, "reason": "edge"}))["distance_note"] == "box touches the edge"


def test_distance_box_height_range():
    bh = {"value": 1.2, "space": "metric", "source": "box_height", "age": 0}
    assert sa.distance(det(dist=bh, box_distance={"distance_cm": 120.0, "reason": None}))["distance_valid"]
    far = dict(bh, value=3.5)
    d = sa.distance(det(dist=far, box_distance={"distance_cm": 350.0, "reason": None}))
    assert not d["distance_valid"] and "outside" in d["distance_note"]


# ---------------------------------------------------------------- 장 단위 판정
def test_frame_verdict_uses_frame_not_track():
    """트랙 점수가 높아도(앞 각도의 근거를 물려받았어도) 그 장이 안 되면 안 된다."""
    found, score, reason = sa.frame_verdict(det(score=41.0, suspect=False, track_score=72.0))
    assert not found and score == 41.0 and reason == "score 41 < 50"


def test_frame_verdict_gate_reason():
    found, _, reason = sa.frame_verdict(det(score=60.0, suspect=False, gates_failed=["colour"]))
    assert not found and reason == "gate failed: colour"


def test_frame_verdict_axes_rule():
    assert sa.frame_verdict(det())[0]                                              # 전역 하나만 잼 → 하나로 충분
    assert sa.frame_verdict(det(geometry=True))[0]                                 # 둘 다 동의
    assert not sa.frame_verdict(det(geometry=False))[0]                            # 기하 반대
    assert not sa.frame_verdict(det(local_declared=True, local=0.01))[0]           # 국소 반대
    assert sa.frame_verdict(det(local_declared=True, local=0.3, geometry=True))[0]
    assert sa.frame_verdict(det(local=0.01))[0]                                    # 국소를 선언 안 했으면 안 셈


def test_first_frame_is_judged_without_waiting():
    """회전 스캔은 각도당 한 장 — 첫 장부터 판정이 나와야 한다(보류 없음)."""
    assert sa.frame_verdict(det(track_frames=1))[0]


def test_pick_prefers_found_then_score_and_ignores_other_labels():
    r = rec(det(score=55.0), det(score=90.0, suspect=False, gates_failed=["x"]), det(label="person", score=99.0))
    d, found, score, _ = sa.pick(r, "demo_door")
    assert found and score == 55.0


def test_pick_reasons_when_nothing():
    assert sa.pick(rec(), "demo_box")[3] == "no candidate"
    r = rec(rejected_by_gates=[{"target": "demo_box"}])
    assert sa.pick(r, "demo_box")[3] == "detector box rejected by gates"


# ---------------------------------------------------------------- 근거
def test_evidence_uses_frame_cards():
    d = det(frame_cards={"demo_door": {"rows": [["class", 80.0, "door 0.62"]]}}, geometry=True)
    cues = sa.evidence(d)
    assert cues[0] == ["class", 0.8, "door 0.62"]
    assert cues[-1] == ["axes", None, "global yes · local - · geometry yes"]


def test_evidence_strips_track_average_on_old_results():
    assert sa.evidence(det())[0][2] == "door 0.6"


# ---------------------------------------------------------------- 모으기
SCAN = {"scan_id": "S1", "target": "door", "query": None, "targets": ["demo_door"], "profile_version": 12}


def test_assemble_best_is_highest_found_score():
    cap = [{"rotation_deg": 0, "frame_n": 1, "rec": rec(det(score=8.0, suspect=False))},
           {"rotation_deg": 90, "frame_n": 2, "rec": rec(det(score=71.0, dist=METRIC, offset=6.4))},
           {"rotation_deg": 180, "frame_n": 3, "rec": rec(det(score=60.0, dist=METRIC, offset=0.0))}]
    out = sa.assemble(SCAN, cap)
    assert [f["found"] for f in out["frames"]] == [False, True, True]
    assert out["frames"][0]["box_xyxy"] is None and out["frames"][0]["reason"] == "score 8 < 50"
    b = out["best"]
    assert b["rotation_deg"] == 90 and b["bearing_deg"] == 96.4 and b["distance_valid"] and b["provisional"]
    assert out["score_basis"] == "frame" and out["profile_version"] == 12
    assert out["evidence"]["cues"]


def test_assemble_tie_prefers_valid_distance():
    cap = [{"rotation_deg": 0, "frame_n": 1, "rec": rec(det(score=70.0))},
           {"rotation_deg": 45, "frame_n": 2, "rec": rec(det(score=70.0, dist=METRIC))}]
    assert sa.assemble(SCAN, cap)["best"]["rotation_deg"] == 45


def test_assemble_frozen_is_excluded():
    cap = [{"rotation_deg": 0, "frame_n": 1, "rec": rec(det(score=70.0))},
           {"rotation_deg": 45, "frame_n": 1, "rec": rec(det(score=90.0)), "frozen_of": 0}]
    out = sa.assemble(SCAN, cap)
    assert out["best"]["rotation_deg"] == 0
    assert out["frames"][1]["frozen"] and out["frames"][1]["reason"] == "frozen frame (same as 0°)"


def test_assemble_not_found_is_best_null():
    out = sa.assemble(SCAN, [{"rotation_deg": 0, "frame_n": 1, "rec": rec()}])
    assert out["best"] is None and out["evidence"] == {"cues": [], "enough": False}
    assert out["frames"][0]["reason"] == "no candidate"


# ---------------------------------------------------------------- 질의 정리 (2026-11-24)
def test_query_ko_table_wins_over_broken_translation():
    """실제 임무 111600: 발화 '사람' → 번역 'celedar or person orhumanbeingorhumanh'."""
    assert sa.resolve_query("celedar or person orhumanbeingorhumanh", "사람") == ("person", "query_ko")
    assert sa.resolve_query(None, "소화기") == ("fire extinguisher", "query_ko")


def test_query_synonyms_and_plain_names():
    assert sa.resolve_query("Human being", None) == ("person", "query_synonym")
    assert sa.resolve_query(" Fire  Extinguisher ", "빨간 통") == ("fire extinguisher", "query")


def test_broken_query_without_table_is_rejected():
    name, why = sa.resolve_query("celedar or person orhumanbeingorhumanh", "낯선 것")
    assert name is None and "plain English" in why
    assert sa.resolve_query("", "") == (None, "no query")
    assert sa.resolve_query("사람", None)[0] is None


def test_ko_table_names_are_valid_target_names():
    import cue_catalog as cc
    for v in list(sa.KO_NAMES.values()) + list(sa.EN_SYNONYMS.values()):
        assert cc._NAME_RE.match(v), v
    db = cc.load_targets()
    for v in ("person", "door", "chair", "desk"):
        assert v in db
