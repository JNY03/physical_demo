"""스캔 탐지 창구의 산출(scan_assess) 시험 — 모델 없이 결과 JSON 만으로.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test_scan_assess.py -q
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


# ---------------------------------------------------------------- 스캔 하한 · 방위 일치 (2026-11-24 박스 오탐)
def test_scan_floor_drops_weak_demo_box():
    """3039번: 선반 아래 골판지 상자 틈 51.5 점 — 관문은 다 통과했지만 한 장 판정으로는 약하다."""
    found, score, reason = sa.pick(rec(det(label="demo_box", score=51.5), provisional=()), "demo_box")[1:]
    assert not found and score == 51.5 and reason == "score 52 < 65 (scan floor)"
    assert sa.pick(rec(det(label="demo_box", score=89.0), provisional=()), "demo_box")[1]
    assert sa.pick(rec(det(score=51.5)), "demo_door")[1]          # 하한은 대상마다 — 문에는 없다


def test_isolated_detection_ranks_below_corroborated():
    """11:06 판: 진짜 박스는 0° · 45° 에서 같은 방위(34.5 · 35.0), 본체 오탐(78.7)은 225° 한 각도뿐."""
    b = lambda **k: det(label="demo_box", **k)
    cap = [{"rotation_deg": 0, "frame_n": 1, "rec": rec(b(score=70.0, offset=34.5), provisional=())},
           {"rotation_deg": 45, "frame_n": 2, "rec": rec(b(score=68.0, offset=-10.0), provisional=())},
           {"rotation_deg": 225, "frame_n": 3, "rec": rec(b(score=78.7, offset=67.5), provisional=())}]
    out = sa.assemble(dict(SCAN, target="box", targets=["demo_box"]), cap)
    f = {e["rotation_deg"]: e for e in out["frames"]}
    assert f[0]["seen_at"] == [0, 45] and not f[0]["isolated"]
    assert f[225]["found"] and f[225]["isolated"] and f[225]["seen_at"] == [225]
    assert out["best"]["rotation_deg"] == 0 and not out["best"]["isolated"] and out["best"]["seen_at"] == [0, 45]


def test_isolated_best_is_kept_and_said():
    cap = [{"rotation_deg": 90, "frame_n": 1, "rec": rec(det(score=70.0, offset=3.0))}]
    out = sa.assemble(SCAN, cap)
    assert out["best"]["isolated"] and out["reason"].endswith("seen at one angle only")


def test_corroboration_wraps_around_180():
    """174.1° 와 -178.8° 는 7.1° 차이 — 같은 물체."""
    cap = [{"rotation_deg": 135, "frame_n": 1, "rec": rec(det(offset=39.1))},
           {"rotation_deg": 180, "frame_n": 2, "rec": rec(det(offset=1.2))}]
    out = sa.assemble(SCAN, cap)
    assert [e["seen_at"] for e in out["frames"]] == [[135, 180], [135, 180]]


# ---------------------------------------------------------------- Go1 거리: 바닥선 · UniDepth 는 valid 아님 (2026-11-24 소화기)
UD = {"value": 4.294, "space": "metric", "source": "measured", "age": 0}
GD = {"distance_cm": 168.7, "reason": None}


def test_go1_floor_object_uses_ground_line():
    """13:35 소화기 180°: UniDepth 4.29 m(실제 1.5~2 m) → 바닥선 1.69 m."""
    d = sa.distance(det(label="fire extinguisher", dist=UD, ground_distance=GD))
    assert d == {"distance_m": 1.687, "distance_valid": True, "distance_source": "ground_line", "distance_note": None}


def test_go1_unidepth_alone_is_not_valid():
    d = sa.distance(det(label="fire extinguisher", dist=UD, ground_distance={"distance_cm": None,
                                                                             "reason": "bottom edge touches the frame"}))
    assert not d["distance_valid"] and d["distance_m"] == 4.294 and d["distance_source"] == "measured"
    assert d["distance_note"].startswith(sa.GO1_DEPTH_NOTE) and "bottom edge touches the frame" in d["distance_note"]


def test_go1_desk_object_does_not_use_ground_line():
    d = sa.distance(det(label="cup", dist=UD, ground_distance=GD))
    assert not d["distance_valid"] and "not standing on the floor" in d["distance_note"]


def test_go1_box_height_still_wins():
    bh = {"value": 1.889, "space": "metric", "source": "box_height", "age": 0}
    d = sa.distance(det(label="demo_box", dist=bh, box_distance={"distance_cm": 188.9, "reason": None},
                        ground_distance={"distance_cm": 218.0, "reason": None}))
    assert d["distance_source"] == "box_height" and d["distance_m"] == 1.889 and d["distance_valid"]


def test_go1_known_by_bearing_source_too():
    d = sa.distance(det(label="person", dist=UD, offset=3.0, f_source="go1_box"))
    assert not d["distance_valid"] and "no ground line" in d["distance_note"]


def test_other_camera_unidepth_unchanged():
    assert sa.distance(det(label="person", dist=UD, offset=3.0, f_source="fov"))["distance_valid"]


def test_floor_objects_cover_targets_and_ko_table_floor_items():
    assert {"demo_box", "demo_door", "person", "door", "fire extinguisher"} <= sa.FLOOR_OBJECTS
    assert not {"cup", "laptop", "monitor", "keyboard", "bottle"} & sa.FLOOR_OBJECTS
