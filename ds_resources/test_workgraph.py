"""작업 그래프 (work/1) — 같은 일 한 번 · 다른 모델 경고 · fps 최대 · GPU 한 번 · producer 고르기 · 재식별 범위.

클러스터 없이 가짜(test_controller.make)로 돈다.
"""
import copy

import pytest

from ds_resources import functions as fnc
from ds_resources import slots as slt
from ds_resources import streams
from ds_resources import workgraph as wg
from ds_resources.test_controller import make

REG = fnc._registry(None)
FUNCS = fnc.load(REG)


def build(functions, picks=None, params=None, sources=("pi12",), fps=None, active=None, reg=REG):
    funcs = fnc.load(reg)
    sel = slt.effective(reg, funcs, slt.check_picks(reg, funcs, picks or {}), None)
    p = wg.check_params(funcs, params, functions, list(sources))
    sel = wg.apply_model_params(funcs, sel, picks, p)
    return wg.build(reg, funcs, functions, list(sources), sel, p, fps or {s: 5.0 for s in sources}, active)


def nodes_of(out, artifact):
    return [n for n in out["work"] if artifact in n["artifacts"]]


# ---------------------------------------------------------------- 같은 일 · 다른 모델
def test_same_model_is_one_shared_node_and_gpu_counted_once():
    # depth_estimate 의 unidepth 선택지(정체 unidepth_l) = 지정 물체 탐지 거리 근거 unidepth_l
    out = build(["targeted_detect", "depth_estimate"],
                picks={"targeted_detect": {"server": {"depth": "unidepth_l"}}, "depth_estimate": {"server": {"model": "unidepth"}}},
                active={"targeted_detect": {"unidepth_l", "yoloe_text"}})
    depth = nodes_of(out, "depth")
    assert len(depth) == 1
    d = depth[0]
    assert d["identity"] == "unidepth_l" and d["shared"] is True
    assert sorted(d["consumers"]) == ["depth_estimate", "targeted_detect"]
    assert d["stage"] == "server" and d["unit"] == "depth_estimate.server"
    assert [w for w in out["work_warnings"] if w["code"] == "dup_kind"] == []
    # GPU 는 노드마다 한 번 — 기능마다 따로 셌을 때보다 그 몫만큼 적다
    assert out["work_gpu"]["naive_mb"] - out["work_gpu"]["total_mb"] == out["work_gpu"]["saved_mb"] == d["gpu_mb"] > 0


def test_different_model_same_kind_is_warned_with_alignable_keys():
    out = build(["targeted_detect", "depth_estimate"],
                picks={"depth_estimate": {"server": {"model": "da3_metric_l"}}},
                active={"targeted_detect": {"depth", "yoloe_text"}})
    assert len(nodes_of(out, "depth")) == 2                      # 둘 다 돈다
    w = [w for w in out["work_warnings"] if w["code"] == "dup_kind"]
    assert len(w) == 1 and w[0]["artifact"] == "depth" and w[0]["source"] == "pi12"
    assert w[0]["functions"] == ["depth_estimate", "targeted_detect"]
    assert set(w[0]["models"]) == {"da3_metric_l", "depth"} and w[0]["models"][0] == "da3_metric_l"
    # 런처 [맞추기] — 기능마다 그 슬롯의 선택지 키
    assert w[0]["align"] == {"depth_estimate": {"stage": "server", "role": "model", "model": "da3_metric_l"},
                             "targeted_detect": {"stage": "server", "role": "depth", "model": "da3_metric_l"}}
    assert w[0]["save_mb"] == 1300 and w[0]["reason"] == "같은 일을 다른 모델로 두 번 함 — 하나로 맞추면 GPU 1300 MB 절약"
    assert "dup_kind" in streams.REASON_CODES


def test_align_uses_each_slots_own_key_for_the_same_weights():
    # unidepth(거리 추정 선택지) ↔ unidepth_l(지정 물체 탐지 depth 역할) — 같은 가중치, 다른 이름
    out = build(["targeted_detect", "depth_estimate"], active={"targeted_detect": {"depth"}})
    w = [w for w in out["work_warnings"] if w["code"] == "dup_kind"][0]
    assert w["align"]["depth_estimate"]["model"] == "unidepth"
    assert w["align"]["targeted_detect"]["model"] == "unidepth_l"
    for fn, a in w["align"].items():                             # 넣을 키는 그 슬롯의 선택지다
        st = FUNCS[fn]["stages"][a["stage"]]
        sl = next(x for x in st["slots"] if x["role"] == a["role"])
        assert a["model"] in slt.choices(sl, a["stage"], st, slt.parts(REG))


def test_internal_arms_of_a_domain_function_are_not_dup_kind():
    out = build(["targeted_detect", "object_detect"])
    assert [w for w in out["work_warnings"] if w["code"] == "dup_kind" and w["artifact"] == "detections"] == []


def test_inactive_secondary_slots_make_no_node():
    out = build(["targeted_detect"], active={"targeted_detect": {"yoloe_text"}})
    assert nodes_of(out, "depth") == []


# ---------------------------------------------------------------- fps
def test_node_fps_is_max_of_consumers():
    reg = copy.deepcopy(REG)
    reg["functions"] = {k: v for k, v in reg["functions"].items() if k in ("object_detect", "segmentation")}
    reg["functions"]["object_detect"]["stages"]["device"]["fps"] = 2
    reg["functions"]["segmentation"]["stages"]["device"]["fps"] = 4
    picks = {"object_detect": {"device": {"detector": "yolo26n_seg_cpu"}},
             "segmentation": {"device": {"segmenter": "yolo26n_seg_cpu"}}}
    out = build(["object_detect", "segmentation"], picks=picks, reg=reg, fps={"pi12": 1.0})
    dev = [n for n in out["work"] if n["stage"] == "device"]
    assert len(dev) == 1 and dev[0]["shared"] and dev[0]["fps"] == 4.0
    assert dev[0]["unit"] in ("object_detect.device@pi12", "segmentation.device@pi12")


# ---------------------------------------------------------------- producer 고르기
def test_tracking_input_default_is_highest_priority_then_name():
    out = build(["object_track", "targeted_detect", "object_detect"])
    row = out["work_inputs"]["object_track"]["pi12"][0]
    assert row["param"] == "input" and row["how"] == "default"
    assert row["choices"] == ["object_detect", "targeted_detect"] and row["function"] == "object_detect"
    track = nodes_of(out, "tracks")[0]
    assert track["model"] == "botsort"
    edge = [e for e in out["work_edges"] if e["function"] == "object_track" and e["how"] == "default"][0]
    assert edge["from"] == row["node"] and edge["to"] == track["node"]
    assert "object_track" in [n for n in out["work"] if n["node"] == row["node"]][0]["consumers"]


def test_tracking_input_param_and_fallback_and_missing():
    out = build(["object_track", "targeted_detect", "object_detect"],
                params={"object_track": {"input": "targeted_detect"}})
    assert out["work_inputs"]["object_track"]["pi12"][0]["function"] == "targeted_detect"
    assert out["work_inputs"]["object_track"]["pi12"][0]["how"] == "param"
    out = build(["object_track", "object_detect"], params={"object_track": {"input": "river_detect"}})
    assert out["work_inputs"]["object_track"]["pi12"][0]["function"] == "object_detect"
    assert [w["code"] for w in out["work_warnings"]] == ["param_fallback"]
    out = build(["object_track"])
    assert out["work_inputs"]["object_track"]["pi12"][0]["how"] == "none"
    assert [w["code"] for w in out["work_warnings"]] == ["no_producer"]


def test_tracker_param_is_used_unless_slot_picked():
    out = build(["object_track", "object_detect"], params={"object_track": {"tracker": "bytetrack"}})
    assert nodes_of(out, "tracks")[0]["model"] == "bytetrack"
    out = build(["object_track", "object_detect"], params={"object_track": {"tracker": "bytetrack"}},
                picks={"object_track": {"server": {"tracker": "botsort"}}})
    assert nodes_of(out, "tracks")[0]["model"] == "botsort"


# ---------------------------------------------------------------- 물체 추적 기능 (계약 · 펼치기 · 배치)
def test_object_track_contract_and_expansion():
    g = FUNCS["object_track"]
    assert g["pattern"] == "cascade" and g["produces"] == ["tracks"] and g["consumes"] == ["detections"]
    srv = {s["role"]: s for s in g["stages"]["server"]["slots"]}
    assert srv["tracker"]["default"] == "botsort"
    ch = slt.choices(srv["tracker"], "server", g["stages"]["server"], slt.parts(REG))
    assert ch[:1] == ["botsort"] and "bytetrack" in ch
    dev = {s["role"]: s for s in g["stages"]["device"]["slots"]}
    assert "bytetrack" in slt.choices(dev["tracker"], "device", g["stages"]["device"], slt.parts(REG))
    st, _r, _o = fnc.expand(FUNCS, ["object_track"], sources=["pi12", "a2"])
    assert sorted(s["unit"] for s in st) == ["object_track.device@a2", "object_track.device@pi12",
                                             "object_track.server"]


def test_plan_carries_work_and_places_tracking(tmp_path):
    k, r, _c = make(tmp_path)
    r.add_frames("pi12")
    p = k.prepare(["pi12"], ["yoloe"], {"function_params": {"object_track": {"input": "object_detect"}}},
                  functions=["object_detect", "object_track"])
    plan = p["plan"]
    assert "object_track.server" in {u["task"] for u in plan["cpu_units"]}
    assert plan["functions"]["object_track"]["verdict"] in ("ok", "degraded")
    row = [n for n in plan["work"] if n["artifact"] == "detections" and n["stage"] == "server"][0]
    assert set(row) >= {"source", "node", "artifact", "model", "stage", "fps", "consumers", "shared", "gpu_mb", "unit"}
    assert row["consumers"] == ["object_detect", "object_track"] and row["shared"]
    assert row["unit"] == "gpu:yoloe" and row["fps"] == 5.0
    assert plan["function_params"]["object_track"]["input"] == "object_detect"


def test_plan_without_overlap_is_unchanged_except_added_work_fields(tmp_path, monkeypatch):
    """골든 — 기능이 겹치지 않으면 배치표는 work 칸이 더해지는 것 말고 그대로다"""
    k, r, _c = make(tmp_path)
    r.add_frames("pi12")
    r.add_frames("a2")
    args = (["pi12", "a2"], ["yoloe"], None)
    kw = dict(functions=["object_detect", "depth_estimate", "slam"])
    new = k.prepare(*args, **kw)["plan"]
    monkeypatch.setattr(k, "work_plan", lambda *a, **kw2: {})
    old = k.prepare(*args, **kw)["plan"]
    added = set(new) - set(old)
    assert added == {"work", "work_edges", "work_warnings", "work_inputs", "work_reid", "work_gpu", "function_params"}
    # 스냅숏 해시는 이 기기의 /proc 순간값(부하 · 메모리)이라 두 번 부르면 갈린다 — 요청 · 레지스트리 해시만 견준다
    new["inputs"].pop("snapshot")
    old["inputs"].pop("snapshot")
    assert {k2: v for k2, v in new.items() if k2 not in added} == old
    assert all(not n["shared"] for n in new["work"]) and new["work_warnings"] == []


def test_estimate_unchanged_without_overlap_and_counts_shared_once():
    from ds_resources import gpu_executor as ge
    cfg = {}
    base, need = ge.estimate_mb(["targeted"], None, cfg)
    assert ge.estimate_mb(["targeted", "yoloe"], None, cfg)[1] == need            # 겹치지 않으면 근거 그대로
    prof = {"functions": {"targeted_detect": {"server": {"depth": "unidepth_l"}}},
            "models": {"targeted": {"targets": [{"name": "chair"}], "distance": True}}}
    _mb, n1 = ge.estimate_mb(["targeted"], prof, cfg)
    _mb, n2 = ge.estimate_mb(["targeted", "unidepth"], prof, cfg)
    assert "unidepth_l" in n1 and "unidepth_l" not in n2                         # unidepth 선택지가 이미 적재한다
    assert wg.shared_need(REG, ["targeted", "yoloe"], set(need)) == set()


# ---------------------------------------------------------------- 재식별 범위
def test_reid_scope_validation():
    on = {"pi12": ["reid"], "a2": ["reid"], "robot1": ["object_detect"]}
    assert wg.check_params(FUNCS, None, on)["reid"] == {"scope": "source", "group_sources": None}
    got = wg.check_params(FUNCS, {"reid": {"scope": "group"}}, on)
    assert got["reid"]["group_sources"] == ["a2", "pi12"]                        # 기본: 그 기능을 켠 소스 전부
    with pytest.raises(ValueError, match="2개 이상"):
        wg.check_params(FUNCS, {"reid": {"scope": "group", "group_sources": ["pi12"]}}, on)
    with pytest.raises(ValueError, match="켜져 있지 않다"):
        wg.check_params(FUNCS, {"reid": {"scope": "group", "group_sources": ["pi12", "robot1"]}}, on)
    with pytest.raises(ValueError, match="중 하나"):
        wg.check_params(FUNCS, {"reid": {"scope": "zone"}}, on)
    with pytest.raises(ValueError, match="매개변수가 없다"):
        wg.check_params(FUNCS, {"reid": {"radius": 3}}, on)
    with pytest.raises(ValueError, match="모르는 기능"):
        wg.check_params(FUNCS, {"nope": {}}, on)


def test_reid_group_plan_shape():
    out = build({"pi12": ["reid", "object_detect"], "a2": ["reid", "object_detect"]},
                params={"reid": {"scope": "group", "group_sources": ["pi12", "a2"]}}, sources=())
    row = out["work_reid"]["reid"]
    assert row["scope"] == "group" and row["group_sources"] == ["a2", "pi12"] and row["group"] == "a2+pi12"
    assert row["gallery"]["redis_out"] == "reid_identity:a2+pi12"
    assert row["embedder"]["server"] == "mobileclip_s0_cpu"
    gal = [n for n in out["work"] if n["node"] == row["gallery"]["node"]][0]
    assert gal["source"] is None and gal["sources"] == ["a2", "pi12"] and gal["artifact"] == "identities"
    assert {e["source"] for e in out["work_edges"] if e["to"] == gal["node"]} == {"a2", "pi12"}
    src = build(["reid", "object_detect"], sources=("pi12",))["work_reid"]["reid"]
    assert src["scope"] == "source" and src["gallery"]["where"] == "per_source"


def test_places_json_has_param_schema_and_artifacts(tmp_path):
    k, r, _c = make(tmp_path)
    r.add_frames("pi12")
    pl = k.places(functions=["object_detect", "targeted_detect", "object_track", "reid"], sources=["pi12"])
    fp = pl["function_params"]
    assert fp["reid"]["scope"]["choices"] == ["source", "group"] and fp["reid"]["scope"]["default"] == "source"
    assert fp["reid"]["scope"]["labels"] == {"source": "소스 안", "group": "소스 묶음"}
    assert fp["reid"]["group_sources"]["min"] == 2
    assert fp["object_track"]["input"]["choices"] == ["object_detect", "targeted_detect"]
    assert fp["object_track"]["input"]["default"] == "object_detect"
    assert pl["artifacts"]["object_track"] == {"produces": ["tracks"], "consumes": ["detections"]}
    assert pl["artifacts"]["depth_estimate"]["produces"] == ["depth"]


def test_bad_function_params_are_refused():
    with pytest.raises(ValueError):
        fnc._params({"x": {"kind": "choice", "choices": ["a"], "default": "b"}}, "t")
    with pytest.raises(ValueError):
        fnc._params({"x": {"kind": "model"}}, "t")
    with pytest.raises(ValueError):
        fnc._artifacts(["pixels"], "t")
