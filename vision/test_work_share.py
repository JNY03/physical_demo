"""비전 작업 나눔 · 물체 추적 · 재식별 갤러리 (work/1) — 가짜 실행기로 (모델 적재 없음).

  python -m pytest vision/test_work_share.py -q
"""
import json
import os

import numpy as np
import pytest

os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import object_track as ot  # noqa: E402
import reid_gallery as rg  # noqa: E402
import work_share as ws  # noqa: E402

IDENT = ("unidepth", "lpiccinelli/test-weights")


class FakeDepth:
    """독립 depth 실행기 — 부른 횟수를 센다"""

    def __init__(self):
        self.calls = 0

    def __call__(self, bgr, fov_x=None):
        self.calls += 1
        h, w = bgr.shape[:2]
        K = np.array([[500.0, 0, w / 2], [0, 500.0, h / 2], [0, 0, 1]])
        return {"infer_ms": 1.0, "checkpoint": "fake"}, np.full((h, w), 3.0, np.float32), K


def _dets():
    return [{"cls": 0, "name": "chair", "conf": 0.9, "xyxy": [10, 10, 60, 90]}]


def _vision():
    import vision_infer as vi
    return vi


def test_targeted_distance_reuses_standalone_depth_once_per_frame():
    vi = _vision()
    share = ws.WorkShare(log=None)
    runner = FakeDepth()
    share.register_depth("unidepth", runner, IDENT)
    # 거리 근거 — 같은 가중치면 모델을 따로 올리지 않고 나눠 쓴다
    dt = vi._depth_tracker({"enabled": True, "kind": "unidepth", "model_id": IDENT[1], "period": 5}, share)
    assert isinstance(dt.provider, ws.SharedDepthProvider) and dt.ready
    bgr = np.zeros((120, 160, 3), np.uint8)
    for n in range(6):
        share.begin("pi12", n, None, ["unidepth"])
        dets = _dets()
        dt(bgr, dets)                                   # 지정 물체 탐지가 먼저 (주기와 무관하게 매 장 받는다)
        assert dets[0]["dist"]["source"] == "measured" and dets[0]["dist"]["value"] == 3.0
        assert dets[0]["real_mm"]["fx"] == 500.0         # 독립 모델의 K 로 실물 치수
        info, depth, K = share.depth_for(IDENT, bgr, "unidepth")   # 독립 depth 는 받는다
        assert depth.shape == (120, 160)
    assert runner.calls == 6                             # 한 장에 한 번
    st = share.summary()["pi12"]["depth:unidepth"]
    assert st == {"runs": 6, "reused": 6, "consumers": ["targeted", "unidepth"]}


def test_without_standalone_the_tracker_loads_its_own():
    vi = _vision()
    share = ws.WorkShare(log=None)
    dt = vi._depth_tracker({"enabled": False, "kind": "unidepth", "model_id": "x"}, share)
    assert not isinstance(dt.provider, ws.SharedDepthProvider)


def test_identity_rules():
    assert ws.standalone_identity("unidepth", None, {"model_name": "lpiccinelli/a"}) == ("unidepth", "lpiccinelli/a")
    assert ws.standalone_identity("unidepth", None, {"model_name": "x", "infer_long_side": 640}) is None
    a = ws.standalone_identity("da3_metric_l", {"runner": "da3_metric:Runner"}, {"model_path": "models/da3-metric-large"})
    assert a[0] == "depth_anything_3" and a[1].endswith("models/da3-metric-large")
    assert ws.standalone_identity("moge2_aerial", None, {}) is None


class FakeTracker:
    def __init__(self):
        self.seen = []

    def __call__(self, bgr, dets=None, stream="default", ts=None):
        self.seen.append(dets)
        return {"track_ms": 0.1}, [dict(d, track_id=i + 1) for i, d in enumerate(dets)]


def test_track_step_reads_detector_output_and_record_shape():
    vi = _vision()
    share = ws.WorkShare(log=None)
    tr = FakeTracker()
    step = ot.ObjectTrackStep("botsort", tr, "object_detect", "yoloe", min_conf=0.25)
    bgr = np.zeros((100, 100, 3), np.uint8)
    for n in range(3):
        share.begin("pi12", n)
        share.put(("detections", "yoloe"), _dets() + [{"cls": 1, "name": "x", "conf": 0.1, "xyxy": [0, 0, 5, 5]}],
                  "yoloe")
        rec = step.step(share, "pi12", {"n": n, "source": "pi12", "frame": "frames/%d.jpg" % n,
                                        "image_wh": [100, 100]}, bgr, 100.0 + n)
    assert len(tr.seen[-1]) == 1                         # min_conf 아래는 거른다
    assert rec["model"] == "track" and rec["function"] == "object_track" and rec["tracker"] == "botsort"
    assert rec["input"] == {"function": "object_detect", "model": "yoloe"}
    t = rec["tracks"][0]
    assert set(t) == {"track_id", "label", "confidence", "bbox", "hits", "first_ts", "last_ts", "trajectory"}
    assert t["hits"] == 3 and len(t["trajectory"]) == 3 and t["trajectory"][-1] == [102.0, 35.0, 50.0]
    v2 = vi.v2_record(rec, {})
    assert v2["detections"][0]["track_id"] == 1 and v2["detections"][0]["label"] == "chair"
    assert v2["stream"] == "pi12" and v2["model"] == "track"
    json.dumps(v2)
    assert share.summary()["pi12"]["detections:yoloe"]["reused"] == 3   # 탐지는 다시 돌리지 않는다
    share.begin("pi12", 9)
    assert step.step(share, "pi12", {"n": 9}, bgr, 109.0) is None   # 그 장에 입력이 없으면 없음


def test_judged_detectors_track_only_suspects():
    step = ot.ObjectTrackStep("botsort", FakeTracker(), "targeted_detect", "targeted")
    got = step._pick([dict(_dets()[0], assess={"suspect": "chair"}), dict(_dets()[0], assess={})])
    assert len(got) == 1 and got[0]["name"] == "chair"


def test_resolve_tracking_from_profile():
    assert ot.resolve({}, ["yoloe"]) is None
    r = ot.resolve({"function_params": {"object_track": {"input": "targeted_detect"}}}, ["targeted", "yoloe"])
    assert r == {"tracker": "botsort", "function": "targeted_detect", "model": "targeted"}
    r = ot.resolve({"functions": {"object_track": {"server": {"tracker": "bytetrack"}}},
                    "function_params": {"object_track": {"tracker": "botsort"}}}, ["yoloe", "targeted"])
    assert r["tracker"] == "bytetrack" and r["function"] == "object_detect" and r["model"] == "yoloe"


def test_real_bytetrack_runner_through_step():
    pytest.importorskip("ultralytics")
    pytest.importorskip("lap")
    from runners import bytetrack
    step = ot.ObjectTrackStep("bytetrack", bytetrack.Runner({}, "cpu"), "object_detect", "yoloe")
    share = ws.WorkShare(log=None)
    for n in range(5):
        share.begin("a2", n)
        share.put(("detections", "yoloe"), [{"cls": 0, "name": "p", "conf": 0.9,
                                             "xyxy": [10 + 3 * n, 10, 60 + 3 * n, 90]}], "yoloe")
        rec = step.step(share, "a2", {"n": n}, np.zeros((120, 160, 3), np.uint8), 1.0 + 0.1 * n)
    assert rec["tracks"][0]["track_id"] >= 1 and rec["tracks"][0]["hits"] >= 4 and rec["tracker"] == "bytetrack"


# ---------------------------------------------------------------- 재식별 갤러리
def _e(v):
    return rg.enc(np.asarray(v, np.float32) / np.linalg.norm(v))


def test_group_gallery_links_sources_and_respects_cooccurrence():
    g = rg.GroupGallery("a2+pi12", ["a2", "pi12"], sim=0.9, tol_s=0.5)
    g.feed("pi12", 10.0, [{"track_id": 1, "label": "person", "emb": _e([1, 0, 0])},
                          {"track_id": 2, "label": "person", "emb": _e([0.99, 0.1, 0])}])   # 같은 장 — 다른 물체
    g.feed("a2", 12.0, [{"track_id": 7, "label": "person", "emb": _e([1, 0.02, 0])}])
    out = g.drain()                                      # 12.0 − 0.5 까지만 — a2 는 아직 기다린다
    assert [(a["source"], a["track_id"]) for a in out] == [("pi12", 1), ("pi12", 2)]
    assert out[0]["global_id"] != out[1]["global_id"]   # 함께 보인 둘은 같은 번호가 아니다
    out = g.drain(flush=True)
    assert out[0]["source"] == "a2" and out[0]["global_id"] == g.assign[("pi12", 1)] and not out[0]["new"]
    g.feed("a2", 13.0, [{"track_id": 8, "label": "chair", "emb": _e([1, 0, 0])}])
    assert g.drain(flush=True)[0]["new"]                 # 이름이 다르면 잇지 않는다


def test_reid_step_scope_source_and_group_outbox():
    class Emb:
        def __call__(self, bgr, boxes):
            return {"infer_ms": 2.0}, np.tile(np.eye(4, dtype=np.float32)[:1], (len(boxes), 1))
    rec = {"tracks": [{"track_id": 3, "label": "chair", "bbox": [0, 0, 10, 10]}]}
    src = ot.ReidStep({"function": "reid", "scope": "source", "group_sources": None, "embedder": "m",
                       "in_process": True}, Emb())
    src(rec, None, "pi12", 1.0)
    assert rec["tracks"][0]["global_id"] == 1 and rec["identity"]["scope"] == "source"
    assert [k for k, _f in src.outbox] == ["reid_identity:pi12"]
    conf = ot.resolve_reid({"function_params": {"reid": {"scope": "group", "group_sources": ["pi12", "a2"]}}},
                           ["pi12"])
    assert conf["in_process"] is False and conf["group_sources"] == ["a2", "pi12"]
    grp = ot.ReidStep(conf, Emb())
    rec = {"tracks": [{"track_id": 3, "label": "chair", "bbox": [0, 0, 10, 10]}]}
    grp(rec, None, "pi12", 1.0)
    key, fields = grp.outbox[0]
    assert key == "reid_embed:pi12"
    doc = json.loads(fields["result"])
    assert doc["group"] == "a2+pi12" and doc["items"][0]["track_id"] == 3
    assert rec["identity"]["gallery"] == "vision/reid_gallery.py"
