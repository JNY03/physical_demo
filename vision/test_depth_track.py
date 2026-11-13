"""거리 단계 — 주기·전파·나이의 경계.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../.venv/bin/python -m pytest test_depth_track.py -q
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "depth_track_under_test", Path(__file__).with_name("depth_track.py"))
dt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dt)

IMG = np.zeros((100, 100, 3), np.uint8)


class FakeDepth:
    """부르면 센다. 모든 화소가 `value` m 인 지도를 돌려준다."""
    space = dt.METRIC

    def __init__(self, value=10.0):
        self.calls, self.value = 0, value

    def __call__(self, bgr):
        self.calls += 1
        return np.full((50, 50), self.value, np.float32)


def tracker(period=5, **over):
    spec = {"enabled": True, "kind": "hf_metric", "period": period}
    spec.update(over)
    t = dt.DepthTracker.__new__(dt.DepthTracker)
    t.period, t.percentile, t.max_age = period, 40.0, int(spec.get("max_age", 30))
    t.iou_match, t.classes = 0.5, tuple(spec.get("classes") or ())
    t.provider = FakeDepth(spec.get("value", 10.0))
    t.space, t.ready, t.why = dt.METRIC, True, None
    t._mem, t._n = {}, 0
    return t


def det(tid=1, box=(10, 10, 30, 30), name="butane_canister"):
    return {"name": name, "conf": 0.5, "xyxy": list(box), "track_id": tid}


def test_it_measures_on_the_first_frame_then_carries():
    t = tracker(period=5)
    d = det()
    t(IMG, [d])
    assert d["dist"]["source"] == "measured" and d["dist"]["age"] == 0
    for i in range(1, 5):
        d = det()
        t(IMG, [d])
        assert d["dist"]["source"] == "carried", i
        assert d["dist"]["age"] == i
    assert t.provider.calls == 1                 # 다섯 장에 한 번


def test_a_new_track_does_not_wait_for_the_period():
    """새 위험물이 다음 주기까지 거리 없이 떠 있으면 안 된다."""
    t = tracker(period=10)
    t(IMG, [det(tid=1)])
    assert t.provider.calls == 1
    newcomer = det(tid=2)
    t(IMG, [det(tid=1), newcomer])
    assert t.provider.calls == 2                 # 주기를 당겼다
    assert newcomer["dist"]["source"] == "measured"


def test_a_carried_distance_is_corrected_by_apparent_size():
    """상자가 2배 커지면 거리는 대략 절반이다(핀홀)."""
    t = tracker(period=99)
    t(IMG, [det(box=(10, 10, 30, 30))])          # 폭 20, 거리 10 m
    d = det(box=(10, 10, 50, 50))                # 폭 40 → 2배
    t(IMG, [d])
    assert d["dist"]["source"] == "carried"
    assert abs(d["dist"]["value"] - 5.0) < 0.6
    assert d["dist"]["scale"] == 2.0


def test_an_old_carried_value_is_dropped_not_stretched():
    """**모르면 모른다고 한다.** 오래된 값을 계속 들고 있지 않는다."""
    t = tracker(period=99, max_age=3)
    t(IMG, [det()])
    for _ in range(3):
        t(IMG, [det()])
    d = det()
    t(IMG, [d])
    assert d["dist"] is None


def test_every_distance_carries_the_frame_it_came_from():
    """상대 깊이를 미터처럼 쓰지 않기 위해 **space 를 함께 낸다**(AI-C-02)."""
    t = tracker()
    d = det()
    t(IMG, [d])
    assert d["dist"]["space"] == dt.METRIC


def test_only_declared_classes_are_measured():
    t = tracker(classes=("butane_canister",))
    other = det(name="wooden_box_mine")
    t(IMG, [other])
    assert t.provider.calls == 0 and "dist" not in other


def test_a_provider_that_will_not_load_disables_distance_only():
    """가중치가 없는 기계에서 **세우기를 막지 않는다**(AI-C-05)."""
    t = dt.DepthTracker({"enabled": True, "kind": "hf_metric",
                         "model_id": "nope/does-not-exist"}, log=None)
    assert t.ready is False and t.why
    d = det()
    assert t(IMG, [d]) == 0.0
    assert "dist" not in d                       # 거리를 **지어내지 않는다**


def test_an_unknown_kind_is_reported_not_guessed():
    t = dt.DepthTracker({"enabled": True, "kind": "telepathy"}, log=None)
    assert t.ready is False and "kind" in t.why
