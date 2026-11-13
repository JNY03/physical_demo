"""river 의 부품 접기(_drop_nested) — 큰 상자 안의 작은 상자는 지우고, 다른 위험물은 남긴다.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../venv/bin/python -m pytest test_river_nested.py -q
"""
from types import SimpleNamespace

from vision_infer import RiverRunner


def _d(box, sus=None, name="unconfirmed", score=50.0):
    return {"xyxy": box, "name": name, "assess": {"suspect": sus, "score": score}}


def _run(dets, ratio=0.7):
    return RiverRunner._drop_nested(SimpleNamespace(nest_ratio=ratio), dets)


def test_parts_inside_whole_are_dropped():
    can = _d([100, 100, 200, 400], "butane_canister", name="butane_canister", score=50)
    cap = _d([130, 100, 170, 140], "butane_canister", score=60)     # 점수가 더 높아도 부품이다
    label = _d([110, 200, 190, 300])                                 # 의심 아님
    kept, n = _run([cap, can, label])
    assert kept == [can] and n == 2


def test_other_hazard_inside_big_box_is_kept():
    box = _d([0, 0, 400, 400], "wooden_box_mine")
    can = _d([50, 50, 150, 350], "butane_canister")                  # 큰 상자에 덮였지만 다른 위험물
    kept, n = _run([box, can])
    assert kept == [box, can] and n == 0


def test_side_by_side_overlap_is_not_nesting():
    a, b = _d([0, 0, 200, 200]), _d([150, 0, 350, 200])              # 25% 만 겹친다
    kept, n = _run([a, b])
    assert len(kept) == 2 and n == 0


def test_use_stream_keeps_caches_per_device():
    v = SimpleNamespace(_last=[])
    dp = SimpleNamespace(_mem={}, _n=0)
    r = SimpleNamespace(verifier=v, depth=dp, _stream=None, _streams={})
    use = lambda k: RiverRunner.use_stream(r, k)
    use(("robot1", "a")); v._last = ["r1"]; dp._mem = {"k": 1}; dp._n = 3
    use(("pi13", "b"))
    assert v._last == [] and dp._mem == {} and dp._n == 0         # 다른 기기는 빈 기억에서
    v._last = ["p13"]
    use(("robot1", "a"))
    assert v._last == ["r1"] and dp._mem == {"k": 1} and dp._n == 3   # 돌아오면 제 것
    use(("pi13", "b"))
    assert v._last == ["p13"]


def test_group_box_of_two_objects_is_dropped():
    r = SimpleNamespace(group_split={"member_min": 0.15, "inside": 0.8, "cover": 0.5})
    group = _d([0, 0, 400, 300])                                     # 둘을 감싼 상자
    a, b = _d([10, 10, 190, 290]), _d([210, 20, 390, 280])
    kept, n = RiverRunner._drop_groups(r, [group, a, b])
    assert kept == [a, b] and n == 1


def test_whole_with_small_parts_is_not_a_group():
    r = SimpleNamespace(group_split={"member_min": 0.15, "inside": 0.8, "cover": 0.5})
    can = _d([0, 0, 100, 300])
    cap, label = _d([30, 0, 70, 30]), _d([5, 100, 95, 250])          # 뚜껑 4% · 라벨 45%
    kept, n = RiverRunner._drop_groups(r, [can, cap, label])
    assert can in kept and n == 0


def test_smoother_holds_decision_near_threshold():
    from vision_infer import AssessSmoother
    s = AssessSmoother({"alpha": 0.4, "hysteresis": 5})
    def det(score):
        return {"track_id": 1, "assess": {"scores": {"butane_canister": score, "wooden_box_mine": 10.0},
                                          "min_score": 45, "suspect": "butane_canister" if score >= 45 else None}}
    seq = [50, 43, 47, 42, 48, 44, 46]                               # 하한 45 를 오가는 날 점수
    out = [s.update([det(v)])[0]["assess"]["suspect"] for v in seq]
    assert out == ["butane_canister"] * len(seq)                     # 판정은 흔들리지 않는다


def test_unnamed_scene_box_does_not_erase_real_can():
    scene = _d([0, 0, 800, 600], "butane_canister", score=55)        # 이름 없는 장면 상자
    can = _d([300, 100, 450, 500], "butane_canister", name="butane_canister", score=48)
    label = _d([320, 250, 430, 400], "butane_canister", score=60)    # 통의 라벨
    kept, n = _run([scene, can, label])
    assert kept == [can] and n == 2                                  # 라벨만 남던 연쇄가 없다
