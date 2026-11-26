"""바꾸면 무엇이 달라지는지 미리 보는 도구 (tools/preview_apply.py).

두 겹으로 나눈다.
  가중치 없이   가짜 실행기(make=)를 넣어 **고르기 · 설정 고치기 · 오버레이 · 차이 요약 · 런처 계약**을 전부 본다.
                vision_infer 의 적재만 가짜고 추론 뒤의 길(오버레이 그리기 · v2 기록)은 진짜를 지난다.
  가중치 있을 때  models/yoloe-11l-seg-pf.pt 와 저장된 장이 있는 기계에서만 실제로 한 번 돌린다 (없으면 skip).
"""
import json
from pathlib import Path

import pytest

from tools import preview_apply as pa

REPO = Path(__file__).resolve().parent.parent
YOLOE_PT = REPO / "models" / "yoloe-11l-seg-pf.pt"
FRAMES = REPO / "stream_data" / "pi12" / "20261126_155152" / "frames"


def _cv2():
    """cv2 가 없는 core venv 에서는 그림이 드는 시험을 건너뛴다 (가짜 실행기여도 오버레이는 진짜로 그린다)"""
    return pytest.importorskip("cv2")


def fake_maker(dets, seen=None):
    """make(모델, 모델설정, 장치) — 가중치 대신 미리 정해 둔 검출을 내놓는다. conf 설정은 지킨다."""
    def make(model, mcfg, device):
        if seen is not None:
            seen.append((model, device))

        def run(bgr):
            floor = float(mcfg.get("conf", 0.25))
            keep = [dict(d) for d in dets if d["conf"] >= floor]
            return {"infer_ms": 1.0, "weight": "fake.pt", "prompt": "free"}, keep
        return run
    return make


DETS = [{"cls": 0, "name": "person", "conf": 0.90, "xyxy": [10.0, 10.0, 60.0, 90.0]},
        {"cls": 1, "name": "chair", "conf": 0.50, "xyxy": [70.0, 20.0, 120.0, 100.0]},
        {"cls": 2, "name": "cup", "conf": 0.30, "xyxy": [20.0, 60.0, 40.0, 80.0]}]


@pytest.fixture
def shot(tmp_path):
    """가짜 그림 한 폴더 — (폴더, 가짜 가중치 파일). 가중치는 **있는지만** 보므로 빈 파일이면 된다"""
    cv2 = _cv2()
    import numpy as np
    pics = tmp_path / "pics"
    pics.mkdir()
    for i in range(1, 6):
        img = np.full((160, 240, 3), 40 * i % 255, np.uint8)
        cv2.imwrite(str(pics / ("%09d.jpg" % i)), img)
    weight = tmp_path / "fake.pt"
    weight.write_bytes(b"not a real weight")
    return pics, weight


def run_preview(shot, tmp_path, **kw):
    pics, weight = shot
    sets = ["models.yoloe.weight=%s" % weight] + list(kw.pop("sets", []))
    return pa.preview(str(pics), ["yoloe"], str(tmp_path / "out"), sets=sets,
                      make=kw.pop("make", fake_maker(DETS)), **kw)


# ---------------------------------------------------------------- 고르기
def test_spread_picks_evenly_and_keeps_both_ends():
    """최근 묶음에서 고르게 — 처음과 끝이 들어가야 가장 오래된 쪽과 최신 쪽을 함께 본다"""
    got = pa.spread(list(range(100)), 5)
    assert len(got) == 5 and got[0] == 0 and got[-1] == 99
    gaps = {b - a for a, b in zip(got, got[1:])}
    assert max(gaps) - min(gaps) <= 1                      # 고르다


def test_spread_one_takes_the_newest():
    assert pa.spread([1, 2, 3], 1) == [3]


def test_spread_never_asks_for_more_than_there_is():
    assert pa.spread([1, 2], 9) == [1, 2]


def test_folder_picks_from_the_recent_pool_only(shot, tmp_path):
    """--recent 는 밑감, --frames 는 볼 장 수 — 오래된 장은 밑감에 들지 않는다"""
    pics, _w = shot
    got = pa.pick_images(str(pics), frames=2, recent=2)
    assert [p.name for p, _h in got] == ["000000004.jpg", "000000005.jpg"]


def test_one_file_is_taken_as_is(shot):
    pics, _w = shot
    got = pa.pick_images(str(pics / "000000003.jpg"))
    assert len(got) == 1 and got[0][0].name == "000000003.jpg"


def test_missing_path_says_so(tmp_path):
    with pytest.raises(pa.Bad):
        pa.pick_images(str(tmp_path / "없는폴더"))


def test_empty_folder_says_so(tmp_path):
    (tmp_path / "빈").mkdir()
    with pytest.raises(pa.Bad):
        pa.pick_images(str(tmp_path / "빈"))


def test_run_folder_brings_the_frame_record_and_the_source_name(tmp_path):
    """stream_data/<소스>/<실행>/ 을 주면 frames.jsonl 의 그 장 기록(sender)과 소스 이름을 같이 읽는다"""
    cv2 = _cv2()
    import numpy as np
    run = tmp_path / "stream_data" / "pi12" / "20261126_000000"
    (run / "frames").mkdir(parents=True)
    cv2.imwrite(str(run / "frames" / "000000001.jpg"), np.zeros((40, 60, 3), np.uint8))
    (run / "frames.jsonl").write_text(json.dumps(
        {"frame_file": "frames/000000001.jpg", "sender": {"camera_id": "pi12", "sequence_id": 1}}) + "\n",
        encoding="utf-8")
    got = pa.pick_images(str(run))
    assert len(got) == 1
    assert (got[0][1] or {}).get("sender", {}).get("camera_id") == "pi12"
    assert pa.source_of(got[0][0]) == "pi12"               # 기기별 오버레이 지정이 걸리는 이름


# ---------------------------------------------------------------- 설정 고치기 (--set)
def test_set_one_makes_the_path_and_reads_json():
    cfg = {}
    assert pa.set_one(cfg, "models.yoloe.conf=0.4") == "models.yoloe.conf=0.4"
    assert cfg["models"]["yoloe"]["conf"] == 0.4
    pa.set_one(cfg, "models.yoloe.prompt.classes=[\"person\"]")
    assert cfg["models"]["yoloe"]["prompt"]["classes"] == ["person"]


def test_set_one_keeps_plain_text_as_text():
    cfg = {}
    pa.set_one(cfg, "models.yoloe.weight=models/yoloe-11l-seg-pf.pt")
    assert cfg["models"]["yoloe"]["weight"] == "models/yoloe-11l-seg-pf.pt"


def test_set_one_refuses_what_is_not_a_pair():
    with pytest.raises(pa.Bad):
        pa.set_one({}, "models.yoloe.conf")


# ---------------------------------------------------------------- 없는 파일 알리기
def test_needs_asks_for_the_weight_the_mode_actually_uses():
    """free 모드는 prompt-free 가중치만, text 모드는 텍스트 가중치·인코더를 본다 —
    쓰지도 않는 가중치가 없다고 알리면 사람이 엉뚱한 파일을 받는다"""
    cfg = {"models": {"yoloe": {"weight": "/w/pf.pt",
                                "prompt": {"mode": "free", "text_weight": "/w/text.pt"}}}}
    paths = [r["path"] for r in pa.needs(cfg, ["yoloe"], {})]
    assert paths == ["/w/pf.pt"]
    cfg["models"]["yoloe"]["prompt"]["mode"] = "text"
    rows = pa.needs(cfg, ["yoloe"], {})
    assert [r["path"] for r in rows if r["required"]] == ["/w/text.pt"]
    assert any("mobileclip" in r["path"] for r in rows if not r["required"])


def test_river_runs_while_one_arm_is_missing_but_not_while_all_are(tmp_path):
    """arm 하나가 없으면 알리고 돈다(degrade). 하나도 없으면 못 돈다 — 둘을 섞으면 안 된다"""
    there = tmp_path / "arm.pt"
    there.write_bytes(b"x")
    cfg = {"models": {"river": {"arms": [{"kind": "yoloe", "weight": str(there)},
                                         {"kind": "free", "weight": "/없는/arm2.pt"}]}}}
    _rows, miss, warn = pa.check_files(cfg, ["river"], {})
    assert not miss and [r["path"] for r in warn] == ["/없는/arm2.pt"]
    cfg["models"]["river"]["arms"][0]["weight"] = "/없는/arm1.pt"
    _rows, miss, warn = pa.check_files(cfg, ["river"], {})
    assert len(miss) == 2 and not warn


def test_missing_weight_blocks_before_loading_and_names_the_file(shot, tmp_path):
    """조용히 실패 금지 — 무엇이 어느 자리에 없어서 못 하는지 돌려주고, 적재는 **하지도 않는다**"""
    seen = []
    res = run_preview(shot, tmp_path, make=fake_maker(DETS, seen),
                      sets=["models.yoloe.weight=/없는/가중치.pt"])
    assert res["ok"] is False and not seen
    assert "/없는/가중치.pt" in res["why"] or "가중치.pt" in res["why"]
    assert [r["path"] for r in res["missing"]] == ["/없는/가중치.pt"]
    assert res["summary"] and res["overlay"] is None


# ---------------------------------------------------------------- 한 쪽 돌리기
def test_preview_draws_the_overlay_and_counts_what_it_found(shot, tmp_path):
    res = run_preview(shot, tmp_path, frames=2)
    assert res["ok"] is True and len(res["images"]) == 2
    one = res["images"][0]["a"]["yoloe"]
    assert one["n"] == 3 and one["labels"] == {"chair": 1, "cup": 1, "person": 1}
    assert one["conf"]["min"] == 0.3 and one["conf"]["max"] == 0.9
    assert one["ms"] >= 0 and one["infer_ms"] == 1.0
    img = Path(one["overlay"])
    assert img.is_file() and img.stat().st_size > 0 and img.parent == Path(res["out"])


def test_the_launcher_four_fields_are_there(shot, tmp_path):
    """런처가 읽는 칸 — ok · overlay(그림 하나) · why · summary(한 줄)"""
    res = run_preview(shot, tmp_path, frames=1)
    assert res["ok"] is True and res["why"] == ""
    assert Path(res["overlay"]).is_file()
    assert isinstance(res["summary"], str) and "yoloe" in res["summary"]
    assert json.loads(json.dumps(res, ensure_ascii=False))["overlay"] == res["overlay"]


def test_only_the_out_folder_is_written(shot, tmp_path):
    """읽기만 한다 — 입력 폴더는 그대로여야 한다 (저장된 장을 건드리지 않는다)"""
    pics, _w = shot
    before = {p.name: p.stat().st_mtime_ns for p in pics.iterdir()}
    run_preview(shot, tmp_path, frames=2)
    assert {p.name: p.stat().st_mtime_ns for p in pics.iterdir()} == before


def test_parameters_reach_the_model(shot, tmp_path):
    """--set 이 진짜로 추론에 걸리는지 — conf 를 올리면 약한 검출이 빠진다"""
    res = run_preview(shot, tmp_path, frames=1, sets=["models.yoloe.conf=0.6"])
    assert res["images"][0]["a"]["yoloe"]["labels"] == {"person": 1}


def test_overlay_min_conf_is_told_apart_from_the_count(shot, tmp_path):
    """그림에 안 그려진 것도 검출 수에는 든다 — 둘을 한 숫자로 내면 그림과 요약이 어긋난다"""
    res = run_preview(shot, tmp_path, frames=1, sets=["overlay.min_conf=0.6"])
    one = res["images"][0]["a"]["yoloe"]
    assert one["n"] == 3 and one["drawn"] == 1


# ---------------------------------------------------------------- 두 설정 견주기
def test_compare_says_what_changed_and_puts_the_two_side_by_side(shot, tmp_path):
    res = run_preview(shot, tmp_path, frames=1, sets_b=["models.yoloe.conf=0.6"])
    assert res["compare"] is True
    d = res["images"][0]["diff"]["yoloe"]
    assert d["n"] == [3, 1] and d["n_delta"] == -2
    assert d["only_a"] == ["chair", "cup"] and d["only_b"] == []
    assert d["same"] is False
    pair = Path(d["image"])
    assert pair.is_file() and pair.name.endswith("_ab_yoloe.jpg")
    assert res["overlay"] == str(pair)                     # 견주기면 결과 창에 나란히 붙인 그림을 띄운다
    assert res["summary"].endswith("1장 중 1장 달라짐")


def test_compare_with_the_same_setting_says_same(shot, tmp_path):
    res = run_preview(shot, tmp_path, frames=1, sets_b=["models.yoloe.conf=0.25"])
    assert res["images"][0]["diff"]["yoloe"]["same"] is True
    assert res["totals"]["changed_images"] == 0


def test_an_overlay_only_change_is_told_as_a_changed_picture(shot, tmp_path):
    """오버레이만 바꾸면 검출은 같고 **그림이 다르다** — 그때 '같음' 으로만 끝내면 바뀐 것을 안 보여 준다"""
    res = run_preview(shot, tmp_path, frames=1, sets_b=["overlay.yoloe.label=[[\"name\"]]"])
    d = res["images"][0]["diff"]["yoloe"]
    assert d["same"] is True and d["image_same"] is False
    assert res["totals"]["changed_images"] == 0 and res["totals"]["changed_overlays"] == 1
    assert "그림만 다른 장 1" in res["summary"]


def test_the_same_weight_is_loaded_once_for_both_sides(shot, tmp_path):
    """오버레이만 다른 두 쪽은 가중치를 함께 쓴다 (적재가 제일 비싸다 — 실측 2.5~3.3초)"""
    seen = []
    res = run_preview(shot, tmp_path, frames=1, make=fake_maker(DETS, seen),
                      sets_b=["overlay.yoloe.label=[[\"name\"]]"])
    assert len(seen) == 1 and res["sides"]["b"]["shared_load"] == ["yoloe"]
    assert res["sides"]["b"]["load_ms"] == 0


def test_a_changed_model_setting_loads_its_own_weights(shot, tmp_path):
    seen = []
    run_preview(shot, tmp_path, frames=1, make=fake_maker(DETS, seen),
                sets_b=["models.yoloe.conf=0.6"])
    assert len(seen) == 2


def test_b_inherits_a_and_changes_only_what_it_says(shot, tmp_path):
    """--set 은 A·B 둘 다에, --set-b 는 B 에만 — B 는 A 위에 얹은 것이다"""
    res = run_preview(shot, tmp_path, frames=1, sets=["models.yoloe.max_det=7"],
                      sets_b=["models.yoloe.conf=0.6"])
    assert "models.yoloe.max_det=7" in res["sides"]["a"]["set"]
    assert "models.yoloe.max_det=7" in res["sides"]["b"]["set"]
    assert "models.yoloe.conf=0.6" in res["sides"]["b"]["set"]
    assert "models.yoloe.conf=0.6" not in res["sides"]["a"]["set"]


def test_diff_one_counts_labels_that_only_changed_in_number():
    a = {"n": 5, "labels": {"chair": 5}, "ms": 1.0}
    b = {"n": 3, "labels": {"chair": 3}, "ms": 2.0}
    d = pa.diff_one(a, b)
    assert d["changed"] == {"chair": [5, 3]} and not d["only_a"] and d["same"] is False


def test_caption_writes_only_the_difference():
    """나란히 붙인 그림의 글자 — 같은 것을 양쪽에 적으면 무엇이 달라졌는지 안 보인다"""
    a = {"profile": None, "config": "/c/x.yaml", "set": ["models.yoloe.max_det=7"]}
    b = {"profile": "/p/B.json", "config": "/c/x.yaml",
         "set": ["models.yoloe.max_det=7", "models.yoloe.conf=0.6"]}
    assert pa.caption("a", a, b) == "A · yaml 기본"
    assert pa.caption("b", b, a) == "B · B.json · models.yoloe.conf=0.6"


# ---------------------------------------------------------------- 프로필
def test_a_wrong_profile_is_refused_with_the_reason(shot, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"models": {"yoloe": {"conf": 0.4}}}), encoding="utf-8")
    with pytest.raises(pa.Bad) as e:
        run_preview(shot, tmp_path, profile=str(bad))
    assert "프로필" in str(e.value)


def test_a_missing_profile_file_is_refused(shot, tmp_path):
    with pytest.raises(pa.Bad):
        run_preview(shot, tmp_path, profile=str(tmp_path / "없는.json"))


def test_the_profile_label_comes_back(shot, tmp_path):
    p = tmp_path / "A.json"
    p.write_text(json.dumps({"label": "검수용", "models": {"yoloe": {"overlay": {"count": True}}}}),
                 encoding="utf-8")
    res = run_preview(shot, tmp_path, frames=1, profile=str(p))
    assert res["sides"]["a"]["label"] == "검수용"


# ---------------------------------------------------------------- 런처 계약 (표준출력)
def test_json_mode_puts_nothing_but_json_on_stdout(monkeypatch, capsys):
    """런처가 표준출력을 통째로 ConvertFrom-Json 한다 — 비전 적재 로그가 섞이면 깨진다"""
    def fake(*_a, **_k):
        print("17:13:47   yoloe prompt-free 4585종")        # vision_infer.log 흉내
        return {"ok": True, "overlay": "/out/x.jpg", "why": "", "summary": "yoloe 검출 1개"}
    monkeypatch.setattr(pa, "preview", fake)
    assert pa.main(["--json", "--image", "x", "--models", "yoloe"]) == 0
    cap = capsys.readouterr()
    assert json.loads(cap.out)["overlay"] == "/out/x.jpg"
    assert "prompt-free" in cap.err                        # 로그는 표준에러로 간다


def test_failure_keeps_the_same_shape_and_returns_one(monkeypatch, capsys):
    def boom(*_a, **_k):
        raise pa.Bad("그림이 없습니다")
    monkeypatch.setattr(pa, "preview", boom)
    assert pa.main(["--json", "--image", "x"]) == 1
    got = json.loads(capsys.readouterr().out)
    assert got["ok"] is False and got["why"] == "그림이 없습니다" and got["overlay"] is None


# ---------------------------------------------------------------- 실제 가중치가 있을 때만
@pytest.mark.skipif(not YOLOE_PT.exists(), reason="models/yoloe-11l-seg-pf.pt 가 없다")
@pytest.mark.skipif(not FRAMES.is_dir(), reason="저장된 장이 없다 (stream_data/pi12)")
def test_real_yoloe_on_a_saved_frame(tmp_path):
    """진짜 가중치로 저장된 장 한 번 — 이 기계에서 실측 1280x720 · 추론 40~62 ms"""
    pytest.importorskip("ultralytics")
    res = pa.preview(str(FRAMES.parent), ["yoloe"], str(tmp_path / "out"), frames=1)
    assert res["ok"] is True, res.get("why")
    one = res["images"][0]["a"]["yoloe"]
    assert one["infer_ms"] > 0 and Path(one["overlay"]).is_file()
    assert one["model_file"] == "yoloe-11l-seg-pf.pt"
    assert isinstance(one["n"], int) and one["n"] == len(one["detections"])
