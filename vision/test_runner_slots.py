"""비전이 역할 슬롯 · 새 실행기를 따르는 자리 (M2 · M4, 2026-11-27)"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vision_infer as vi  # noqa: E402


def test_old_model_lists_unchanged_without_runner_models():
    """레지스트리에 runner 모델이 없으면 옛 목록 그대로 (다른 모델이 등록되면 뒤에 붙는다)"""
    assert vi.MODELS[:5] == ("yoloe", "river", "targeted", "unidepth", "moge2_aerial")
    assert vi.DETECT_MODELS[:3] == ("yoloe", "river", "targeted")
    assert set(vi.RUNNER_SPECS).isdisjoint(set(vi.RUNNERS))


def test_slot_models_picks_server_model_slots():
    slots = {"object_detect": {"server": {"model": "yoloe"}, "device": {"detector": "pi_x"}},
             "depth_estimate": {"server": {"model": "unidepth"}},
             "targeted_detect": {"server": {"ovd": ["yolo_world"]}},
             "x": {"server": {"model": "not_a_model"}}}
    assert vi.slot_models(slots) == ["unidepth", "yoloe"]          # 기능 이름순 · 비전이 모르는 것은 뺀다
    assert vi.slot_models({}) == []


def test_apply_profile_keeps_slots():
    cfg = {"models": {}}
    vi.apply_profile(cfg, {"slots": {"object_detect": {"server": {"model": "yoloe"}}}})
    assert cfg["_slots"] == {"object_detect": {"server": {"model": "yoloe"}}}


def test_make_runner_uses_loader_for_registry_runners(monkeypatch):
    made = {}

    def fake_load(key, spec, device, cfg):
        made.update(key=key, spec=spec, device=device, cfg=cfg)
        return "runner"
    monkeypatch.setattr(vi, "load_runner", fake_load)
    monkeypatch.setitem(vi.RUNNER_SPECS, "dummy_det", {"runner": "dummy:D", "config": {"a": 1}})
    assert vi.make_runner("dummy_det", {"models": {}}, "cpu") == "runner"
    assert made["cfg"] == {"a": 1} and made["device"] == "cpu"
    vi.make_runner("dummy_det", {"models": {"dummy_det": {"b": 2}}}, "cpu")
    assert made["cfg"] == {"b": 2}                                  # vision_config models.<키> 가 이긴다
