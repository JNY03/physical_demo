"""settings — DS_HOME · config/ · DS_* 우선순위와 경로 풀이 (Docker 계획 D1)."""
import json
import sys

from ds_resources import memory as mm
from ds_resources import settings


def test_ds_home_defaults_to_repo_and_env_wins(monkeypatch, tmp_path):
    monkeypatch.delenv("DS_HOME", raising=False)
    assert settings.ds_home() == settings.REPO
    monkeypatch.setenv("DS_HOME", str(tmp_path))
    assert settings.ds_home() == tmp_path.resolve()
    assert settings.resolve("models/a.pt") == tmp_path.resolve() / "models/a.pt"
    assert str(settings.resolve("/abs/x")) == "/abs/x"


def test_hosts_prefers_config_then_legacy(monkeypatch, tmp_path):
    monkeypatch.setenv("DS_HOME", str(tmp_path))
    monkeypatch.delenv("DS_CONFIG_DIR", raising=False)
    assert settings.load_hosts() == {}
    (tmp_path / "hosts.local.json").write_text(json.dumps({"robot1": "1.1.1.1"}))
    assert settings.load_hosts() == {"robot1": "1.1.1.1"}           # 옛 이름도 읽는다
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "hosts.json").write_text(json.dumps({"robot1": "2.2.2.2"}))
    assert settings.load_hosts() == {"robot1": "2.2.2.2"}           # config/ 가 이긴다
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "hosts.json").write_text(json.dumps({"a2": "3.3.3.3"}))
    monkeypatch.setenv("DS_CONFIG_DIR", str(other))
    assert settings.load_hosts() == {"a2": "3.3.3.3"}


def test_python_and_node_name(monkeypatch):
    monkeypatch.delenv("DS_PYTHON", raising=False)
    assert settings.python() == sys.executable
    monkeypatch.setenv("DS_PYTHON", "/opt/py")
    assert settings.python() == "/opt/py"
    monkeypatch.setenv("DS_NODE_NAME", "ds-node")
    assert settings.node_name() == settings.host_label() == "ds-node"
    from ds_resources import probe
    assert probe.node_name() == "ds-node"


def test_gpu_executor_python_default(monkeypatch):
    from ds_resources import gpu_executor as ge
    monkeypatch.delenv("DS_PYTHON", raising=False)
    assert ge.GpuExecutor(state=object()).python == sys.executable
    monkeypatch.setenv("DS_PYTHON", "/opt/py")
    assert ge.GpuExecutor(state=object()).python == "/opt/py"


def test_resolve_tree_only_touches_model_roots(monkeypatch, tmp_path):
    monkeypatch.setenv("DS_HOME", str(tmp_path))
    cfg = {"weight": "models/y.pt", "enc": "models", "src": "external/UniDepth", "source": "box_height",
           "home": "~/.cache/x", "abs": "/abs/models/y.pt", "url": "http://models/x",
           "extra": ["external/a", "vision/assets/b"], "n": 3}
    out = settings.resolve_tree(cfg)
    root = str(tmp_path.resolve())
    assert out["weight"] == root + "/models/y.pt" and out["enc"] == root + "/models"
    assert out["src"] == root + "/external/UniDepth" and out["extra"] == [root + "/external/a", "vision/assets/b"]
    for k in ("source", "home", "abs", "url", "n"):
        assert out[k] == cfg[k]
    assert cfg["weight"] == "models/y.pt"                       # 원본은 그대로
    assert settings.relative_tree(out) == cfg                    # 되돌리면 같다


def test_cfg_hash_does_not_depend_on_ds_home(monkeypatch, tmp_path):
    raw = {"models": {"yoloe": {"weight": "models/y.pt", "conf": 0.3}}}
    monkeypatch.setenv("DS_HOME", str(tmp_path / "a"))
    ha = mm.cfg_hash("/nonexistent", settings.resolve_tree(raw), ["yoloe"])
    monkeypatch.setenv("DS_HOME", str(tmp_path / "b"))
    hb = mm.cfg_hash("/nonexistent", settings.resolve_tree(raw), ["yoloe"])
    assert ha == hb == mm.cfg_hash("/nonexistent", raw, ["yoloe"])


def test_vision_config_overlay(monkeypatch, tmp_path):
    base = tmp_path / "vision_config.yaml"
    base.write_text("models:\n  yoloe:\n    weight: models/a.pt\n    conf: 0.3\nlive: {fps: 5}\n")
    monkeypatch.setenv("DS_CONFIG_DIR", str(tmp_path / "cfg"))
    assert settings.load_vision_config(base)["models"]["yoloe"]["weight"] == "models/a.pt"
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / settings.VISION_OVERLAY).write_text("models:\n  yoloe:\n    weight: /w/b.pt\n")
    cfg = settings.load_vision_config(base)
    assert cfg["models"]["yoloe"] == {"weight": "/w/b.pt", "conf": 0.3} and cfg["live"] == {"fps": 5}
