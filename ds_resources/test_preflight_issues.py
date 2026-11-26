"""시작 점검 (preflight issues · cli fix) — 런처 배너가 읽는 것. subprocess 는 가짜로 (클러스터 · docker 를 건드리지 않는다)"""
import json
import subprocess

import pytest

from ds_resources import cli
from ds_resources import preflight as pf

MACH_OK = {"host": "h", "cpu": 8, "mem_gb": 16, "disk_gb": 500, "gpu": "RTX", "gpu_mem_gb": 6, "docker": "29",
           "nvidia_toolkit": True, "kubectl": True, "redis": True, "container": False}


def _fake_run(answers):
    """명령 앞부분 → 표준출력(None 이면 실패). 부른 명령과 env 를 적어 둔다"""
    calls = []

    def run(cmd, timeout=4, env=None):
        calls.append((list(cmd), env))
        key = " ".join(cmd[1:4]) if "kubectl" in cmd[0] else " ".join(cmd[:2])
        return answers.get(key)
    return run, calls


def test_priorityclass_names_read_from_manifests_without_yaml(tmp_path):
    """orchestrator/k3s/*.yaml 의 PriorityClass 이름 — PyYAML 없이, 다른 kind 는 빼고"""
    (tmp_path / "a.yaml").write_text("apiVersion: v1\nkind: PriorityClass\nmetadata:\n  name: ds-mission  # 임무\n"
                                     "value: 1\n---\nkind: Namespace\nmetadata:\n  name: not-me\n", encoding="utf-8")
    (tmp_path / "b.yml").write_text("---\nkind: PriorityClass\nmetadata:\n  labels:\n    name: x\n  name: 'ds-b'\n",
                                    encoding="utf-8")
    assert pf.k3s_priorityclasses(tmp_path) == {"ds-mission", "ds-b"}
    assert {"ds-mission", "ds-safety-high"} <= pf.k3s_priorityclasses()     # 저장소의 실제 매니페스트


def test_missing_priorityclass_is_an_error_with_apply_fix(monkeypatch):
    """k3s 에 닿는데 등급이 빠졌으면 error + kubectl apply -f orchestrator/k3s/ — KUBECONFIG 는 ~/.kube/config 로 채운다"""
    monkeypatch.delenv("KUBECONFIG", raising=False)
    run, calls = _fake_run({"get priorityclass -o": "priorityclass.scheduling.k8s.io/ds-vision-low\n"
                                                     "priorityclass.scheduling.k8s.io/system-node-critical"})
    monkeypatch.setattr(pf, "_run", run)
    iss = pf._check_priorityclass(dict(MACH_OK))
    assert len(iss) == 1 and iss[0]["level"] == "error" and "ds-mission" in iss[0]["title"]
    assert iss[0]["fix"]["argv"] == ["kubectl", "apply", "-f", "orchestrator/k3s/"]
    assert calls[0][1]["KUBECONFIG"].endswith(".kube/config")


def test_priorityclass_all_present_or_unknown_is_quiet(monkeypatch):
    names = "\n".join("priorityclass.scheduling.k8s.io/" + n for n in pf.k3s_priorityclasses())
    monkeypatch.setattr(pf, "_run", _fake_run({"get priorityclass -o": names})[0])
    assert pf._check_priorityclass(dict(MACH_OK)) == []
    monkeypatch.setattr(pf, "_run", _fake_run({})[0])                       # 못 읽음 = 모름 → 말하지 않는다
    assert pf._check_priorityclass(dict(MACH_OK)) == []
    assert pf._check_priorityclass(dict(MACH_OK, kubectl=False)) == []     # k3s 없음은 따로(info) 알린다


def test_kubectl_probe_uses_kubeconfig_fallback(monkeypatch):
    """machine() 의 k3s 확인도 ~/.kube/config 로 묻는다 — 빈 KUBECONFIG 로 물어 「없음」이 나오던 것"""
    monkeypatch.delenv("KUBECONFIG", raising=False)
    run, calls = _fake_run({"get nodes -o": "node/a"})
    monkeypatch.setattr(pf, "_run", run)
    monkeypatch.setattr(pf.shutil, "which", lambda n: "/usr/bin/" + n)
    m = pf.machine(probe_gpus=lambda: [])
    assert m["kubectl"] is True
    kc = [env for cmd, env in calls if "nodes" in cmd]
    assert kc and kc[0]["KUBECONFIG"].endswith(".kube/config")


def test_weights_fix_fetches_first_missing_option_per_blocked_function():
    rows = [{"key": "object_detect", "options": ["yoloe", "river"], "needs": ["weights"]},
            {"key": "depth_estimate", "options": ["unidepth", "moge2_aerial"], "needs": ["weights"]},
            {"key": "segmentation", "options": ["yoloe"], "needs": ["weights"]},
            {"key": "motor_control", "options": [], "needs": ["kubectl"]}]
    ws = {"yoloe": (False, "x"), "river": (False, "y"), "unidepth": (None, "모름"), "moge2_aerial": (False, "z")}
    iss = pf._check_weights(rows, ws)
    assert len(iss) == 1 and "3개" in iss[0]["title"]
    fx = iss[0]["fix"]
    assert fx["argv"][1:] == ["tools/fetch_models.py", "--models", "yoloe", "moge2_aerial"] and fx["long"]
    assert pf._check_weights([rows[3]], ws) == []


def test_registry_stopped_gives_docker_start(monkeypatch):
    monkeypatch.setattr(pf, "_run", _fake_run({"docker ps": "exited"})[0])
    iss = pf._check_registry(dict(MACH_OK))
    assert iss[0]["id"] == "docker_registry" and iss[0]["fix"]["argv"] == ["docker", "start", "ds-registry"]
    monkeypatch.setattr(pf, "_run", _fake_run({"docker ps": "running"})[0])
    assert pf._check_registry(dict(MACH_OK)) == []
    monkeypatch.setattr(pf, "_run", _fake_run({})[0])                       # 레지스트리를 안 둔 기기
    assert pf._check_registry(dict(MACH_OK)) == []
    assert pf._check_registry(dict(MACH_OK, docker=False)) == []           # docker 없으면 묻지도 않는다


def test_simple_checks_redis_gpu_disk_have_no_auto_fix():
    iss = pf._check_simple(dict(MACH_OK, redis=False, gpu=False, disk_gb=2.0))
    by = {i["id"]: i for i in iss}
    assert by["redis"]["level"] == "warn" and by["redis"]["fix"] is None and "6380" in by["redis"]["title"]
    assert by["gpu"]["level"] == "warn" and by["disk"]["level"] == "error"
    assert not {"redis", "gpu", "disk"} & {i["id"] for i in pf._check_simple(dict(MACH_OK))}


def test_issues_sorted_by_level_and_one_broken_check_does_not_hide_others(monkeypatch):
    def boom(m):
        raise RuntimeError("깨짐")
    monkeypatch.setattr(pf, "_check_priorityclass", boom)
    monkeypatch.setattr(pf, "_check_registry", lambda m: [pf._issue("r", "warn", "t", "d")])
    monkeypatch.setattr(pf, "_check_simple", lambda m: [pf._issue("x", "info", "t", "d"),
                                                        pf._issue("y", "error", "t", "d")])
    out = pf.issues(dict(MACH_OK), [], {})
    assert [i["level"] for i in out] == ["error", "warn", "info", "info"]
    assert "preflight_error" in {i["id"] for i in out}


REP = {"issues": [pf._issue("docker_registry", "warn", "t", "d",
                            pf._fix("docker_registry", "켜기", ["docker", "start", "ds-registry"])),
                  pf._issue("redis", "warn", "t", "Redis 켜기", None),
                  pf._issue("s", "warn", "t", "d", pf._fix("s", "s", ["apt", "install", "x"], needs_sudo=True))]}


def test_run_fix_runs_argv_from_recheck_in_repo_root(monkeypatch):
    monkeypatch.delenv("KUBECONFIG", raising=False)
    seen = {}

    def runner(argv, **kw):
        seen.update(argv=argv, **kw)
        return subprocess.CompletedProcess(argv, 0, "ds-registry\n", "")
    res = pf.run_fix("docker_registry", rep=REP, runner=runner)
    assert res["ok"] and res["ran"] and res["returncode"] == 0
    assert seen["argv"] == ["docker", "start", "ds-registry"] and seen["cwd"] == str(pf.REPO)
    assert seen["env"]["KUBECONFIG"].endswith(".kube/config")


def test_run_fix_dry_run_unknown_nofix_sudo_and_failure():
    def never(*a, **k):
        pytest.fail("돌리면 안 된다")
    r = pf.run_fix("docker_registry", dry_run=True, rep=REP, runner=never)
    assert r["ok"] and not r["ran"] and r["argv"] == ["docker", "start", "ds-registry"]
    r = pf.run_fix("gone", rep=REP, runner=never)                          # 이미 고쳐짐 — 점검에 없다
    assert r["ok"] and not r["ran"]
    r = pf.run_fix("redis", rep=REP, runner=never)
    assert not r["ok"] and "자동으로" in r["why"]
    r = pf.run_fix("s", rep=REP, runner=never)
    assert not r["ok"] and "sudo apt install x" in r["why"]

    def bad(argv, **k):
        return subprocess.CompletedProcess(argv, 1, "", "Error: no such container\n")
    r = pf.run_fix("docker_registry", rep=REP, runner=bad)
    assert not r["ok"] and r["returncode"] == 1 and "no such container" in r["why"]


def test_cli_fix_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(pf, "report", lambda: REP)
    assert cli.main(["fix", "docker_registry", "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["argv"] == ["docker", "start", "ds-registry"] and out["dry_run"]
    with pytest.raises(SystemExit):
        cli.main(["fix"])


def test_report_carries_issues_and_text_lists_them():
    rep = {"machine": dict(MACH_OK), "features": [], "install": [],
           "issues": [pf._issue("weights", "warn", "모델 가중치 없음", "d", pf._fix("weights", "받기", ["x"]))],
           "summary": {"usable": 0, "blocked": 0, "total": 0}}
    assert "[warn] 모델 가중치 없음 → 고치기: cli fix weights" in pf.text(rep)
