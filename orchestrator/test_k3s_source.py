import time

from orchestrator import k3s_source as ks

NOW = 1_800_000_000.0


def node(name, ip, ready=True, role="robot", boot="b1"):
    return {"metadata": {"name": name, "labels": {"ds.role": role}},
            "spec": {"taints": [{"key": "ds.role"}]} if role == "robot" else {},
            "status": {"addresses": [{"type": "InternalIP", "address": ip}],
                       "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
                       "nodeInfo": {"architecture": "arm64", "bootID": boot, "kernelVersion": "6.18", "osImage": "Debian"},
                       "capacity": {"cpu": "4", "memory": "8255888Ki", "ephemeral-storage": "122527104Ki"},
                       "allocatable": {"cpu": "4", "memory": "8255888Ki"}}}


def metric(name, age_s, cpu="400000000n", mem="819200Ki"):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(NOW - age_s))
    return {"metadata": {"name": name}, "timestamp": ts, "usage": {"cpu": cpu, "memory": mem}}


def fake(monkeypatch, nodes, metrics):
    def kj(*args, timeout=None):
        if args[:2] == ("get", "nodes"):
            return {"items": nodes}
        return {"items": metrics}
    monkeypatch.setattr(ks, "kubectl_json", kj)
    monkeypatch.setattr(ks, "local_ts_ip", lambda: "100.125.104.23")
    monkeypatch.setattr(ks, "local_node_names", lambda: {"book-test-host"})   # 시험 기기 이름에 기대지 않는다


def test_units():
    assert ks.cpu_m("4") == 4000 and ks.cpu_m("250m") == 250 and ks.cpu_m("400000000n") == 400
    assert round(ks.mem_mb("8255888Ki")) == 8062 and ks.mem_mb("512Mi") == 512


def test_read_skips_local_marks_stale_and_converts(monkeypatch):
    fake(monkeypatch, [node("wsl", "100.125.104.23", role="gpu"), node("pi7", "100.72.109.9"),
                       node("pi1", "100.83.132.16", ready=False)],
         [metric("pi7", 20), metric("pi1", 20)])
    src = ks.K3sNodeSource(probe_net=False, clock=lambda: NOW)
    rows = {r["name"]: r for r in src._read()}
    assert set(rows) == {"pi7", "pi1"}                                  # data_stream 기기는 로컬 기록기 몫
    p7 = rows["pi7"]
    assert p7["cpu_alloc_m"] == 4000 and p7["cpu_now_m"] == 400 and abs(p7["cpu_pct"] - 10.0) < 1e-6
    assert p7["identity"]["boot_id"] == "b1" and p7["baseline"]["cpu_count"] == 4 and not p7["stale"]
    assert rows["pi1"]["stale"]                                         # NotReady


def test_old_or_missing_metrics_are_stale(monkeypatch):
    fake(monkeypatch, [node("pi7", "100.72.109.9"), node("pi6", "100.76.251.100", role="sensor")],
         [metric("pi7", ks.STALE_S + 5)])
    rows = {r["name"]: r for r in ks.K3sNodeSource(probe_net=False, clock=lambda: NOW)._read()}
    assert rows["pi7"]["stale"] and rows["pi6"]["stale"] and rows["pi6"]["cpu_pct"] is None


def test_nodes_attach_service_share_and_net(monkeypatch):
    fake(monkeypatch, [node("pi7", "100.72.109.9")], [metric("pi7", 10)])

    class Mem:
        def node(self, name):
            return {"net": {"bw_mbps": 80.0, "relay": False}}

        def service_share(self, name):
            return {"cpu_pct": 25.0, "mem_mb": 900.0, "active_measured": True, "mode": "active"}
    n = ks.K3sNodeSource(memory=Mem(), probe_net=False, clock=lambda: NOW).nodes()[0]
    assert n["service"] == {"cpu_p95_m": 1000.0, "mem_p95_mb": 900.0, "active_measured": True, "mode": "active"}
    assert n["net"]["bw_mbps"] == 80.0


def test_robot_without_memory_is_unmeasured(monkeypatch):
    fake(monkeypatch, [node("pi7", "100.72.109.9")], [metric("pi7", 10)])
    n = ks.K3sNodeSource(memory=None, probe_net=False, clock=lambda: NOW).nodes()[0]
    assert n["service"]["active_measured"] is False


def test_check_reports_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(ks, "kubectl_json", boom)
    monkeypatch.setattr(ks, "local_ts_ip", lambda: None)
    ok, why, caps = ks.K3sNodeSource(probe_net=False).check()
    assert not ok and "connection refused" in why and caps == {}


# ---------------------------------------------------------------- 클러스터 객체 → caps (M11 R2)
def fake_kubectl(monkeypatch, missing=(), calls=None):
    """가짜 kubectl — get -f orchestrator/k3s/ 는 missing 이 있으면 NotFound 로 실패"""
    def kj(*args, timeout=None):
        if calls is not None:
            calls.append(args)
        if args[:2] == ("get", "-f"):
            assert args[2] == str(ks.K3S_DIR)
            if missing:
                raise RuntimeError("\n".join('Error from server (NotFound): priorityclasses.scheduling.k8s.io "%s" '
                                             "not found" % n for n in missing))
            return {"kind": "PriorityClass"}
        return {"items": []}
    monkeypatch.setattr(ks, "kubectl_json", kj)
    monkeypatch.setattr(ks, "local_ts_ip", lambda: "100.125.104.23")


def test_check_reports_missing_priorityclass_as_caps(monkeypatch):
    fake_kubectl(monkeypatch, missing=["ds-vision-low"])
    ok, why, caps = ks.K3sNodeSource(probe_net=False, clock=lambda: NOW).check()
    assert ok and why == ""                                             # 노드는 읽힌다 — k3s 모드 그대로
    assert caps["cpu_tasks"] is False and "PriorityClass ds-vision-low 없음" in caps["why"]
    assert "orchestrator/k3s/" in caps["why"]


def test_check_caps_ok_and_cached(monkeypatch):
    calls = []
    fake_kubectl(monkeypatch, calls=calls)
    t = {"now": NOW}
    src = ks.K3sNodeSource(probe_net=False, clock=lambda: t["now"])
    assert src.check() == (True, "", {"cpu_tasks": True})
    src.check()
    assert sum(1 for a in calls if a[:2] == ("get", "-f")) == 1          # poll 마다 묻지 않는다
    t["now"] += ks.CAPS_TTL_S + 1
    src.check()
    assert sum(1 for a in calls if a[:2] == ("get", "-f")) == 2


def test_k3s_folder_holds_priorityclass():
    import yaml
    docs = [d for p in sorted(ks.K3S_DIR.glob("*.yaml")) for d in yaml.safe_load_all(p.read_text(encoding="utf-8")) if d]
    assert ("PriorityClass", "ds-vision-low") in [(d["kind"], d["metadata"]["name"]) for d in docs]


def test_scripts_apply_and_check_the_folder():
    from pathlib import Path
    cl = Path(ks.__file__).resolve().parent / "cluster"
    assert 'apply -f "$HERE/../k3s/"' in (cl / "install_server.sh").read_text(encoding="utf-8")
    assert 'get -f "$HERE/../k3s/"' in (cl / "smoke_test.sh").read_text(encoding="utf-8")


# ---------------------------------------------------------------- 대역폭 결과 상태 (M11 R1)
def test_measure_net_status(monkeypatch):
    monkeypatch.setattr(ks, "tailscale_ping", lambda ip: (6.0, False))
    monkeypatch.setattr(ks, "local_ts_ip", lambda: None)
    r = ks.K3sNodeSource(probe_net=False).measure_net("pi7", "100.72.109.9")
    assert r["net_status"] == "unsupported" and r["bw_mbps"] is None and "테일넷" in r["error"]
    monkeypatch.setattr(ks, "local_ts_ip", lambda: "100.125.104.23")
    monkeypatch.setattr(ks, "bandwidth_probe_ex", lambda node, ip: (None, "측정 파드 거절: no PriorityClass"))
    r = ks.K3sNodeSource(probe_net=False).measure_net("pi7", "100.72.109.9")
    assert r["net_status"] == "failed" and "PriorityClass" in r["error"] and r["latency_ms"] == 6.0
    monkeypatch.setattr(ks, "bandwidth_probe_ex", lambda node, ip: (82.0, None))
    r = ks.K3sNodeSource(probe_net=False).measure_net("pi7", "100.72.109.9")
    assert r["net_status"] == "ok" and r["bw_mbps"] == 82.0


def test_want_net_thread_failure_is_reported_as_failed(monkeypatch):
    monkeypatch.setattr(ks, "local_ts_ip", lambda: "100.125.104.23")
    src = ks.K3sNodeSource(probe_net=True)

    def boom(node, ip):
        raise OSError("tailscale 없음")
    src.measure_net = boom
    src.want_net("pi7", "100.72.109.9")
    for _ in range(100):
        if src.net_pending:
            break
        time.sleep(0.01)
    r = src.net_updates()["pi7"]
    assert r["net_status"] == "failed" and "tailscale" in r["error"] and r["measured_at"]


# ---------------------------------------------------------------- 로컬 노드 가려내기 (M11 R5)
def test_local_by_name_when_ip_differs(monkeypatch):
    # 검토 기기: InternalIP 172.28.x (테일넷 아님) · k3s 노드 이름 == probe.node_name()
    fake(monkeypatch, [node("desktop-0ib285f", "172.28.32.223", role="gpu"), node("pi7", "100.72.109.9")], [])
    monkeypatch.setattr(ks, "local_ts_ip", lambda: "100.83.37.9")
    src = ks.K3sNodeSource(probe_net=False, clock=lambda: NOW, local_names={"desktop-0ib285f"})
    assert [r["name"] for r in src._read()] == ["pi7"]


def test_local_by_ip_when_name_differs(monkeypatch):
    # 작업 기기: InternalIP == 테일넷 주소 · 이름은 book-4vg1edpvi5-wsl (probe 는 book-4vg1edpvi5)
    fake(monkeypatch, [node("book-4vg1edpvi5-wsl", "100.125.104.23", role="gpu"), node("pi7", "100.72.109.9")], [])
    src = ks.K3sNodeSource(probe_net=False, clock=lambda: NOW, local_names={"book-4vg1edpvi5"})
    assert [r["name"] for r in src._read()] == ["pi7"]


def test_local_by_explicit_env(monkeypatch):
    # IP 도 이름도 안 맞는 설치 — DS_K3S_LOCAL_NODE 로 직접 지정
    real = ks.local_node_names
    fake(monkeypatch, [node("gpu-box", "10.0.0.5", role="gpu"), node("pi7", "100.72.109.9")], [])
    monkeypatch.setattr(ks, "local_node_names", real)                  # 이 시험은 진짜 이름 모으기를 쓴다
    monkeypatch.setattr(ks.socket, "gethostname", lambda: "Some-Host")
    monkeypatch.delenv("DS_NODE_NAME", raising=False)
    monkeypatch.delenv("DS_K3S_LOCAL_NODE", raising=False)
    assert [r["name"] for r in ks.K3sNodeSource(probe_net=False, clock=lambda: NOW)._read()] == ["gpu-box", "pi7"]
    monkeypatch.setenv("DS_K3S_LOCAL_NODE", "gpu-box")
    assert [r["name"] for r in ks.K3sNodeSource(probe_net=False, clock=lambda: NOW)._read()] == ["pi7"]
    monkeypatch.delenv("DS_K3S_LOCAL_NODE")
    monkeypatch.setenv("DS_NODE_NAME", "gpu-box")
    assert [r["name"] for r in ks.K3sNodeSource(probe_net=False, clock=lambda: NOW)._read()] == ["pi7"]


def test_local_names_include_hostname_and_probe(monkeypatch):
    monkeypatch.setattr(ks.socket, "gethostname", lambda: "Book-4VG1EDPVI5")
    monkeypatch.delenv("DS_NODE_NAME", raising=False)
    monkeypatch.setenv("DS_K3S_LOCAL_NODE", "a, b")
    names = ks.local_node_names()
    assert {"book-4vg1edpvi5", "Book-4VG1EDPVI5", "a", "b"} <= names
