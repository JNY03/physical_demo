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
    ok, why = ks.K3sNodeSource(probe_net=False).check()
    assert not ok and "connection refused" in why
