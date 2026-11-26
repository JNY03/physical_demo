"""이 기기에서 **무엇이 되고 무엇이 왜 안 되는지** 한 번에 본다 (core, 2026-11-26).

처음 보는 사람이 제일 먼저 묻는 것은 「내 컴퓨터로 이게 돌아갑니까」다. 그 답을 사람이 찾아다니게 두지 않는다:
자원(CPU · 메모리 · GPU · 디스크)과 둘레의 것들(docker · NVIDIA 컨테이너 툴킷 · kubectl · Redis · 가중치)을
읽어서, 기능마다 **된다 · 안 된다 · 왜 · 어떻게 하면 되는지**를 한 줄씩 돌려준다.

    report(...) → {"machine": {...}, "features": [{key, shown, usable, why, how, needs}], "install": [...]}

원칙
  * 읽기만 한다. 무엇도 깔지 않고 띄우지 않는다 — 깔지 말지는 사람이 고른다(그래서 how 를 적어 준다).
  * **모름과 없음을 가른다.** 못 읽은 것은 unknown 이고, 그것 때문에 기능을 막지 않는다(placement 와 같은 규칙).
  * 기능 목록은 레지스트리 하나에서 온다(vision/model_registry.yaml 의 functions · options) — 기능이 늘면
    이 파일을 고치지 않아도 같이 나온다.
  * 가중치가 없어 안 되는 것은 **받는 길**(tools/fetch_models.py)과 **넣는 길**(런처 「가중치」)을 함께 적는다.
"""
import os
import shutil
import socket
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
#: 기능이 돌려면 있어야 하는 것 → 사람이 읽는 이름과 고치는 길
NEEDS = {
    "gpu": {"shown": "NVIDIA GPU", "how": "GPU 가 없으면 CPU 로 되는 기능만 씁니다 (서버 쪽 무거운 모델은 빠집니다)"},
    "weights": {"shown": "모델 가중치", "how": "venv/bin/python tools/fetch_models.py --models <선택지> "
                                             "또는 런처 2단계 「가중치」 단추로 파일을 넣으세요"},
    "docker": {"shown": "Docker", "how": "bash tools/docker/install_docker_wsl.sh (WSL 안 docker-ce)"},
    "nvidia_toolkit": {"shown": "NVIDIA 컨테이너 툴킷", "how": "컨테이너에서 GPU 를 쓰려면 필요합니다 — "
                                                              "nvidia-container-toolkit 설치 후 docker 재시작"},
    "kubectl": {"shown": "kubectl", "how": "파이에 작업을 올리려면 k3s 와 kubeconfig 가 있어야 합니다 "
                                           "(없으면 이 기기 안에서만 돕니다)"},
    "redis": {"shown": "Redis", "how": "docker start mission-runtime-data-stream-redis 또는 "
                                       "redis-server --port 6380"},
    "camera": {"shown": "영상 소스", "how": "런처 소스 탭에서 소스를 켜고 그 기기 전원과 네트워크를 확인하세요"},
}


def _run(cmd, timeout=4):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _port_open(port, host="127.0.0.1", timeout=0.4):
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def _in_container():
    """이 프로세스가 컨테이너 안인가 — 그러면 「없다」고 단정하지 않는다(그 컨테이너에 안 넣었을 뿐일 수 있다)"""
    if Path("/.dockerenv").exists():
        return True
    try:
        txt = Path("/proc/1/cgroup").read_text(encoding="utf-8")
        return any(x in txt for x in ("docker", "kubepods", "containerd", "podman"))
    except OSError:
        return False


def machine(probe_gpus=None):
    """이 기기의 자원과 둘레 — 못 읽은 칸은 None(모름)이다."""
    out = {"host": socket.gethostname(), "wsl": None, "cpu": os.cpu_count(), "mem_gb": None, "disk_gb": None,
           "gpu": None, "gpu_mem_gb": None, "docker": None, "nvidia_toolkit": None, "kubectl": None,
           "redis": _port_open(6380), "python": None}
    try:
        out["wsl"] = "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        pass
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal"):
                out["mem_gb"] = round(int(line.split()[1]) / 1048576.0, 1)
                break
    except (OSError, ValueError, IndexError):
        pass
    try:
        st = os.statvfs(str(REPO))
        out["disk_gb"] = round(st.f_bavail * st.f_frsize / 1e9, 1)
    except OSError:
        pass
    out["container"] = _in_container()
    gs = None
    if probe_gpus is not None:
        try:
            gs = probe_gpus()
        except Exception:  # noqa: BLE001 — nvidia-smi 가 없거나 막혀도 나머지는 낸다
            gs = None
    if gs:
        out["gpu"] = gs[0].get("name")
        out["gpu_mem_gb"] = round(float(gs[0].get("mem_total_mb") or 0) / 1024.0, 1)
    elif shutil.which("nvidia-smi") is None:
        # 컨테이너 안에서는 nvidia-smi 가 없다고 GPU 가 없는 것이 아니다 — 그 컨테이너에 안 넣었을 뿐이다.
        # 모름과 없음을 가른다(ds-core 에는 일부러 넣지 않는다 — Docker 계획 §5)
        out["gpu"] = None if out["container"] else False
    ver = _run(["docker", "version", "--format", "{{.Server.Version}}"])
    out["docker"] = ver or (False if shutil.which("docker") is None else None)
    info = _run(["docker", "info", "--format", "{{json .Runtimes}}"], timeout=6)
    if info is not None:
        out["nvidia_toolkit"] = "nvidia" in info
    elif shutil.which("nvidia-container-runtime") is not None:
        out["nvidia_toolkit"] = True
    kube = shutil.which("kubectl")
    if kube is None:
        out["kubectl"] = False
    else:
        out["kubectl"] = _run([kube, "get", "nodes", "-o", "name"], timeout=5) is not None
    py = REPO / "venv" / "bin" / "python"
    out["python"] = str(py) if py.exists() else None
    return out


def _weight_state(models, dest=None):
    """{선택지: (됨?, 못 되는 까닭)} — tools/fetch_models.py 의 검증을 그대로 쓴다(한 곳에서만 센다)."""
    import sys
    sys.path.insert(0, str(REPO))
    try:
        from tools import fetch_models as fm
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for m in models:
        try:
            keys, _trained = fm.needed([m])              # (모델 키 집합, 대상 학습 가중치)
            items = fm.items_for(keys, dest=dest)
        except SystemExit as e:                          # 레지스트리에 없는 선택지
            out[m] = (None, str(e))
            continue
        except Exception as e:  # noqa: BLE001
            out[m] = (None, "확인 못 함: %s" % e)
            continue
        bad = []
        for it in items:
            st, why = fm.check_item(it)
            if st in ("missing", "mismatch", "manual", "nosource"):
                bad.append("%s(%s)" % (Path(it["where"]).name if it["where"] != "-" else it["model"], st))
        out[m] = (not bad, ", ".join(bad[:4]))
    return out


def features(mach, funcs, catalog=None, dest=None):
    """기능마다 된다 · 안 된다 · 왜 · 어떻게.

    funcs    functions.load() 결과 (레지스트리 하나에서 온다 — 기능이 늘면 여기도 같이 는다)
    catalog  cue_catalog 의 선택지 모음 (모델 이름 · 가중치 유무). 없으면 가중치는 fetch_models 로만 본다"""
    want = sorted({o for g in funcs.values() for o in (g.get("options") or [])})
    wstate = _weight_state(want, dest)
    rows = []
    for name, g in sorted(funcs.items()):
        opts = list(g.get("options") or [])
        needs, why = [], []
        usable = True
        # 1) 가중치 — 그 기능이 쓸 수 있는 선택지 중 하나라도 되면 된다
        if opts:
            ok_opts = [o for o in opts if wstate.get(o, (None, ""))[0]]
            if not ok_opts:
                usable = False
                needs.append("weights")
                miss = "; ".join("%s: %s" % (o, wstate.get(o, (None, "모름"))[1] or "없음") for o in opts)
                why.append("쓸 수 있는 모델이 없음 — %s" % miss)
        # 2) 서버 단계에 GPU 가 필요한가
        gpu_stage = any((s.get("needs") or {}).get("accel") == "gpu" for s in g["stages"].values())
        if gpu_stage and mach.get("gpu") is False:
            usable = False
            needs.append("gpu")
            why.append("서버 단계가 GPU 를 요구하는데 이 기기에 GPU 가 없음")
        # 3) 기기(파이)에서 도는 단계가 있는데 k3s 가 없으면 그 쪽만 못 올린다
        dev = g["stages"].get("device")
        if dev is not None and dev.get("pin") == "device" and mach.get("kubectl") is not True:
            if g["pattern"] in ("device_only", "resident") or not g["stages"].get("server"):
                usable = False
                why.append("기기 안에서만 도는 기능인데 k3s(파이)에 닿지 않음")
            else:
                why.append("기기 단계는 못 올림 — 서버 쪽만 돈다 (k3s 없음)")
            needs.append("kubectl")
        # 4) 실행체가 아직 없는 계약
        runnable = any(s.get("image") or s.get("command") or s.get("local_command") for s in g["stages"].values())
        rows.append({"key": name, "shown": name, "pattern": g["pattern"], "criticality": g["criticality"],
                     "options": opts, "usable": bool(usable), "runnable": bool(runnable),
                     "why": " / ".join(why), "needs": sorted(set(needs)),
                     "how": " / ".join(NEEDS[n]["how"] for n in sorted(set(needs)) if n in NEEDS)})
    return rows, wstate


def install_steps(mach, rows):
    """사람이 고를 수 있는 「더 깔기」 목록 — 무엇을, 왜, 어떻게. 깔지는 않는다."""
    out = []
    need_counts = {}
    for r in rows:
        for n in r["needs"]:
            need_counts.setdefault(n, []).append(r["key"])
    for n, who in sorted(need_counts.items()):
        # 예전에는 venv 가 없으면(= 컨테이너) 가중치 안내를 숨겼다 — 컨테이너에서도 **호스트에서 받으라고**
        # 알려 줘야 한다. 받는 도구(tools/fetch_models.py)가 있으면 안내한다 (2026-11-26)
        if n == "weights" and not (REPO / "tools" / "fetch_models.py").exists():
            continue
        out.append({"what": NEEDS.get(n, {}).get("shown", n), "key": n, "how": NEEDS.get(n, {}).get("how", ""),
                    "unlocks": sorted(who)[:12], "count": len(who)})
    if mach.get("redis") is False:
        out.append({"what": NEEDS["redis"]["shown"], "key": "redis", "how": NEEDS["redis"]["how"],
                    "unlocks": ["모든 기능 (스트림이 Redis 로 간다)"], "count": 0})
    if mach.get("docker") in (False, None):
        out.append({"what": NEEDS["docker"]["shown"], "key": "docker", "how": NEEDS["docker"]["how"],
                    "unlocks": ["컨테이너로 돌리기 (선택)"], "count": 0})
    elif mach.get("nvidia_toolkit") is False and mach.get("gpu") not in (False, None):
        out.append({"what": NEEDS["nvidia_toolkit"]["shown"], "key": "nvidia_toolkit",
                    "how": NEEDS["nvidia_toolkit"]["how"], "unlocks": ["컨테이너 안에서 GPU 쓰기"], "count": 0})
    return out


def report(funcs=None, probe_gpus=None, dest=None):
    """{machine, features, install, summary} — 읽기만 한다."""
    from ds_resources import functions as fnc
    if funcs is None:
        funcs = fnc.load()
    if probe_gpus is None:
        from ds_resources import probe
        probe_gpus = probe.gpus
    mach = machine(probe_gpus)
    rows, wstate = features(mach, funcs, dest=dest)
    ok = [r["key"] for r in rows if r["usable"]]
    no = [r["key"] for r in rows if not r["usable"]]
    return {"machine": mach, "features": rows, "install": install_steps(mach, rows),
            "weights": {k: {"ok": v[0], "missing": v[1]} for k, v in sorted(wstate.items())},
            "summary": {"usable": len(ok), "blocked": len(no), "total": len(rows)}}


def text(rep):
    """사람이 읽는 한 장 — 런처 · 컨테이너 첫 실행이 그대로 찍는다"""
    m = rep["machine"]
    lines = ["이 기기: %s, CPU %s코어, 메모리 %s GB, 디스크 여유 %s GB" % (
        m.get("host"), m.get("cpu"), m.get("mem_gb"), m.get("disk_gb"))]
    gpu = m.get("gpu")
    lines.append("GPU: %s" % ("없음" if gpu is False else (("모름 (컨테이너 안에서는 안 보일 수 있습니다)"
                                                            if m.get("container") else "모름") if gpu is None else
                                                          "%s (%s GB)" % (gpu, m.get("gpu_mem_gb")))))
    lines.append("둘레: docker %s, NVIDIA 툴킷 %s, k3s %s, Redis %s" % (
        m.get("docker") or "없음", _yn(m.get("nvidia_toolkit")), _yn(m.get("kubectl")), _yn(m.get("redis"))))
    s = rep["summary"]
    lines.append("")
    lines.append("기능 %d개 중 **%d개 지금 됨**, %d개는 조건이 모자람" % (s["total"], s["usable"], s["blocked"]))
    for r in rep["features"]:
        if r["usable"]:
            continue
        lines.append("  - %s: %s" % (r["shown"], r["why"] or "조건 모자람"))
    if rep["install"]:
        lines.append("")
        lines.append("더 깔면 되는 것:")
        for i in rep["install"]:
            lines.append("  - %s: %s" % (i["what"], i["how"]))
            if i["count"]:
                lines.append("      기능 %d개가 풀립니다: %s" % (i["count"], ", ".join(i["unlocks"][:6])))
    return "\n".join(lines)


def _yn(v):
    return "모름" if v is None else ("있음" if v else "없음")
