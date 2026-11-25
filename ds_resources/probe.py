"""이 기기의 자원 재기 — k3s 없이 (core, 2026-11-24). 리눅스 /proc 와 nvidia-smi 만 쓴다.

호출마다 시간 제한(기본 2초)이 있다. 못 재면 그 칸을 None 으로 두고 멈추지 않는다.
"""
import os
import platform
import shutil
import subprocess
import time

NVSMI_TIMEOUT = 2.0


def node_name():
    """로컬 노드 이름 — DS_NODE_NAME 이 있으면 그것, 없으면 호스트 이름(소문자). 규칙은 settings.node_name 한 곳."""
    from ds_resources import settings
    return settings.node_name()


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def identity():
    """다시 재야 하는지 가르는 신원 — 재부팅(boot_id) · 커널 · OS 가 바뀌면 달라진다."""
    os_name = None
    txt = _read("/etc/os-release") or ""
    for line in txt.splitlines():
        if line.startswith("PRETTY_NAME="):
            os_name = line.split("=", 1)[1].strip().strip('"')
    return {"boot_id": (_read("/proc/sys/kernel/random/boot_id") or "").strip() or None,
            "kernel": platform.release(), "os": os_name}


def gpus(timeout=NVSMI_TIMEOUT):
    """[{name, mem_total_mb, mem_used_mb, util_pct, temp_c}] — nvidia-smi 가 없거나 늦으면 []"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.stdout.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 5:
            continue

        def num(x):
            try:
                return float(x)
            except ValueError:
                return None
        rows.append({"name": p[0], "mem_total_mb": num(p[1]), "mem_used_mb": num(p[2]),
                     "util_pct": num(p[3]), "temp_c": num(p[4])})
    return rows


def _cpu_times():
    line = (_read("/proc/stat") or "cpu 0 0 0 0").splitlines()[0].split()[1:]
    v = [int(x) for x in line[:8]]
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    return sum(v), idle


class UsageProbe:
    """CPU % 는 두 번 읽은 차이로 낸다 — 첫 호출은 None."""

    def __init__(self):
        self.prev = None

    def sample(self, with_gpu=True):
        tot, idle = _cpu_times()
        cpu = None
        if self.prev is not None and tot > self.prev[0]:
            cpu = 100.0 * (1.0 - (idle - self.prev[1]) / float(tot - self.prev[0]))
        self.prev = (tot, idle)
        mem = {}
        for line in (_read("/proc/meminfo") or "").splitlines():
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = int(v.split()[0]) / 1024.0
        total, avail = mem.get("MemTotal"), mem.get("MemAvailable")
        used = (total - avail) if total and avail is not None else None
        temp = None
        t = _read("/sys/class/thermal/thermal_zone0/temp")
        if t and t.strip().lstrip("-").isdigit():
            temp = int(t) / 1000.0
        return {"cpu_pct": cpu, "mem_mb": used, "mem_pct": (100.0 * used / total) if used and total else None,
                "temp_c": temp, "gpu": gpus() if with_gpu else []}


def jpeg_bench(seconds=2.0):
    """짧은 CPU 벤치마크 — 1280x720 JPEG 인코딩을 seconds 동안 몇 장 하나 (장/초). cv2 가 없으면 None."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    img = (np.random.default_rng(0).integers(0, 255, (720, 1280, 3))).astype("uint8")
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        n += 1
    return round(n / (time.perf_counter() - t0), 1)


def baseline(bench=True):
    """기본 자원 — 잘 안 바뀌는 것만 (사용량은 넣지 않는다. 넣으면 매번 '바뀜' 이 된다)."""
    mem_total = None
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemTotal:"):
            mem_total = round(int(line.split()[1]) / 1024.0)
    model = None
    info = (_read("/proc/cpuinfo") or "").splitlines()
    for prefix in ("model name", "Model"):          # x86 · 일부 ARM 은 model name, 라즈베리파이는 Model (보드 이름)
        for line in info:
            if line.startswith(prefix) and ":" in line:
                model = line.split(":", 1)[1].strip()
                break
        if model:
            break
    disk = shutil.disk_usage("/")
    out = {"arch": platform.machine(), "cpu_count": os.cpu_count(), "cpu_model": model,
           "mem_total_mb": mem_total, "disk_total_gb": round(disk.total / 1e9),
           "gpus": [{"name": g["name"], "mem_total_mb": g["mem_total_mb"]} for g in gpus()]}
    if bench:
        out["jpeg_720p_fps"] = jpeg_bench()
    return out
