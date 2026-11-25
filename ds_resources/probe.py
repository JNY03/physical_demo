"""이 기기의 자원 재기 — k3s 없이 (core, 2026-11-24). 리눅스 /proc 와 nvidia-smi 만 쓴다.

호출마다 시간 제한(기본 2초)이 있다. 못 재면 그 칸을 None 으로 두고 멈추지 않는다.

노드별 특성(2026-11-25): caps() 는 잘 안 바뀌는 특성(아키텍처 · 기기 이름 · 센서 · 가속기 · 남길 코어),
thermal() 은 자주 바뀌는 온도 · throttle 을 낸다. 둘 다 **확인된 것만** 낸다 — 못 읽은 칸은 None 이고
추측하지 않는다(배치가 모르는 것을 안다고 믿으면 엉뚱한 곳에 작업을 올린다).
"""
import glob
import os
import platform
import re
import shutil
import subprocess
import time

NVSMI_TIMEOUT = 2.0
VCGENCMD_TIMEOUT = 2.0

#: platform.machine() → 컨테이너 이미지가 쓰는 이름
ARCH_ALIAS = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
#: 보드 · 기기 이름이 적힌 곳 — 라즈베리파이는 device-tree, x86 은 DMI
MODEL_FILES = ("/proc/device-tree/model", "/sys/devices/virtual/dmi/id/product_name")
#: 리눅스 일반 온도 (파이가 아닌 기기)
THERMAL_ZONES = "/sys/class/thermal/thermal_zone*/temp"
#: USB 카메라를 찾을 장치 번호 상한 (/dev/video0 ~ /dev/video9)
CAM_MAX = 10
#: V4L2 기능 비트 — 캡처(단일 · 다중 평면)이고 메모리↔메모리(코덱 · ISP)가 아니어야 카메라다
V4L2_CAPTURE = 0x0001 | 0x1000
V4L2_M2M = 0x4000 | 0x8000
#: VIDIOC_QUERYCAP — _IOR('V', 0, struct v4l2_capability(104바이트))
VIDIOC_QUERYCAP = 0x80685600


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


# ---------------------------------------------------------------- 노드별 특성 (2026-11-25)
def norm_arch(machine=None):
    """아키텍처 이름 정규화 — x86_64 → amd64, aarch64 → arm64. 모르는 이름은 소문자로 그대로, 비면 None."""
    m = str(machine if machine is not None else platform.machine() or "").strip().lower()
    return ARCH_ALIAS.get(m, m or None)


def model_name(files=MODEL_FILES):
    """보드 · 기기 이름 — /proc/device-tree/model(파이) 또는 DMI product_name. 없으면 None.
    device-tree 값은 끝에 NUL 이 붙는다."""
    for p in files:
        txt = _read(p)
        if txt:
            txt = txt.replace("\x00", "").strip()
            if txt:
                return txt
    return None


def v4l2_capture(path):
    """그 /dev/video* 가 실제 캡처 장치인가 — V4L2 기능 비트로 본다. 못 물어보면 False(모르면 세지 않는다).

    라즈베리파이 5 의 /dev/video19 이상은 ISP · 코덱(메모리↔메모리) 노드라 **카메라가 없어도 생긴다**
    (pi6 실측: /dev/video19~35, video19 = rpi-hevc-dec, Device Caps 0x04204000 = M2M 다중 평면).
    그래서 번호만 보고 카메라로 세면 안 된다 — 캡처 비트가 있고 M2M 비트가 없어야 카메라다."""
    import fcntl
    import struct
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        buf = fcntl.ioctl(fd, VIDIOC_QUERYCAP, bytes(104))
        caps_all, dev_caps = struct.unpack_from("<II", buf, 16 + 32 + 32 + 4)
        bits = dev_caps or caps_all
        return bool(bits & V4L2_CAPTURE) and not (bits & V4L2_M2M)
    except OSError:
        return False
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def sensors(dev="/dev", is_capture=v4l2_capture):
    """확인된 센서만 — i2c · serial · camera. 알 수 없으면 넣지 않는다.

    camera 는 /dev/video0~9 중 V4L2 캡처 장치(USB 카메라)만 센다. 파이의 CSI 카메라는 libcamera 쪽을
    봐야 알 수 있어 여기서는 가리지 않는다 — 달려 있어도 'camera' 가 안 나올 수 있다(모르면 모른다)."""
    out = []
    if glob.glob(os.path.join(dev, "i2c-*")):
        out.append("i2c")
    if glob.glob(os.path.join(dev, "ttyACM*")) or glob.glob(os.path.join(dev, "ttyUSB*")):
        out.append("serial")
    for i in range(CAM_MAX):
        p = os.path.join(dev, "video%d" % i)
        if os.path.exists(p) and is_capture(p):
            out.append("camera")
            break
    return out


def accel(dev="/dev", gpu_list=None):
    """가속기 — /dev/hailo* 있으면 npu, NVIDIA GPU 가 보이면 gpu, 아니면 none."""
    if glob.glob(os.path.join(dev, "hailo*")):
        return "npu"
    return "gpu" if (gpu_list if gpu_list is not None else gpus()) else "none"


def cores_reserved():
    """제어용으로 남길 코어 수 — DS_CORES_RESERVED > 설정 파일 > 0. 규칙은 settings 한 곳."""
    from ds_resources import settings
    return settings.cores_reserved()


def caps(dev="/dev", model_files=MODEL_FILES, is_capture=v4l2_capture):
    """이 기기의 특성 {arch, model, sensors, accel, cores_reserved} — 잘 안 바뀌는 것만.
    읽을 수 없는 값은 None 이다(추측하지 않는다)."""
    return {"arch": norm_arch(), "model": model_name(model_files),
            "sensors": sensors(dev, is_capture), "accel": accel(dev),
            "cores_reserved": cores_reserved()}


def _vcgencmd(exe, arg, timeout):
    try:
        out = subprocess.run([exe, arg], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def thermal(timeout=VCGENCMD_TIMEOUT, which=None, zones=THERMAL_ZONES):
    """{temp_c, throttled} — 지금 뜨거운가. 파이는 vcgencmd(measure_temp · get_throttled),
    그 밖의 리눅스는 /sys/class/thermal/thermal_zone*/temp 중 가장 높은 값. 둘 다 없으면 둘 다 None.

    throttled 는 파이의 get_throttled 비트가 0 이 아니면 참이다(지금 또는 부팅 뒤 한 번이라도 조여졌음).
    일반 리눅스에서는 알 수 없어 None 으로 둔다."""
    exe = (which or shutil.which)("vcgencmd")
    if exe:
        temp = thr = None
        m = re.search(r"-?\d+(?:\.\d+)?", _vcgencmd(exe, "measure_temp", timeout) or "")   # temp=55.4'C
        if m:
            temp = float(m.group())
        txt = _vcgencmd(exe, "get_throttled", timeout) or ""                                # throttled=0x0
        if "=" in txt:
            try:
                thr = int(txt.split("=", 1)[1].strip(), 0) != 0
            except ValueError:
                thr = None
        if temp is not None or thr is not None:
            return {"temp_c": temp, "throttled": thr}
    temps = []
    for p in sorted(glob.glob(zones)):                  # CPU 가 몇 번 zone 인지는 기기마다 달라 가장 높은 값을 쓴다
        v = (_read(p) or "").strip()
        if v.lstrip("-").isdigit():
            temps.append(int(v) / 1000.0)
    return {"temp_c": max(temps) if temps else None, "throttled": None}
