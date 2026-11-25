"""requirements/ 층 파일이 서로 맞는가 (Docker 계획 D2, 2026-11-25). 표준 라이브러리만 — core venv 에서 돈다."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQ = ROOT / "requirements"
#: core 에 들어오면 안 되는 무거운 패키지
HEAVY = {"torch", "torchvision", "ultralytics", "onnxruntime", "onnxruntime-gpu", "transformers", "triton", "timm"}


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def pins(path, seen=None):
    """{패키지: 버전} — -r 을 따라간다. git+ 줄은 버전 자리에 URL."""
    seen = seen if seen is not None else set()
    out = {}
    if path in seen:
        return out
    seen.add(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("--"):
            continue
        if line.startswith("-r "):
            out.update(pins((path.parent / line[3:].strip()).resolve(), seen))
            continue
        if " @ " in line:
            name, url = line.split(" @ ", 1)
            out[norm(name)] = url.strip()
            continue
        name, sep, ver = line.partition("==")
        assert sep, "%s: 버전을 고정하지 않은 줄 %r" % (path.name, line)
        out[norm(name)] = ver.strip()
    return out


def test_core_has_no_heavy_packages():
    core = pins(REQ / "core.txt")
    assert not (set(core) & HEAVY), set(core) & HEAVY


def test_shared_pins_agree_between_layers():
    gpu = pins(REQ / "vision-gpu.txt")
    for layer in ("core.txt", "vision-cpu.txt", "dev.txt"):
        for name, ver in pins(REQ / layer).items():
            other = gpu.get(name) or gpu.get(name.replace("-headless", ""))
            if other is not None:
                assert other == ver, "%s 의 %s==%s 가 vision-gpu.txt 의 %s 와 다르다" % (layer, name, ver, other)


def test_legacy_requirements_points_at_full_set():
    assert pins(ROOT / "requirements.txt") == pins(REQ / "vision-gpu.txt")
