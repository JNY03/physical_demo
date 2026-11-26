#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ds-device-runner 스테이징 — 빌드 컨텍스트 폴더를 만든다 (build.sh 가 부른다, 2026-11-26).

넣는 것
  requirements.txt                       docker/device-runner/requirements.txt
  vision/device_stage.py                 기기 단계 실행체
  vision/runners/**                      실행기 전부 (*.py · README) — 이름만 더하면 들어간다
  vision/model_registry.yaml · vision/vision_config.yaml   실행기 · 매개변수를 키로 찾는다
  models/…                               models.txt 에 적은 모델의 파일만

models.txt 한 줄 = 하나 (# 뒤는 주석, 빈 줄은 건너뜀)
  <모델 키>      레지스트리 models.<키>.files + 그 키의 매개변수(vision_config.yaml models.<키> → 없으면
                 레지스트리 models.<키>.config) 안의 models/… 경로 중 있는 것. 폴더면 통째로.
  models/<경로>  그 파일 · 폴더를 그대로 (레지스트리에 없는 것을 넣을 때)
  없는 파일은 알리고 건너뛴다(--strict 면 멈춘다). 가중치는 DS_HOME(없으면 저장소 루트) 기준으로 찾는다.

  python docker/device-runner/stage.py --out /tmp/ctx [--models docker/device-runner/models.txt] [--strict]
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PATH_ROOTS = ("models",)


def home():
    v = os.environ.get("DS_HOME")
    return Path(v).expanduser() if v else REPO


def _yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _paths_in(v):
    """매개변수 안의 models/… 문자열들"""
    if isinstance(v, dict):
        for x in v.values():
            yield from _paths_in(x)
    elif isinstance(v, list):
        for x in v:
            yield from _paths_in(x)
    elif isinstance(v, str) and v.replace("\\", "/").split("/", 1)[0] in PATH_ROOTS and "/" in v and "{" not in v:   # {imgsz} 같은 틀은 files 가 맡는다
        yield v.replace("\\", "/")


def model_paths(entry, reg, vcfg):
    """models.txt 한 줄 → [models/… 상대 경로]"""
    if "/" in entry:
        return [entry]
    m = (reg.get("models") or {}).get(entry)
    cfg = (vcfg.get("models") or {}).get(entry)
    if m is None and cfg is None:
        raise KeyError(entry)
    m = m or {}
    if cfg is None:
        cfg = m.get("config") or {}
    out = [str(f).replace("\\", "/") for f in (m.get("files") or [])]
    out += list(_paths_in(cfg))
    return list(dict.fromkeys(p for p in out if p.split("/", 1)[0] in PATH_ROOTS))


def read_manifest(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(src, dst)


def stage(out, manifest, strict=False, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    shutil.copy2(HERE / "requirements.txt", out / "requirements.txt")
    vis = out / "vision"
    vis.mkdir(exist_ok=True)
    for f in ("device_stage.py", "model_registry.yaml", "vision_config.yaml"):
        shutil.copy2(REPO / "vision" / f, vis / f)
    _copy(REPO / "vision" / "runners", vis / "runners")
    reg = _yaml(REPO / "vision" / "model_registry.yaml")
    vcfg = _yaml(REPO / "vision" / "vision_config.yaml")
    added, missing = [], []
    for entry in read_manifest(manifest) if manifest and Path(manifest).exists() else []:
        try:
            paths = model_paths(entry, reg, vcfg)
        except KeyError:
            missing.append("%s (레지스트리 · vision_config 에 없는 키)" % entry)
            continue
        if not paths:
            missing.append("%s (파일 목록이 비었다 — 레지스트리 files 를 채우세요)" % entry)
        for p in paths:
            src = home() / p
            if not src.exists():
                missing.append("%s → %s (없음)" % (entry, p))
                continue
            _copy(src, out / p)
            added.append(p)
    for m in missing:
        log("가중치 빠짐: " + m)
    if missing and strict:
        raise SystemExit("models.txt 의 파일이 다 있지 않다 (--strict)")
    log("스테이징 %s — 실행기 %d 개, 가중치 %d 개" % (out, len(list((vis / "runners").glob("*.py"))), len(added)))
    return added, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description="ds-device-runner 빌드 컨텍스트 만들기")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=str(HERE / "models.txt"))
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    stage(a.out, a.models, a.strict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
