#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""바꾸면 무엇이 달라지는지 — 저장된 이미지에 지금 설정을 **한 번** 걸어 본다 (2026-11-26).

    venv/bin/python tools/preview_apply.py --image stream_data/pi12/20261126_155152 --models yoloe
    venv/bin/python tools/preview_apply.py --image <폴더|파일> --models yoloe \
        --profile A.json --profile-b B.json --out state/preview --json

왜 있나: 런처에서 모델·가중치·파라미터·프롬프트·오버레이를 바꾸면 지금은 **실행해야** 화면이 어떻게
달라지는지 안다. 저장된 장에 미리 걸어 결과 그림과 숫자를 내놓는다. 실시간이 아니다 — 몇 장에 한 번 돈다.

하는 일
  1. 그림 고르기  --image 가 파일이면 그 하나. 폴더면 **최근 --recent 장에서 --frames 개를 고르게**(기본 3장).
                  stream_data/<소스>/<실행>/ (또는 그 안의 frames/)을 주면 frames.jsonl 의 그 장 기록도 읽는다.
  2. 걸어 보기    vision/vision_infer.py 의 적재·추론·오버레이를 **그대로 불러 쓴다**(infer_frame).
                  프로필은 vision/cue_catalog.py 의 resolve_profile 로 확정한다. 여기서 다시 만드는 것은 없다.
  3. 내놓기       --out 폴더에 오버레이 그림을 저장하고, 화면(또는 --json)에
                  {검출 수 · 라벨 · 신뢰도 · 걸린 ms · 결과 그림 경로} 를 돌려준다.
                  검출 수(n)와 **그린 것(drawn)은 다를 수 있다** — 오버레이 min_conf 아래는 결과에는
                  남고 그림에는 없다(운영과 같은 규칙). 둘이 다르면 그렇게 적는다.

런처가 읽는 네 칸 (--json 의 맨 위, 2026-11-26)
  ok        되었는가 (참/거짓)
  overlay   결과 창에 띄울 그림 하나 — 견주기면 나란히 붙인 그림, 아니면 첫 장의 오버레이
  why       안 됐을 때의 까닭 한 줄 (됐으면 빈 글자)
  summary   한 줄 요약 (「yoloe 검출 7개, 61 ms, 1장」)
  나머지 칸(images · sides · totals · missing · warn)은 더 자세히 보려는 쪽이 읽는다.

두 설정 견주기 — 「바꾸면 무엇이 달라지는가」
  --profile-b B.json      프로필(오버레이 · 지정 클래스 대상 · 근거 · 기능)
  --config-b <yaml>       설정 파일 전체. **가중치 · 임계값 · 프롬프트는 프로필이 아니라 여기 있다**
  --set-b models.yoloe.conf=0.4
                          그 자리만 고쳐서. --set 은 A 쪽, --set-b 는 B 쪽(A 위에 더 얹는다)
  셋 중 하나라도 주면 같은 그림에 둘을 돌려 나란히 붙인 그림(<이름>_ab_<모델>.jpg)과
  차이(검출 수 · 한쪽에만 있는 라벨 · 개수가 바뀐 라벨 · ms)를 낸다.

읽기만 한다
  Redis · 스트림 · stream_data 에 아무것도 쓰지 않는다. 그림은 --out 폴더(기본 state/preview)에만 쓴다.
  가시화가 읽는 vision_stream:* 은 건드리지 않는다.

무거운 것
  모델 적재가 오래 걸린다(이 PC 에서 yoloe-11l-seg-pf 실측 2.5~3.3초, 추론은 장당 25~76 ms)
  — **한 번 적재해 여러 장을 돌린다**.
  A·B 의 그 모델 설정(models.<모델>)이 같으면 가중치도 한 번만 적재해 둘이 함께 쓴다.
  가중치가 없으면 **무엇이 없어 못 하는지** 돌려준다 (ok=false · missing). 조용히 빈 결과를 내지 않는다.

미리보기와 실시간이 다른 점 (일부러)
  - 장 사이 안정화·추적(stabilize)을 끈다. 고른 장이 연속이 아니라 이어 붙일 것이 없고,
    min_hits 때문에 첫 장의 검출이 전부 사라진다. 장마다 처음부터 본다.
  - 탐지 영역(ROI)은 없다 — 웹 화면에서 끌어 정하는 것이라 미리보기에 들어오지 않는다.
  - 오버레이 제목의 실행 이름은 `<실행>-a` · `<실행>-b` 다 (두 쪽의 기억을 섞지 않으려고).
"""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
#: 폴더를 줬을 때 고를 밑감 — 최근 이만큼에서 고르게 뽑는다
DEFAULT_RECENT = 200
#: JSON 에 남기는 검출 칸 (polygon 은 빼고 — 한 장에 수만 점이 된다)
DET_KEEP = ("label", "label_raw", "lead", "class_id", "confidence", "bbox", "track_id", "held",
            "depth_m", "range_m", "assess")


class Bad(Exception):
    """사람에게 그대로 보여 줄 까닭"""


def _vision():
    """(vision_infer, cue_catalog) — 늦게 불러온다 (cv2 적재가 무겁다). 그 파일들을 고치지 않고 쓴다."""
    for p in (str(REPO), str(REPO / "vision")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import cue_catalog as cc
    import vision_infer as vi
    return vi, cc


def _why(exc):
    """SystemExit(까닭 글자) · 예외 → 한 줄"""
    if isinstance(exc, SystemExit):
        return str(exc.code)
    return "%s: %s" % (type(exc).__name__, str(exc)[:300])


# ---------------------------------------------------------------- 그림 고르기
def spread(items, k):
    """묶음에서 k 개를 고르게 (처음과 끝을 넣는다). k 가 1 이면 가장 최근 한 장"""
    k = max(1, min(int(k), len(items)))
    if k == 1:
        return [items[-1]]
    step = (len(items) - 1) / (k - 1)
    return [items[round(i * step)] for i in range(k)]


def _run_dir(p):
    """그 그림이 든 실행 폴더 (stream_data/<소스>/<실행>/frames/*.jpg) 또는 None"""
    if p.is_file() and p.parent.name == "frames":
        return p.parents[1]
    if p.is_dir() and (p / "frames").is_dir():
        return p
    if p.is_dir() and p.name == "frames":
        return p.parent
    return None


def source_of(p):
    """실행 폴더 위의 소스 이름 (stream_data/<소스>/<실행>) — 아니면 None"""
    run = _run_dir(p)
    if run is None:
        return None
    name = run.parent.name
    try:
        vi, _ = _vision()
    except Exception:                                      # noqa: BLE001 — 이름만 쓰는 길
        return name or None
    aliases = dict(vi.DEVICE_ALIASES)
    name = aliases.get(name, name)
    return name if name in vi.DEVICES else None


def header_of(p):
    """그 한 장의 frames.jsonl 기록(sender 포함) — 없으면 None. 줄 전체를 읽지 않고 이름으로 찾는다"""
    run = _run_dir(p)
    jl = None if run is None else run / "frames.jsonl"
    if jl is None or not jl.exists():
        return None
    want = p.name.encode()
    with open(jl, "rb") as f:
        for line in f:
            if want not in line:
                continue
            try:
                h = json.loads(line)
            except ValueError:
                continue
            got = str(h.get("frame_file") or h.get("file") or "")
            if Path(got).name == p.name:
                return h
    return None


def pick_images(path, frames=3, recent=DEFAULT_RECENT):
    """[(그림 경로, 그 장의 기록 또는 None)] — 파일 하나이거나, 폴더면 최근 묶음에서 고르게"""
    vi, _ = _vision()
    p = Path(path).expanduser()
    if p.is_file():
        return [(p.resolve(), header_of(p))]
    if not p.is_dir():
        raise Bad("--image 에 준 자리가 없습니다: %s" % p)
    run = _run_dir(p)
    if run is not None:
        items = [(q.resolve(), h) for q, h in vi.read_frame_list(run)]
    else:
        items = [(q.resolve(), None) for q in sorted(p.iterdir()) if q.suffix.lower() in IMAGE_EXTS]
    if not items:
        raise Bad("그림이 없습니다: %s" % p)
    return spread(items[-max(1, int(recent)):], frames)


# ---------------------------------------------------------------- 설정 고치기 (--set)
def set_one(cfg, item):
    """`models.yoloe.conf=0.4` → cfg["models"]["yoloe"]["conf"] = 0.4.
    값은 JSON 으로 읽고(0.4 · 1280 · true · ["person"]) 안 되면 글자 그대로."""
    key, sep, raw = str(item).partition("=")
    parts = [s for s in key.strip().split(".") if s]
    if not sep or not parts:
        raise Bad("--set 은 자리=값 꼴입니다: %r  (예: models.yoloe.conf=0.4)" % item)
    try:
        val = json.loads(raw)
    except ValueError:
        val = raw
    node = cfg
    for s in parts[:-1]:
        nxt = node.get(s)
        if not isinstance(nxt, dict):
            nxt = node[s] = {}
        node = nxt
    node[parts[-1]] = val
    return "%s=%s" % (".".join(parts), json.dumps(val, ensure_ascii=False))


# ---------------------------------------------------------------- 있어야 하는 파일
def _need(model, what, path, required=True):
    return None if not path else {"model": model, "what": what, "path": str(path), "required": bool(required)}


def needs(cfg, models, resolved):
    """적재하기 **전에** 무엇이 있어야 하는지 — [{model, what, path, required}]"""
    _vi, cc = _vision()
    mc = cfg.get("models") or {}
    out = []
    for m in models:
        c = mc.get(m) or {}
        if m == "yoloe":
            pr = c.get("prompt") or {}
            if str(pr.get("mode", "free")).lower() == "text":
                w = pr.get("text_weight") or c.get("weight")
                out.append(_need(m, "yoloe 텍스트 가중치", w))
                enc = Path(str(pr.get("text_encoder_dir") or Path(str(w)).parent)) / "mobileclip_blt.ts"
                # 없으면 ultralytics 가 받는다(~570MB) — 막는 것이 아니라 알리는 것
                out.append(_need(m, "MobileCLIP 텍스트 인코더", enc, required=False))
                out.append(_need(m, "학습 임베딩(learned_pe)", pr.get("learned_pe")))
            else:
                out.append(_need(m, "yoloe prompt-free 가중치", c.get("weight")))
        elif m == "river":
            for i, a in enumerate(c.get("arms") or [], 1):
                # arm 하나가 없어도 on_missing_arm: degrade 로 돈다 — **전부** 없을 때만 못 한다
                out.append(_need(m, "arm %d(%s) 가중치" % (i, a.get("kind") or "?"), a.get("weight"),
                                 required=False))
        elif m == "targeted":
            files = cc.model_files(c)
            for key in sorted(cc.needed_models((resolved or {}).get("targeted") or {}, c)):
                for q in files.get(key, []):
                    out.append(_need(m, "지정 클래스 %s" % key, q))
        elif m == "unidepth":
            out.append(_need(m, "UniDepth 코드(repo_path)", c.get("repo_path")))
            # model_name 은 HuggingFace 이름일 수도 있다 — 없다고 막지 않는다
            out.append(_need(m, "UniDepth 가중치", c.get("model_name"), required=False))
        elif m == "moge2_aerial":
            out.append(_need(m, "MoGe 코드(repo_path)", c.get("repo_path")))
            out.append(_need(m, "MoGe2-Aerial 가중치", c.get("checkpoint")))
    return [x for x in out if x]


def check_files(cfg, models, resolved):
    """(전부, 없어서 못 하는 것, 없지만 돌아가는 것)"""
    rows = needs(cfg, models, resolved)
    for r in rows:
        r["exists"] = Path(r["path"]).exists()
    miss = [r for r in rows if r["required"] and not r["exists"]]
    warn = [r for r in rows if not r["required"] and not r["exists"]]
    arms = [r for r in rows if r["model"] == "river" and r["what"].startswith("arm")]
    if arms and not any(r["exists"] for r in arms):        # arm 이 하나도 없으면 river 는 못 돈다
        miss += arms
        warn = [r for r in warn if r not in arms]
    return rows, miss, warn


# ---------------------------------------------------------------- 한 쪽(A · B) 준비
def pick_device(cfg, want):
    gpu = want or (cfg.get("runtime") or {}).get("device", "cuda:0")
    if str(gpu).startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
        except Exception:                                  # noqa: BLE001
            return "cpu"
    return gpu


def build_side(name, models, config_path, profile_path, sets, gpu):
    """설정 + 프로필 → 그 쪽의 cfg (적재는 아직 하지 않는다)"""
    vi, cc = _vision()
    cfg = vi.load_config(config_path)
    applied = [set_one(cfg, s) for s in sets or []]
    try:
        vi.check_overlay_cfg(cfg)
    except SystemExit as exc:
        raise Bad("설정 overlay 가 틀렸습니다 — %s" % _why(exc))
    raw = {}
    if profile_path:
        if not Path(profile_path).exists():
            raise Bad("프로필 파일이 없습니다: %s" % profile_path)
        try:
            raw = vi.load_profile_file(profile_path)
        except (ValueError, OSError) as exc:
            raise Bad("프로필을 읽지 못했습니다 (%s) — %s" % (profile_path, exc))
    try:
        resolved = cc.resolve_profile(raw, cfg, models)
    except (cc.ProfileError, ValueError, OSError) as exc:
        raise Bad("프로필이 틀렸습니다 (%s) — %s" % (profile_path or "yaml 기본", exc))
    vi.apply_profile(cfg, resolved)
    cfg["_device"] = gpu
    # 미리보기는 장 사이 안정화를 끈다 (머리말 「미리보기와 실시간이 다른 점」)
    cfg["stabilize"] = {"enabled": False}
    return {"name": name, "cfg": cfg, "resolved": resolved,
            "profile": str(profile_path) if profile_path else None, "config": str(config_path),
            "set": applied, "label": resolved.get("label"), "version": resolved.get("version"),
            "runners": {}, "load_ms": 0, "shared": []}


def _fingerprint(mcfg):
    return json.dumps(mcfg, sort_keys=True, ensure_ascii=False, default=str)


def load_runners(side, models, gpu, cache, make=None):
    """가중치 적재 — 설정이 같은 모델은 cache 에서 그대로 쓴다(한 번만 적재)"""
    vi, _cc = _vision()
    make = make or (lambda m, mcfg, dev: vi.RUNNERS[m](mcfg, dev))
    for m in models:
        mcfg = (side["cfg"].get("models") or {}).get(m) or {}
        key = (m, gpu, _fingerprint(mcfg))
        if key in cache:
            side["runners"][m] = cache[key]
            side["shared"].append(m)
            continue
        t0 = time.time()
        try:
            side["runners"][m] = cache[key] = make(m, mcfg, gpu)
        except (SystemExit, Exception) as exc:             # noqa: BLE001 — 까닭을 사람에게 그대로
            raise Bad("%s 모델을 적재하지 못했습니다 — %s" % (m, _why(exc)))
        side["load_ms"] += int((time.time() - t0) * 1000)
    return side


# ---------------------------------------------------------------- 한 장 돌리기
def summarize(rec, overlay_path, wall_ms):
    """결과 기록(v2) → {검출 수 · 라벨 · 신뢰도 · ms · 그림 경로}"""
    dets = rec.get("detections") or []
    labels, confs = {}, []
    for d in dets:
        name = str(d.get("label"))
        labels[name] = labels.get(name, 0) + 1
        c = d.get("confidence")
        if isinstance(c, (int, float)):
            confs.append(float(c))
    counts = rec.get("counts") or {}
    out = {"n": len(dets),
           # 그린 것 — 오버레이 min_conf 아래는 **결과에는 남고 그림에는 없다**(운영과 같은 규칙).
           # 둘이 다르면 "검출 2 (그린 것 1)" 로 보여 준다. 안 그러면 차이 요약과 그림이 어긋나 보인다
           "drawn": sum(counts.values()),
           "labels": dict(sorted(labels.items(), key=lambda t: (-t[1], t[0]))),
           "conf": ({"min": round(min(confs), 4), "max": round(max(confs), 4),
                     "mean": round(sum(confs) / len(confs), 4)} if confs else None),
           "infer_ms": rec.get("infer_ms"), "ms": round(float(wall_ms), 1),
           "overlay": str(overlay_path) if overlay_path else None,
           "counts": counts,
           "detections": [{k: v for k, v in d.items() if k in DET_KEEP} for d in dets]}
    for k in ("depth_stats", "missing_arms", "prompt", "model_file", "roi"):
        if rec.get(k) is not None:
            out[k] = rec[k]
    return out


def run_one(side, models, bgr, header, base, outdir, quality=90):
    """그 쪽 설정으로 한 장 — ({모델: 요약}, {모델: 오버레이 그림})"""
    vi, _cc = _vision()
    cfg = side["cfg"]
    for k in ("_stab", "_lab", "_asm", "_axes"):           # 장마다 처음부터 (고른 장이 연속이 아니다)
        cfg.pop(k, None)
    depth = [m for m in models if m in vi.DEPTH_MODELS]
    save = {m for m in models if m in vi.DETECT_MODELS}
    t0 = time.perf_counter()
    results = vi.infer_frame(bgr, base, header, base["source"], save, depth, side["runners"], cfg)
    wall = (time.perf_counter() - t0) * 1000
    out, imgs = {}, {}
    for m, rec, overlay, _d in results:
        path = None
        if overlay is not None:
            path = Path(outdir) / ("%s_%s_%s.jpg" % (Path(base["frame"]).stem, side["name"], m))
            vi.cv2.imwrite(str(path), overlay, [int(vi.cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            imgs[m] = overlay
        out[m] = summarize(vi.v2_record(rec, header), path, wall)
    return out, imgs


# ---------------------------------------------------------------- 견주기
def diff_one(a, b, img_a=None, img_b=None):
    """한 모델의 두 결과 차이 — 검출 수 · 한쪽에만 있는 라벨 · 개수가 바뀐 라벨 · ms.

    **그림도 견준다**(image_same). 오버레이만 바꾸면 검출은 글자까지 같은데 화면은 달라진다 —
    그때 "같음" 이라고만 하면 바로 그 바뀐 것을 안 보여 주는 셈이다."""
    la, lb = a["labels"], b["labels"]
    changed = {k: [la[k], lb[k]] for k in sorted(set(la) & set(lb)) if la[k] != lb[k]}
    only_a, only_b = sorted(set(la) - set(lb)), sorted(set(lb) - set(la))
    out = {"n": [a["n"], b["n"]], "n_delta": b["n"] - a["n"], "only_a": only_a, "only_b": only_b,
           "changed": changed, "drawn": [a.get("drawn"), b.get("drawn")], "ms": [a["ms"], b["ms"]],
           "same": not (only_a or only_b or changed) and a["n"] == b["n"]}
    if img_a is not None and img_b is not None:
        vi, _cc = _vision()
        out["image_same"] = bool(img_a.shape == img_b.shape and vi.np.array_equal(img_a, img_b))
    return out


def caption(name, me, other):
    """나란히 붙인 그림의 글자 띠 — **무엇이 다른지**를 적는다 (같은 것은 적지 않는다)"""
    bits = [Path(me["profile"]).name if me["profile"] else "yaml 기본"]
    if me["config"] != other["config"]:
        bits.append(Path(me["config"]).name)
    bits += [s for s in me["set"] if s not in other["set"]]
    return "%s · %s" % (name.upper(), " · ".join(bits))


def side_by_side(img_a, img_b, cap_a, cap_b, gap=8):
    """두 오버레이를 나란히 — 글자 띠를 얹고 높이를 맞춰 붙인다 (왼쪽 A · 오른쪽 B)"""
    vi, _cc = _vision()
    np = vi.np
    a, b = vi.add_header(img_a, [cap_a], {}), vi.add_header(img_b, [cap_b], {})
    h = max(a.shape[0], b.shape[0])

    def pad(im):
        if im.shape[0] == h:
            return im
        return np.vstack([im, np.zeros((h - im.shape[0], im.shape[1], 3), im.dtype)])
    bar = np.full((h, max(1, int(gap)), 3), 60, np.uint8)
    return np.hstack([pad(a), bar, pad(b)])


# ---------------------------------------------------------------- 전체
def preview(image, models=None, out="state/preview", config=None, profile=None, sets=(),
            config_b=None, profile_b=None, sets_b=(), frames=3, recent=DEFAULT_RECENT,
            source=None, gpu=None, make=None):
    """저장된 그림에 설정을 걸어 본다 → JSON 사전. B 쪽을 주면 같은 그림에 둘을 돌려 견준다.

    make(모델, 모델설정, 장치) 를 주면 그것으로 실행기를 만든다 (시험에서 가짜를 넣는 자리)."""
    vi, _cc = _vision()
    config = str(config or (REPO / "vision" / "vision_config.yaml"))
    cfg0 = vi.load_config(config)
    want = models or (cfg0.get("defaults") or {}).get("models") or ["yoloe"]
    want = [want] if isinstance(want, str) else list(want)
    try:
        models = vi.normalize(want, vi.MODELS, vi.MODEL_ALIASES, "모델")
    except SystemExit as exc:
        raise Bad(_why(exc))
    models = [m for m in vi.MODELS if m in models]          # 검출을 먼저 (depth 결합 순서)
    gpu = pick_device(cfg0, gpu)
    compare = bool(profile_b or config_b or sets_b)

    picked = pick_images(image, frames, recent)
    src = source or source_of(picked[0][0]) or "preview"
    outdir = Path(out).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    outdir = outdir.resolve()

    sides = [build_side("a", models, config, profile, list(sets), gpu)]
    if compare:
        # B 는 **A 위에 바꾼 것만** — 설정·프로필은 준 것이 이기고, --set 은 A 것에 더 얹는다
        sides.append(build_side("b", models, config_b or config, profile_b or profile,
                                list(sets) + list(sets_b), gpu))
    # 런처가 읽는 네 칸을 맨 위에 둔다 (ok · overlay · why · summary)
    res = {"ok": True, "overlay": None, "why": "", "summary": "",
           "models": models, "device": gpu, "source": src, "out": str(outdir),
           "compare": compare, "frames": len(picked), "sides": {}, "missing": [], "warn": [],
           "images": []}
    for s in sides:
        _rows, miss, warn = check_files(s["cfg"], models, s["resolved"])
        for r in miss:
            r["side"] = s["name"]
        for r in warn:
            r["side"] = s["name"]
        res["missing"] += [r for r in miss if r not in res["missing"]]
        res["warn"] += [r for r in warn if r not in res["warn"]]
    if res["missing"]:
        res["ok"] = False
        res["why"] = "없는 파일이 있어 모델을 적재하지 않았습니다: " + " · ".join(
            "%s(%s)" % (r["what"], Path(r["path"]).name) for r in res["missing"])
        res["summary"] = res["why"]
        return res

    cache = {}
    for s in sides:
        load_runners(s, models, gpu, cache, make)
        res["sides"][s["name"]] = {"profile": s["profile"], "config": s["config"], "set": s["set"],
                                   "label": s["label"], "version": s["version"],
                                   "load_ms": s["load_ms"], "shared_load": sorted(s["shared"])}

    quality = (cfg0.get("output") or {}).get("jpeg_quality", 90)
    for path, header in picked:
        bgr = vi.cv2.imread(str(path), vi.cv2.IMREAD_COLOR)
        if bgr is None:
            res["images"].append({"image": str(path), "error": "이미지를 읽지 못함"})
            continue
        h, w = bgr.shape[:2]
        run = _run_dir(path)
        row = {"image": str(path), "n": vi.frame_number(path), "width": w, "height": h}
        imgs = {}
        for s in sides:
            # frame 은 실행 폴더 기준 이름(운영 결과와 같은 자리). 실행 폴더 밖의 그림이면 파일 이름만
            base = {"n": vi.frame_number(path), "source": src,
                    "frame": "frames/%s" % path.name if run is not None else path.name,
                    "run": "%s-%s" % (run.name if run is not None else "preview", s["name"]),
                    "image_wh": [w, h]}
            row[s["name"]], imgs[s["name"]] = run_one(s, models, bgr, header, base, outdir, quality)
        if compare:
            row["diff"] = {}
            for m in models:
                a, b = row["a"].get(m), row["b"].get(m)
                if not (a and b):
                    continue
                ia, ib = imgs["a"].get(m), imgs["b"].get(m)
                d = diff_one(a, b, ia, ib)
                if ia is not None and ib is not None:
                    pair = outdir / ("%s_ab_%s.jpg" % (path.stem, m))
                    sa, sb = res["sides"]["a"], res["sides"]["b"]
                    vi.cv2.imwrite(str(pair), side_by_side(ia, ib, caption("a", sa, sb),
                                                           caption("b", sb, sa)),
                                   [int(vi.cv2.IMWRITE_JPEG_QUALITY), int(quality)])
                    d["image"] = str(pair)
                row["diff"][m] = d
        res["images"].append(row)
    res["totals"] = summarize_all(res, models, compare)
    res["overlay"] = primary_overlay(res)
    res["summary"] = one_line(res)
    if not res["images"] or all(r.get("error") for r in res["images"]):
        res["ok"] = False
        res["why"] = "그림을 읽지 못했습니다"
    return res


def primary_overlay(res):
    """결과 창에 띄울 그림 하나 — 견주기면 나란히 붙인 것, 아니면 첫 장의 오버레이"""
    for row in res.get("images") or []:
        for d in (row.get("diff") or {}).values():
            if d.get("image"):
                return d["image"]
        for m in res.get("models") or []:
            v = (row.get("a") or {}).get(m) or {}
            if v.get("overlay"):
                return v["overlay"]
    return None


def one_line(res):
    """한 줄 요약 — 런처 결과 창의 머리글"""
    t = res.get("totals") or {}
    if not t.get("images"):
        return "결과 없음"
    bits = []
    for m, v in (t.get("a") or {}).items():
        bits.append("%s 검출 %d개, %.0f ms" % (m, v["n"], v["ms"]))
    head = " / ".join(bits) or "검출 없음"
    if res.get("compare"):
        delta = []
        for m, v in (t.get("b") or {}).items():
            delta.append("%s %+d" % (m, v["n"] - ((t.get("a") or {}).get(m) or {}).get("n", 0)))
        tail = ""
        if t.get("changed_overlays"):
            tail = " (그림만 다른 장 %d)" % t["changed_overlays"]
        return "%s, B 와 견줌(%s), %d장 중 %d장 달라짐%s" % (head, " ".join(delta), t["images"],
                                                      t.get("changed_images", 0), tail)
    return "%s, %d장" % (head, t["images"])


def summarize_all(res, models, compare):
    """전체 요약 — 쪽·모델마다 검출 수 합과 ms 평균, 달라진 장 수"""
    out = {"images": len([r for r in res["images"] if not r.get("error")])}
    for side in ("a", "b") if compare else ("a",):
        part = {}
        for m in models:
            rows = [r[side][m] for r in res["images"] if r.get(side, {}).get(m)]
            if not rows:
                continue
            part[m] = {"n": sum(r["n"] for r in rows),
                       "ms": round(sum(r["ms"] for r in rows) / len(rows), 1),
                       "labels": sorted({k for r in rows for k in r["labels"]})}
        out[side] = part
    if compare:
        diffs = [(r.get("diff") or {}).values() for r in res["images"]]
        out["changed_images"] = len([ds for ds in diffs if any(not d.get("same") for d in ds)])
        # 검출은 같은데 **화면만** 달라진 장 (오버레이만 바꿨을 때)
        out["changed_overlays"] = len([ds for ds in diffs
                                       if any(d.get("same") and d.get("image_same") is False for d in ds)])
    return out


# ---------------------------------------------------------------- 사람이 읽는 꼴
def text(res):
    L = []
    if not res.get("ok"):
        L.append("못 했습니다 — %s" % res.get("why", "까닭 없음"))
        for r in res.get("missing") or []:
            L.append("  없는 파일  %-28s %s" % (r["what"], r["path"]))
        return "\n".join(L)
    L.append("소스 %s · 모델 %s · 장치 %s · %d장 → %s"
             % (res["source"], " ".join(res["models"]), res["device"], res["frames"], res["out"]))
    for s, v in res["sides"].items():
        L.append("  %s: 프로필 %s · 설정 %s%s · 적재 %.1fs%s"
                 % (s.upper(), Path(v["profile"]).name if v["profile"] else "(yaml 기본)",
                    Path(v["config"]).name, (" · " + " ".join(v["set"])) if v["set"] else "",
                    v["load_ms"] / 1000.0, " (함께 쓴 가중치: %s)" % " ".join(v["shared_load"])
                    if v["shared_load"] else ""))
    for r in res.get("warn") or []:
        L.append("  알림: %s 가 없습니다 (%s) — 그대로 돕니다" % (r["what"], r["path"]))
    for row in res["images"]:
        if row.get("error"):
            L.append("%s  %s" % (Path(row["image"]).name, row["error"]))
            continue
        L.append("%s  %dx%d" % (Path(row["image"]).name, row["width"], row["height"]))
        for s in ("a", "b"):
            for m, v in (row.get(s) or {}).items():
                c = v["conf"]
                L.append("  %s %-9s 검출 %3d%s  %6.1f ms  %s%s" % (
                    s, m, v["n"], "" if v["drawn"] == v["n"] else " (그린 것 %d)" % v["drawn"], v["ms"],
                    ", ".join("%s %d" % (k, n) for k, n in list(v["labels"].items())[:6]) or "-",
                    "  신뢰도 %.2f~%.2f" % (c["min"], c["max"]) if c else ""))
                if v["overlay"]:
                    L.append("      → %s" % v["overlay"])
        for m, d in (row.get("diff") or {}).items():
            bits = ["검출 %+d (%d→%d)" % (d["n_delta"], d["n"][0], d["n"][1])]
            if d["only_a"]:
                bits.append("A 만: " + ", ".join(d["only_a"]))
            if d["only_b"]:
                bits.append("B 만: " + ", ".join(d["only_b"]))
            if d["changed"]:
                bits.append("바뀜: " + ", ".join("%s %d→%d" % (k, v[0], v[1]) for k, v in d["changed"].items()))
            if d["same"]:
                bits = ["검출 같음" + ("" if d.get("image_same") is not False else " · **그림이 다름**(오버레이)")]
            L.append("  차이 %-9s %s" % (m, "  ".join(bits)))
            if d.get("image"):
                L.append("      → %s" % d["image"])
    L.append("→ %s" % res.get("summary", ""))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="저장된 이미지에 지금 설정을 걸어 본다 (읽기만 · 실시간 아님)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--image", required=True, metavar="파일|폴더",
                    help="그림 하나, 또는 폴더(stream_data/<소스>/<실행> 이면 frames/ 와 그 기록을 읽는다)")
    ap.add_argument("--models", "-m", nargs="+", metavar="모델", help="yoloe river targeted unidepth moge2_aerial")
    ap.add_argument("--out", default="state/preview", metavar="폴더", help="결과 그림을 쓸 곳 (기본 state/preview)")
    ap.add_argument("--config", metavar="파일", help="vision_config.yaml (기본: vision/vision_config.yaml)")
    ap.add_argument("--profile", metavar="파일", help="프로필 JSON (런처가 만든다)")
    ap.add_argument("--set", action="append", default=[], metavar="자리=값",
                    help="설정 한 자리 고치기. 여러 번. 예) models.yoloe.conf=0.4")
    g = ap.add_argument_group("두 설정 견주기 (주면 같은 그림에 둘을 돌린다)")
    g.add_argument("--config-b", metavar="파일", help="B 쪽 설정 파일 (가중치·임계값·프롬프트는 여기 있다)")
    g.add_argument("--profile-b", metavar="파일", help="B 쪽 프로필 JSON")
    g.add_argument("--set-b", action="append", default=[], metavar="자리=값", help="B 쪽만 고치기 (A 위에 더)")
    ap.add_argument("--frames", type=int, default=3, metavar="N", help="폴더를 줬을 때 볼 장 수 (기본 3)")
    ap.add_argument("--recent", type=int, default=DEFAULT_RECENT, metavar="N",
                    help="최근 N장에서 고르게 뽑는다 (기본 %d)" % DEFAULT_RECENT)
    ap.add_argument("--source", metavar="기기", help="기기별 오버레이 지정에 쓸 이름 (기본: 경로에서 읽는다)")
    ap.add_argument("--gpu", metavar="장치", help="cuda:0 / cpu (기본: config runtime.device)")
    ap.add_argument("--json", action="store_true", help="JSON 한 줄로")
    a = ap.parse_args(argv)
    # **--json 이면 표준출력에 JSON 말고 아무것도 없다**(런처가 통째로 ConvertFrom-Json 한다).
    # 비전 쪽 적재 로그(vision_infer.log) · 진행 상황은 표준에러로 보낸다.
    with contextlib.redirect_stdout(sys.stderr if a.json else sys.stdout):
        try:
            res = preview(a.image, a.models, a.out, a.config, a.profile, a.set,
                          a.config_b, a.profile_b, a.set_b, a.frames, a.recent, a.source, a.gpu)
        except Bad as exc:
            res = {"ok": False, "overlay": None, "why": str(exc), "summary": str(exc),
                   "missing": [], "warn": [], "images": []}
        except KeyboardInterrupt:
            return 130
    print(json.dumps(res, ensure_ascii=False) if a.json else text(res))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
