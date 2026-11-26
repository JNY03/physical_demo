"""역할 슬롯 — 기능 → 단계 → 역할 슬롯 → 모델 (slots/1, core, 2026-11-27).

표준 라이브러리 + yaml 만 쓴다(functions.py 와 같은 층). torch · vision 을 import 하지 않는다.

레지스트리(vision/model_registry.yaml)에서
  부품(parts)   options.<키> (파이프라인 — 지금 비전이 돌리는 선택지)과 models.<키> (모델 · 구성품)을 한 표로 편다.
                같은 키가 둘에 있으면(moge2_aerial) 선택지가 이긴다 — 혼자 도는 쪽이 그것이다.
  슬롯(slots)   functions.load 가 단계마다 편 `slots` 목록 (옛 options · cue_model_choices 도 슬롯으로 바뀌어 있다)

을 읽어 **슬롯마다 고를 수 있는 것**을 계산한다.

  choices(slot, side, stage, parts)   → [부품 키…] 기본값이 맨 앞. 명시 선택지(choices) 다음에 능력으로 저절로
                                       맞는 것(auto). 저절로 맞는 조건:
                                         - 혼자 도는 부품(선택지 · runner 가 있는 모델). 슬롯이 component 면
                                           (기능 코드가 모델 파일을 직접 부르는 자리) runner 없는 구성 모델도
                                         - slot.needs ⊆ part.provides
                                         - 그 단계에서 돌 수 있다: 기기 단계 → runs_on 에 cpu_arm64
                                           서버 단계 → gpu (needs.accel 이 gpu 가 아니면 cpu_x86 도)
                                         - slot.size 가 small 이면 part.size 도 small
  function_slots(reg, funcs, have)    → 런처가 받는 모양 (cli places --json 의 function_slots)
  check_picks(reg, funcs, picks)      → 프로필 functions: {기능: {단계: {역할: 모델 | [모델]}}} 검사 (ValueError)
  effective(reg, funcs, picks, names) → 기본값으로 채운 선택 (고른 것이 이긴다)
  stage_costs(reg, selection)         → 단계마다 GPU MB · p90 — 같은 모델은 한 번만 센다

라이선스는 **안내만** 한다 — 모델은 바꿔 끼우는 부품이라 라이선스로 막지 않는다.
"""
import json
import sys

from ds_resources import functions as fnc
from ds_resources import settings

FORMAT = "slots/1"
RUNS_ON = ("gpu", "cpu_x86", "cpu_arm64")
#: 능력 이름 — 적는 쪽의 약속이다(모르는 능력도 받는다 — 새 능력은 모델과 슬롯이 같이 쓰면 된다)
KNOWN_CAPS = ("detect", "segment", "open_vocab", "ovd", "depth_metric", "depth_relative", "track", "vlm", "embed",
              # 지정 물체 탐지의 근거 모델 자리 (2026-11-27) — 약속은 레지스트리 targeted_detect 서버 단계 slots 주석
              "detect_text", "detect_free", "embed_image_text")
#: 부품 종류 — runner 의 kind 와 같은 이름
KINDS = ("detect", "segment", "depth", "track", "vlm", "embed")


# ---------------------------------------------------------------- 부품 표
def _uses(key, o, reg):
    """선택지가 쓰는 구성 모델 키 — 가중치 유무를 볼 때"""
    cm = o.get("cue_models")
    if cm == "targeted":
        return ["yoloe_text"]
    if isinstance(cm, dict):
        return sorted({m for ms in cm.values() for m in (ms or [])})
    return [key] if key in (reg.get("models") or {}) else []


def _kind(spec):
    k = spec.get("kind")
    if k:
        return str(k)
    caps = set(spec.get("provides") or ())
    if caps & {"depth_metric", "depth_relative", "depth"}:
        return "depth"
    for k in ("segment", "detect", "track", "vlm", "embed"):
        if k in caps:
            return k
    return None


def is_runner_ref(runner):
    """runner 칸이 vision/runners 의 「모듈:클래스」 인가 (그 밖은 옛 선택지 키)"""
    return isinstance(runner, str) and ":" in runner


def parts(reg):
    """{키: 부품}. 부품 = {key, shown, kind, pipeline, domain, runnable, provides, runs_on, size, runner,
    license, gpu_mb, p90_ms, uses, note}"""
    out = {}
    models = reg.get("models") or {}
    for k, o in (reg.get("options") or {}).items():
        o = o or {}
        out[str(k)] = {"key": str(k), "shown": str(o.get("shown") or k), "kind": _kind(o),
                       "pipeline": True, "domain": bool(o.get("domain")), "runnable": True,
                       "provides": [str(x) for x in (o.get("provides") or [])],
                       "runs_on": [str(x) for x in (o.get("runs_on") or ["gpu"])],
                       "size": o.get("size"), "runner": str(k), "license": o.get("license"),
                       "gpu_mb": o.get("gpu_mb"), "p90_ms": o.get("p90_ms") or {},
                       "uses": _uses(k, o, reg), "note": str(o.get("note") or "")}
    for k, m in models.items():
        if k in out:
            continue
        m = m or {}
        dev = str(m.get("device") or "gpu")
        runs = m.get("runs_on") or (["gpu"] if dev == "gpu" else ["cpu_x86"])
        runner = m.get("runner")
        out[str(k)] = {"key": str(k), "shown": str(m.get("shown") or k), "kind": _kind(m),
                       "pipeline": False, "domain": False, "runnable": is_runner_ref(runner),
                       "provides": [str(x) for x in (m.get("provides") or [])],
                       "runs_on": [str(x) for x in runs], "size": m.get("size"),
                       "runner": None if runner is None else str(runner), "license": m.get("license"),
                       "gpu_mb": m.get("gpu_mb"), "p90_ms": m.get("p90_ms") if m.get("p90_ms") is not None else {},
                       "uses": [str(k)], "note": str(m.get("note") or "")}
    return out


def check_parts(reg):
    """부품 선언 검사 — 틀리면 ValueError (runs_on 이름 · size)"""
    for k, p in parts(reg).items():
        bad = [x for x in p["runs_on"] if x not in RUNS_ON]
        if bad:
            raise ValueError("%s 의 runs_on 에 모르는 곳 %s (가능: %s)" % (k, ", ".join(bad), ", ".join(RUNS_ON)))
        if p["size"] not in (None, "small", "any"):
            raise ValueError("%s 의 size 는 small 이나 any (받은 것: %r)" % (k, p["size"]))
    return True


# ---------------------------------------------------------------- 선택지 계산
def platforms(side, stage=None):
    """그 단계에서 돌 수 있는 곳 — 기기 단계는 파이(cpu_arm64), 서버 단계는 GPU (가속 요구가 없으면 cpu_x86 도)"""
    if side == "device":
        return {"cpu_arm64"}
    accel = ((stage or {}).get("needs") or {}).get("accel")
    return {"gpu"} if accel == "gpu" else {"gpu", "cpu_x86"}


def fits(part, slot, side, stage=None):
    """그 부품이 그 슬롯에 저절로 맞는가 (능력 · 도는 곳 · 크기)"""
    if part.get("domain"):
        return False
    if not part.get("runnable") and not (slot.get("component") and not part.get("pipeline")):
        return False
    if not set(slot.get("needs") or ()) <= set(part.get("provides") or ()):
        return False
    if not set(part.get("runs_on") or ()) & platforms(side, stage):
        return False
    if slot.get("size") == "small" and part.get("size") != "small":
        return False
    return True


def choices(slot, side, stage, table):
    """슬롯의 선택지 [키…] — 기본값이 맨 앞, 다음 명시 선택지(적은 순서), 다음 능력으로 맞는 것(키 순서)"""
    out = [k for k in slot.get("choices") or [] if k in table]
    if slot.get("auto"):
        out += sorted(k for k, p in table.items() if k not in out and fits(p, slot, side, stage))
    dflt = slot.get("default")
    first = [d for d in (dflt if isinstance(dflt, list) else [dflt]) if d in out]
    return first + [k for k in out if k not in first]


def check_registry(reg, funcs=None):
    """슬롯의 명시 선택지 · 기본값이 부품 표에 있는가 — 틀리면 ValueError"""
    funcs = fnc.load(reg) if funcs is None else funcs
    table = parts(reg)
    check_parts(reg)
    for fn, g in funcs.items():
        for side, st in g["stages"].items():
            for sl in st.get("slots") or []:
                what = "functions.%s.stages.%s.slots.%s" % (fn, side, sl["role"])
                bad = [k for k in sl["choices"] if k not in table]
                if bad:
                    raise ValueError("%s 의 choices 에 모르는 모델 %s" % (what, ", ".join(bad)))
                d = sl["default"]
                ch = choices(sl, side, st, table)
                for x in (d if isinstance(d, list) else ([] if d is None else [d])):
                    if x not in ch:
                        raise ValueError("%s 의 default %r 가 선택지에 없다 (선택지: %s)"
                                         % (what, x, ", ".join(ch) or "없음"))
    return True


def role_files(model_key, reg, needs=()):
    """그 슬롯(능력 needs)에서 쓸 때 있어야 하는 파일 — files_for.<능력> 이 있으면 그것을 더한다
    (예: clip_fp32 는 river 에는 비전 인코더만, embed_image_text 로는 글 인코더 · 토크나이저까지)"""
    m = (reg.get("models") or {}).get(model_key) or {}
    out = list(m.get("files") or [])
    for cap in needs or ():
        for f in ((m.get("files_for") or {}).get(cap) or []):
            if f not in out:
                out.append(f)
    return out


def _have_files(model_key, reg, needs=()):
    return all(settings.resolve(f).exists() for f in role_files(model_key, reg, needs))


def available(part, reg, have=None, needs=()):
    """가중치가 제자리에 있는가 — 구성 모델의 files 가 모두 있으면 참. have(모델 키) 로 바꿔 끼울 수 있다.
    needs(슬롯 능력)를 주면 그 능력으로 쓸 때의 파일(files_for)까지 본다"""
    have = have or (lambda k: _have_files(k, reg, needs))
    return all(have(k) for k in part.get("uses") or [])


def _choice_row(part, reg, have, needs=()):
    ok = available(part, reg, have, needs)
    return {"key": part["key"], "shown": part["shown"], "provides": list(part["provides"]),
            "runs_on": list(part["runs_on"]), "size": part.get("size"), "gpu_mb": part.get("gpu_mb"),
            "p90_ms": part.get("p90_ms") or {}, "license": part.get("license"), "available": bool(ok),
            "note": (part.get("note") or "") if ok else ("가중치 없음 — " + (part.get("note") or part["key"]))}


def function_slots(reg, funcs=None, have=None, names=None):
    """런처 모양 — {기능: {단계: [{role, label, label_en, needs, optional, multi, size, choices: [...], default}]}}.
    슬롯이 없는 기능 · 단계는 빠진다. names 를 주면 그 기능만"""
    funcs = fnc.load(reg) if funcs is None else funcs
    table = parts(reg)
    cache = {}

    def row(k, needs=()):
        ck = (k, tuple(needs))
        if ck not in cache:
            cache[ck] = _choice_row(table[k], reg, have, needs)
        return cache[ck]

    out = {}
    for fn, g in sorted(funcs.items()):
        if names is not None and fn not in names:
            continue
        for side in fnc.SIDES:
            st = g["stages"].get(side)
            if not st or not st.get("slots"):
                continue
            rows = []
            for sl in st["slots"]:
                ch = choices(sl, side, st, table)
                rows.append({"role": sl["role"], "label": sl["label"], "label_en": sl["label_en"],
                             "needs": list(sl["needs"]), "optional": sl["optional"], "multi": sl["multi"],
                             "size": sl["size"], "origin": sl.get("origin"), "component": bool(sl.get("component")),
                             "choices": [dict(row(k, sl["needs"])) for k in ch], "default": sl["default"]})
            out.setdefault(fn, {})[side] = rows
    return out


# ---------------------------------------------------------------- 프로필 (M4)
def _slot_of(funcs, fn, side, role):
    st = (funcs.get(fn) or {}).get("stages", {}).get(side)
    if st is None:
        return None, None
    return next((x for x in st.get("slots") or [] if x["role"] == role), None), st


def check_picks(reg, funcs, picks):
    """프로필 functions 칸 → 고르게 편 사전 {기능: {단계: {역할: 키 | [키] | None}}}. 틀리면 ValueError.

    None(또는 multi 의 빈 목록)은 「그 역할을 비운다」 — optional 슬롯만 된다"""
    if picks in (None, {}):
        return {}
    if not isinstance(picks, dict):
        raise ValueError("functions 는 {기능: {단계: {역할: 모델}}} 사전이어야 한다")
    funcs = fnc.load(reg) if funcs is None else funcs
    table = parts(reg)
    out = {}
    for fn, stages in picks.items():
        if fn not in funcs:
            raise ValueError("functions 에 모르는 기능 %r" % fn)
        if not isinstance(stages, dict):
            raise ValueError("functions.%s 는 {단계: {역할: 모델}} 사전" % fn)
        for side, roles in stages.items():
            if side not in funcs[fn]["stages"]:
                raise ValueError("functions.%s 에 %r 단계가 없다 (있는 것: %s)"
                                 % (fn, side, ", ".join(sorted(funcs[fn]["stages"]))))
            if not isinstance(roles, dict):
                raise ValueError("functions.%s.%s 는 {역할: 모델} 사전" % (fn, side))
            for role, v in roles.items():
                sl, st = _slot_of(funcs, fn, side, role)
                what = "functions.%s.%s.%s" % (fn, side, role)
                if sl is None:
                    have = [x["role"] for x in st.get("slots") or []]
                    raise ValueError("%s — 그런 역할이 없다 (있는 것: %s)" % (what, ", ".join(have) or "없음"))
                ch = choices(sl, side, st, table)
                vals = [] if v is None else (list(v) if isinstance(v, (list, tuple)) else [v])
                if not vals:
                    if not sl["optional"]:
                        raise ValueError("%s 는 비울 수 없다 (필수 역할)" % what)
                    val = [] if sl["multi"] else None
                else:
                    if len(vals) > 1 and not sl["multi"]:
                        raise ValueError("%s 는 하나만 고른다" % what)
                    for m in vals:
                        if m not in ch:
                            raise ValueError("%s 에 고를 수 없는 모델 %r (가능: %s)"
                                             % (what, m, ", ".join(ch) or "없음"))
                    val = [str(m) for m in dict.fromkeys(vals)] if sl["multi"] else str(vals[0])
                out.setdefault(fn, {}).setdefault(side, {})[role] = val
    return out


def effective(reg, funcs, picks=None, names=None):
    """{기능: {단계: {역할: 값}}} — 고른 것이 없으면 기본값. names 를 주면 그 기능만"""
    funcs = fnc.load(reg) if funcs is None else funcs
    picks = picks or {}
    out = {}
    for fn, g in sorted(funcs.items()):
        if names is not None and fn not in names:
            continue
        for side, st in g["stages"].items():
            for sl in st.get("slots") or []:
                given = ((picks.get(fn) or {}).get(side) or {})
                val = given[sl["role"]] if sl["role"] in given else sl["default"]
                out.setdefault(fn, {}).setdefault(side, {})[sl["role"]] = val
    return out


def _p90(p90, bucket):
    if isinstance(p90, dict):
        return p90.get(bucket)
    return p90


def stage_costs(reg, selection, buckets=("s", "m", "l", "xl")):
    """단계마다 {models, gpu_mb, p90_ms: {구간: ms}} — **같은 모델은 한 번만 센다**(여러 기능 · 슬롯이 한 프로세스에서
    같은 모델을 쓴다). 프로세스 기본 몫(base_gpu_mb)은 넣지 않는다 — 단계 합에 더하는 쪽이 정한다.
    선택지(파이프라인)의 gpu_mb 가 없으면(targeted) 0 으로 센다 — 근거 모델 몫은 cue_catalog.needed_models 가 센다"""
    table = parts(reg)
    seen = {}
    for fn, stages in (selection or {}).items():
        for side, roles in stages.items():
            for _role, v in roles.items():
                for k in (v if isinstance(v, list) else ([] if v is None else [v])):
                    seen.setdefault(side, [])
                    if k in table and k not in seen[side]:
                        seen[side].append(k)
    out = {}
    for side, keys in sorted(seen.items()):
        mb = sum(int(table[k].get("gpu_mb") or 0) for k in keys)
        p90 = {b: float(sum(float(_p90(table[k].get("p90_ms"), b) or 0) for k in keys)) for b in buckets}
        out[side] = {"models": sorted(keys), "gpu_mb": mb, "p90_ms": p90}
    return out


def main(argv=None):
    """기능마다 슬롯과 선택지를 찍는다 (읽기만)"""
    import argparse
    ap = argparse.ArgumentParser(description="역할 슬롯 · 선택지 (slots/1)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    reg = fnc._registry(None)
    funcs = fnc.load(reg)
    check_registry(reg, funcs)
    fs = function_slots(reg, funcs)
    if a.json:
        print(json.dumps(fs, ensure_ascii=False))
        return 0
    for fn, stages in fs.items():
        for side, rows in stages.items():
            for r in rows:
                ch = ", ".join(("%s%s" % (c["key"], "" if c["available"] else "(없음)")) for c in r["choices"])
                print("%-18s %-6s %-10s %-5s needs=%-22s %s" % (fn, side, r["role"],
                      "선택" if r["optional"] else "필수", ",".join(r["needs"]) or "-", ch or "(선택지 없음)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
