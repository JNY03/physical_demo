"""기능 계약 — 레지스트리 functions 절을 읽고 · 검증하고 · 단계로 펼친다 (functions/1, core, 2026-11-25).

표준 라이브러리 + yaml 만 쓴다. orchestrator/ 를 import 하지 않는다 — 계약은 실행 방식보다 아래에 있다.

왜 있나: cpu_tasks 에는 중요도 · 마감 · 고정 · 센서 요구를 적는 칸이 없어 「안전 제어는 로봇 안에서만,
1차 추론은 기기, 무거운 모델은 서버」(비전_오케스트레이션_의사결정_스케줄링_261125.md §10)를 코드로 적을 수
없었다. 이 모듈이 그 계약을 읽는다. 배치 알고리즘(placement/1)은 아직 이것을 쓰지 않는다 — 계약이 먼저다.

  load(경로 | 사전)         → {기능: {pattern, criticality, trigger, note, forbid, stages, ...}}
                             모르는 칸 이름 · 범위 밖 값은 ValueError 다. 메시지에 가능한 값을 나열한다
                             (streams.check_code 와 같은 방식 — 문구 비교로 되돌아가지 않게)
                             레지스트리에 없고 여기서 더하는 칸: priority_class (criticality 로 고른 k3s
                             PriorityClass), legacy (옛 cpu_tasks 에서 온 별칭인가), 단계의 unit (배치표 이름)
  expand(funcs, names)     → (배치_단계들, 상주들, 그밖). resident 는 상주로, config · pipeline 은 그밖으로
                             빠진다 — 오케스트레이터는 그것을 「놓는」 게 아니라 있는지 보고 없으면 알린다
  pairs(stages, placed)    → {기능: {verdict: ok | degraded | rejected, ...}}. 판정만 돌려준다 — 스트림에 쓰지 않는다
  as_cpu_task(stage)       → placement.place_cpu 가 받는 작업 사전 (옛 cpu_tasks 와 같은 칸 이름 · 같은 뜻)

역할 슬롯 (slots/1, 2026-11-27) — 단계마다 「어떤 능력의 모델이 들어갈 자리」를 선언한다.
  stages.<단계>.slots.<역할>: {label, label_en, needs: [능력], optional, multi, size: small|any,
                              default, choices: [명시 선택지], auto}
  읽은 단계에는 `slots` 목록(역할 이름순이 아니라 적은 순서)이 붙는다. 선택지 계산은 ds_resources/slots.py 가 한다
  (여기서는 선언만 고르게 편다 — 모델 표를 모른다).
  옛 칸은 그대로 읽히고 슬롯으로도 바뀐다(별칭):
    functions.<기능>.options: [선택지]   → 서버 단계 `model` 슬롯 (choices = options, 기본 = 첫째)
    cue_model_choices.<근거>             → targeted_detect 서버 단계의 `ovd` · `depth` 슬롯

옛 cpu_tasks.<이름> 도 단일 단계 기능으로 읽힌다(별칭). 그 별칭 단계의 unit 은 기능 이름 그대로여서
배치표 · 가시화가 쓰던 이름(cpu:<이름>)이 바뀌지 않는다. functions 에 같은 이름이 있으면 functions 가 이긴다.
그 밖의 단계는 unit 이 「<기능>.<단계>」 다 — 한 기능이 기기 · 서버 둘에 놓일 수 있으니 이름이 갈려야 한다.
"""
import re

import yaml

from ds_resources.settings import REPO

FORMAT = "functions/1"
REGISTRY = REPO / "vision" / "model_registry.yaml"

PATTERNS = ("resident", "device_only", "cascade", "split", "server_only", "config", "pipeline")
#: 배치 대상 — 오케스트레이터가 노드에 놓는 패턴
PLACEABLE = ("device_only", "cascade", "split", "server_only")
#: 상주 — 기기 안에 늘 떠 있어야 하는 것 (놓지 않고 있는지만 본다)
RESIDENT = ("resident",)
#: 그밖 — 값을 만들어 두거나 받아서 적용하는 흐름 (놓지 않고 있는지만 본다)
DECLARED = ("config", "pipeline")
CRITICALITY = ("safety", "mission", "best_effort")
TRIGGERS = ("always", "on_activity", "on_request", "periodic")
SIDES = ("device", "server")
PINS = ("device", "server", "none")
ARCHS = ("arm64", "amd64", "any")
SENSORS = ("camera", "imu", "lidar", "serial")
ACCELS = ("none", "gpu", "npu")
STATES = ("stateless", "sticky")
PLACEMENTS = ("any", "measured")
#: 둘 수 있는 노드 역할 — orchestrator/cluster/nodes.json 과 같은 이름
ROLES = ("gpu", "robot", "sensor", "small")
VERDICTS = ("ok", "degraded", "rejected")
#: criticality → k3s PriorityClass (orchestrator/k3s/priorityclass*.yaml). 여기서는 알려만 준다 — 붙이는 건 실행 쪽 일이다
PRIORITY_CLASSES = {"safety": "ds-safety-high", "mission": "ds-mission", "best_effort": "ds-vision-low"}

#: 기능 칸 — 칸 추가만 한다. 기존 칸의 이름 · 뜻은 바꾸지 않는다
FUNCTION_KEYS = ("pattern", "criticality", "trigger", "note", "forbid", "stages", "options")
FUNCTION_REQUIRED = ("pattern", "criticality", "trigger", "note", "stages")
NEEDS_KEYS = ("arch", "sensors", "accel", "cores")
#: 자원 칸 — cpu_tasks 와 같은 이름 · 같은 뜻
RESOURCE_KEYS = ("cpu_m", "mem_mb", "frame_kb", "fps", "proc_ms", "roles", "placement",
                 "image", "command", "local_command", "env")
STAGE_KEYS = ("pin", "needs", "deadline_ms", "egress_kb", "state", "optional", "note",
              "per_source", "slots") + RESOURCE_KEYS
#: 역할 슬롯 칸 (slots/1) — 칸 추가만 한다
SLOT_KEYS = ("label", "label_en", "needs", "optional", "multi", "size", "default", "choices", "auto")
SLOT_SIZES = ("small", "any")
#: 옛 options 가 바뀌어 들어가는 슬롯 역할 · 옛 cue_model_choices 가 들어가는 기능
MODEL_ROLE = "model"
CUE_SLOT_FUNCTION = "targeted_detect"
#: cue_model_choices 근거 → 슬롯 이름표 (런처 · 카드)
CUE_SLOT_LABELS = {"ovd": ("OVD 동의 모델", "OVD agreement"), "depth": ("거리 모델", "distance model")}
#: 수로 적는 자원 칸 (없으면 null)
NUMERIC_KEYS = ("cpu_m", "mem_mb", "frame_kb", "fps", "proc_ms")
NAME_RE = re.compile(r"[a-z][a-z0-9_]*\Z")


# ---------------------------------------------------------------- 검사 (streams.check_code 방식)
def _one(value, allowed, what):
    if value not in allowed:
        raise ValueError("모르는 %s %r (가능: %s)" % (what, value, ", ".join(allowed)))
    return value


def _dict(body, allowed, what):
    if not isinstance(body, dict):
        raise ValueError("%s 는 사전이어야 한다 (받은 것: %s)" % (what, type(body).__name__))
    extra = sorted(set(body) - set(allowed))
    if extra:
        raise ValueError("%s 에 모르는 칸 %s (가능: %s)" % (what, ", ".join(extra), ", ".join(allowed)))
    return body


def _num(value, what, low=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s 는 수여야 한다 (받은 것: %r)" % (what, value))
    if value < low:
        raise ValueError("%s 는 %g 이상이어야 한다 (받은 것: %r)" % (what, low, value))
    return value


def _opt_num(body, key, what, low=0.0):
    return None if body.get(key) is None else _num(body[key], "%s.%s" % (what, key), low)


def _bool(value, what):
    if not isinstance(value, bool):
        raise ValueError("%s 는 참/거짓이어야 한다 (받은 것: %r)" % (what, value))
    return value


def _list(value, what):
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("%s 는 목록이어야 한다 (받은 것: %r)" % (what, value))
    return list(value)


def _subset(value, allowed, what):
    out = _list(value, what)
    bad = sorted(set(map(str, out)) - set(allowed))
    if bad:
        raise ValueError("모르는 %s %s (가능: %s)" % (what, ", ".join(bad), ", ".join(allowed)))
    return [str(x) for x in out]


def _cmd(value, what):
    if value is None:
        return None
    out = _list(value, what)
    if not out or not all(isinstance(x, str) for x in out):
        raise ValueError("%s 는 글자 목록이어야 한다 (예: [python, e2p/slice360_live.py])" % what)
    return out


def _env(value, what):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("%s 는 사전이어야 한다 (받은 것: %r)" % (what, value))
    return {str(k): str(v) for k, v in value.items()}


def _needs(value, what):
    body = _dict({} if value is None else value, NEEDS_KEYS, what)
    return {"arch": _one(body.get("arch", "any"), ARCHS, what + ".arch"),
            "sensors": sorted(_subset(body.get("sensors"), SENSORS, what + ".sensors")),
            "accel": _one(body.get("accel", "none"), ACCELS, what + ".accel"),
            "cores": _opt_num(body, "cores", what)}


# ---------------------------------------------------------------- 역할 슬롯
def _slot(role, body, what):
    """slots.<역할> 선언 → 고르게 편 사전. 선택지 계산(모델 표)은 slots.py 몫이다"""
    if not NAME_RE.match(str(role)):
        raise ValueError("%s 의 역할 이름 %r 은 영어 소문자 snake_case 여야 한다" % (what, role))
    body = _dict({} if body is None else body, SLOT_KEYS, what)
    needs = [str(x) for x in _list(body.get("needs"), what + ".needs")]
    choices = [str(x) for x in _list(body.get("choices"), what + ".choices")]
    multi = _bool(body.get("multi", False), what + ".multi")
    default = body.get("default")
    if default is not None:
        if multi:
            default = [str(x) for x in (default if isinstance(default, (list, tuple)) else [default])]
        elif isinstance(default, (list, tuple)):
            raise ValueError("%s.default 는 multi 가 아니라서 하나여야 한다 (받은 것: %r)" % (what, default))
        else:
            default = str(default)
    return {"role": str(role), "label": str(body.get("label") or role),
            "label_en": str(body.get("label_en") or body.get("label") or role),
            "needs": needs, "optional": _bool(body.get("optional", False), what + ".optional"),
            "multi": multi, "size": _one(body.get("size", "any"), SLOT_SIZES, what + ".size"),
            "default": default, "choices": choices,
            # auto: 능력(needs)으로 모델 표에서 저절로 채우는가. 적지 않으면 needs 가 있을 때만 그렇다
            "auto": _bool(body.get("auto", bool(needs)), what + ".auto"),
            "origin": "slots"}


def _slots(body, what):
    if body is None:
        return []
    if not isinstance(body, dict):
        raise ValueError("%s 는 역할 → 슬롯 사전이어야 한다 (받은 것: %s)" % (what, type(body).__name__))
    return [_slot(r, b, "%s.%s" % (what, r)) for r, b in body.items()]


def _legacy_model_slot(stage, options):
    """옛 options → 서버 단계 `model` 슬롯. 이미 model 슬롯이 있으면 options 를 명시 선택지 앞에 얹는다"""
    slots = stage.setdefault("slots", [])
    have = next((x for x in slots if x["role"] == MODEL_ROLE), None)
    if have is None:
        slots.insert(0, {"role": MODEL_ROLE, "label": "모델", "label_en": "model", "needs": [], "optional": False,
                         "multi": False, "size": "any", "default": options[0], "choices": list(options),
                         "auto": False, "origin": "options"})
        return
    have["choices"] = list(dict.fromkeys(list(options) + have["choices"]))
    if have["default"] is None:
        have["default"] = options[0]


def _cue_slots(stage, cue_choices):
    """옛 cue_model_choices → targeted_detect 서버 단계의 근거 모델 슬롯 (이미 적었으면 그것이 이긴다)"""
    slots = stage.setdefault("slots", [])
    roles = {x["role"] for x in slots}
    for need, ch in (cue_choices or {}).items():
        if need in roles or not isinstance(ch, dict):
            continue
        choices = [str(m) for m in (ch.get("choices") or [])]
        if not choices:
            continue
        multi = bool(ch.get("multi"))
        lab, lab_en = CUE_SLOT_LABELS.get(need, (need, need))
        slots.append({"role": str(need), "label": lab, "label_en": lab_en, "needs": [],
                      "optional": True, "multi": multi, "size": "any",
                      "default": [choices[0]] if multi else choices[0], "choices": choices,
                      "auto": False, "origin": "cue_model_choices"})


# ---------------------------------------------------------------- 읽기
def _stage(fn, side, body, what):
    body = _dict({} if body is None else body, STAGE_KEYS, what)
    st = {"function": fn, "stage": side, "unit": "%s.%s" % (fn, side),
          "pin": _one(body.get("pin", "none"), PINS, what + ".pin"),
          "needs": _needs(body.get("needs"), what + ".needs"),
          "deadline_ms": _opt_num(body, "deadline_ms", what),
          "egress_kb": _opt_num(body, "egress_kb", what),
          "state": _one(body.get("state", "stateless"), STATES, what + ".state"),
          "optional": _bool(body.get("optional", False), what + ".optional"),
          # 소스(카메라)마다 하나인 단계인가. 적지 않으면 **카메라를 요구하는 기기 단계**가 그렇다 —
          # 영상 한 줄기마다 그 기기에서 도는 일이 따로 있기 때문이다. SLAM 처럼 기기에 하나뿐인 일은 false 로 적는다
          "per_source": (None if body.get("per_source") is None
                         else _bool(body["per_source"], what + ".per_source")),
          "note": str(body.get("note") or ""),
          "roles": sorted(_subset(body.get("roles"), ROLES, what + ".roles")),
          "placement": _one(body.get("placement", "any"), PLACEMENTS, what + ".placement"),
          "image": None if body.get("image") is None else str(body["image"]),
          "command": _cmd(body.get("command"), what + ".command"),
          "local_command": _cmd(body.get("local_command"), what + ".local_command"),
          "env": _env(body.get("env"), what + ".env"),
          "slots": _slots(body.get("slots"), what + ".slots")}
    for k in NUMERIC_KEYS:
        st[k] = _opt_num(body, k, what)
    if st["pin"] not in ("none", side):
        raise ValueError("%s.pin 은 %r 일 수 없다 (그 단계 쪽인 %r 이나 none)" % (what, st["pin"], side))
    return st


def _rules(what, fn):
    """패턴 · 중요도가 지키는 규칙. 어기면 ValueError — 계약은 읽을 때 거부한다"""
    p, st = fn["pattern"], fn["stages"]
    if p in ("cascade", "split") and set(st) != set(SIDES):
        raise ValueError("%s 는 %s 라서 기기 · 서버 단계가 모두 있어야 한다 (있는 것: %s)"
                         % (what, p, ", ".join(sorted(st)) or "없음"))
    if p == "cascade" and not st["device"]["optional"]:
        raise ValueError("%s 는 cascade 라서 기기 단계가 optional: true 여야 한다 (없으면 품질만 낮춘다)" % what)
    if p == "split" and any(s["optional"] for s in st.values()):
        raise ValueError("%s 는 split 라서 어느 단계도 optional 일 수 없다 (한쪽이 없으면 기능 거절)" % what)
    if p in ("resident", "device_only") and "device" not in st:
        raise ValueError("%s 는 %s 라서 기기 단계가 있어야 한다" % (what, p))
    if p == "device_only" and "server" in st and not st["server"]["optional"]:
        raise ValueError("%s 는 device_only 라서 서버 단계는 optional: true 여야 한다 (보조다)" % what)
    if p == "server_only":
        if "server" not in st:
            raise ValueError("%s 는 server_only 라서 서버 단계가 있어야 한다" % what)
        if "device" in st and not st["device"]["optional"]:
            raise ValueError("%s 는 server_only 라서 기기 단계는 optional: true 여야 한다" % what)
    if p == "resident" and st["device"]["pin"] != "device":
        raise ValueError("%s 는 resident 라서 기기 단계가 pin: device 여야 한다 (상주는 그 기기에 붙박이다)" % what)
    if fn["criticality"] == "safety" and ("device" not in st or st["device"]["pin"] != "device"):
        raise ValueError("%s 는 safety 라서 기기 단계가 있고 pin: device 여야 한다 (안전은 서버로 옮기지 않는다)" % what)
    if p in PLACEABLE:
        for side, s in sorted(st.items()):
            for k in ("cpu_m", "mem_mb"):
                if s[k] is None:
                    raise ValueError("%s.stages.%s 에 %s 가 없다 (배치 대상 패턴 %s 의 단계는 cpu_m · mem_mb 를 적는다)"
                                     % (what, side, k, p))


def _function(name, body, cue_choices=None):
    what = "functions.%s" % name
    if not NAME_RE.match(str(name)):
        raise ValueError("기능 이름 %r 은 영어 소문자 snake_case 여야 한다" % name)
    body = _dict(body, FUNCTION_KEYS, what)
    for k in FUNCTION_REQUIRED:
        if body.get(k) in (None, "", {}, []):
            raise ValueError("%s 에 %s 가 없다 (있어야 하는 칸: %s)" % (what, k, ", ".join(FUNCTION_REQUIRED)))
    fn = {"function": name,
          "pattern": _one(body["pattern"], PATTERNS, what + ".pattern"),
          "criticality": _one(body["criticality"], CRITICALITY, what + ".criticality"),
          "trigger": _one(body["trigger"], TRIGGERS, what + ".trigger"),
          "note": str(body["note"]),
          "forbid": sorted(_subset(body.get("forbid"), SIDES, what + ".forbid")),
          # 그 기능이 쓸 수 있는 비전 선택지(모델) — 런처가 「기능 → 모델」로 고르게 한다. 없으면 모델 없이 도는 기능이다
          # (추적처럼 트래커가 ultralytics 안에 있는 것). 이름은 레지스트리 options 의 열쇠다
          "options": [str(x) for x in (body.get("options") or [])],
          "legacy": False,
          "stages": {}}
    fn["priority_class"] = PRIORITY_CLASSES[fn["criticality"]]
    stages = _dict(body["stages"], SIDES, what + ".stages")
    for side in SIDES:
        if side in stages:
            fn["stages"][side] = _stage(name, side, stages[side], "%s.stages.%s" % (what, side))
    for side in fn["forbid"]:
        if side in fn["stages"]:
            raise ValueError("%s 는 %s 단계를 두면 안 된다 (forbid: %s)" % (what, side, ", ".join(fn["forbid"])))
    _rules(what, fn)
    # 옛 칸 → 역할 슬롯 (별칭). options 는 서버 단계의 model 슬롯이 된다 — 서버 단계가 없으면 기기 단계
    if fn["options"]:
        side = "server" if "server" in fn["stages"] else ("device" if "device" in fn["stages"] else None)
        if side is not None:
            _legacy_model_slot(fn["stages"][side], fn["options"])
    if name == CUE_SLOT_FUNCTION and "server" in fn["stages"]:
        _cue_slots(fn["stages"]["server"], cue_choices)
    for side, st in fn["stages"].items():
        roles = [x["role"] for x in st["slots"]]
        dup = sorted({r for r in roles if roles.count(r) > 1})
        if dup:
            raise ValueError("%s.stages.%s.slots 에 같은 역할이 둘: %s" % (what, side, ", ".join(dup)))
    for s in fn["stages"].values():                 # 단계가 혼자서도 읽히게 — 배치 쪽은 단계만 들고 간다
        s.update({"pattern": fn["pattern"], "criticality": fn["criticality"], "trigger": fn["trigger"],
                  "priority_class": fn["priority_class"]})
    return fn


def _from_cpu_task(name, body):
    """옛 cpu_tasks.<이름> → 단일 단계 기능 (별칭). 배치표 이름(unit)은 그대로 <이름> 이다"""
    what = "cpu_tasks.%s" % name
    stage = _stage(name, "device", body, what)
    stage["unit"] = name
    fn = {"function": name, "pattern": "device_only", "criticality": "best_effort", "trigger": "on_request",
          "note": stage["note"] or ("%s — 옛 cpu_tasks 작업" % name), "forbid": [], "options": [], "legacy": True,
          "priority_class": PRIORITY_CLASSES["best_effort"], "stages": {"device": stage}}
    _rules(what, fn)
    stage.update({"pattern": fn["pattern"], "criticality": fn["criticality"], "trigger": fn["trigger"],
                  "priority_class": fn["priority_class"]})
    return fn


def _registry(src=None):
    if isinstance(src, dict):
        return src
    path = REGISTRY if src is None else src
    with open(path, encoding="utf-8") as f:
        reg = yaml.safe_load(f) or {}
    if not isinstance(reg, dict):
        raise ValueError("레지스트리 %s 가 사전이 아니다" % path)
    return reg


def load(src=None):
    """레지스트리(경로 · 사전) → {기능: 기능 계약}. 이름순이다. 어긴 계약은 ValueError"""
    reg = _registry(src)
    funcs = {}
    cue_choices = reg.get("cue_model_choices") or {}
    for section, maker in (("functions", _function), ("cpu_tasks", _from_cpu_task)):
        body = reg.get(section) or {}
        if not isinstance(body, dict):
            raise ValueError("레지스트리 %s 절은 사전이어야 한다" % section)
        for name, item in sorted(body.items()):
            if name in funcs:                       # functions 에 같은 이름이 있으면 그것이 이긴다
                continue
            funcs[name] = (_function(name, item, cue_choices) if section == "functions"
                           else _from_cpu_task(name, item))
    return dict(sorted(funcs.items()))


# ---------------------------------------------------------------- 펼치기 · 판정
def expand(funcs, names=None, sources=None):
    """기능 이름들 → (배치_단계들, 상주들, 그밖).

    배치_단계들  device_only · cascade · split · server_only 의 단계 — 오케스트레이터가 놓는 것
    상주들       resident 의 단계 — 놓는 게 아니라 있는지 보고 없으면 알린다
    그밖         config · pipeline 의 단계 — 같다 (값을 만들어 두거나 받아서 적용하는 흐름)
    세 목록 모두 같은 모양의 단계 사전이고 기능 이름순 · 기기 → 서버 순이다. 모르는 이름은 ValueError

    sources (2026-11-26)
      소스 이름들을 주면 **기기 단계를 소스마다 하나씩** 만든다 — unit 은 `<기능>.device@<소스>` 이고
      단계에 `source` 가 붙는다. 카메라마다 그 기기에서 돌 일이 따로 있기 때문이다(카메라 3대면 셋).
      **서버 단계는 기능마다 하나다** — 여러 소스를 모아 보는 몫이라 소스 수만큼 늘지 않는다.
      names 가 사전({소스: [기능...]})이면 소스마다 고른 기능이 다른 것으로 본다."""
    per_source = None
    if isinstance(names, dict):
        per_source = {s: sorted(set(v or ())) for s, v in names.items()}
        picked = sorted({n for v in per_source.values() for n in v})
        sources = sorted(per_source)
    else:
        picked = sorted(funcs) if names is None else sorted(set(names))
    srcs = sorted(set(sources or ()))
    placeable, residents, others = [], [], []
    for n in picked:
        if n not in funcs:
            raise ValueError("모르는 기능 %r (가능: %s)" % (n, ", ".join(sorted(funcs))))
        fn = funcs[n]
        bucket = placeable if fn["pattern"] in PLACEABLE else (residents if fn["pattern"] in RESIDENT else others)
        mine = [s for s in srcs if per_source is None or n in per_source.get(s, ())] if srcs else []
        for side in SIDES:
            if side not in fn["stages"]:
                continue
            st = fn["stages"][side]
            # 소스마다 늘릴 것은 **그 카메라 영상을 보는 단계**뿐이다 — 모터 제어 · SLAM 처럼 기기 하나에
            # 하나인 일을 카메라 수만큼 늘리면 거짓이다 (needs.sensors 에 camera 가 있는 단계만 늘린다)
            per_src = st.get("per_source")
            if per_src is None:
                per_src = "camera" in ((st.get("needs") or {}).get("sensors") or ())
            if side == "device" and mine and per_src:
                for s in mine:                         # 소스마다 하나 — 그 카메라에서 도는 몫이다
                    one = dict(st)
                    one["source"] = s
                    one["unit"] = "%s@%s" % (st["unit"], s)
                    bucket.append(one)
            elif side == "device" and per_src and srcs and per_source is not None and not mine:
                continue                               # 그 기능을 고른 소스가 없다 — 기기 단계도 없다
            else:
                bucket.append(dict(st))
    return placeable, residents, others


def _pair_key(x):
    if isinstance(x, dict):
        return x["function"], x["stage"]
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return str(x[0]), str(x[1])
    raise ValueError("놓인 단계는 (기능, 단계) 쌍이나 단계 사전이어야 한다 (받은 것: %r)" % (x,))


def _unit_key(x):
    """놓인 것의 unit 이름 — 문자열이면 그대로, 단계 사전이면 그 unit. (기능, 단계) 쌍이면 None (쌍으로만 본다)"""
    if isinstance(x, dict):
        return x.get("unit")
    if isinstance(x, (tuple, list)):
        return None
    return str(x)


def pairs(stages, placed=None):
    """같은 기능의 단계 쌍 규칙 — 판정만 돌려준다. 스트림에 쓰지 않는다 (코드로 옮기는 건 오케스트레이터 일이다).

    stages  expand 가 낸 단계 목록 (그 기능이 **선언한** 단계 전부)
    placed  실제로 놓인 단계 — (기능, 단계) 쌍이나 단계 사전의 모음. None 이면 전부 놓였다고 본다
    → {기능: {pattern, criticality, verdict, placed, missing, missing_optional, reason}}

    규칙은 하나다: 없는 단계가 optional 이면 degraded(품질만 낮춘다), 아니면 rejected(기능 거절).
    그래서 cascade 는 기기 단계가 빠져도 degraded 고(기기 단계는 optional 이다 — load 가 확인한다),
    split 은 어느 쪽이 빠져도 rejected 다(어느 단계도 optional 일 수 없다)."""
    # 놓인 것은 unit 이름으로 본다 — 기기 단계가 소스마다 여럿이면 쌍(기능, 단계)만으로는 가릴 수 없다.
    # 옛 호출(쌍)도 받는다: 그 쌍의 단계는 모두 놓였다고 본다
    units = coarse = None
    declared = {st["unit"] for st in stages}
    if placed is not None:
        units = {u for u in (_unit_key(x) for x in placed) if u}
        coarse = {_pair_key(x) for x in placed if isinstance(x, (tuple, list)) and len(x) == 2}
        # 선언에 없는 이름은 거부한다 — 「reid」처럼 단계를 안 적은 것이 조용히 「안 놓임」이 되지 않게
        unknown = sorted(u for u in units if u not in declared)
        if unknown:
            raise ValueError("놓인 단계 %s 를 모른다 (선언된 것: %s)"
                             % (", ".join(unknown), ", ".join(sorted(declared))))
    out = {}
    for st in stages:
        g = out.setdefault(st["function"], {"pattern": st.get("pattern"), "criticality": st.get("criticality"),
                                            "placed": [], "missing": [], "missing_optional": [],
                                            "sources": [], "sources_missing": []})
        ok = placed is None or st["unit"] in units or (coarse and (st["function"], st["stage"]) in coarse)
        src = st.get("source")
        if src and src not in g["sources"]:
            g["sources"].append(src)
        if ok:
            if st["stage"] not in g["placed"]:
                g["placed"].append(st["stage"])
        else:
            if src:
                g["sources_missing"].append("%s(%s)" % (src, st["stage"]))
            if st["stage"] not in g["missing"]:
                g["missing"].append(st["stage"])
            if st["optional"] and st["stage"] not in g["missing_optional"]:
                g["missing_optional"].append(st["stage"])
    for g in out.values():
        both = [s for s in g["missing"] if s in g["placed"]]      # 어떤 소스는 놓이고 어떤 소스는 빠졌다
        for s in both:
            g["missing"].remove(s)
            if s in g["missing_optional"]:
                g["missing_optional"].remove(s)
        need = [s for s in g["missing"] if s not in g["missing_optional"]]
        if not g["placed"]:
            g["verdict"] = "rejected"
            g["reason"] = "놓인 단계가 없음 — 기능 거절 (%s)" % g["pattern"]
        elif need:
            g["verdict"] = "rejected"
            g["reason"] = "%s 단계가 없음 — 필수 단계라 기능 거절 (%s)" % (", ".join(need), g["pattern"])
        elif g["missing"]:
            g["verdict"] = "degraded"
            g["reason"] = "%s 단계가 없음 — optional 이라 품질만 낮춤 (%s)" % (", ".join(g["missing"]), g["pattern"])
        elif g["sources_missing"]:
            g["verdict"] = "degraded"
            g["reason"] = "일부 소스에서 빠짐: %s (%s)" % (", ".join(sorted(g["sources_missing"])), g["pattern"])
        else:
            g["verdict"] = "ok"
            g["reason"] = "선언한 단계가 모두 놓임 (%s)" % g["pattern"]
    return dict(sorted(out.items()))


def as_cpu_task(stage, measured=None):
    """단계 → placement.place_cpu 가 받는 작업 사전 (옛 cpu_tasks 와 같은 칸 이름 · 같은 뜻).

    cpu_m · mem_mb 가 없는 단계는 CPU 작업으로 낼 수 없다 — ValueError.
    measured: {노드: ms} 노드별 비교 측정 (기억에서 온다). placement: measured 인 작업은 이것이 없으면 옮기지 않는다"""
    for k in ("cpu_m", "mem_mb"):
        if stage.get(k) is None:
            raise ValueError("%s 단계에 %s 가 없어 CPU 작업으로 낼 수 없다" % (stage.get("unit"), k))
    task = {"task": stage["unit"], "cpu_m": stage["cpu_m"], "mem_mb": stage["mem_mb"],
            "roles": list(stage["roles"]), "placement": stage["placement"],
            "frame_kb": stage["frame_kb"], "fps": stage["fps"], "proc_ms": stage["proc_ms"],
            "image": stage["image"], "command": stage["command"], "local_command": stage["local_command"]}
    if stage.get("note"):
        task["note"] = stage["note"]
    if stage.get("env"):
        task["env"] = dict(stage["env"])
    if measured:
        task["measured"] = dict(measured)
    return task
