"""웹 런처 화면 상태(ui) → 원하는 상태(ds_desired/1) (core, 2026-11-25).

화면 상태는 WinForms 런처의 settings.json 과 **같은 모양**이다(그대로 옮겨 붙여도 된다):

    {"sources": [...], "robotCam": 1, "vision": false, "orchestrate": true, "models": ["yoloe"],
     "slice": false, "sliceU": "", "targets": null | [...], "direct": [{"name", "size", "colour"}],
     "distance": true, "cueMode": {대상: "default"|"custom"}, "cues": {대상: [근거 키]},
     "overlay": {모델: {바꾼 항목만}}, "features": {기능: 참/거짓},
     "assign": {기기: [모델...]},                      # 모델 매칭 탭 — 적지 않은 기기는 고른 모델 전부 (2026-11-26)
     "manual": false, "manualPlace": {작업: 노드}}     # 수동 배치 탭 — 자동이면 false

targets 가 null 이면 「대상을 건드리지 않음」 — 프로필에 대상을 넣지 않고 yaml 기본값을 쓴다(런처 TargetsTouched).
build_profile 은 tools/launcher/launcher.ps1 의 Build-Profile 을 한 줄씩 옮긴 것이다(profile_version 만 뺀다 —
감독자가 단다). 화면(static/app.js)은 같은 규칙으로 체크 상태 · 켜짐만 그린다. 프로필은 여기서만 만든다.
"""
import copy
import re

#: 런처 $Sources 와 같은 순서 (cam360_slice 는 슬라이스 켜기로만 붙는다)
SOURCES = ("cam360", "drone", "drone2", "robot1", "ep-001", "realsense", "a2", "pi11", "pi12", "pi13")
#: 소스 → 데이터 서버 포트 (server_stream_multi_source.SOURCES 와 같아야 한다 — 시험이 맞춰 본다)
PORTS = {"cam360": 10001, "drone": 10002, "robot1": 10003, "ep-001": 10004, "realsense": 10005,
         "cam360_slice": 10006, "a2": 10007, "pi11": 10008, "pi12": 10009, "pi13": 10011, "drone2": 10012}
ROBOT_CAMS = (1, 2, 3, 4, 5)
SLICE_V = -30
LABEL_ITEMS = (("name", "클래스명"), ("conf", "신뢰도"), ("id", "트랙 번호"), ("dist", "거리"), ("gps", "GPS"))
FLAG_ITEMS = (("count", "탐지 개수"), ("roi", "탐지 영역"), ("show_non_suspect", "판정 아닌 검출"),
              ("show_rejected", "관문에 떨어진 검출"), ("title", "제목 띠"), ("pose", "드론 위치 띠"))
NAME_RE = re.compile(r"^[a-z][a-z0-9 _\-]{0,40}$")
SIZE_RE = re.compile(r"^\d+(\.\d+)?x\d+(\.\d+)?x\d+(\.\d+)?$")


def default_ui(cat):
    """처음 여는 화면 — 런처의 '마지막 선택' 기본값과 같다"""
    return {"sources": list(SOURCES), "robotCam": 1, "vision": False, "models": ["yoloe"], "orchestrate": True,
            "slice": False, "sliceU": "", "targets": None, "direct": [], "distance": bool(cat.get("distance_default", True)),
            "cueMode": {}, "cues": {}, "overlay": {}, "features": {},
            "assign": {}, "manual": False, "manualPlace": {}}


def normalize_ui(ui, cat):
    out = default_ui(cat)
    for k, v in (ui or {}).items():
        if v is not None or k == "targets":
            out[k] = copy.deepcopy(v)
    for k in ("cueMode", "cues", "overlay", "features", "assign", "manualPlace"):
        if not isinstance(out.get(k), dict):
            out[k] = {}
    out["manual"] = bool(out.get("manual"))
    # 모델 매칭은 「그 기기에 올릴 모델 목록」 — 글자 하나를 적었으면 목록으로 고친다(화면이 잘못 보내도 받는다)
    out["assign"] = {str(s): ([ms] if isinstance(ms, str) else [str(m) for m in (ms or [])])
                     for s, ms in out["assign"].items()}
    out["manualPlace"] = {str(k): str(v) for k, v in out["manualPlace"].items() if str(v or "").strip()}
    if not isinstance(out.get("direct"), list):
        out["direct"] = []
    out["direct"] = [{"name": str(d.get("name") or ""), "size": str(d.get("size") or ""), "colour": str(d.get("colour") or "")}
                     for d in out["direct"] if isinstance(d, dict) and d.get("name")]
    return out


def ui_from_desired(d, cat):
    """ui 칸이 없는 원하는 상태(다른 도구가 쓴 것)에서 화면 상태를 되살린다 — 아는 칸만"""
    if isinstance(d.get("ui"), dict):
        return normalize_ui(d["ui"], cat)
    ui = default_ui(cat)
    if d.get("version"):
        s, sl, v = d.get("server") or {}, d.get("slice") or {}, d.get("vision") or {}
        man = v.get("manual") or {}
        ui.update(sources=list(s.get("sources") or []), robotCam=int(s.get("robot_cam") or 1),
                  slice=bool(sl.get("enabled")), sliceU="" if sl.get("u") in (None, "") else str(sl.get("u")),
                  vision=bool(v.get("enabled")), orchestrate=bool(v.get("orchestrate", True)),
                  models=list(v.get("models") or []),
                  assign={k: list(x) for k, x in (v.get("assign") or {}).items()},
                  manual=bool(man.get("enabled")), manualPlace=dict(man.get("place") or {}))
    return ui


# ------------------------------------------------------------ 런처 함수 옮김
def model_keys(cat):
    return [m["key"] for m in cat.get("models") or []]


def chosen_models(ui, cat):
    """Chosen-Models — 카탈로그 순서"""
    on = set(ui.get("models") or [])
    return [m for m in model_keys(cat) if m in on]


def depth_models(cat):
    """깊이 모델 = 오버레이 '드론 위치 띠'(pose)를 쓸 수 있는 모델 (cue_catalog.DEPTH_MODELS 와 같다)"""
    return [m for m, o in (cat.get("overlay") or {}).items() if o.get("pose") is True]


def distance_on(ui, cat):
    """거리 측정 체크 — 깊이 모델 파일이 없으면 꺼져 있다"""
    return bool(ui.get("distance")) and bool(cat.get("distance_available", True))


def _direct(ui, name):
    for d in ui.get("direct") or []:
        if d.get("name") == name:
            return d
    return None


def target_names(ui, cat):
    """Target-Names — 카탈로그 대상 다음에 직접 입력"""
    n = [t["name"] for t in cat.get("targets") or []]
    for d in ui.get("direct") or []:
        if d["name"] not in n:
            n.append(d["name"])
    return n


def checked_targets(ui, cat):
    """Checked-Targets — 건드리지 않았으면 카탈로그 기본 대상"""
    keep = ui.get("targets")
    if keep is None:
        keep = [t["name"] for t in cat.get("targets") or [] if t.get("default")]
    keep = set(keep)
    return [n for n in target_names(ui, cat) if n in keep]


def cue_list(ui, cat, name):
    """Cue-List — 이 대상에서 쓸 수 있는 근거 [{key, label, default, needs_distance}]"""
    d = _direct(ui, name)
    if d is None:
        for t in cat.get("targets") or []:
            if t["name"] == name:
                return list(t.get("cues") or [])
        return []
    dr = cat.get("direct") or {}
    out = [dict(c, default=True) for c in dr.get("always") or []]
    if d.get("size"):
        out += [dict(c, default=True) for c in dr.get("with_size") or []]
    if d.get("colour"):
        out += [dict(c, default=True) for c in dr.get("with_colour") or []]
    return out


def overlay_value(ui, cat, m, key):
    o = (ui.get("overlay") or {}).get(m) or {}
    if key in o:
        return o[key]
    return ((cat.get("overlay_defaults") or {}).get(m) or {}).get(key)


def overlay_allowed(ui, cat, m, key):
    """Overlay-Allowed — 고정 가능표는 카탈로그, 선택에 걸린 셋(yoloe 거리 · targeted 거리)은 여기서"""
    if ((cat.get("overlay") or {}).get(m) or {}).get(key) is not True:
        return False
    if m == "yoloe" and key in ("dist", "gps"):
        return any(x in depth_models(cat) for x in chosen_models(ui, cat))
    if m == "targeted" and key in ("dist", "gps"):
        return distance_on(ui, cat)
    return True


def build_profile(ui, cat):
    """Build-Profile (profile_version 은 넣지 않는다 — 감독자가 단다)"""
    ui = normalize_ui(ui, cat)
    chosen = chosen_models(ui, cat)
    checked = checked_targets(ui, cat)
    dist = distance_on(ui, cat)
    pm, tm = {}, {}
    if "targeted" in chosen and ui.get("targets") is not None:
        ts = []
        for n in checked:
            t = {"name": n}
            d = _direct(ui, n)
            if d is not None:
                if d.get("size"):
                    t["size_mm"] = [float(x) for x in d["size"].split("x")]
                if d.get("colour"):
                    t["colours"] = [d["colour"]]
            cues = (ui.get("cues") or {}).get(n)
            if (ui.get("cueMode") or {}).get(n) == "custom" and cues is not None:
                # 지금 쓸 수 있는 근거만 — 거리 측정을 끈 뒤 남은 '크기 거부' 같은 것은 뺀다(비전이 거절한다)
                ok = [c["key"] for c in cue_list(ui, cat, n) if not (c.get("needs_distance") and not dist)]
                t["cues"] = [c for c in cues if c in ok]
            ts.append(t)
        tm["targets"] = ts
        tm["distance"] = dist
    for m in chosen:
        o = (ui.get("overlay") or {}).get(m)
        if not o:
            continue
        ov = {}
        for k, v in o.items():
            if k == "label":
                ov["label"] = [[t for t in (v or []) if overlay_allowed(ui, cat, m, "name" if t == "lead" else t)]]
            elif k == "show_rejected" and "demo_box" not in checked:
                pass
            else:
                ov[k] = v
        if m == "targeted":
            tm["overlay"] = ov
        else:
            pm[m] = {"overlay": ov}
    if tm:
        pm["targeted"] = tm
    prof = {"models": pm}
    feat = {f["key"]: True for f in cat.get("features") or []
            if f.get("available") and (ui.get("features") or {}).get(f["key"])}
    if feat:
        prof["features"] = feat
    return prof


def validate_ui(ui, cat):
    """런처가 입력할 때 막던 것(대화 상자) — 웹에서는 저장 전에 한 번에 본다"""
    errs = []
    for d in ui.get("direct") or []:
        if not NAME_RE.match(d["name"]):
            errs.append("영문 이름으로 적으세요 (예: traffic cone): %s" % d["name"])
        if d.get("size") and not SIZE_RE.match(d["size"]):
            errs.append("크기는 가로x세로x높이 mm (예: 300x300x700): %s" % d["name"])
        if d.get("colour") and d["colour"] not in (cat.get("colours") or []):
            errs.append("모르는 색 %s: %s" % (d["colour"], d["name"]))
    try:
        if int(ui.get("robotCam") or 1) not in ROBOT_CAMS:
            errs.append("로봇 캠은 1~5 중 하나")
    except (TypeError, ValueError):
        errs.append("로봇 캠은 1~5 중 하나")
    bad = [s for s in ui.get("sources") or [] if s not in SOURCES]
    if bad:
        errs.append("모르는 소스: %s" % ", ".join(bad))
    if ui.get("vision"):
        srcs, a, chosen = vision_source_list(ui), assign_map(ui, cat), chosen_models(ui, cat)
        if srcs and chosen and not [s for s in srcs if a.get(s, chosen)]:
            errs.append("모델 매칭에서 모든 기기를 비웠습니다 — 하나 이상은 모델을 맞추세요.")
        if ui.get("manual") and not ui.get("orchestrate"):
            errs.append("수동 배치는 자원 배분을 켠 채로 씁니다.")
    return errs


def _slice_u(text):
    t = str(text or "").strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return t                                     # desired.validate 가 「u 는 숫자로」라고 알린다


def _int_or(v):
    try:
        return int(v or 1)
    except (TypeError, ValueError):
        return v


def vision_source_list(ui):
    """비전이 볼 소스 — desired.vision_sources 와 같은 규칙(서버 소스 + 슬라이스)"""
    on = set(ui.get("sources") or [])
    return [s for s in SOURCES if s in on] + sorted(on - set(SOURCES)) + (["cam360_slice"] if ui.get("slice") else [])


def assign_map(ui, cat):
    """모델 매칭 탭 → {기기: [모델...]}. **기본값(고른 모델 전부)은 적지 않는다** — 적어 두면 소스를 켜고 끌 때마다
    구성 요소 열쇠가 바뀌어 비전이 괜히 다시 뜬다. 지금 꺼진 기기 · 안 고른 모델은 버린다."""
    chosen = chosen_models(ui, cat)
    srcs = set(vision_source_list(ui))
    out = {}
    for s, ms in sorted((ui.get("assign") or {}).items()):
        if s not in srcs:
            continue
        keep = [m for m in chosen if m in set(ms)]
        if keep != chosen:                            # 전부와 같으면 기본값이다 — 적지 않는다
            out[s] = keep
    return out


def to_desired(ui, running, cat, prev=None):
    """화면 상태 → 원하는 상태 문서(version 은 desired.write 가 단다). prev 의 vision.max_fps 는 이어받는다."""
    ui = normalize_ui(ui, cat)
    on = set(ui.get("sources") or [])
    prev = prev or {}
    return {"schema": "ds_desired/1", "running": bool(running),
            "server": {"sources": [s for s in SOURCES if s in on] + sorted(on - set(SOURCES)),
                       "robot_cam": _int_or(ui.get("robotCam"))},
            "slice": {"enabled": bool(ui.get("slice")), "u": _slice_u(ui.get("sliceU")), "v": SLICE_V},
            "vision": {"enabled": bool(ui.get("vision")), "orchestrate": bool(ui.get("orchestrate")),
                       "models": chosen_models(ui, cat), "profile": build_profile(ui, cat),
                       "max_fps": copy.deepcopy(((prev.get("vision") or {}).get("max_fps")) or {}),
                       "assign": assign_map(ui, cat),
                       "manual": {"enabled": bool(ui.get("manual")),
                                  "place": {k: v for k, v in sorted((ui.get("manualPlace") or {}).items()) if v}}},
            "ui": ui}
