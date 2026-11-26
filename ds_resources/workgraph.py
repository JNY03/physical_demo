"""작업 그래프 — 기능 사이 중복 작업 조율 (work/1, core, 2026-11-27).

표준 라이브러리만 쓴다(functions.py · slots.py 와 같은 층 — 그 둘이 읽어 둔 것을 받는다). torch · vision 을 모른다.

왜 있나: 한 소스에 기능 여럿(지정 물체 탐지의 거리 근거 + 거리 추정 + 재식별 …)을 고르면 **같은 일**(같은 소스 ·
같은 모델 · 같은 입력 크기)을 기능마다 따로 돌리게 된다. 이 모듈이 소스마다 작업 그래프를 만들어 같은 일을 한 번만
세고(노드 하나), 그 결과를 쓰는 기능들을 노드의 consumers 로 묶는다.

  노드   (소스, 단계 쪽, 모델 정체, 매개변수 해시) — 정체는 선택지가 쓰는 모델 하나로 푼다(unidepth → unidepth_l).
         매개변수 해시 = 레지스트리 models.<정체>.config (imgsz 등)의 해시. 같으면 같은 일이다
  간선   consumes → produces. 기능의 consumes 는 그 소스에 함께 고른 다른 기능의 주 노드가 낸다(producer 고르기):
           1) 기능 매개변수(kind: function, 예 object_track.input)로 사람이 고른 기능
           2) 없으면 우선순위가 가장 높은 기능 (criticality safety → mission → best_effort, 같으면 이름순)
         target_detections 는 detections 를 바라는 쪽도 받는다(판정된 검출도 검출이다)
         단계의 둘째 슬롯(targeted 의 depth · ovd 처럼 기능 안에서 쓰는 모델)은 그 기능의 주 노드로 가는 내부 간선이다
  fps    노드 fps = 그 노드를 쓰는 기능들의 fps 중 가장 큰 것 (기기 단계는 단계 fps, 서버는 소스 목표 fps)
  GPU    노드마다 한 번만 센다 (gpu_mb). naive = 기능마다 따로 셌을 때, saved = 그 차이
  경고   dup_kind — 한 소스 · 한 쪽에서 같은 종류의 일(depth · detections …)을 **다른 모델**로 두 번 한다.
         둘 다 돌리되 「같은 일을 다른 모델로 두 번 함 — 하나로 맞추면 GPU n MB 절약」을 남긴다 (n = 합 − 가장 작은 것)
         no_producer — consumes 를 낼 기능이 그 소스에 없다 · param_fallback — 고른 입력 기능이 그 소스에 없어 기본으로

  build(reg, funcs, functions, sources, selection, params, fps, active, units) → {"work": [...], "work_warnings": [...],
        "work_inputs": {...}, "work_reid": {...}, "work_gpu": {...}}   (계획 plan 에 그대로 얹는다 — 칸 추가만)
  check_params(funcs, params, functions, sources) → 고르게 편 function_params (틀리면 ValueError)
  params_schema(funcs, functions) → places --json 의 function_params (런처가 그릴 것)
  artifacts(funcs) → places --json 의 artifacts {기능: {produces, consumes}}
  shared_need(reg, models, need) → 비전 프로세스 하나의 GPU 어림에서 같은 정체를 두 번 세지 않게 뺄 근거 모델

재식별 범위 (reid · multi_cam_track)
  scope: source  그 소스 트래커 옆 갤러리 — 기본 매칭은 features.reid (SuperPoint+LightGlue) 그대로
  scope: group   group_sources(사람이 고른 소스 묶음, 기본: 그 기능을 켠 소스 전부)의 소스마다 임베딩 →
                 서버 갤러리 하나(vision/reid_gallery.py)가 묶음 전체에 전역 번호(identities)를 붙인다.
                 묶음은 소스 2개 이상이어야 하고 그 소스 모두에 그 기능이 켜져 있어야 한다 (자동 묶음 없음)
"""
import hashlib
import json

FORMAT = "work/1"
RANKS = {"safety": 0, "mission": 1, "best_effort": 2}
#: 부품 종류 → 그 부품이 내는 결과물 (functions.ARTIFACTS 이름)
KIND_ARTIFACTS = {"detect": ["detections"], "segment": ["detections", "masks"], "depth": ["depth"],
                  "track": ["tracks"], "embed": ["embeddings"], "vlm": ["scene_text"]}
#: 바라는 결과물 → 받아 줄 수 있는 결과물
SATISFIES = {"detections": ("detections", "target_detections")}
GALLERY_MODEL = "reid_gallery"
#: 재식별 결과 Redis 키 (vision/reid_gallery.py 와 같게)
IDENTITY_KEY = "reid_identity:%s"
EMBED_KEY = "reid_embed:%s"
DUP_TEXT = "같은 일을 다른 모델로 두 번 함 — 하나로 맞추면 GPU %d MB 절약"


def _hash(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:8]


def _rank(g):
    return RANKS.get(g.get("criticality"), 2)


def _per_source(functions, sources):
    """{소스: [기능...]} — 목록이면 모든 소스에 같은 기능"""
    if isinstance(functions, dict):
        return {s: sorted(set(v or ())) for s, v in sorted(functions.items())}
    return {s: sorted(set(functions or ())) for s in sorted(sources or ())}


# ---------------------------------------------------------------- 정체 · 매개변수
def identity(key, table):
    """부품 키 → 모델 정체. 선택지(파이프라인, 도메인 아님)가 모델 하나만 쓰면 그 모델이다 (unidepth → unidepth_l)"""
    p = table.get(key) or {}
    uses = list(p.get("uses") or [])
    if p.get("pipeline") and not p.get("domain") and len(uses) == 1:
        return uses[0]
    return key


def params_hash(key, reg):
    """그 정체의 실행 매개변수 해시 — 레지스트리 models.<키>.config (입력 크기 imgsz 등). 없으면 빈 사전"""
    cfg = ((reg.get("models") or {}).get(key) or {}).get("config") or {}
    return _hash(cfg)


# ---------------------------------------------------------------- 매개변수 (function_params)
def check_params(funcs, params, functions=None, sources=None):
    """프로필 최상위 function_params → {기능: {이름: 값}} (적지 않은 값은 기본). 틀리면 ValueError.

    functions(고른 기능 — 목록 · {소스: [...]})를 주면 소스 묶음을 그것으로 검사한다
    (묶음 소스 ≥ min, 묶음의 소스 모두에 그 기능이 켜져 있다). 주지 않으면 모양만 본다"""
    params = params or {}
    if not isinstance(params, dict):
        raise ValueError("function_params 는 {기능: {이름: 값}} 사전이어야 한다")
    per = _per_source(functions, sources) if functions is not None else None
    out = {}
    for fn, vals in params.items():
        if fn not in funcs:
            raise ValueError("function_params 에 모르는 기능 %r" % fn)
        if not isinstance(vals, dict):
            raise ValueError("function_params.%s 는 {이름: 값} 사전" % fn)
        decl = funcs[fn].get("params") or {}
        for k in vals:
            if k not in decl:
                raise ValueError("function_params.%s.%s — 그런 매개변수가 없다 (있는 것: %s)"
                                 % (fn, k, ", ".join(decl) or "없음"))
    for fn, g in funcs.items():
        decl = g.get("params") or {}
        if not decl:
            continue
        vals = params.get(fn) or {}
        on = sorted(s for s, fs in (per or {}).items() if fn in fs)
        if per is not None and not on and fn not in params:
            continue
        got = {}
        for k, d in decl.items():
            what = "function_params.%s.%s" % (fn, k)
            v = vals.get(k)
            if d["kind"] == "choice":
                v = d["default"] if v is None else str(v)
                if v not in d["choices"]:
                    raise ValueError("%s 는 %s 중 하나 (받은 것: %r)" % (what, ", ".join(d["choices"]), v))
            elif d["kind"] in ("function", "model"):
                v = None if v in (None, "") else str(v)
                if d["kind"] == "function" and v is not None and v not in funcs:
                    raise ValueError("%s 에 모르는 기능 %r" % (what, v))
            elif d["kind"] == "sources":
                if v is not None and (isinstance(v, (str, bytes)) or not isinstance(v, (list, tuple))):
                    raise ValueError("%s 는 소스 목록" % what)
                v = None if v is None else sorted(dict.fromkeys(str(x) for x in v))
            got[k] = v
        # 소스 묶음 — when 이 맞을 때만 따진다 (reid scope: group)
        for k, d in decl.items():
            if d["kind"] != "sources":
                continue
            if any(got.get(wk) != wv for wk, wv in (d.get("when") or {}).items()):
                got[k] = None
                continue
            what = "function_params.%s.%s" % (fn, k)
            if per is not None:
                if got[k] is None:
                    got[k] = list(on)                      # 기본: 그 기능을 켠 소스 전부
                off = [s for s in got[k] if s not in on]
                if off:
                    raise ValueError("%s 의 %s 에는 %s 기능이 켜져 있지 않다 (켠 소스: %s)"
                                     % (what, ", ".join(off), fn, ", ".join(on) or "없음"))
            if got[k] is not None and len(got[k]) < d["min"]:
                raise ValueError("%s 는 소스가 %d개 이상이어야 한다 (받은 것: %s)"
                                 % (what, d["min"], ", ".join(got[k]) or "없음"))
        out[fn] = got
    return out


def params_schema(funcs, functions=None, sources=None):
    """places --json 의 function_params — {기능: {이름: {kind, label, choices, default, labels, min …}}}.
    kind function 의 choices 는 고른 기능 중 그 결과물을 내는 것(고른 것이 없으면 그 결과물을 내는 모든 기능),
    default 는 그중 우선순위가 가장 높은 것"""
    picked = None
    if functions is not None:
        picked = sorted({f for fs in _per_source(functions, sources).values() for f in fs})
    out = {}
    for fn, g in sorted(funcs.items()):
        decl = g.get("params") or {}
        if not decl:
            continue
        rows = {}
        for k, d in decl.items():
            r = {"kind": d["kind"], "label": d["label"], "label_en": d["label_en"]}
            if d["kind"] == "choice":
                r.update(choices=list(d["choices"]), default=d["default"], labels=dict(d["labels"]),
                         labels_en=dict(d["labels_en"]))
            elif d["kind"] == "function":
                pool = [f for f in (picked if picked else sorted(funcs)) if f != fn and f in funcs
                        and _gives(funcs[f], d["artifact"])]
                pool.sort(key=lambda f: (_rank(funcs[f]), f))
                r.update(artifact=d["artifact"], choices=pool, default=pool[0] if pool else None)
            elif d["kind"] == "sources":
                r.update(min=d["min"], when=dict(d["when"]))
            elif d["kind"] == "model":
                r.update(slot=d["slot"])
            rows[k] = r
        out[fn] = rows
    return out


def artifacts(funcs):
    """{기능: {produces, consumes}} — 결과물을 하나라도 적은 기능만 (단계에 적은 것도 합친다)"""
    out = {}
    for fn, g in sorted(funcs.items()):
        prod = list(g.get("produces") or [])
        cons = list(g.get("consumes") or [])
        for st in g["stages"].values():
            prod += [a for a in st.get("produces") or [] if a not in prod]
            cons += [a for a in st.get("consumes") or [] if a not in cons]
        if prod or cons:
            out[fn] = {"produces": prod, "consumes": cons}
    return out


def _gives(g, art):
    ok = SATISFIES.get(art, (art,))
    prods = list(g.get("produces") or []) + [a for st in g["stages"].values() for a in st.get("produces") or []]
    return any(a in ok for a in prods)


# ---------------------------------------------------------------- 그래프
def _picks(v):
    return [str(x) for x in (v if isinstance(v, list) else ([] if v is None else [v]))]


def build(reg, funcs, functions, sources=None, selection=None, params=None, fps=None, active=None, units=None,
          table=None):
    """작업 그래프 → 계획에 얹을 칸들 (work · work_warnings · work_inputs · work_reid · work_gpu).

    reg        레지스트리 사전           funcs      functions.load 결과
    functions  고른 기능 (목록 | {소스: [기능]})          sources  소스 이름들 (목록일 때)
    selection  slots.effective 결과 {기능: {단계: {역할: 값}}}
    params     check_params 결과 {기능: {이름: 값}}
    fps        {소스: 목표 fps}           active  {기능: {모델 키}} 그 기능 안에서 실제로 적재할 근거 모델
               (targeted 의 depth · ovd 처럼 대상 · 거리 설정에 따라 적재 여부가 갈리는 둘째 슬롯). 없으면 모두 적재로 본다
    units      {(소스, 쪽, 기능): 단위 이름} — 계획의 GPU 단위 · CPU 작업 이름 (없으면 단계 unit)"""
    from ds_resources import slots as slt
    table = slt.parts(reg) if table is None else table
    per = _per_source(functions, sources)
    selection = selection or {}
    params = params or {}
    fps = fps or {}
    units = units or {}
    nodes, order = {}, []
    primary = {}                       # (소스, 기능) → [노드 id] (그 기능의 주 노드 — 다른 기능이 consumes 로 받는다)
    edges, warnings, inputs = [], [], {}

    def unit_of(s, side, fn):
        st = funcs[fn]["stages"].get(side) or {}
        default = "%s@%s" % (st.get("unit"), s) if side == "device" else st.get("unit")
        return units.get((s, side, fn)) or units.get((s, side, None)) or default

    def add(s, side, fn, role, key, arts, internal, stage):
        ident = identity(key, table)
        nid = "%s|%s|%s|%s" % (s, side, ident, params_hash(ident, reg))
        n = nodes.get(nid)
        part = table.get(key) or {}
        mb = int(part.get("gpu_mb") or 0) if (side == "server" and "gpu" in (part.get("runs_on") or ())) else 0
        f = (stage.get("fps") if side == "device" else None) or fps.get(s)
        if n is None:
            n = nodes[nid] = {"source": s, "node": nid, "artifact": arts[0] if arts else None, "artifacts": [],
                              "model": key, "identity": ident, "models": [], "kind": part.get("kind"),
                              "domain": bool(part.get("domain")), "stage": side,
                              "params_hash": nid.rsplit("|", 1)[1], "fps": None, "consumers": [], "uses": [],
                              "shared": False, "internal": True, "gpu_mb": 0,
                              "unit": unit_of(s, side, fn)}
            order.append(nid)
        for a in arts:
            if a not in n["artifacts"]:
                n["artifacts"].append(a)
        if key not in n["models"]:
            n["models"].append(key)
        if fn not in n["consumers"]:
            n["consumers"].append(fn)
        n["uses"].append({"function": fn, "role": role, "stage": side, "gpu_mb": mb})
        n["gpu_mb"] = max(n["gpu_mb"], mb)
        n["internal"] = n["internal"] and internal
        if f is not None:
            n["fps"] = max(float(f), n["fps"] or 0.0)
        return nid

    for s, fns in per.items():
        for fn in sorted(fns, key=lambda f: (_rank(funcs.get(f) or {}), f)):
            g = funcs.get(fn)
            if g is None:
                continue
            for side in ("device", "server"):
                st = g["stages"].get(side)
                if not st:
                    continue
                if side == "device":
                    per_src = st.get("per_source")
                    if per_src is None:
                        per_src = "camera" in ((st.get("needs") or {}).get("sensors") or ())
                    if not per_src:
                        continue
                sel = (selection.get(fn) or {}).get(side) or {}
                slots = st.get("slots") or []
                main = None
                for i, sl in enumerate(slots):
                    v = sel.get(sl["role"], sl["default"])
                    picks = _picks(v)
                    if not picks:
                        continue
                    first = i == 0
                    if not first and active is not None and fn in active:
                        picks = [k for k in picks if k in active[fn] or identity(k, table) in active[fn]]
                    for k in picks:
                        part = table.get(k) or {}
                        if first:
                            arts = list(st.get("produces") or g.get("produces") or KIND_ARTIFACTS.get(part.get("kind"), []))
                        else:
                            arts = list(KIND_ARTIFACTS.get(part.get("kind"), []))
                        nid = add(s, side, fn, sl["role"], k, arts, not first, st)
                        if first:
                            main = main or nid
                            primary.setdefault((s, fn), []).append(nid)
                        elif main:
                            edges.append({"source": s, "from": nid, "to": main, "artifact": arts[0] if arts else None,
                                          "function": fn, "how": "internal"})
    # consumes — 그 소스에 함께 고른 다른 기능의 주 노드에서
    for s, fns in per.items():
        for fn in fns:
            g = funcs.get(fn)
            if g is None:
                continue
            for art in g.get("consumes") or []:
                ok = SATISFIES.get(art, (art,))
                cands = sorted({f for f in fns if f != fn and f in funcs and _gives(funcs[f], art)
                                and primary.get((s, f))}, key=lambda f: (_rank(funcs[f]), f))
                pname = next((k for k, d in (g.get("params") or {}).items()
                              if d["kind"] == "function" and d.get("artifact") in (art,) + tuple(ok)), None)
                want = (params.get(fn) or {}).get(pname) if pname else None
                how, chosen = "default", (cands[0] if cands else None)
                if want:
                    if want in cands:
                        how, chosen = "param", want
                    else:
                        warnings.append({"source": s, "artifact": art, "functions": [fn, want], "models": [],
                                         "save_mb": 0, "code": "param_fallback",
                                         "reason": "%s 의 입력으로 고른 %s 가 %s 에 없다 — %s 로" % (
                                             fn, want, s, chosen or "입력 없음")})
                row = {"param": pname, "artifact": art, "function": chosen, "how": how if chosen else "none",
                       "choices": cands, "node": None}
                if chosen is None:
                    warnings.append({"source": s, "artifact": art, "functions": [fn], "models": [], "save_mb": 0,
                                     "code": "no_producer",
                                     "reason": "%s 가 받을 %s 를 내는 기능이 %s 에 없다" % (fn, art, s)})
                else:
                    src_nodes = [n for n in primary[(s, chosen)]
                                 if set(nodes[n]["artifacts"]) & set(ok)] or primary[(s, chosen)]
                    # 서버 쪽 결과를 먼저 (기기 단계는 cascade 의 1차 거름이다)
                    src_nodes.sort(key=lambda n: nodes[n]["stage"] != "server")
                    row["node"] = src_nodes[0]
                    to = (primary.get((s, fn)) or [fn])[0]
                    edges.append({"source": s, "from": src_nodes[0], "to": to, "artifact": art, "function": fn,
                                  "how": how})
                    if fn not in nodes[src_nodes[0]]["consumers"]:
                        nodes[src_nodes[0]]["consumers"].append(fn)
                        nodes[src_nodes[0]]["uses"].append({"function": fn, "role": "input", "stage": "server",
                                                             "gpu_mb": 0})
                    nodes[src_nodes[0]]["internal"] = False
                    to_fps = nodes[to]["fps"] if to in nodes else fps.get(s)
                    if to_fps is not None:
                        nodes[src_nodes[0]]["fps"] = max(float(to_fps), nodes[src_nodes[0]]["fps"] or 0.0)
                inputs.setdefault(fn, {}).setdefault(s, []).append(row)
    # 재식별 범위 — scope group 이면 묶음 갤러리 노드 하나
    reid = {}
    for fn in sorted({f for fs in per.values() for f in fs}):
        g = funcs.get(fn) or {}
        decl = g.get("params") or {}
        if not any(d["kind"] == "sources" for d in decl.values()):
            continue
        pv = params.get(fn) or {}
        scope = pv.get("scope") or ("group" if "scope" not in decl else (decl["scope"]["default"]))
        on = sorted(s for s, fs in per.items() if fn in fs)
        grp = pv.get("group_sources") or (on if scope == "group" else None)
        emb = ((selection.get(fn) or {}).get("server") or {}).get("embedder")
        dev_emb = ((selection.get(fn) or {}).get("device") or {}).get("embedder")
        row = {"scope": scope, "group_sources": grp, "embedder": {"device": dev_emb, "server": emb}}
        if scope == "group" and grp:
            gid = "+".join(grp)
            nid = "group:%s|server|%s|%s" % (gid, GALLERY_MODEL, _hash(grp))
            nodes[nid] = {"source": None, "sources": list(grp), "node": nid, "artifact": "identities",
                          "artifacts": ["identities"], "model": GALLERY_MODEL, "identity": GALLERY_MODEL,
                          "models": [GALLERY_MODEL], "kind": None, "domain": False, "stage": "server",
                          "params_hash": nid.rsplit("|", 1)[1],
                          "fps": max([fps.get(s) or 0.0 for s in grp] or [0.0]) or None, "consumers": [fn],
                          "uses": [{"function": fn, "role": "gallery", "stage": "server", "gpu_mb": 0}],
                          "shared": False, "internal": False, "gpu_mb": 0,
                          "unit": unit_of(None, "server", fn) if "server" in g["stages"] else fn}
            order.append(nid)
            for s in grp:
                edges.append({"source": s, "from": "%s:embeddings" % s, "to": nid, "artifact": "embeddings",
                              "function": fn, "how": "group"})
            row.update(group=gid, gallery={"node": nid, "where": "server", "redis_in": EMBED_KEY % "<소스>",
                                           "redis_out": IDENTITY_KEY % gid})
        else:
            row.update(gallery={"where": "per_source", "matcher": "features.reid (SuperPoint+LightGlue)",
                                "redis_out": [IDENTITY_KEY % s for s in on]})
        reid[fn] = row
    # 공유 · 경고 · GPU
    out = []
    for nid in order:
        n = nodes[nid]
        n["shared"] = len(n["consumers"]) > 1
        out.append(n)
    naive = sum(u["gpu_mb"] for n in out for u in n["uses"])
    total = sum(n["gpu_mb"] for n in out)
    groups = {}
    for n in out:
        if n["source"] is None or n["domain"]:
            continue
        for a in KIND_ARTIFACTS.get(n["kind"], []):
            # 도메인 기능 안의 구성 모델(targeted 의 텍스트 · prompt-free arm · OVD)은 「같은 일」로 보지 않는다 —
            # 그 기능만의 문구 · 판정을 위해 돈다. 거리(depth)만은 같은 일이다(같은 물체 거리)
            if n["internal"] and a != "depth":
                continue
            groups.setdefault((n["source"], n["stage"], a), []).append(n)
    for (s, side, a), ns in sorted(groups.items()):
        if len(ns) < 2 or a == "masks":
            continue
        mbs = [n["gpu_mb"] for n in ns]
        save = sum(mbs) - min(mbs)
        fns = sorted({f for n in ns for f in n["consumers"]})
        # 맞추기 — 한 모델로 모을 때 기능마다 그 슬롯에 넣을 **그 슬롯의 선택지 키** (같은 정체의 다른 이름을 푼다:
        # depth_estimate 의 unidepth ↔ targeted 의 unidepth_l). 모든 기능 슬롯에 넣을 수 있는 것 중 GPU 가 작은 것이 먼저
        opts = []
        for c in ns:
            align = _align(c, ns, funcs, table)
            if align is not None:
                opts.append((c["gpu_mb"], c["model"], c, align))
        opts.sort(key=lambda o: (o[0], o[1]))
        first = opts[0][2] if opts else None
        order = ([first] if first else []) + [n for n in ns if n is not first]
        row = {"source": s, "stage": side, "artifact": a, "models": [n["model"] for n in order],
               "functions": fns, "nodes": [n["node"] for n in order], "save_mb": int(save), "code": "dup_kind",
               "reason": DUP_TEXT % save}
        if opts:                                         # 실제로 맞출 수 있는 쪽으로 모을 때 아끼는 몫
            row["align"] = opts[0][3]
            row["save_mb"] = int(sum(mbs) - opts[0][0])
            row["reason"] = DUP_TEXT % row["save_mb"]
        warnings.append(row)
    return {"work": out, "work_edges": edges, "work_warnings": warnings, "work_inputs": inputs, "work_reid": reid,
            "work_gpu": {"format": FORMAT, "total_mb": int(total), "naive_mb": int(naive),
                         "saved_mb": int(naive - total)}}


def apply_model_params(funcs, selection, picks, params):
    """kind: model 매개변수(object_track.tracker) → 서버 단계 슬롯. **프로필 functions 에서 그 슬롯을 고르지 않았을 때만**
    (슬롯이 이긴다). selection 을 고쳐 돌려준다"""
    for fn, vals in (params or {}).items():
        for k, d in ((funcs.get(fn) or {}).get("params") or {}).items():
            v = (vals or {}).get(k)
            if d["kind"] != "model" or not v:
                continue
            if d["slot"] in (((picks or {}).get(fn) or {}).get("server") or {}):
                continue
            if fn in selection and "server" in selection[fn]:
                selection[fn]["server"][d["slot"]] = v
    return selection


def _align(cand, ns, funcs, table):
    """cand 의 정체로 모을 때 {기능: {stage, role, model}} — 넣을 수 없는 기능이 하나라도 있으면 None"""
    from ds_resources import slots as slt
    ident = cand["identity"]
    out = {}
    for n in ns:
        for u in n["uses"]:
            if u["role"] == "input":
                continue
            st = (funcs.get(u["function"]) or {}).get("stages", {}).get(u["stage"]) or {}
            sl = next((x for x in st.get("slots") or [] if x["role"] == u["role"]), None)
            if sl is None:
                return None
            if n is cand:
                key = next((k for k in n["models"] if k in slt.choices(sl, u["stage"], st, table)), n["model"])
            else:
                key = next((k for k in slt.choices(sl, u["stage"], st, table) if identity(k, table) == ident), None)
            if key is None:
                return None
            out[u["function"]] = {"stage": u["stage"], "role": u["role"], "model": key}
    return out


def shared_need(reg, models, need, table=None):
    """비전 프로세스 하나의 GPU 어림에서 **같은 정체를 두 번 세지 않게** 뺄 근거 모델 — 같은 묶음의 선택지 · 새 모델이
    이미 그 정체를 적재한다. 예: unidepth 선택지(정체 unidepth_l) + targeted 거리 모델 unidepth_l → 근거 쪽 unidepth_l 을 뺀다.
    겹치지 않으면 빈 집합 — 옛 어림은 그대로다"""
    from ds_resources import slots as slt
    table = slt.parts(reg) if table is None else table
    have = {identity(m, table) for m in models if m != "targeted"}
    return {n for n in need if identity(n, table) in have}
