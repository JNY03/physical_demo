"""물체 추적 기능 (object_track) — 비전 프로세스 안의 한 단계 (work/1, 2026-11-27).

고른 탐지 기능(입력)의 검출을 **그 장의 FrameWork 에서 받아**(탐지를 다시 돌리지 않는다) 트래커(botsort ·
bytetrack — vision/runners)로 번호를 붙이고, 번호마다 궤적을 쌓아 Redis vision_stream:<소스>:track 에 낸다.

프로필에서 켠다 (런처가 쓴다):
  functions.object_track.server.tracker   트래커 모델 (슬롯 — 이것이 이긴다)
  function_params.object_track            {"input": "<탐지 기능 키>", "tracker": "<모델 키>"}
둘 중 하나라도 있으면 켜진다. input 이 없으면 이 프로세스에서 도는 탐지 모델을 낸 기능 중 우선순위가 가장 높은 것
(중요도 → 이름순 — ds_resources/workgraph.py 의 기본 규칙과 같다).

기록 (vision_stream:<소스>:track 의 result — 다른 모델 기록과 같은 v2 머리, 칸 추가만):
  model "track" · function "object_track" · tracker · input {function, model} · infer_ms(트래커 ms)
  detections  입력 검출 사본에 track_id (번호 못 받은 것도 on_untracked: keep 이면 그대로)
  tracks      [{track_id, label, confidence, bbox, hits, first_ts, last_ts, trajectory: [[ts, cx, cy], ...]}]
              이 장에 보인 번호만. 궤적은 최근 TRAIL 개 (중심점, 원본 px)
"""
from __future__ import annotations

import time
from collections import deque

TRAIL = 30
FORGET_S = 10.0
#: 판정형 탐지(검출 중 판정된 것만 따라간다) — vision_infer.count_dets 와 같은 기준
JUDGED = ("river", "targeted")


def _registry():
    from ds_resources import functions as fnc
    return fnc._registry(None)


def function_model(fn, picks, reg=None):
    """탐지 기능 → 그 기능의 서버 주 모델 키 (프로필 슬롯이 이기고 없으면 기본값)"""
    from ds_resources import functions as fnc
    from ds_resources import slots as slt
    reg = reg or _registry()
    funcs = fnc.load(reg)
    if fn not in funcs:
        return None
    sel = slt.effective(reg, funcs, slt.check_picks(reg, funcs, picks or {}), {fn})
    st = funcs[fn]["stages"].get("server") or {}
    slots = st.get("slots") or []
    if not slots:
        return None
    v = ((sel.get(fn) or {}).get("server") or {}).get(slots[0]["role"])
    return v[0] if isinstance(v, list) and v else v


def resolve(profile, running, reg=None):
    """프로필 → (트래커 키, 입력 기능, 입력 모델) 또는 None (추적이 꺼져 있다). running = 이 프로세스의 탐지 모델"""
    profile = profile or {}
    picks = profile.get("functions") or {}
    params = (profile.get("function_params") or {}).get("object_track")
    slot = ((picks.get("object_track") or {}).get("server") or {}).get("tracker")
    if params is None and "object_track" not in picks:
        return None
    params = params or {}
    reg = reg or _registry()
    tracker = slot or params.get("tracker") or "botsort"
    fn = params.get("input")
    model = function_model(fn, picks, reg) if fn else None
    if fn and model not in running:
        model = None                                    # 고른 입력이 이 프로세스에 없다 — 기본으로
    if model is None:
        from ds_resources import functions as fnc
        from ds_resources import workgraph as wg
        funcs = fnc.load(reg)
        cands = []
        for f, g in funcs.items():
            if f == "object_track" or not wg._gives(g, "detections"):
                continue
            m = function_model(f, picks, reg)
            if m in running:
                cands.append((wg._rank(g), f, m))
        if not cands:
            return {"tracker": tracker, "function": fn, "model": None}
        _r, fn, model = min(cands)
    return {"tracker": tracker, "function": fn, "model": model}


class ObjectTrackStep:
    """한 프로세스에 하나 — 소스마다 트래커 스트림 · 궤적을 따로 둔다"""

    def __init__(self, tracker_key, runner, input_function, input_model, min_conf=None):
        self.key, self.runner = tracker_key, runner
        self.function, self.model = input_function, input_model
        self.min_conf = min_conf
        self.trails = {}                                 # (소스, track_id) → {deque, hits, first, last, label, conf}
        self.reid = None                                 # ReidStep (재식별 범위를 켰을 때)

    def _pick(self, dets):
        out = []
        for d in dets or []:
            if self.model in JUDGED:
                name = (d.get("assess") or {}).get("suspect")
                if not name:
                    continue
            else:
                if self.min_conf is not None and d.get("conf", 0) < float(self.min_conf):
                    continue
                name = d.get("name")
            out.append({"cls": int(d.get("cls") or 0), "name": str(name), "conf": float(d.get("conf") or 0.0),
                        "xyxy": [float(v) for v in d["xyxy"]]})
        return out

    def step(self, share, device, base, bgr, ts):
        """그 장의 입력 검출을 받아 (rec, 오버레이에 그릴 것) — 입력이 이 장에 없으면 None"""
        dets = share.get(("detections", self.model), "object_track") if share is not None else None
        if dets is None:
            return None
        ts = float(ts if ts is not None else time.time())
        pick = self._pick(dets)
        info, out = self.runner(bgr, dets=pick, stream=device, ts=ts)
        tracks = []
        for d in out:
            tid = d.get("track_id")
            if tid is None:
                continue
            k = (device, tid)
            t = self.trails.get(k)
            if t is None:
                t = self.trails[k] = {"trail": deque(maxlen=TRAIL), "hits": 0, "first": ts}
            x1, y1, x2, y2 = d["xyxy"]
            t["trail"].append([round(ts, 3), round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)])
            t["hits"] += 1
            t["last"] = ts
            tracks.append({"track_id": int(tid), "label": d["name"], "confidence": round(d["conf"], 3),
                           "bbox": [round(v, 1) for v in d["xyxy"]], "hits": t["hits"],
                           "first_ts": round(t["first"], 3), "last_ts": round(ts, 3),
                           "trajectory": [list(p) for p in t["trail"]]})
        for k in [k for k, t in self.trails.items() if k[0] == device and ts - t.get("last", ts) > FORGET_S]:
            del self.trails[k]
        rec = dict(base, model="track", function="object_track", tracker=self.key,
                   input={"function": self.function, "model": self.model},
                   infer_ms=info.get("track_ms"), tracks=tracks,
                   detections=[{k: v for k, v in d.items()} for d in out])
        if self.reid is not None:
            self.reid(rec, bgr, device, ts)
        return rec


def overlay(bgr, rec):
    """번호 · 궤적을 그린다"""
    import cv2
    img = bgr.copy()
    for t in rec.get("tracks") or []:
        h = (t["track_id"] * 47) % 180
        col = tuple(int(c) for c in cv2.cvtColor(
            __import__("numpy").uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        x1, y1, x2, y2 = (int(v) for v in t["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, "#%d %s" % (t["track_id"], t["label"]), (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        pts = [(int(p[1]), int(p[2])) for p in t["trajectory"]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, col, 2)
    cv2.putText(img, "track (%s <- %s)  %s #%s" % (rec.get("tracker"), (rec.get("input") or {}).get("model"),
                                                    rec.get("source"), rec.get("n")),
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- 재식별 범위 (reid · multi_cam_track)
def resolve_reid(profile, devices):
    """프로필 → 재식별 설정 또는 None. {function, scope, group_sources, embedder, in_process}

    scope source: 소스마다 갤러리(이 프로세스 안). scope group: group_sources(기본 — 이 프로세스의 소스 전부) 하나의
    갤러리 — 묶음의 소스가 모두 이 프로세스에 있으면 안에서, 아니면 임베딩만 reid_embed:<소스> 로 내고 갤러리는
    vision/reid_gallery.py 를 따로 띄운다. 추적(object_track)이 켜져 있어야 돈다(트랙마다 임베딩)"""
    profile = profile or {}
    fp = profile.get("function_params") or {}
    picks = profile.get("functions") or {}
    for fn in ("multi_cam_track", "reid"):
        if fn not in fp and fn not in picks:
            continue
        p = fp.get(fn) or {}
        scope = p.get("scope") or ("group" if fn == "multi_cam_track" else "source")
        grp = sorted(p.get("group_sources") or devices) if scope == "group" else None
        emb = ((picks.get(fn) or {}).get("server") or {}).get("embedder") or "mobileclip_s0_cpu"
        return {"function": fn, "scope": scope, "group_sources": grp, "embedder": emb,
                "in_process": scope == "source" or set(grp or ()) <= set(devices)}
    return None


class ReidStep:
    """트랙마다 임베딩 → 갤러리(이 프로세스 안이면) → 트랙 기록에 global_id. 내보낼 것은 outbox 에 쌓는다"""

    def __init__(self, conf, embedder, every=10):
        from reid_gallery import GroupGallery
        self.conf, self.embedder, self.every = conf, embedder, int(every)
        self._GG = GroupGallery
        self.galleries = {}
        self.group = "+".join(conf["group_sources"]) if conf["scope"] == "group" else None
        if conf["in_process"] and self.group:
            self.galleries[self.group] = GroupGallery(self.group, conf["group_sources"])
        self.last = {}                                   # (소스, track_id) → 마지막으로 임베딩한 장 번호
        self.frame_no = {}
        self.outbox = []                                 # [(키, 필드)] — live 가 Redis 에 쓴다

    def gallery_for(self, device):
        if self.conf["scope"] == "group":
            return self.galleries.get(self.group)
        g = self.galleries.get(device)
        if g is None:
            g = self.galleries[device] = self._GG(device, [device])
        return g

    def __call__(self, rec, bgr, device, ts):
        import json
        from reid_gallery import EMBED_KEY, IDENTITY_KEY, enc
        n = self.frame_no[device] = self.frame_no.get(device, 0) + 1
        todo = [t for t in rec.get("tracks") or []
                if (device, t["track_id"]) not in self.last or n - self.last[(device, t["track_id"])] >= self.every]
        embed_ms = 0.0
        if todo:
            info, embs = self.embedder(bgr, [t["bbox"] for t in todo])
            embed_ms = float(info.get("infer_ms") or 0.0)
            items = []
            for t, e in zip(todo, embs):
                self.last[(device, t["track_id"])] = n
                items.append({"track_id": t["track_id"], "label": t["label"], "emb": enc(e)})
            if self.conf["scope"] == "group":           # 묶음 갤러리가 다른 프로세스여도 받게 늘 낸다
                self.outbox.append((EMBED_KEY % device, {"result": json.dumps(
                    {"schema": "reid_embed/1", "source": device, "ts": ts, "group": self.group,
                     "embedder": self.conf["embedder"], "items": items}, ensure_ascii=False)}))
            g = self.gallery_for(device)
            if g is not None:
                if self.conf["scope"] == "group":
                    g.feed(device, ts, items)
                    done = g.drain()
                else:
                    done = g.ingest(device, ts, items)
                if done:
                    self.outbox.append((IDENTITY_KEY % g.group, {"result": json.dumps(
                        {"schema": "reid/1", "group": g.group, "ts": ts, "assigned": done}, ensure_ascii=False)}))
        for g in self.galleries.values():                # 배정된 번호를 트랙에 단다 (칸 추가만)
            for t in rec.get("tracks") or []:
                gid = g.assign.get((device, t["track_id"]))
                if gid is not None:
                    t["global_id"] = gid
        rec["identity"] = {"function": self.conf["function"], "scope": self.conf["scope"],
                           "group": self.group or device, "embedder": self.conf["embedder"],
                           "gallery": "in_process" if self.conf["in_process"] else "vision/reid_gallery.py",
                           "embed_ms": round(embed_ms, 1)}
        return rec

    def flush(self, r):
        """쌓인 것을 Redis 에 (live). r 이 없으면 버린다(저장된 이미지 처리)"""
        box, self.outbox = self.outbox, []
        if r is None or not box:
            return
        pipe = r.pipeline(transaction=False)
        for key, fields in box:
            pipe.xadd(key, fields, maxlen=1000, approximate=True)
        pipe.execute()
