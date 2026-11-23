#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reid.py — 객체 재식별 (2026-11-23).

가시화(VIZ) · 웹 화면에서 「재식별 선택」 → 상자 클릭으로 고른 물체를 **끝날 때까지** 따라간다.

  고르기      서버 /control/reid 가 Redis `vision_reid` 에 {id, device, model, bbox, track_id, n, label} 을 적는다
  갤러리      그 기기를 처리하는 비전 프로세스가 다음 장에서 그 물체를 찾아 **모습(view)** 을 만든다:
              전역 = CLIP 이미지 임베딩, 국소 = 특징점 매칭용 crop(256). Redis `vision_reid_gallery:<id>` —
              **모든 카메라의 비전 프로세스가 같은 갤러리를 본다**(다른 카메라 연결)
  추적        트랙 번호(또는 앞 장 상자와 IoU)가 이어지는 동안은 그 물체다. 2초마다, 지금 모습이 갤러리의
              어느 모습과도 많이 다르면(CLIP 코사인 < 0.92) 새 모습을 더한다(최대 8)
  다시 찾기   놓쳤거나 다른 카메라라면 같은 이름의 후보를 **전역(CLIP)으로 2 개까지 좁히고**, **국소(특징점
              매칭 수)** 로 가린다 — 매칭 수 ≥ 15 이고 둘째 후보의 두 배 이상이면 그 물체다
              (실측: 같은 물체 31~74 쌍, 다른 물체 5~7 쌍). 한 트랙은 10 장에 한 번만 다시 잰다
  기록        Redis `vision_reid_seen:<id>` — 마지막으로 본 기기 · 트랙 · 상자 · 방법(tracked · reacquired ·
              other_camera) · 시각. 검출에 `reid` 칸을 붙이고 오버레이에 자홍색 상자로 그린다

값(0.92 · 15 · 2 배 · 10 장 · 8 모습)은 대상과 무관한 일반 상수다.
"""
from __future__ import annotations

import base64
import json
import time

import cv2
import numpy as np

VIEW_SIM_NEW = 0.92        # 지금 모습이 갤러리 모습들과 이보다 덜 닮았으면 새 모습으로 더한다
VIEW_EVERY_S = 2.0
MAX_VIEWS = 8
SHORTLIST = 2              # 전역으로 좁힐 후보 수
MIN_MATCHES = 15
MARGIN = 2.0               # 첫째 후보 매칭 수 ≥ 둘째의 이 배
RECHECK_FRAMES = 10
CONFIRM = 2                # 다시 찾기는 같은 후보가 **연달아 이만큼** 확인돼야 한다(한 번은 우연일 수 있다 —
                           # robot2 저장본: 다른 사람이 한 번 17 쌍으로 붙었다가 다음 확인에서 근거가 사라졌다)
REID_BGR = (255, 0, 255)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class MemStore:
    """Redis 없이(저장된 이미지 처리) — 한 프로세스 안에서만 나눈다."""

    def __init__(self):
        self.kv = {}

    def get(self, k):
        return self.kv.get(k)

    def set(self, k, v):
        self.kv[k] = v

    def delete(self, k):
        self.kv.pop(k, None)


class RedisStore:
    def __init__(self, r):
        self.r = r

    def get(self, k):
        try:
            v = self.r.get(k)
        except Exception:
            return None
        return v.decode() if isinstance(v, bytes) else v

    def set(self, k, v):
        try:
            self.r.set(k, v)
        except Exception:
            pass

    def delete(self, k):
        try:
            self.r.delete(k)
        except Exception:
            pass


def _jpg_b64(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _b64_rgb(s):
    img = cv2.imdecode(np.frombuffer(base64.b64decode(s), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class ReidEngine:
    """한 비전 프로세스에 하나. step() 을 장마다 (기기, 모델, 그림, 검출) 로 부른다."""

    def __init__(self, store, clip, matcher, log=print):
        self.store, self.clip, self.matcher, self.log = store, clip, matcher, log
        self.bound = {}            # (id, device) → {"key": 트랙 키, "box", "n"}
        self.cache = {}            # (id, device, 트랙 키) → (매칭 수, 판정 장 번호)
        self.view_t = {}           # id → 마지막으로 모습을 더한 시각
        self.gal = {}              # id → {"label", "emb": np (V,512), "rgb": [..], "ver"}
        self.last_try = {}         # id → 마지막 다시 찾기 시도(결과 · 화면 설명용)
        self.pending = {}          # (id, device) → {"key": 후보 키, "box", "count", "n"} — 확인 중인 후보
        self.n = 0

    # -- 갤러리
    def _load(self, rid):
        raw = self.store.get("vision_reid_gallery:%s" % rid)
        if not raw:
            return None
        g = json.loads(raw)
        cur = self.gal.get(rid)
        if cur is None or cur["ver"] != g.get("ver"):
            cur = {"label": g["label"], "ver": g.get("ver"),
                   "emb": np.array([v["emb"] for v in g["views"]], np.float32),
                   "rgb": [_b64_rgb(v["jpg"]) for v in g["views"]], "raw": g}
            self.gal[rid] = cur
        return cur

    def _save(self, rid, g):
        g["ver"] = int(g.get("ver", 0)) + 1
        self.store.set("vision_reid_gallery:%s" % rid, json.dumps(g))
        self.gal.pop(rid, None)

    def _crop(self, bgr, box):
        H, W = bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        c = bgr[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
        return c if c.size and min(c.shape[:2]) >= 8 else None

    def _view(self, crop, device):
        from feature_match import to_rgb_square
        emb = self.clip.images([crop])[0]
        return {"emb": [round(float(x), 5) for x in emb], "jpg": _jpg_b64(to_rgb_square(crop)),
                "device": device, "t": time.time()}, emb

    @staticmethod
    def _tkey(d):
        return ("t", d["track_id"]) if d.get("track_id") is not None else None

    # -- 한 장
    def step(self, items, device, model, bgr, dets, n):
        """items = 고른 물체 목록. dets 에 `reid` 를 붙인다. 걸린 ms 를 돌려준다."""
        t0 = time.perf_counter()
        self.n += 1
        taken = set()
        for it in items:
            rid = it["id"]
            if it.get("model") and it["model"] != model:
                continue
            g = self._load(rid)
            if g is None:
                if it.get("device") == device:
                    self._init(it, device, bgr, dets, n)
                continue
            label = g["label"]
            same = [d for d in dets if d.get("name") == label and id(d) not in taken]
            if not same:
                continue
            hit, how, score = self._follow(rid, device, same)
            if hit is None:
                hit, how, score = self._reacquire(rid, g, device, bgr, same)
            if hit is None:
                continue
            taken.add(id(hit))
            self.bound[(rid, device)] = {"key": self._tkey(hit), "box": list(hit["xyxy"]), "n": self.n}
            hit["reid"] = dict({"id": rid, "how": how}, **score)
            self._seen(rid, device, hit, how, score, n)
            if how == "tracked":
                self._maybe_add_view(rid, g, device, bgr, hit)
        return round((time.perf_counter() - t0) * 1000, 1)

    def _init(self, it, device, bgr, dets, n):
        """고른 장 근처에서 그 물체를 찾아 첫 모습을 만든다."""
        cand = [d for d in dets if (not it.get("label") or d.get("name") == it["label"])]
        pick = None
        if it.get("track_id") is not None:
            pick = next((d for d in cand if d.get("track_id") == it["track_id"]), None)
        if pick is None and it.get("bbox"):
            best = max(cand, key=lambda d: _iou(d["xyxy"], it["bbox"]), default=None)
            if best is not None and _iou(best["xyxy"], it["bbox"]) >= 0.3:
                pick = best
        if pick is None:
            return
        crop = self._crop(bgr, pick["xyxy"])
        if crop is None:
            return
        view, _e = self._view(crop, device)
        g = {"id": it["id"], "label": pick["name"], "views": [view], "from": {"device": device, "n": n}}
        self._save(it["id"], g)
        self.view_t[it["id"]] = time.time()
        self.bound[(it["id"], device)] = {"key": self._tkey(pick), "box": list(pick["xyxy"]), "n": self.n}
        pick["reid"] = {"id": it["id"], "how": "selected"}
        self._seen(it["id"], device, pick, "selected", {}, n)
        if self.log:
            self.log("재식별 %s: %s 를 %s 에서 골랐다 (장 %s)" % (it["id"], pick["name"], device, n))

    def _follow(self, rid, device, same):
        b = self.bound.get((rid, device))
        if b is None or self.n - b["n"] > 3:
            return None, None, {}
        if b["key"] is not None:
            # **번호만 믿지 않는다** — 트래커가 다시 세워지면(프레임 사이가 벌어짐) 번호를 새로 매겨 다른 사람이
            # 같은 번호를 받는다(robot1 저장본: 오른쪽 사람의 2 번이 왼쪽 사람에게). 앞 장 상자와도 겹쳐야 한다
            d = next((d for d in same if self._tkey(d) == b["key"] and _iou(d["xyxy"], b["box"]) >= 0.3), None)
            if d is not None:
                return d, "tracked", {}
        d = max(same, key=lambda d: _iou(d["xyxy"], b["box"]))
        if _iou(d["xyxy"], b["box"]) >= 0.3:
            return d, "tracked", {}
        return None, None, {}

    def _reacquire(self, rid, g, device, bgr, same):
        """전역으로 좁히고 국소로 가린다."""
        crops, ds = [], []
        for d in same:
            c = self._crop(bgr, d["xyxy"])
            if c is not None:
                crops.append(c)
                ds.append(d)
        if not ds:
            return None, None, {}
        embs = self.clip.images(crops)
        sims = (embs @ g["emb"].T).max(axis=1)
        order = np.argsort(-sims)[:SHORTLIST]
        from feature_match import to_rgb_square
        results = []
        for i in order:
            d = ds[i]
            key = (rid, device, self._tkey(d) or ("b", round(d["xyxy"][0] / 16), round(d["xyxy"][1] / 16)))
            c = self.cache.get(key)
            pend = self.pending.get((rid, device))
            confirming = pend is not None and _iou(pend["box"], d["xyxy"]) >= 0.3 and self.n - pend["n"] <= RECHECK_FRAMES
            if c is not None and self.n - c[1] < RECHECK_FRAMES and not confirming:
                m = c[0]
            else:
                vi = np.argsort(-(g["emb"] @ embs[i]))[:2]
                q = to_rgb_square(crops[i])
                m = max(self.matcher.count([(q, g["rgb"][j]) for j in vi])) if self.matcher else 0
                self.cache[key] = (m, self.n)
            results.append((m, float(sims[i]), d))
        results.sort(key=lambda t: -t[0])
        top = results[0]
        second = results[1][0] if len(results) > 1 else 0
        self.last_try[rid] = {"device": device, "matches": top[0], "second": second, "sim": round(top[1], 3),
                              "candidates": len(ds),
                              "all": [[m, round(sv, 3), [round(v) for v in dd["xyxy"]]] for m, sv, dd in results],
                              "sims": sorted([round(float(x), 3) for x in sims], reverse=True)}
        if top[0] >= MIN_MATCHES and top[0] >= MARGIN * max(second, 1):
            p = self.pending.get((rid, device))
            if p is not None and _iou(p["box"], top[2]["xyxy"]) >= 0.3 and self.n - p["n"] <= RECHECK_FRAMES:
                p["count"] += 1
            else:
                p = {"count": 1}
            p.update(box=list(top[2]["xyxy"]), n=self.n)
            self.pending[(rid, device)] = p
            self.last_try[rid].update(confirm=p["count"])
            if p["count"] < CONFIRM:
                return None, None, {}
            self.pending.pop((rid, device), None)
            last = self._last_device(rid)
            how = "other_camera" if last and last != device else "reacquired"
            return top[2], how, {"matches": top[0], "sim": round(top[1], 3), "second": second}
        self.pending.pop((rid, device), None)
        return None, None, {}

    def _maybe_add_view(self, rid, g, device, bgr, d):
        if time.time() - self.view_t.get(rid, 0) < VIEW_EVERY_S or len(g["raw"]["views"]) >= MAX_VIEWS:
            return
        crop = self._crop(bgr, d["xyxy"])
        if crop is None:
            return
        self.view_t[rid] = time.time()
        e = self.clip.images([crop])[0]
        if float((g["emb"] @ e).max()) >= VIEW_SIM_NEW:
            return
        view, _ = self._view(crop, device)
        raw = dict(g["raw"])
        raw["views"] = list(raw["views"]) + [view]
        self._save(rid, raw)

    def _last_device(self, rid):
        raw = self.store.get("vision_reid_seen:%s" % rid)
        return json.loads(raw).get("device") if raw else None

    def _seen(self, rid, device, d, how, score, n):
        prev = self.store.get("vision_reid_seen:%s" % rid)
        trail = (json.loads(prev).get("trail") if prev else None) or []
        if not trail or trail[-1]["device"] != device:
            trail = (trail + [{"device": device, "t": time.time()}])[-20:]
        self.store.set("vision_reid_seen:%s" % rid, json.dumps(
            {"id": rid, "device": device, "track_id": d.get("track_id"), "bbox": [round(v, 1) for v in d["xyxy"]],
             "how": how, "n": n, "t": time.time(), "trail": trail, **score}))


def draw_reid(img, dets):
    """재식별 물체 — 자홍색 굵은 상자와 'REID <id> · 방법'."""
    for d in dets or []:
        r = d.get("reid")
        if not r:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        lw = max(3, round(img.shape[1] / 320))
        cv2.rectangle(img, (x1, y1), (x2, y2), REID_BGR, lw)
        how = {"selected": "selected", "tracked": "tracked", "reacquired": "found again",
               "other_camera": "other camera"}.get(r.get("how"), r.get("how", ""))
        s = "REID %s %s%s" % (r["id"], how, (" %d" % r["matches"]) if r.get("matches") else "")
        fs = max(0.4, img.shape[1] / 1600.0)
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        y = min(img.shape[0] - 2, y2 + th + 6)
        cv2.rectangle(img, (x1, y - th - 6), (x1 + tw + 6, y), REID_BGR, -1)
        cv2.putText(img, s, (x1 + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return img
