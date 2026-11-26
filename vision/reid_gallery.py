"""재식별 갤러리 — 소스 묶음에 전역 번호를 붙인다 (reid scope, work/1, 2026-11-27).

범위 (프로필 function_params.reid.scope — 런처 「소스 안 / 소스 묶음」)
  source  소스 하나에 갤러리 하나 — 그 소스 트래커 옆(같은 비전 프로세스)에서 돈다. 다시 나타난 물체에 같은 번호.
          사람이 고른 물체를 찾는 기본 매칭(features.reid — SuperPoint+LightGlue, vision/reid.py)은 그대로 따로 돈다
  group   사람이 고른 소스 묶음(group_sources — 자동 묶음은 없다) 전체에 갤러리 하나. 소스마다 트랙 임베딩
          (embed 슬롯: 서버 mobileclip_s0_cpu · 기기 embedder)을 받아 같은 물체에 같은 전역 번호(global_id)를 단다

자료 흐름 (group)

   소스 A 비전 ─ 탐지 → 추적(track_id) → 임베딩 ─┐  Redis reid_embed:A   {ts, source, items:[{track_id, label, emb}]}
   소스 B 비전 ─ 탐지 → 추적(track_id) → 임베딩 ─┤  Redis reid_embed:B
                                                  ▼
                       GroupGallery (같은 프로세스면 그 안에서, 아니면 이 파일을 따로 띄운다)
                         · 시각 맞춤: 들어온 것을 ts 순으로, 가장 늦은 ts 보다 tol_s 이상 앞선 것만 처리한다
                           (소스마다 늦게 오는 것을 기다린다)
                         · 같은 소스에 같은 시각(tol_s 안)에 함께 보인 두 트랙은 같은 물체가 아니다
                         · 같은 이름(label) · 코사인 ≥ sim 인 전역 번호 중 가장 닮은 것, 없으면 새 번호
                                                  ▼
                       Redis reid_identity:<묶음>  {ts, group, assigned:[{source, track_id, global_id, sim, new}]}
                       Redis reid_identity:<묶음>:map (hash)  "<소스>:<track_id>" → global_id

따로 띄우기 (묶음의 소스가 서로 다른 비전 프로세스에 있을 때):
  python vision/reid_gallery.py --group pi12+a2 --sources pi12 a2 [--redis redis://127.0.0.1:6380/0]
"""
from __future__ import annotations

import argparse
import base64
import json
import time

import numpy as np

EMBED_KEY = "reid_embed:%s"
IDENTITY_KEY = "reid_identity:%s"


def enc(v):
    return base64.b64encode(np.asarray(v, np.float16).tobytes()).decode()


def dec(s):
    return np.frombuffer(base64.b64decode(s), np.float16).astype(np.float32)


def _unit(v):
    v = np.asarray(v, np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class GroupGallery:
    """전역 번호 갤러리 — 소스 묶음 하나(scope source 면 소스 하나)"""

    def __init__(self, group, sources, sim=0.80, tol_s=1.0, alpha=0.2, forget_s=600.0):
        self.group, self.sources = str(group), list(sources)
        self.sim, self.tol, self.alpha, self.forget = float(sim), float(tol_s), float(alpha), float(forget_s)
        self.ids = {}           # global_id → {emb, label, last, members: {(소스, track_id)}}
        self.assign = {}        # (소스, track_id) → global_id
        self.seen = {}          # (소스, track_id) → 마지막 ts
        self.buf = []           # [(ts, 소스, item)]
        self.next_id = 1
        self.latest = None

    # -- 시각 맞춤
    def feed(self, source, ts, items):
        for it in items or []:
            self.buf.append((float(ts), str(source), it))
        self.latest = max(self.latest or float(ts), float(ts))

    def drain(self, flush=False):
        """가장 늦게 들어온 ts 보다 tol_s 이상 앞선 것(flush 면 전부)을 ts 순으로 처리 → [배정]"""
        if not self.buf:
            return []
        cut = float("inf") if flush else (self.latest - self.tol)
        ready = sorted((b for b in self.buf if b[0] <= cut), key=lambda b: (b[0], b[1]))
        self.buf = [b for b in self.buf if b[0] > cut]
        return [self._one(ts, src, it) for ts, src, it in ready]

    def ingest(self, source, ts, items):
        """바로 처리 (한 소스 · 시각 맞춤이 필요 없을 때)"""
        return [self._one(float(ts), str(source), it) for it in items or []]

    # -- 배정
    def _busy(self, source, ts, key):
        """그 시각에 그 소스에서 다른 트랙이 쥐고 있는 전역 번호 — 함께 보인 둘은 같은 물체가 아니다"""
        out = set()
        for k, last in self.seen.items():
            if k[0] == source and k != key and abs(ts - last) <= self.tol and k in self.assign:
                out.add(self.assign[k])
        return out

    def _one(self, ts, source, it):
        key = (source, int(it["track_id"]))
        emb = _unit(it["emb"] if not isinstance(it["emb"], str) else dec(it["emb"]))
        label = it.get("label")
        self.seen[key] = ts
        gid = self.assign.get(key)
        best, sim = None, None
        if gid is None:
            busy = self._busy(source, ts, key)
            for g, e in self.ids.items():
                if g in busy or (label and e["label"] and e["label"] != label):
                    continue
                s = float(np.dot(e["emb"], emb))
                if s >= self.sim and (sim is None or s > sim):
                    best, sim = g, s
            new = best is None
            gid = best if best is not None else self.next_id
            if new:
                self.next_id += 1
                self.ids[gid] = {"emb": emb, "label": label, "last": ts, "members": set()}
            self.assign[key] = gid
        else:
            new = False
            sim = float(np.dot(self.ids[gid]["emb"], emb))
        e = self.ids[gid]
        e["emb"] = _unit((1 - self.alpha) * e["emb"] + self.alpha * emb)
        e["last"] = max(e["last"], ts)
        e["members"].add(key)
        for g in [g for g, x in self.ids.items() if ts - x["last"] > self.forget]:
            for k in self.ids[g]["members"]:
                self.assign.pop(k, None)
                self.seen.pop(k, None)
            del self.ids[g]
        return {"source": source, "track_id": key[1], "global_id": gid, "label": label,
                "sim": None if sim is None else round(sim, 3), "new": new, "ts": round(ts, 3)}


def publish(r, group, assigned, ts=None):
    """배정 → Redis reid_identity:<묶음> (스트림) · :map (hash)"""
    if not assigned:
        return
    pipe = r.pipeline(transaction=False)
    pipe.xadd(IDENTITY_KEY % group, {"result": json.dumps({"schema": "reid/1", "group": group,
                                                          "ts": ts or time.time(), "assigned": assigned},
                                                         ensure_ascii=False)},
              maxlen=1000, approximate=True)
    pipe.hset((IDENTITY_KEY % group) + ":map",
              mapping={"%s:%s" % (a["source"], a["track_id"]): a["global_id"] for a in assigned})
    pipe.execute()


def run(redis_url, group, sources, sim=0.80, tol_s=1.0, stop=lambda: False, log=print):
    """따로 띄운 갤러리 — reid_embed:<소스> 를 읽어 reid_identity:<묶음> 에 쓴다"""
    import redis
    r = redis.Redis.from_url(redis_url)
    g = GroupGallery(group, sources, sim, tol_s)
    last = {EMBED_KEY % s: "$" for s in sources}
    log("갤러리 %s — 소스 %s · 읽기 reid_embed:<소스> · 쓰기 %s" % (group, " ".join(sources), IDENTITY_KEY % group))
    while not stop():
        got = r.xread(last, count=50, block=500) or []
        for key, rows in got:
            key = key.decode() if isinstance(key, bytes) else key
            for eid, f in rows:
                last[key] = eid
                try:
                    doc = json.loads(f.get(b"result") or f.get("result"))
                except (TypeError, ValueError):
                    continue
                g.feed(doc["source"], doc["ts"], doc.get("items"))
        publish(r, group, g.drain())


def main(argv=None):
    ap = argparse.ArgumentParser(description="재식별 갤러리 (소스 묶음 — reid scope group)")
    ap.add_argument("--group", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--redis", default="redis://127.0.0.1:6380/0")
    ap.add_argument("--sim", type=float, default=0.80)
    ap.add_argument("--tol-s", type=float, default=1.0)
    a = ap.parse_args(argv)
    if len(a.sources) < 2:
        raise SystemExit("소스 묶음은 2개 이상이어야 합니다")
    run(a.redis, a.group, a.sources, a.sim, a.tol_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
