"""오케스트레이션 컨트롤러 — 요청 → 클러스터 확인 → 배치 → 실행 → 건강 확인 → 스트림 (core, 2026-11-24).

k3s 는 선택이다. cluster 백엔드(check() · nodes())가 없거나 check 가 실패하면 local 로 떨어지고
(fallback, basis=saved) GPU 작업은 data_stream 기기의 GPU Executor 로만 돈다. GPU 새 작업은 늘 지금 nvidia-smi 를
확인한 뒤에 들인다(Executor).

applied 와 running 은 다르다. running 은 (1) 프로세스가 살아 있고 (2) 비전 보고에 처리한 장이 있으며
(3) 맡은 소스의 입력 영상(camera_stream)이 INPUT_FRESH_S 안의 것일 때만 남긴다.

실행에 꼭 필요한 상태는 StateFile(state/orchestrator_state.json). 스트림을 못 쓰면 outbox 에 쌓았다가
다음 poll 에서 다시 쓴다(Redis 장애 동안 이벤트를 잃지 않는다).
"""
import json
import time
import uuid
from pathlib import Path

from ds_resources import gpu_executor as ge
from ds_resources import memory as mm
from ds_resources import placement, probe
from ds_resources.streams import ORCH_STREAM, StreamWriter, orch_record

REPO = Path(__file__).resolve().parent.parent
REGISTRY = REPO / "vision" / "model_registry.yaml"
INPUT_FRESH_S = 5.0
COMBO_STALE_S = 7 * 86400.0
DEGRADED_RATIO = 0.8
DEGRADED_AFTER_S = 30.0


def jpeg_size(data):
    """JPEG 바이트에서 (w, h) — SOF 표지만 읽는다(풀지 않는다)."""
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            return (data[i + 7] << 8) | data[i + 8], (data[i + 5] << 8) | data[i + 6]
        i += 2 + ((data[i + 2] << 8) | data[i + 3])
    return None, None


def source_info(r, src, now_ms=None):
    """camera_stream:<소스> 마지막 장 → {age_s, w, h, fps(최근 10초)} — 없으면 None"""
    now_ms = now_ms or int(time.time() * 1000)
    try:
        last = r.xrevrange("camera_stream:" + src, "+", "-", count=1)
        if not last:
            return None
        eid, fields = last[0]
        ts = int(eid.split(b"-")[0] if isinstance(eid, bytes) else str(eid).split("-")[0])
        w, h = jpeg_size(fields.get(b"image") or fields.get("image") or b"")
        recent = r.xrevrange("camera_stream:" + src, "+", "%d-0" % (now_ms - 10000), count=600)
        return {"age_s": (now_ms - ts) / 1000.0, "w": w, "h": h, "fps": round(len(recent) / 10.0, 2)}
    except Exception:  # noqa: BLE001
        return None


def load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Controller:
    def __init__(self, redis_client, state, executor, memory, local_node=None, cluster=None, cfg=None,
                 clock=time.time, log=print, gpus=probe.gpus):
        self.r, self.state, self.ex, self.mem = redis_client, state, executor, memory
        self.local = local_node or probe.node_name()
        self.cluster, self.clock, self.log, self.gpus = cluster, clock, log, gpus
        self.cfg = cfg if cfg is not None else load_yaml(REPO / "vision" / "vision_config.yaml")
        self.reg = load_yaml(REGISTRY)
        self.reg_hash = mm.cfg_hash(REGISTRY, {}, [])
        self.writer = StreamWriter(redis_client, maxlen=2000)

    # ------------------------------------------------------------ 스트림 (outbox)
    def emit(self, event, request_id, **kw):
        rec = orch_record(event, request_id, **kw)
        if not self.writer.add(ORCH_STREAM, rec):
            self.state.data.setdefault("outbox", []).append(rec)
            self.state._save()
        return rec

    def flush_outbox(self):
        box = self.state.data.get("outbox") or []
        if not box:
            return 0
        left = [rec for rec in box if not self.writer.add(ORCH_STREAM, rec)]
        self.state.data["outbox"] = left
        self.state._save()
        return len(box) - len(left)

    # ------------------------------------------------------------ 비용 · 스냅숏
    def device_class(self):
        g = self.gpus()
        return ("gpu:" + g[0]["name"]) if g else None

    def costs(self, models, sources, profile=None):
        est, need = ge.estimate_mb(models, profile, self.cfg)
        dev = self.device_class()
        h = mm.cfg_hash(REGISTRY, self.cfg, models)
        now = self.clock()
        mv, ms = self.mem.combo(mm.combo_key("gpu", models, h, dev))
        stale = mv is None or now - (self.mem.data["combos"].get(mm.combo_key("gpu", models, h, dev)) or {}).get(
            "updated_at", 0) > COMBO_STALE_S
        req, basis = ge.required_mb(est, (mv, ms, stale) if mv else None, ge.untracked_mb(need))
        p90, p90_basis = {}, {}
        for s, info in sources.items():
            bucket = mm.res_bucket(info.get("w"), info.get("h"))
            cv, cs = self.mem.combo(mm.combo_key("cost", models, h, dev, bucket))
            if cs == "measured" and cv and cv.get("infer_ms_p90"):
                p90[s], p90_basis[s] = float(cv["infer_ms_p90"]), "measured"
            else:                                           # 레지스트리 기본값 (조합이면 큰 쪽 합)
                p90[s] = float(sum((self.reg["options"].get(m, {}).get("p90_ms") or {}).get(bucket, 500) for m in models))
                p90_basis[s] = "registry"
        return {"need_mb": round(req), "basis": basis, "estimate_mb": est, "needs": need,
                "p90_ms": p90, "p90_basis": p90_basis, "cfg_hash": h}

    def local_node_snapshot(self):
        g = self.gpus()
        loaded = [u for u in self.ex.active_units().values() if u.get("status") == "loaded"]
        return {"name": self.local, "role": "gpu", "ready": bool(g), "stale": not g,
                "gpus": [{"mem_total_mb": x["mem_total_mb"], "mem_used_mb": x["mem_used_mb"]} for x in g],
                "gpu_util_committed": sum(float(u.get("gpu_util") or 0) for u in loaded)}

    # ------------------------------------------------------------ 요청
    def submit(self, sources, models, profile=None, profile_path=None, target_fps=None):
        rid = "req-" + uuid.uuid4().hex[:10]
        request = {"sources": sorted(sources), "models": sorted(models), "target_fps": target_fps or {}}
        self.state.put_request(rid, dict(request, status="requested", at=self.clock()))
        self.emit("requested", rid, request=request)
        ok, why, nodes = True, "", []
        if self.cluster is None:
            ok, why = False, "k3s 백엔드 없음 (로컬 전용)"
        else:
            try:
                ok, why = self.cluster.check()
                nodes = self.cluster.nodes() if ok else []
            except Exception as e:  # noqa: BLE001
                ok, why = False, "k3s 확인 실패: %s" % e
        self.emit("cluster_checked", rid, reason=why, detail={"ok": ok})
        mode, basis = ("k3s", "live") if ok else ("local", "saved")
        if not ok:
            self.emit("fallback", rid, mode="local", basis="saved", reason=why)
        infos, missing = {}, []
        for s in sorted(sources):
            info = source_info(self.r, s, int(self.clock() * 1000))
            if info is None or info["age_s"] > INPUT_FRESH_S:
                missing.append(s)
            else:
                infos[s] = info
        snapshot = {"nodes": [self.local_node_snapshot()] + [n for n in nodes if n.get("name") != self.local],
                    "costs": {}}
        if infos:
            snapshot["costs"]["+".join(sorted(models))] = self.costs(models, infos, profile)
        fps = {s: float((target_fps or {}).get(s) or max(1.0, infos[s]["fps"] or 10.0)) for s in infos}
        req = {"request_id": rid, "gpu": [{"models": models, "sources": sorted(infos), "target_fps": fps}] if infos else []}
        plan = placement.plan(req, snapshot, self.reg_hash, self.local)
        for s in missing:
            plan["refused"].append({"item": "input:%s" % s, "reason": "입력 영상 없음 또는 %d초 넘게 멈춤" % INPUT_FRESH_S})
        self.emit("planned", rid, mode=mode, basis=basis, plan=plan)
        xid = "exe-" + uuid.uuid4().hex[:10]
        self.state.put_request(rid, dict(request, status="planned", plan=plan, mode=mode, execution_id=xid, at=self.clock()))
        for item in plan["degraded"]:
            self.emit("degraded", rid, execution_id=xid, mode=mode, reason=item["reason"], detail=item)
        for item in plan["refused"]:
            self.emit("failed", rid, execution_id=xid, mode=mode, reason=item["reason"], detail=item)
        for u in plan["gpu_units"]:
            if u["node"] != self.local:                 # GPU 노드가 늘면 그 노드의 Executor 로 (지금은 data_stream 기기 하나)
                self.emit("failed", rid, execution_id=xid, mode=mode, reason="원격 GPU Executor 없음: %s" % u["node"])
                continue
            c = snapshot["costs"]["+".join(u["models"])]
            okk, why2, calc = self.ex.admit(u["need_mb"], c["basis"], c["estimate_mb"])
            if not okk:
                self.emit("failed", rid, execution_id=xid, mode=mode, reason=why2, detail=calc)
                continue
            uid = self.ex.launch({"devices": u["sources"], "models": u["models"], "profile_path": profile_path,
                                  "fps_cap": u["fps_cap"], "gpu_util": u["gpu_util"], "request_id": rid,
                                  "execution_id": xid}, u["need_mb"], calc)
            self.emit("applied", rid, execution_id=xid, mode=mode, reason=why2 or "GPU Executor 로 띄움",
                      detail={"unit_id": uid, "admit": calc, "fps_cap": u["fps_cap"]})
        return rid

    # ------------------------------------------------------------ 감시
    def poll(self):
        self.flush_outbox()
        events = []
        for uid, change in self.ex.poll():
            u = self.state.units().get(uid) or {}
            rid, xid = u.get("request_id") or "-", u.get("execution_id")
            if change == "loaded":
                continue                                    # running 은 아래 건강 확인에서
            if change == "restarted":
                events.append(self.emit("degraded", rid, execution_id=xid, mode="local",
                                        reason="프로세스가 끝나 WSL 안에서 다시 띄움 (%d번째)" % u.get("restarts", 0)))
            else:
                events.append(self.emit("failed" if change == "failed" else "stopped", rid, execution_id=xid,
                                        reason=u.get("reason", change), detail={"unit_id": uid}))
        now = self.clock()
        for uid, u in self.ex.active_units().items():
            if u.get("status") != "loaded":
                continue
            rid, xid = u.get("request_id") or "-", u.get("execution_id")
            tel = self.ex._telemetry(u["pid"]) or {}
            streams = tel.get("streams") or {}
            now_ms = int(now * 1000)
            fresh = {s: (source_info(self.r, s, now_ms) or {}).get("age_s", 1e9) <= INPUT_FRESH_S for s in u["devices"]}
            frames = sum(int(x.get("frames") or 0) for x in streams.values())
            if not u.get("running_emitted"):
                if frames > 0 and all(fresh.values()):
                    self.state.update_unit(uid, running_emitted=True, running_at=now)
                    events.append(self.emit("running", rid, execution_id=xid, mode="local",
                                            detail={"unit_id": uid, "fps": {s: x.get("fps") for s, x in streams.items()}}))
                continue
            low = [s for s, cap in (u.get("fps_cap") or {}).items()
                   if (streams.get(s) or {}).get("fps") is not None and streams[s]["fps"] < DEGRADED_RATIO * cap]
            since = u.get("low_since")
            if low and since is None:
                self.state.update_unit(uid, low_since=now)
            elif low and now - since >= DEGRADED_AFTER_S and not u.get("degraded_emitted"):
                self.state.update_unit(uid, degraded_emitted=True)
                events.append(self.emit("degraded", rid, execution_id=xid, mode="local",
                                        reason="목표 fps 의 %d%% 미달 %d초: %s" % (DEGRADED_RATIO * 100, DEGRADED_AFTER_S,
                                                                             ", ".join(low)),
                                        detail={"fps": {s: streams[s].get("fps") for s in low}, "cap": u.get("fps_cap")}))
            elif not low and since is not None:
                self.state.update_unit(uid, low_since=None, degraded_emitted=False)
        return events

    def stop(self, request_id, reason="요청"):
        n = 0
        for uid, u in self.ex.active_units().items():
            if u.get("request_id") == request_id:
                self.ex.stop(uid, reason)
                self.emit("stopped", request_id, execution_id=u.get("execution_id"), reason=reason,
                          detail={"unit_id": uid})
                n += 1
        return n
