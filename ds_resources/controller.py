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
CPU_HEARTBEAT_FRESH_S = 30.0          # cpu_telemetry:<unit> 이 이 안에 있어야 running
CPU_REPLACE_AFTER_S = 60.0            # running 이던 CPU 작업이 이만큼 하트비트가 없으면 다시 배치
NODE_BAD_GRACE_S = 30.0               # 노드가 Ready 아님 · stale 인 채로 이만큼 이어져야 옮긴다 — metrics 가 잠깐
                                      # 끊긴 것만으로 작업이 노드 사이를 오가지 않게 (2026-11-24 실측: 끊긴 지 15초에 옮겼다)


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
                 clock=time.time, log=print, gpus=probe.gpus, cpu_executor=None):
        self.r, self.state, self.ex, self.mem = redis_client, state, executor, memory
        self.cpu_ex = cpu_executor                    # k3s CPU 실행기 (orchestrator 가 주입) — 없으면 파이에 못 둔다
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
        mem = {}
        try:
            for line in open("/proc/meminfo", encoding="utf-8"):
                k, _, v = line.partition(":")
                if k in ("MemTotal", "MemAvailable"):
                    mem[k] = int(v.split()[0]) / 1024.0
            load1 = float(open("/proc/loadavg", encoding="utf-8").read().split()[0])
        except (OSError, ValueError):
            load1 = 0.0
        import os
        return {"name": self.local, "role": "gpu", "ready": bool(g), "stale": not g,
                "gpus": [{"mem_total_mb": x["mem_total_mb"], "mem_used_mb": x["mem_used_mb"]} for x in g],
                "gpu_util_committed": sum(float(u.get("gpu_util") or 0) for u in loaded),
                "cpu_alloc_m": (os.cpu_count() or 1) * 1000.0, "cpu_now_m": load1 * 1000.0,
                "mem_alloc_mb": mem.get("MemTotal", 0.0),
                "mem_now_mb": mem.get("MemTotal", 0.0) - mem.get("MemAvailable", 0.0), "net": {}}

    # ------------------------------------------------------------ 요청
    def cpu_specs(self, names, task_specs=None):
        specs = task_specs if task_specs is not None else (self.reg.get("cpu_tasks") or {})
        out = []
        for n in sorted(names or []):
            if n not in specs:
                raise ValueError("모르는 CPU 작업 %r (레지스트리 cpu_tasks)" % n)
            t = dict(specs[n], task=n)
            if not t.get("image"):                      # 이미지가 없으면 파이에 못 둔다 → data_stream 기기만
                t["roles"] = [r for r in t.get("roles", []) if r == "gpu"]
            out.append(t)
        return out

    def prepare(self, sources, models, profile=None, target_fps=None, cpu_tasks=None, task_specs=None, request_id=None):
        """클러스터 확인 → 입력 영상 → 스냅숏 → placement/1. 아무것도 띄우지 않고 기록도 하지 않는다(미리보기와 같은 계산)."""
        ok, why, nodes = True, "", []
        if self.cluster is None:
            ok, why = False, "k3s 백엔드 없음 (로컬 전용)"
        else:
            try:
                ok, why = self.cluster.check()
                nodes = self.cluster.nodes() if ok else []
            except Exception as e:  # noqa: BLE001
                ok, why = False, "k3s 확인 실패: %s" % e
        mode, basis = ("k3s", "live") if ok else ("local", "saved")
        infos, missing = {}, []
        for s in sorted(sources):
            info = source_info(self.r, s, int(self.clock() * 1000))
            if info is None or info["age_s"] > INPUT_FRESH_S:
                missing.append(s)
            else:
                infos[s] = info
        snapshot = {"nodes": [self.local_node_snapshot()] + [n for n in nodes if n.get("name") != self.local],
                    "costs": {}}
        if infos and models:
            snapshot["costs"]["+".join(sorted(models))] = self.costs(models, infos, profile)
        fps = {s: float((target_fps or {}).get(s) or max(1.0, infos[s]["fps"] or 10.0)) for s in infos}
        specs = self.cpu_specs(cpu_tasks, task_specs)
        if mode != "k3s" or self.cpu_ex is None:        # k3s 가 없으면 CPU 작업도 data_stream 기기에만
            specs = [dict(t, roles=[r for r in t.get("roles", []) if r == "gpu"]) for t in specs]
        req = {"request_id": request_id,
               "gpu": [{"models": models, "sources": sorted(infos), "target_fps": fps}] if infos and models else [],
               "cpu": specs}
        self._specs = {t["task"]: t for t in specs}
        plan = placement.plan(req, snapshot, self.reg_hash, self.local)
        for s in missing:
            plan["refused"].append({"item": "input:%s" % s, "reason": "입력 영상 없음 또는 %d초 넘게 멈춤" % INPUT_FRESH_S})
        return {"ok": ok, "why": why, "mode": mode, "basis": basis, "plan": plan, "snapshot": snapshot}

    def preview(self, sources, models, profile=None, target_fps=None, cpu_tasks=None):
        """런처 「배치 보기」 — prepare + GPU 단위마다 지금 들일 수 있는지(Executor admit, 읽기만)"""
        p = self.prepare(sources, models, profile, target_fps, cpu_tasks)
        for u in p["plan"]["gpu_units"]:
            c = p["snapshot"]["costs"]["+".join(u["models"])]
            ok, why, calc = self.ex.admit(u["need_mb"], c["basis"], c["estimate_mb"])
            u["admit"] = {"ok": ok, "why": why, **calc}
            u["p90_ms"], u["p90_basis"] = c["p90_ms"], c["p90_basis"]
        return p

    def submit(self, sources, models, profile=None, profile_path=None, target_fps=None, cpu_tasks=None,
               task_specs=None):
        rid = "req-" + uuid.uuid4().hex[:10]
        request = {"sources": sorted(sources), "models": sorted(models), "target_fps": target_fps or {},
                   "cpu_tasks": sorted(cpu_tasks or [])}
        self.state.put_request(rid, dict(request, status="requested", at=self.clock()))
        self.emit("requested", rid, request=request)
        p = self.prepare(sources, models, profile, target_fps, cpu_tasks, task_specs, request_id=rid)
        ok, why, mode, basis, plan, snapshot = p["ok"], p["why"], p["mode"], p["basis"], p["plan"], p["snapshot"]
        self.emit("cluster_checked", rid, reason=why, detail={"ok": ok})
        if not ok:
            self.emit("fallback", rid, mode="local", basis="saved", reason=why)
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
        for u in plan["cpu_units"]:
            self.launch_cpu(rid, xid, mode, u)
        return rid

    # ------------------------------------------------------------ CPU 작업
    def launch_cpu(self, rid, xid, mode, u, replaces=None):
        uid = "cpu-%s-%s" % (u["task"], uuid.uuid4().hex[:6])
        rec = {"unit_id": uid, "kind": "cpu", "task": u["task"], "node": u["node"], "requests": u["requests"],
               "limits": u["limits"], "priority_class": u["priority_class"], "request_id": rid, "execution_id": xid,
               "status": "starting", "started_at": self.clock(), "replaces": replaces}
        if u["node"] == self.local or self.cpu_ex is None:
            rec.update(status="local", where="data_stream 기기 (런처가 띄우는 기존 방식)")
            self.state.put_unit(uid, rec)
            self.emit("applied", rid, execution_id=xid, mode=mode, reason="CPU 작업 %s → data_stream 기기" % u["task"],
                      detail={"unit_id": uid, "node": u["node"], "note": u.get("reason")})
            return uid
        try:
            self.cpu_ex.launch(rec)
        except Exception as e:  # noqa: BLE001
            rec.update(status="failed", reason=str(e)[:200])
            self.state.put_unit(uid, rec)
            self.emit("failed", rid, execution_id=xid, mode=mode, reason="CPU 작업을 못 띄움: %s" % e, detail={"unit_id": uid})
            return uid
        self.state.put_unit(uid, rec)
        self.emit("applied", rid, execution_id=xid, mode=mode, reason="CPU 작업 %s → %s" % (u["task"], u["node"]),
                  detail={"unit_id": uid, "node": u["node"], "requests": u["requests"], "limits": u["limits"],
                          "priority_class": u["priority_class"]})
        return uid

    def heartbeat_age(self, uid):
        try:
            raw = self.r.get("cpu_telemetry:" + uid)
            ts = json.loads(raw).get("ts") if raw else None
            return self.clock() - float(ts) if ts else None
        except Exception:  # noqa: BLE001
            return None

    def replace_cpu(self, u, reason, nodes):
        """CPU 작업만 다시 배치한다(GPU 는 옮기지 않는다). 그 노드를 빼고 placement 의 CPU 규칙으로."""
        self.cpu_ex.stop(u)
        self.state.update_unit(u["unit_id"], status="replaced", reason=reason, ended_at=self.clock())
        spec = (getattr(self, "_specs", {}) or {}).get(u["task"]) or self.cpu_specs([u["task"]])[0]
        out = {"cpu_units": [], "refused": []}
        placement.place_cpu(spec, [n for n in nodes if n["name"] != u["node"]] + [self.local_node_snapshot()],
                            out, self.local)
        if not out["cpu_units"]:
            return self.emit("failed", u["request_id"], execution_id=u["execution_id"], mode="k3s",
                             reason="%s — 옮길 노드 없음" % reason, detail={"unit_id": u["unit_id"]})
        new = out["cpu_units"][0]
        nid = self.launch_cpu(u["request_id"], u["execution_id"], "k3s", new, replaces=u["unit_id"])
        return self.emit("replanned", u["request_id"], execution_id=u["execution_id"], mode="k3s",
                         reason="%s → %s 에서 %s 로" % (reason, u["node"], new["node"]),
                         detail={"from": u["unit_id"], "to": nid, "from_node": u["node"], "to_node": new["node"]})

    def poll_cpu(self):
        units = {k: u for k, u in self.state.units().items() if u.get("kind") == "cpu"
                 and u.get("status") in ("starting", "running")}
        if not units or self.cpu_ex is None:
            return []
        events = []
        try:
            ok, why = self.cluster.check() if self.cluster is not None else (False, "k3s 백엔드 없음")
            nodes = self.cluster.nodes() if ok else []
        except Exception as e:  # noqa: BLE001
            ok, why, nodes = False, str(e), []
        if not ok:                                      # 파드는 노드에서 계속 돈다 — 상태만 모른다
            for uid, u in units.items():
                if not u.get("k3s_down_emitted"):
                    self.state.update_unit(uid, k3s_down_emitted=True)
                    events.append(self.emit("degraded", u["request_id"], execution_id=u["execution_id"], mode="local",
                                            basis="saved", reason="k3s 응답 없음 — 파이 작업 상태를 확인할 수 없음 (%s)" % why))
            return events
        by_name = {n["name"]: n for n in nodes}
        now = self.clock()
        for uid, u in units.items():
            if u.get("k3s_down_emitted"):
                self.state.update_unit(uid, k3s_down_emitted=False)
            node = by_name.get(u["node"])
            st = self.cpu_ex.status(u)
            age = self.heartbeat_age(uid)
            if node is None or not node.get("ready") or node.get("stale"):
                since = u.get("node_bad_since")
                if since is None:
                    self.state.update_unit(uid, node_bad_since=now)
                elif now - since >= NODE_BAD_GRACE_S:
                    events.append(self.replace_cpu(u, "노드 %s 빠짐 (Ready 아님 · stale %d초)" % (u["node"], now - since),
                                                   nodes))
                continue
            if u.get("node_bad_since") is not None:
                self.state.update_unit(uid, node_bad_since=None)
            if st.get("waiting") in ("CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull") or st.get("phase") == "Failed":
                events.append(self.replace_cpu(u, "파드 실패 (%s)" % (st.get("waiting") or st.get("phase")), nodes))
                continue
            if u["status"] == "starting":
                if st.get("ready") and age is not None and age <= CPU_HEARTBEAT_FRESH_S:
                    self.state.update_unit(uid, status="running", running_at=now)
                    events.append(self.emit("running", u["request_id"], execution_id=u["execution_id"], mode="k3s",
                                            detail={"unit_id": uid, "node": u["node"]}))
                elif now - u["started_at"] > ge.LOAD_TIMEOUT_S:
                    events.append(self.replace_cpu(u, "%d초 안에 하트비트 없음" % ge.LOAD_TIMEOUT_S, nodes))
            elif age is None or age > CPU_REPLACE_AFTER_S:
                events.append(self.replace_cpu(u, "하트비트 %s초 없음" % ("?" if age is None else int(age)), nodes))
        return events

    # ------------------------------------------------------------ 감시
    def poll(self):
        self.flush_outbox()
        events = self.poll_cpu()
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

    def cleanup_previous(self, reason="이전 오케스트레이터 실행이 남긴 것"):
        """새 run 을 시작할 때: 지난 실행이 남긴 starting · loaded GPU 단위를 정리한다. 살아 있는 고아는 멈추고(GPU 를
        두 번 쓰지 않게), 죽은 것은 stopped 로 적는다 — Executor.poll 이 그것을 다시 띄우지 않게."""
        n = 0
        for uid, u in self.ex.active_units().items():
            if self.ex._alive(uid, u["pid"]):
                self.ex.stop(uid, reason)
            else:
                self.state.update_unit(uid, status="stopped", reason=reason, ended_at=self.clock())
            n += 1
        for uid, u in self.state.units().items():
            if u.get("kind") == "cpu" and u.get("status") in ("starting", "running"):
                if self.cpu_ex is not None:
                    self.cpu_ex.stop(u)
                self.state.update_unit(uid, status="stopped", reason=reason, ended_at=self.clock())
                n += 1
        return n

    def stop(self, request_id, reason="요청"):
        n = 0
        for uid, u in self.state.units().items():
            if u.get("kind") == "cpu" and u.get("request_id") == request_id and u.get("status") in ("starting", "running"):
                if self.cpu_ex is not None:
                    self.cpu_ex.stop(u)
                self.state.update_unit(uid, status="stopped", reason=reason, ended_at=self.clock())
                self.emit("stopped", request_id, execution_id=u.get("execution_id"), reason=reason,
                          detail={"unit_id": uid})
                n += 1
        for uid, u in self.ex.active_units().items():
            if u.get("request_id") == request_id:
                self.ex.stop(uid, reason)
                self.emit("stopped", request_id, execution_id=u.get("execution_id"), reason=reason,
                          detail={"unit_id": uid})
                n += 1
        return n
