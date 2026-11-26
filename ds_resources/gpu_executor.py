"""GPU Executor — GPU 비전 작업을 들이고 · 예약하고 · 띄우고 · 감시한다 (core, k3s 없이 돈다, 2026-11-24).

GPU 는 k3s 스케줄링 자원이 아니다. k3s 는 WSL GPU 의 실제 사용량을 모른다. 그래서 이 Executor 가 맡는다.

필요 메모리   기준 × MARGIN.
              기준 = 믿을 만한 실측(measured 상태 · stale 아님)이 있으면 그 최대치. 근거가 torch+ctx 이면 torch 밖에서
                     GPU 를 쓰는 모델(onnxruntime — CLIP 등)의 레지스트리 몫을 더한다(torch 예약에 안 잡힌다).
                     실측이 없으면 레지스트리 추정치(프로세스 기본값 + 선택지 + 고른 단서 · 기능의 모델).
              추정치는 보수적이라 실제로 도는 조합보다 클 수 있다(targeted 기본 5370 MB × 1.2 > 6 GB).
들이는 조건  필요 메모리 ≤ 지금 nvidia-smi 여유 − 아직 적재 중인 예약 합 − SPARE_MB  → normal
              안 되면, 실측이 없고 GPU 에 다른 비전 작업이 없으며 추정치 ≤ 여유 − SPARE_MB 일 때만 단독 측정 실행
              (exclusive) 으로 들인다 — 그래야 실측이 생긴다. 단독 측정 중에는 다른 GPU 작업을 들이지 않는다.
              실행 직전에 다시 잰다(기억한 값만으로 들이지 않는다).
예약         띄우면 starting 으로 적는다. 그 프로세스의 vision_telemetry 가 처리한 장과 함께 나오면 loaded —
              그때부터는 기기 사용량에 이미 들어 있으므로 여유 계산에서 빼지 않는다(두 번 세지 않게).
              loaded = 그 프로세스의 비전 보고가 나왔다(모델 적재가 끝나 루프가 돈다) — 처리한 장이 0 이어도 된다.
              영상이 아직 안 오는 소스(로봇이 꺼짐)는 실패가 아니라 입력 대기다 — running 판정은 컨트롤러 몫.
              LOAD_TIMEOUT_S 안에 보고가 없으면 failed (모델 적재 실패).
감시         프로세스가 죽으면 MAX_RESTARTS 번까지 WSL 안에서 다시 띄운다. 넘으면 멈춤(stopped, 이유).
              GPU 작업은 다른 노드(파이)로 옮기지 않는다.
              다시 띄우기는 단위 기록의 spec(배치표가 준 실행 조건 전체 — devices · models · profile_path · fps_cap ·
              gpu_util · request_id · execution_id …)으로 한다(M11 R3). 실행 중 상태(pid · status …)와 섞지 않으므로
              칸이 늘어도 복사 목록을 고칠 일이 없다. spec 이 없는 옛 기록은 남은 칸(fps_cap · gpu_util 포함)으로 되살린다.
              exclusive 는 admit.mode 에서 다시 계산된다.
들이기 거절  계산 근거(calc)에 이유 코드 code 를 단다 — gpu_mem · gpu_exclusive · gpu_unreadable (streams.REASON_CODES).
우선순위    (2026-11-26, 「GPU 는 오케스트레이터가 우선순위로 판단」) 단위는 rank(placement.gpu_rank — safety 0 ·
              mission 1 · best_effort 2, 기능 없이 온 옛 단위는 mission)를 갖는다. admit 에 rank 를 주면, 메모리가
              모자랄 때 **더 낮은** 우선순위의 starting · loaded 단위를 비우면 들어가는지 계산해 calc["evict"] 로 제안한다
              (낮은 우선순위 → 비우면 풀리는 메모리가 큰 것 → unit_id 순, 들어갈 만큼만 — 모자라면 아무것도 제안하지 않는다).
              같거나 높은 우선순위는 비우지 않는다. admit 은 읽기만 한다 — 실제로 멈추는 것은 preempt() 다.
밀려남      preempt() 는 단위를 멈추고(status preempted) 미룬 목록(state 의 gpu_deferred)에 넣는다 — 메모리가 비면
              컨트롤러가 우선순위순으로 다시 들인다. 밀려난 단위는 READMIT_HOLD_S 동안 다시 들이지 않는다(되풀이 방지).
"""
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from ds_resources import placement, probe
from ds_resources.state import StateFile

REPO = Path(__file__).resolve().parent.parent
MARGIN = 1.2
SPARE_MB = 300.0
LOAD_TIMEOUT_S = 180.0
STOP_WAITS = (6.0, 2.0, 1.0)  # 멈춤: SIGINT 뒤 6초 · SIGTERM 뒤 2초 · SIGKILL 뒤 1초 기다린다 — GPU 메모리를 돌려받은 뒤 돌아간다
                              # (기다리지 않으면 바로 다음 요청이 아직 반납 전 메모리를 보고 거절됐다, 2026-11-24 런처 E2E)
MAX_RESTARTS = 2
TELEMETRY_PREFIX = "vision_telemetry:"
#: 밀려난(preempted) 단위를 다시 들이기 전에 기다리는 초 — 높은 우선순위 일이 잠깐 비운 틈에 다시 들어왔다가
#: 곧바로 또 밀리는 되풀이(flapping)를 막는다. 모델 적재가 수십 초 걸리므로 그보다 길게 둔다
READMIT_HOLD_S = 60.0
#: 미룬 단위를 다시 들여 볼 간격 — GPU 단위가 멈추거나 끝난 poll 에서는 바로, 그 밖에는 이 간격마다
#: (다른 프로세스가 GPU 를 놓은 것은 이벤트가 없으니 주기적으로 본다)
DEFERRED_RETRY_S = 10.0
#: 단위 기록 중 실행 중에 생기는 칸 — spec 이 없는 옛 기록을 되살릴 때 이것을 뺀 나머지를 실행 조건으로 본다
RUNTIME_KEYS = ("unit_id", "kind", "pid", "status", "reserved_mb", "admit", "exclusive", "started_at", "restarts", "log",
                "loaded_at", "ended_at", "reason", "running_emitted", "running_at", "sources_up", "low_since",
                "degraded_emitted", "spec", "preempted_at", "preempted_by", "resumed_from")


def unit_rank(u):
    """단위의 우선순위 (작을수록 높다). rank · criticality 칸이 없는 옛 단위는 mission"""
    return placement.gpu_rank(u or {})


def _catalog():
    sys.path.insert(0, str(REPO / "vision"))
    import cue_catalog  # noqa: E402  (core — vision 폴더)
    return cue_catalog


def estimate_mb(models, profile=None, cfg=None):
    """레지스트리 추정치 (MB) 와 근거 목록. targeted 는 프로필을 풀어 실제로 적재할 모델을 센다."""
    cc = _catalog()
    need = set()
    if "targeted" in models:
        tcfg = ((cfg or {}).get("models") or {}).get("targeted") or {}
        resolved = cc.resolve_profile(profile or {}, cfg or {}, list(models))
        need |= cc.needed_models(resolved.get("targeted"), tcfg)      # vision_infer 와 같게 targeted 부분만
        if (resolved.get("features") or {}).get("reid"):
            need.add("reid")
        if (resolved.get("targeted") or {}).get("distance"):
            need.add(cc.depth_key((resolved.get("targeted") or {}).get("cue_models")))   # 근거 표에서 고른 거리 모델
        # 같은 묶음의 선택지 · 새 모델이 이미 적재하는 정체는 근거 쪽에서 빼고 센다 (work/1 — 같은 일은 한 번).
        # 겹치지 않으면 아무것도 빠지지 않는다 — 옛 어림 그대로
        from ds_resources import workgraph as wg
        need -= wg.shared_need(cc._REG, list(models), need)
    elif (profile or {}).get("features", {}).get("reid"):
        need.add("reid")
    mb = cc.gpu_mb(list(models), sorted(need))
    return float(mb), sorted(need)


def untracked_mb(need):
    """torch 예약에 안 잡히는 모델(onnxruntime-gpu)의 레지스트리 GPU 몫 합"""
    reg = _catalog()._REG["models"]
    return float(sum(int(reg.get(m, {}).get("gpu_mb") or 0) for m in need
                     if "onnxruntime-gpu" in (reg.get(m, {}).get("packages") or [])))


def required_mb(estimate, measured=None, untracked=0.0):
    """(필요 MB, 근거). measured: (값 사전, 상태, stale 여부) — measured 상태이고 stale 아닐 때만 실측을 쓴다."""
    if measured:
        value, state, stale = measured
        peak = (value or {}).get("gpu_mb_peak")
        if state == "measured" and not stale and peak is not None:
            extra = untracked if (value or {}).get("gpu_basis") == "torch+ctx" else 0.0
            return (float(peak) + extra) * MARGIN, "measured"
    return float(estimate) * MARGIN, "estimate"


class GpuExecutor:
    def __init__(self, state=None, redis_client=None, gpus=probe.gpus, popen=subprocess.Popen, clock=time.time,
                 python=None, log=print):
        self.state = state or StateFile(REPO / "state" / "orchestrator_state.json")
        self.r, self.gpus, self.popen, self.clock, self.log = redis_client, gpus, popen, clock, log
        self.python = python or os.environ.get("DS_PYTHON") or sys.executable   # D1: venv 를 가정하지 않는다
        self.procs = {}                                   # unit_id → Popen (이 프로세스가 띄운 것)

    # ------------------------------------------------------------ 들이기
    def live_free_mb(self):
        g = self.gpus()
        if not g or g[0].get("mem_total_mb") is None or g[0].get("mem_used_mb") is None:
            return None, "nvidia-smi 를 못 읽음"
        return float(g[0]["mem_total_mb"]) - float(g[0]["mem_used_mb"]), None

    def pending_mb(self):
        return sum(u["reserved_mb"] for u in self.state.units().values() if u.get("status") == "starting")

    def active_units(self):
        return {k: u for k, u in self.state.units().items()
                if u.get("kind") == "gpu" and u.get("status") in ("starting", "loaded")}

    def admit(self, need_mb, basis="estimate", estimate=None, rank=None):
        """(들일 수 있나, 이유, 계산 근거). 계산 근거의 mode: normal | exclusive. 지금 nvidia-smi 를 읽는다.

        rank 를 주면(우선순위로 판단할 때) 거절인 경우 **더 낮은** 우선순위 단위를 비우면 들어가는지 계산해
        calc 에 evict(비울 unit_id 목록) · evict_mode · evict_mb 를 단다. 읽기만 한다 — 멈추지 않는다."""
        free, err = self.live_free_mb()
        if free is None:
            return False, "GPU 상태를 지금 확인할 수 없어 들이지 않음 (%s)" % err, {"code": "gpu_unreadable"}
        active = self.active_units()
        est = need_mb / MARGIN if estimate is None else float(estimate)
        if any(u.get("exclusive") for u in active.values()):
            calc = {"code": "gpu_exclusive"}
            if rank is not None:
                self._propose(calc, active, free, self.pending_mb(), need_mb, est, basis, rank)
            return False, "단독 측정 실행 중 — 실측이 끝날 때까지 다른 GPU 작업을 들이지 않음", calc
        pend = self.pending_mb()
        room = free - pend - SPARE_MB
        calc = {"free_mb": round(free), "pending_mb": round(pend), "spare_mb": SPARE_MB,
                "room_mb": round(room), "need_mb": round(need_mb), "basis": basis, "mode": "normal"}
        if need_mb <= room:
            return True, "", calc
        if basis == "estimate" and not active and est <= free - SPARE_MB:
            calc["mode"] = "exclusive"
            return True, "실측이 없어 단독 측정 실행으로 들임 (추정 %d MB, 여유 %d MB)" % (est, free), calc
        calc["code"] = "gpu_mem"
        if rank is not None:
            self._propose(calc, active, free, pend, need_mb, est, basis, rank)
        return False, "GPU 메모리 부족: 필요 %d MB > 여유 %d MB (지금 여유 %d − 적재 중 예약 %d − 남김 %d)" % (
            need_mb, room, free, pend, SPARE_MB), calc

    @staticmethod
    def freeable_mb(u):
        """그 단위를 멈추면 풀린다고 볼 메모리 — 적재 중(starting)은 예약 그대로(여유 계산에서 빼던 몫),
        적재 끝(loaded)은 예약 ÷ MARGIN(이미 기기 사용량에 든 실제 몫 — 여유분은 빼고 보수적으로)"""
        mb = float(u.get("reserved_mb") or 0)
        return mb if u.get("status") == "starting" else mb / MARGIN

    def _propose(self, calc, active, free, pend, need_mb, est, basis, rank):
        """더 낮은 우선순위 단위를 비우면 들어가는가 — 들어가면 calc 에 evict 를 단다 (읽기만).
        고르는 순서: 낮은 우선순위 → 풀리는 메모리가 큰 것 → unit_id (되풀이 가능하게). 들어갈 만큼만 고르고,
        모두 비워도 안 들어가면 아무것도 제안하지 않는다(헛되이 멈추지 않는다)."""
        victims = sorted((k for k, u in active.items() if unit_rank(u) > rank),
                         key=lambda k: (-unit_rank(active[k]), -self.freeable_mb(active[k]), k))
        if not victims:
            return
        f, p, chosen = float(free), float(pend), []
        for uid in victims:
            u = active[uid]
            chosen.append(uid)
            if u.get("status") == "starting":
                p -= float(u.get("reserved_mb") or 0)
            else:
                f += self.freeable_mb(u)
            rest = [k for k in active if k not in chosen]
            if any(active[k].get("exclusive") for k in rest):
                continue                                    # 남는 단독 측정이 있으면 아직 못 들인다
            if need_mb <= f - p - SPARE_MB:
                calc.update(evict=list(chosen), evict_mode="normal", evict_mb=round(f - free + pend - p))
                return
        if basis == "estimate" and len(chosen) == len(active) and est <= f - SPARE_MB:
            calc.update(evict=list(chosen), evict_mode="exclusive", evict_mb=round(f - free + pend - p))

    # ------------------------------------------------------------ 우선순위 — 밀어내기 · 미루기
    def preempt(self, uids, reason, by=None):
        """단위들을 멈추고(status preempted) 미룬 목록에 넣는다 — 메모리가 비면 다시 들인다(READMIT_HOLD_S 뒤).
        멈춘 단위 기록들을 돌려준다(멈추기 전 모습 — request_id · execution_id 로 이벤트를 쓴다)."""
        now = self.clock()
        out = []
        for uid in uids:
            u = self.state.units().get(uid)
            if not u or u.get("kind") != "gpu" or u.get("status") not in ("starting", "loaded"):
                continue
            self.stop(uid, reason, status="preempted")
            self.state.update_unit(uid, preempted_at=now, preempted_by=by)
            self.defer(uid, {"unit_id": uid, "unit": "gpu:%s" % "+".join(u.get("models") or []),
                             "spec": self.relaunch_spec(u), "need_mb": float(u.get("reserved_mb") or 0),
                             "basis": (u.get("admit") or {}).get("basis") or "estimate", "estimate_mb": None,
                             "rank": unit_rank(u), "trigger": u.get("trigger"),
                             "request_id": u.get("request_id"), "execution_id": u.get("execution_id"),
                             "since": now, "hold_until": now + READMIT_HOLD_S, "reason": reason,
                             "code": "gpu_preempted"})
            out.append(u)
        return out

    def deferred(self):
        """미룬 단위들 {열쇠: 항목} — 항목: unit_id · unit · spec · need_mb · basis · estimate_mb · rank · trigger ·
        request_id · execution_id · since · hold_until · reason · code (gpu_preempted | gpu_deferred)"""
        with self.state.lock:
            if not self.state.writer:
                self.state.load()
            return json.loads(json.dumps(self.state.data.get("gpu_deferred") or {}))

    def defer(self, key, entry):
        with self.state.lock:
            self.state.data.setdefault("gpu_deferred", {})[key] = json.loads(json.dumps(entry))
            self.state._save()

    def undefer(self, key):
        with self.state.lock:
            if (self.state.data.get("gpu_deferred") or {}).pop(key, None) is not None:
                self.state._save()

    # ------------------------------------------------------------ 띄우기
    def command(self, unit):
        cmd = [self.python, str(REPO / "vision" / "vision_infer.py"), "--devices", *unit["devices"],
               "--models", *unit["models"]]
        if unit.get("profile_path"):
            cmd += ["--profile", unit["profile_path"]]
        if unit.get("function_params"):                     # 추적 입력 · 재식별 범위 (work/1) — 프로필 칸을 덮는다
            cmd += ["--function-params", json.dumps(unit["function_params"], ensure_ascii=False, sort_keys=True)]
        if unit.get("fps_cap"):                             # 배치표의 기기별 fps 상한을 실제로 건다
            cmd += ["--max-fps", *["%s=%g" % (d, f) for d, f in sorted(unit["fps_cap"].items())]]
        return cmd

    def launch(self, unit, need_mb, calc=None):
        """들인 뒤에 부른다. 예약을 적고 프로세스를 띄운다."""
        uid = unit.get("unit_id") or "gpu-" + uuid.uuid4().hex[:8]
        env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        log_path = REPO / "state" / "logs" / ("%s.log" % uid)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as lf:
            p = self.popen(self.command(unit), cwd=str(REPO), env=env, stdout=lf, stderr=subprocess.STDOUT,
                           start_new_session=True)
        self.procs[uid] = p
        spec = json.loads(json.dumps(unit.get("spec") or {k: v for k, v in unit.items() if k not in RUNTIME_KEYS}))
        rec = dict(unit, unit_id=uid, kind="gpu", pid=p.pid, status="starting", reserved_mb=round(need_mb),
                   admit=calc or {}, exclusive=(calc or {}).get("mode") == "exclusive",
                   started_at=self.clock(), restarts=unit.get("restarts", 0), log=str(log_path), spec=spec)
        self.state.put_unit(uid, rec)
        return uid

    @staticmethod
    def relaunch_spec(u):
        """다시 띄울 실행 조건 — spec 이 있으면 그대로, 없으면(옛 기록) 실행 중 칸을 뺀 나머지(fps_cap · gpu_util 포함)"""
        spec = u.get("spec")
        if not isinstance(spec, dict):
            spec = {k: v for k, v in u.items() if k not in RUNTIME_KEYS}
        return json.loads(json.dumps(spec))

    # ------------------------------------------------------------ 감시
    def _alive(self, uid, pid):
        p = self.procs.get(uid)
        if p is not None:
            return p.poll() is None                         # 이 프로세스가 띄운 것 — 좀비까지 거둔다
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _telemetry(self, pid):
        if self.r is None:
            return None
        try:
            for k in self.r.scan_iter(TELEMETRY_PREFIX + "*:%d" % pid, count=100):
                raw = self.r.get(k)
                return json.loads(raw) if raw else None
        except Exception:  # noqa: BLE001
            return None
        return None

    def poll(self):
        """상태 전이: starting → loaded | failed, loaded 인데 죽음 → 다시 띄움 | stopped. 바뀐 것 목록을 돌려준다."""
        changes = []
        now = self.clock()
        for uid, u in list(self.state.units().items()):
            if u.get("kind") != "gpu" or u.get("status") not in ("starting", "loaded"):
                continue
            alive = self._alive(uid, u["pid"])
            if not alive:
                if u.get("restarts", 0) < MAX_RESTARTS:
                    self.log("%s 프로세스가 끝남 — WSL 안에서 다시 띄움 (%d/%d)" % (uid, u.get("restarts", 0) + 1, MAX_RESTARTS))
                    nu = self.relaunch_spec(u)
                    nu.update(unit_id=uid, restarts=u.get("restarts", 0) + 1)
                    self.launch(nu, u["reserved_mb"], u.get("admit"))
                    changes.append((uid, "restarted"))
                else:
                    self.state.update_unit(uid, status="stopped", reason="다시 띄우기 %d번 넘음" % MAX_RESTARTS,
                                           ended_at=now)
                    changes.append((uid, "stopped"))
                continue
            if u["status"] == "starting":
                tel = self._telemetry(u["pid"])
                if tel:
                    self.state.update_unit(uid, status="loaded", loaded_at=now)
                    changes.append((uid, "loaded"))
                elif now - u["started_at"] > LOAD_TIMEOUT_S:
                    self.stop(uid, "%d초 안에 비전 보고 없음 (모델 적재 실패)" % LOAD_TIMEOUT_S, status="failed")
                    changes.append((uid, "failed"))
        return changes

    def stop(self, uid, reason="요청", status="stopped"):
        u = self.state.units().get(uid)
        if not u:
            return False
        pid = u["pid"]
        for sig, wait in zip((signal.SIGINT, signal.SIGTERM, signal.SIGKILL), STOP_WAITS):
            try:
                os.killpg(pid, sig)                         # vision_infer 는 SIGINT 에 지금 장까지 저장하고 끝낸다
            except OSError:
                break                                       # 이미 없다
            t0 = time.time()
            while time.time() - t0 < wait and self._alive(uid, pid):
                time.sleep(0.1)
            if not self._alive(uid, pid):
                break
        self.state.update_unit(uid, status=status, reason=reason, ended_at=self.clock())
        self.procs.pop(uid, None)
        return True
