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
"""
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from ds_resources import probe
from ds_resources.state import StateFile

REPO = Path(__file__).resolve().parent.parent
MARGIN = 1.2
SPARE_MB = 300.0
LOAD_TIMEOUT_S = 180.0
MAX_RESTARTS = 2
TELEMETRY_PREFIX = "vision_telemetry:"


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
            need.add("depth")
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
        self.python = python or str(REPO / "venv" / "bin" / "python")
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

    def admit(self, need_mb, basis="estimate", estimate=None):
        """(들일 수 있나, 이유, 계산 근거). 계산 근거의 mode: normal | exclusive. 지금 nvidia-smi 를 읽는다."""
        free, err = self.live_free_mb()
        if free is None:
            return False, "GPU 상태를 지금 확인할 수 없어 들이지 않음 (%s)" % err, {}
        active = self.active_units()
        if any(u.get("exclusive") for u in active.values()):
            return False, "단독 측정 실행 중 — 실측이 끝날 때까지 다른 GPU 작업을 들이지 않음", {}
        pend = self.pending_mb()
        room = free - pend - SPARE_MB
        calc = {"free_mb": round(free), "pending_mb": round(pend), "spare_mb": SPARE_MB,
                "room_mb": round(room), "need_mb": round(need_mb), "basis": basis, "mode": "normal"}
        if need_mb <= room:
            return True, "", calc
        est = need_mb / MARGIN if estimate is None else float(estimate)
        if basis == "estimate" and not active and est <= free - SPARE_MB:
            calc["mode"] = "exclusive"
            return True, "실측이 없어 단독 측정 실행으로 들임 (추정 %d MB, 여유 %d MB)" % (est, free), calc
        return False, "GPU 메모리 부족: 필요 %d MB > 여유 %d MB (지금 여유 %d − 적재 중 예약 %d − 남김 %d)" % (
            need_mb, room, free, pend, SPARE_MB), calc

    # ------------------------------------------------------------ 띄우기
    def command(self, unit):
        cmd = [self.python, str(REPO / "vision" / "vision_infer.py"), "--devices", *unit["devices"],
               "--models", *unit["models"]]
        if unit.get("profile_path"):
            cmd += ["--profile", unit["profile_path"]]
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
        rec = dict(unit, unit_id=uid, kind="gpu", pid=p.pid, status="starting", reserved_mb=round(need_mb),
                   admit=calc or {}, exclusive=(calc or {}).get("mode") == "exclusive",
                   started_at=self.clock(), restarts=unit.get("restarts", 0), log=str(log_path))
        self.state.put_unit(uid, rec)
        return uid

    # ------------------------------------------------------------ 감시
    def _alive(self, uid, pid):
        p = self.procs.get(uid)
        if p is not None:
            return p.poll() is None
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
                    nu = {k: u[k] for k in ("unit_id", "devices", "models", "profile_path", "request_id",
                                            "execution_id") if k in u}
                    nu["restarts"] = u.get("restarts", 0) + 1
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
        try:
            os.killpg(u["pid"], signal.SIGINT)              # vision_infer 는 SIGINT 에 지금 장까지 저장하고 끝낸다
        except OSError:
            pass
        self.state.update_unit(uid, status=status, reason=reason, ended_at=self.clock())
        self.procs.pop(uid, None)
        return True
