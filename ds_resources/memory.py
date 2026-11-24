"""자원 기억 — 노드 기본 자원 · 로봇 서비스 몫 · 모델 조합 사용량을 재고 기억하고 갱신한다 (core, 2026-11-24).

저장: state/resource_memory.json (git 에 올리지 않음). 원자적으로 쓴다(임시 파일 → 이름 바꾸기).
k3s 가 없거나 멈춰도 이 기억만으로 배치를 정한다(basis = saved).

노드 기본 자원  열쇠 = 노드 이름. 신원(boot_id · 커널 · OS)이 바뀌면 바로 다시 잰다. 그 밖에는
               recheck 주기로 다시 재는데, 다시 재서 같으면 주기를 두 배(최대 30일), 달라지면 1일로 되돌린다.
로봇 서비스 몫  노드 사용량 표본으로 p95 와 지수 이동 평균. 배치는 「가용 − p95 − 여유」만 쓴다.
모델 조합      kind "gpu"  (선택지 묶음 · 설정 해시 · 장치 종류) → 프로세스 GPU 메모리 최대치
               kind "cost" (선택지 묶음 · 설정 해시 · 장치 종류 · 해상도 구간) → 처리 시간 p50/p90 · fps
               처음에는 추정치(아는 값 합 × 1.2)로 쓰고, 표본이 200장 또는 60초 쌓이면 실측(measured)으로
               바꾼 뒤 지수 이동 평균으로 갱신한다. 설정 · 레지스트리가 바뀌면 해시가 달라 새 열쇠가 된다.
"""
import json
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

DAY = 86400.0
RECHECK_MIN_S = 1 * DAY
RECHECK_MAX_S = 30 * DAY
PROMOTE_FRAMES = 200
PROMOTE_SECONDS = 60.0
EWMA = 0.2
ESTIMATE_MARGIN = 1.2
MAX_COMBOS = 300
USAGE_WINDOW = 240                      # p95 를 낼 최근 표본 수 (노드마다, 메모리에만)


def res_bucket(w, h):
    """해상도 구간 — 긴 변 기준. 처리 시간이 해상도에 따라 갈린다(자원표 5장)."""
    long_side = max(int(w or 0), int(h or 0))
    for name, top in (("s", 640), ("m", 1280), ("l", 1920)):
        if long_side <= top:
            return name
    return "xl"


def combo_key(kind, models, cfg_hash, device_class, bucket=None):
    parts = [kind, "+".join(sorted(models)), str(cfg_hash), str(device_class)]
    if kind == "cost":
        parts.append(str(bucket))
    return "|".join(parts)


#: 잴 때마다 조금씩 다른 기본 자원 칸 — 이 비율 안이면 같은 것으로 본다(벤치마크)
VOLATILE = {"jpeg_720p_fps": 0.15}


def same_baseline(old, new):
    if not isinstance(old, dict) or not isinstance(new, dict) or set(old) != set(new):
        return False
    for k, v in new.items():
        o = old[k]
        tol = VOLATILE.get(k)
        if tol is not None and isinstance(o, (int, float)) and isinstance(v, (int, float)):
            if abs(v - o) > tol * max(abs(o), 1e-9):
                return False
        elif o != v:
            return False
    return True


def _p95(values):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))] if v else None


class ResourceMemory:
    def __init__(self, path, clock=time.time):
        self.path = Path(path)
        self.clock = clock
        self.lock = threading.RLock()
        self.data = {"version": 1, "nodes": {}, "combos": {}}
        self.usage = {}                   # 노드 → deque[(cpu_pct, mem_mb)] (저장하지 않음)
        self.dirty = False
        self.load()

    # ------------------------------------------------------------ 저장
    def load(self):
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(d, dict) and d.get("version") == 1:
                d.setdefault("nodes", {})
                d.setdefault("combos", {})
                self.data = d
        except (OSError, ValueError):
            pass

    def save(self, force=False):
        with self.lock:
            if not (self.dirty or force):
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".mem-", dir=str(self.path.parent))
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=1, sort_keys=True)
            os.replace(tmp, self.path)
            self.dirty = False
            return True

    # ------------------------------------------------------------ 노드 기본 자원
    def node(self, name):
        with self.lock:
            return json.loads(json.dumps(self.data["nodes"].get(name))) if name in self.data["nodes"] else None

    def baseline_due(self, name, identity):
        """다시 재야 하나: 처음 · 신원(boot_id · 커널 · OS) 변화 · recheck 시각 지남"""
        with self.lock:
            n = self.data["nodes"].get(name)
            if n is None or n.get("identity") != identity:
                return True
            return self.clock() >= n.get("next_check", 0)

    def record_baseline(self, name, identity, baseline):
        """잰 기본 자원을 기억한다. 바뀌었는지(True/False)를 돌려준다."""
        now = self.clock()
        with self.lock:
            n = self.data["nodes"].get(name) or {}
            same = same_baseline(n.get("baseline"), baseline)
            interval = min(RECHECK_MAX_S, max(RECHECK_MIN_S, n.get("interval", RECHECK_MIN_S) * 2)) if same \
                else RECHECK_MIN_S
            n.update({"identity": identity, "baseline": baseline, "measured_at": now,
                      "interval": interval, "next_check": now + interval})
            n.setdefault("first_seen", now)
            self.data["nodes"][name] = n
            self.dirty = True
            return not same

    # ------------------------------------------------------------ 로봇 서비스 몫 (사용량 p95)
    def update_usage(self, name, cpu_pct, mem_mb):
        with self.lock:
            q = self.usage.setdefault(name, deque(maxlen=USAGE_WINDOW))
            q.append((float(cpu_pct or 0.0), float(mem_mb or 0.0)))
            n = self.data["nodes"].setdefault(name, {})
            u = n.setdefault("usage", {})
            for key, val in (("cpu_ewma", cpu_pct), ("mem_ewma_mb", mem_mb)):
                if val is not None:
                    u[key] = val if key not in u else (1 - EWMA) * u[key] + EWMA * float(val)
            if len(q) >= 10:                              # 표본이 조금은 있어야 p95 가 뜻이 있다
                u["cpu_p95"] = _p95([c for c, _ in q])
                u["mem_p95_mb"] = _p95([m for _, m in q])
            u["samples"] = int(u.get("samples", 0)) + 1
            u["updated_at"] = self.clock()
            self.dirty = True

    def service_share(self, name):
        """배치가 그 노드에서 뺄 몫 (cpu_pct, mem_mb) — p95 가 없으면 지수 이동 평균, 그것도 없으면 None"""
        with self.lock:
            u = (self.data["nodes"].get(name) or {}).get("usage") or {}
            cpu = u.get("cpu_p95", u.get("cpu_ewma"))
            mem = u.get("mem_p95_mb", u.get("mem_ewma_mb"))
            return (cpu, mem) if cpu is not None or mem is not None else None

    # ------------------------------------------------------------ 모델 조합
    def combo(self, key, estimate=None):
        """기억한 값(measured · learning) 또는 추정치(estimate × 1.2). (값 사전, basis) — basis: measured · learning · estimate"""
        with self.lock:
            c = self.data["combos"].get(key)
            if c is not None:
                c["used_at"] = self.clock()
                return dict(c["value"]), c["state"]
        if estimate is None:
            return None, None
        return {k: (v * ESTIMATE_MARGIN if isinstance(v, (int, float)) else v) for k, v in estimate.items()}, "estimate"

    def record_combo(self, key, value, frames=0, seconds=0.0):
        """실행 중 잰 값을 더한다. 200장 또는 60초 전에는 learning(그 구간 평균), 넘으면 measured 로 바꾼 뒤 EWMA."""
        now = self.clock()
        with self.lock:
            c = self.data["combos"].get(key)
            if c is None:
                c = {"state": "learning", "value": dict(value), "frames": 0, "seconds": 0.0, "first": now}
                self.data["combos"][key] = c
            elif c["state"] == "learning":
                n0, n1 = max(c["frames"], 1), max(frames, 1)
                for k, v in value.items():
                    if isinstance(v, (int, float)) and isinstance(c["value"].get(k), (int, float)):
                        c["value"][k] = (c["value"][k] * n0 + v * n1) / (n0 + n1)
                    else:
                        c["value"][k] = v
            else:
                for k, v in value.items():
                    old = c["value"].get(k)
                    if k.endswith("_peak") and isinstance(old, (int, float)):
                        c["value"][k] = max(old * (1 - EWMA), v) if v is not None else old  # 최대치는 천천히만 내려간다
                    elif isinstance(v, (int, float)) and isinstance(old, (int, float)):
                        c["value"][k] = (1 - EWMA) * old + EWMA * v
                    else:
                        c["value"][k] = v
            c["frames"] += int(frames)
            c["seconds"] += float(seconds)
            if c["state"] == "learning" and (c["frames"] >= PROMOTE_FRAMES or c["seconds"] >= PROMOTE_SECONDS):
                c["state"] = "measured"
            c["updated_at"] = c["used_at"] = now
            self._prune()
            self.dirty = True
            return c["state"]

    def _prune(self):
        combos = self.data["combos"]
        if len(combos) <= MAX_COMBOS:
            return
        for k in sorted(combos, key=lambda k: combos[k].get("used_at", 0))[:len(combos) - MAX_COMBOS]:
            del combos[k]


class PollSchedule:
    """사용량을 보는 주기 — 작업 없음 60초 · 배치 중 10초 · 부하 80% 넘음 5초.
    값이 change_pct(5%) 넘게 안 변하면 주기를 두 배(최대 60초), 크게 변하면 그 상태의 기본 주기로 되돌린다."""

    BASE = {"idle": 60.0, "active": 10.0, "hot": 5.0}
    MAX = 60.0

    def __init__(self, change_pct=5.0, hot_pct=80.0):
        self.change, self.hot = change_pct, hot_pct
        self.interval, self.last, self.mode = self.BASE["idle"], None, "idle"

    def next(self, active, values):
        """values: 노드 사용량 % 목록 (cpu · mem · gpu). 다음까지 기다릴 초를 돌려준다."""
        vals = [v for v in values if v is not None]
        mode = "hot" if any(v >= self.hot for v in vals) else ("active" if active else "idle")
        changed = self.last is None or len(vals) != len(self.last) or \
            any(abs(a - b) > self.change for a, b in zip(vals, self.last))
        if mode != self.mode or changed:
            self.interval = self.BASE[mode]
        else:
            self.interval = min(self.MAX, self.interval * 2)
        self.mode, self.last = mode, vals
        return self.interval
