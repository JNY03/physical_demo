"""로컬 자원 기록기 — data_stream 기기 자신을 재고 기억하고 node_resource_stream 에 쓴다 (core, 2026-11-24).

데이터 서버(server_stream_multi_source.py) 안에서 스레드로 돈다. k3s 가 없어도 돈다.
  * 처음 · 재부팅 · 커널/OS 변화 · recheck 시각에 기본 자원을 잰다 (JPEG 벤치마크 2초 포함)
  * 사용량(CPU · 메모리 · GPU)을 PollSchedule 주기로 재서 로봇 서비스 몫 기억과 기록 억제를 거쳐 스트림에 쓴다
  * 비전 프로세스가 남긴 vision_telemetry:<프로세스> 를 읽어 모델 조합 사용량을 기억한다
이 기기(로컬 노드)의 기록은 이 기록기만 쓴다. k3s 쪽 감시는 다른 노드만 쓴다.
"""
import json
import threading
import time
from pathlib import Path

from ds_resources import memory as mem_mod
from ds_resources import probe
from ds_resources.streams import NodeChangeFilter, StreamWriter

TELEMETRY_PREFIX = "vision_telemetry:"
#: torch 예약량에 잡히지 않는 CUDA 컨텍스트 몫 — vision/model_registry.yaml gpu_context_mb 가 이긴다
GPU_CONTEXT_MB = 150.0


def registry_context_mb(path=Path(__file__).resolve().parent.parent / "vision" / "model_registry.yaml"):
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return float((yaml.safe_load(f) or {}).get("gpu_context_mb", GPU_CONTEXT_MB))
    except Exception:  # noqa: BLE001
        return GPU_CONTEXT_MB


def process_gpu_mb(g, context_mb=GPU_CONTEXT_MB):
    """프로세스 GPU 메모리 실측 (MB, 근거). 프로세스별 값이 있으면 그것(process), 없으면 torch 예약 + 컨텍스트(torch+ctx).
    torch 만으로는 CUDA 컨텍스트 · onnxruntime 몫이 빠져 실제보다 작다 — 그대로 쓰면 배치가 OOM 을 낸다."""
    g = g or {}
    if g.get("proc_mb") is not None:
        return float(g["proc_mb"]), "process"
    if g.get("torch_peak_mb") is not None:
        return float(g["torch_peak_mb"]) + context_mb, "torch+ctx"
    return None, None
DEFAULT_MEMORY = Path(__file__).resolve().parent.parent / "state" / "resource_memory.json"


def telemetry_records(tel, context_mb=GPU_CONTEXT_MB):
    """vision_telemetry 한 건 → [(combo 열쇠, 값, 장 수, 초)]"""
    models = tel.get("models") or []
    cfg_hash, dev = tel.get("cfg_hash"), tel.get("device_class")
    window = float(tel.get("window_s") or 0.0)
    streams = tel.get("streams") or {}
    out = []
    frames_all = sum(int(s.get("frames") or 0) for s in streams.values())
    peak, basis = process_gpu_mb(tel.get("gpu"), context_mb)
    if peak is not None and frames_all:
        out.append((mem_mod.combo_key("gpu", models, cfg_hash, dev),
                    {"gpu_mb_peak": peak, "gpu_basis": basis, "streams": len(streams)}, frames_all, window))
    for s in streams.values():
        n = int(s.get("frames") or 0)
        if not n or s.get("infer_ms_p50") is None:
            continue
        out.append((mem_mod.combo_key("cost", models, cfg_hash, dev, mem_mod.res_bucket(s.get("w"), s.get("h"))),
                    {"infer_ms_p50": float(s["infer_ms_p50"]), "infer_ms_p90": float(s.get("infer_ms_p90") or s["infer_ms_p50"]),
                     "fps": float(s.get("fps") or 0.0)}, n, window))
    return out


class LocalRecorder:
    def __init__(self, redis_client, memory=None, node=None, role="gpu", log=print, bench=True):
        self.r = redis_client
        self.mem = memory or mem_mod.ResourceMemory(DEFAULT_MEMORY)
        self.node = node or probe.node_name()
        self.role, self.log, self.bench = role, log, bench
        self.filter = NodeChangeFilter()
        self.writer = StreamWriter(redis_client, maxlen=2000)
        self.sched = mem_mod.PollSchedule()
        self.usage = probe.UsageProbe()
        self.seen = {}                                # 프로세스 → 마지막으로 반영한 seq
        self.stop = threading.Event()
        self.thread = None
        self.last_error = None
        self.context_mb = registry_context_mb()

    # ------------------------------------------------------------ 한 번
    def check_baseline(self):
        ident = probe.identity()
        if self.mem.baseline_due(self.node, ident):
            changed = self.mem.record_baseline(self.node, ident, probe.baseline(bench=self.bench))
            self.log("자원 기억: %s 기본 자원을 %s" % (self.node, "새로 잼" if changed else "다시 잼 (같음)"))

    def read_telemetry(self):
        active = False
        try:
            keys = list(self.r.scan_iter(TELEMETRY_PREFIX + "*", count=100))
        except Exception as e:  # noqa: BLE001
            self.last_error = "telemetry: %s" % e
            return False
        for k in keys:
            try:
                raw = self.r.get(k)
                tel = json.loads(raw) if raw else None
            except Exception:  # noqa: BLE001
                continue
            if not tel:
                continue
            active = True
            proc = k.decode() if isinstance(k, bytes) else str(k)
            if self.seen.get(proc) == tel.get("seq"):
                continue
            self.seen[proc] = tel.get("seq")
            for key, value, frames, secs in telemetry_records(tel, self.context_mb):
                self.mem.record_combo(key, value, frames, secs)
        return active

    def tick(self):
        self.check_baseline()
        active = self.read_telemetry()
        u = self.usage.sample()
        self.mem.update_usage(self.node, u["cpu_pct"], u["mem_mb"])
        g = u.get("gpu") or []
        sample = {"role": self.role, "source": "local", "ready": True, "cpu_pct": u["cpu_pct"],
                  "mem_pct": u["mem_pct"], "mem_mb": u["mem_mb"],
                  "temp_c": u["temp_c"] if u["temp_c"] is not None else (g[0].get("temp_c") if g else None),
                  "gpu": g, "baseline": (self.mem.node(self.node) or {}).get("baseline")}
        self.writer.nodes(self.filter.observe(self.node, sample))
        self.mem.save()
        vals = [u["cpu_pct"], u["mem_pct"]] + [100.0 * x["mem_used_mb"] / x["mem_total_mb"]
                                               for x in g if x.get("mem_total_mb")]
        return self.sched.next(active, vals)

    # ------------------------------------------------------------ 스레드
    def _run(self):
        self.usage.sample(with_gpu=False)             # CPU % 는 두 번째 표본부터
        time.sleep(1.0)
        while not self.stop.is_set():
            try:
                wait = self.tick()
                self.last_error = None
            except Exception as e:  # noqa: BLE001 — 기록기는 서버를 멈추지 않는다
                self.last_error = "%s: %s" % (type(e).__name__, e)
                self.log("자원 기록 오류 (%s) — 60초 뒤 다시" % self.last_error)
                wait = 60.0
            self.stop.wait(wait)
        try:
            self.mem.save()
        except OSError:
            pass

    def start(self):
        self.thread = threading.Thread(target=self._run, name="resource-recorder", daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
