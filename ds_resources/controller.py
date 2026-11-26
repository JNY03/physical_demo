"""오케스트레이션 컨트롤러 — 요청 → 클러스터 확인 → 배치 → 실행 → 건강 확인 → 스트림 (core, 2026-11-24).

k3s 는 선택이다. cluster 백엔드(check() · nodes())가 없거나 check 가 실패하면 local 로 떨어지고
(fallback, basis=saved) GPU 작업은 data_stream 기기의 GPU Executor 로만 돈다. GPU 새 작업은 늘 지금 nvidia-smi 를
확인한 뒤에 들인다(Executor).

check() 는 (ok, why) 또는 (ok, why, caps) 를 돌려준다(M11 R2a — 옛 플러그인 호환). caps 예
{"cpu_tasks": False, "why": "PriorityClass ds-vision-low 없음"} — 노드는 읽히지만 CPU 작업은 못 올리는 셋째 상태.
그때도 mode=k3s · 노드 정보는 그대로이고, 파이 CPU 배치만 이유 코드 no_cpu_capability 로 막힌다.

거절 · 실패 · 낮춤 · 대기 기록에는 이유 코드(streams.REASON_CODES)를 붙인다. 실행 직전 들이기(admit) 거절도
같은 코드로 요청 기록(requests.<id>.refused)에 남긴다 — cli 재시도는 문구가 아니라 코드(gpu_mem)를 본다.

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
from ds_resources import functions as fnc
from ds_resources import placement, probe, settings
from ds_resources.streams import ORCH_STREAM, StreamWriter, orch_record, reason_item

REPO = Path(__file__).resolve().parent.parent
REGISTRY = REPO / "vision" / "model_registry.yaml"
INPUT_FRESH_S = 5.0
INPUT_DOWN_S = 15.0                   # 처리하던 소스 영상이 이만큼 끊기면 입력 끊김(degraded) — 들어오면 바로 다시
WAIT_FPS = 10.0                       # 입력 대기 소스의 예산을 미리 잡을 fps (target_fps 가 이긴다)
WAIT_RES = (1920, 1080)               # 입력 대기 소스의 해상도를 모를 때 (보수적으로 큰 쪽)
COMBO_STALE_S = 7 * 86400.0
DEGRADED_RATIO = 0.8
DEGRADED_AFTER_S = 30.0
THERMAL_TTL_S = 10.0                  # 이 기기 온도를 다시 읽는 간격 (vcgencmd · sysfs 를 매 tick 부르지 않게)
WARMUP_S = 30.0                     # running 뒤 이만큼은 fps 저하 판정을 하지 않는다 (2026-11-25 실측: 첫 30초 0.1~0.2 fps)
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


def _is_wsl():
    try:
        return "microsoft" in open("/proc/version", encoding="utf-8").read().lower()
    except OSError:
        return False


def _fps_text(fps):
    """{"robot1": 2.02} → "robot1 2.0 fps" (None 이면 '?')"""
    return ", ".join("%s %s fps" % (s, "?" if f is None else "%.1f" % f) for s, f in sorted((fps or {}).items())) or "-"


def _stream_tags(x):
    """배치표 항목 · 단위에서 스트림 칸으로 옮길 것만 (function · stage · quality). 없는 칸은 넣지 않는다."""
    return {k: x[k] for k in ("function", "stage", "quality") if x.get(k)}


def check_cluster(cluster):
    """cluster.check() → (ok, why, caps). 옛 (ok, why) 반환도 받는다. 예외는 (False, 이유, {})."""
    if cluster is None:
        return False, "k3s 백엔드 없음 (로컬 전용)", {}
    try:
        res = tuple(cluster.check())
    except Exception as e:  # noqa: BLE001
        return False, "k3s 확인 실패: %s" % e, {}
    ok, why = bool(res[0]), (res[1] if len(res) > 1 else "") or ""
    caps = res[2] if len(res) > 2 and isinstance(res[2], dict) else {}
    return ok, why, dict(caps)


def load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Controller:
    def __init__(self, redis_client, state, executor, memory, local_node=None, cluster=None, cfg=None,
                 clock=time.time, log=print, gpus=probe.gpus, cpu_executor=None):
        self._resubmit = {}                                # request_id → submit 인자 (빠진 입력 대기 소스가 들어오면 다시 배치)
        self.r, self.state, self.ex, self.mem = redis_client, state, executor, memory
        self.cpu_ex = cpu_executor                    # k3s CPU 실행기 (orchestrator 가 주입) — 없으면 파이에 못 둔다
        self.local = local_node or probe.node_name()
        self.cluster, self.clock, self.log, self.gpus = cluster, clock, log, gpus
        self.cfg = cfg if cfg is not None else settings.load_vision_config()   # + config/vision_config.local.yaml (D1)
        self.reg = load_yaml(REGISTRY)
        self.reg_hash = mm.cfg_hash(REGISTRY, {}, [])
        self.writer = StreamWriter(redis_client, maxlen=2000)

    # ------------------------------------------------------------ 스트림 (outbox)
    def emit(self, event, request_id, **kw):
        if kw.get("code") is None and isinstance(kw.get("detail"), dict) and kw["detail"].get("code"):
            kw["code"] = kw["detail"]["code"]               # 배치표 항목 {item, code, reason} 을 그대로 넘긴 경우
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

    def local_caps(self):
        """이 기기의 특성 · 온도 (배치가 보는 칸). 특성은 잘 안 바뀌어 한 번만 읽고, 온도는 THERMAL_TTL_S 마다.
        못 읽으면 빈 사전 — 배치는 「모름」을 통과로 다룬다(모른다고 막으면 지금 동작이 퇴행한다)."""
        if not hasattr(self, "_caps_cache"):
            try:
                self._caps_cache = probe.caps()
            except Exception:  # noqa: BLE001 — 특성을 못 읽어도 배치는 돌아야 한다
                self._caps_cache = {}
            self._therm_cache, self._therm_at = {}, None
        now = self.clock()
        if self._therm_at is None or now - self._therm_at >= THERMAL_TTL_S:
            try:
                self._therm_cache = probe.thermal() or {}
            except Exception:  # noqa: BLE001
                self._therm_cache = {}
            self._therm_at = now
        return dict(self._caps_cache), dict(self._therm_cache)

    def local_node_snapshot(self):
        g = self.gpus()
        # 적재 중(starting)도 처리량 몫을 이미 쥐고 있다 — gpu_executor.active_units() 와 같은 기준으로 센다.
        # (loaded 만 세면 적재 창 안에서 예산이 비어 보여 헛요청 · gpu_mem 재시도가 생긴다)
        active = list(self.ex.active_units().values())
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
        caps, therm = self.local_caps()
        return {"name": self.local, "role": "gpu", "ready": bool(g), "stale": not g,
                "caps": caps, "thermal": therm,
                "gpus": [{"mem_total_mb": x["mem_total_mb"], "mem_used_mb": x["mem_used_mb"]} for x in g],
                "gpu_util_committed": sum(float(u.get("gpu_util") or 0) for u in active),
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

    def function_contract(self):
        """작업 계약 (vision/model_registry.yaml 의 functions · 옛 cpu_tasks 별칭). 한 번만 읽는다.
        계약이 깨져 있으면 빈 사전으로 두고 알린다 — 기능 배치만 못 하고 옛 경로는 그대로 돈다."""
        if not hasattr(self, "_funcs"):
            try:
                self._funcs = fnc.load(self.reg)
            except Exception as e:  # noqa: BLE001
                self._funcs = {}
                self.log("작업 계약을 읽지 못함 (%s) — 기능 배치 없이 옛 경로로만 돈다" % e)
        return self._funcs

    def vision_groups(self, models, assign, sources):
        """기기별 모델 매칭 → GPU 묶음들. **묶음 하나 = 프로세스 하나**다 (2026-11-26).

        assign  {소스: [모델...]} 그 기기에 올릴 모델. 없는 소스는 고른 모델 전부(옛 동작 그대로),
                빈 목록은 「그 기기에는 비전을 올리지 않음」이다.
        모델 조합이 같은 소스끼리 묶는다 — 조합이 다르면 프로세스가 갈리고 GPU 메모리도 따로 든다.
        → ([{models, sources}...], 비전을 올리지 않는 소스들). 이름순이라 같은 입력이면 같은 묶음이다."""
        groups, off = {}, []
        pick = set(models)
        for s in sorted(sources):
            want = (assign or {}).get(s)
            ms = sorted(pick & set(want)) if want is not None else sorted(pick)
            if not ms:
                off.append(s)
                continue
            groups.setdefault(tuple(ms), []).append(s)
        return [{"models": list(k), "sources": sorted(v)} for k, v in sorted(groups.items())], off

    def places(self, limits=None, functions=None, sources=None, assign=None, manual=None, profile=None):
        """수동 배치가 고를 것 — {nodes, units} (읽기만, 아무것도 띄우지 않는다).

        functions 를 주면 **고른 기능만** 소스별로 펼쳐서 돌려준다(그러지 않으면 계약 전체). 그때는 배치까지
        한 번 돌려 각 작업의 **자동 예정 노드**(auto_node)와 못 놓는 이유(refused)를 함께 적는다 —
        「자동일 때 어디로 갈지」를 고르기 전에 보여 주려는 것이다. 아무것도 띄우지 않는다.

        nodes  지금 보이는 노드: 이름 · 역할 · 상태 · 특성 · **한도를 거는 쪽인가**(limited — 서버는 아니다)
        units  놓을 수 있는 작업 단위: 기능 계약의 배치 단계 + 옛 cpu_tasks 별칭. 어디에 둘 수 있는지(roles · pin)와
               반드시 그 기기에서 해야 하는지(must_device — 그러면 한도가 면제된다)를 함께 낸다"""
        ok, why, _caps = check_cluster(self.cluster)
        nodes = []
        if ok:
            try:
                nodes = self.cluster.nodes()
            except Exception as e:  # noqa: BLE001 — 노드를 못 읽어도 서버 하나는 고를 수 있다
                ok, why = False, "k3s 확인 실패: %s" % e
                nodes = []
        snap = self.with_limits([self.local_node_snapshot()] + [n for n in nodes if n.get("name") != self.local],
                                limits)
        units = []
        contract = self.function_contract()
        if functions:
            srcs = None if isinstance(functions, dict) else sorted(sources or ())
            stages, _res, _oth = fnc.expand(contract, functions, sources=srcs)
            for st in stages:
                g = contract[st["function"]]
                units.append({"unit": st["unit"], "function": st["function"], "stage": st["stage"],
                              "source": st.get("source"), "pin": st["pin"], "roles": list(st["roles"]),
                              "options": list(g.get("options") or []),
                              "pattern": g["pattern"], "criticality": g["criticality"], "optional": st["optional"],
                              "legacy": g["legacy"], "note": st["note"],
                              "cpu_m": st["cpu_m"], "mem_mb": st["mem_mb"],
                              "must_device": placement.must_on_device(st)})
        else:
            for fn, g in sorted(contract.items()):
                if g["pattern"] not in fnc.PLACEABLE:
                    continue
                for side, st in sorted(g["stages"].items()):
                    units.append({"unit": st["unit"], "function": fn, "stage": side, "pin": st["pin"],
                                  "roles": list(st["roles"]), "pattern": g["pattern"],
                                  "criticality": g["criticality"], "optional": st["optional"],
                                  "legacy": g["legacy"], "note": st["note"],
                                  # 그 단계가 달라는 몫 (계약에 적힌 선언값) — 화면이 「예상 사용」으로 보여 준다
                                  "cpu_m": st["cpu_m"], "mem_mb": st["mem_mb"],
                                  "must_device": placement.must_on_device(
                                      {"pin": st["pin"], "criticality": g["criticality"]})})
        if functions:
            # 자동이면 어디로 갈지 — 같은 계산을 한 번 돌려 본다 (띄우지 않는다)
            try:
                p = self.prepare(sorted(sources or ()), [], profile, functions=functions, assign=assign,
                                 manual=manual, limits=limits)
                where = {u["task"]: u for u in p["plan"]["cpu_units"]}
                why = {x["item"].split(":", 1)[-1]: x for x in p["plan"]["refused"] if x["item"].startswith("cpu:")}
                for u in units:
                    hit = where.get(u["unit"])
                    if hit:
                        u["auto_node"] = hit["node"]
                        u["auto_mode"] = hit.get("limit_mode")
                    elif u["unit"] in why:
                        u["auto_why"] = why[u["unit"]]["reason"]
                        u["auto_code"] = why[u["unit"]]["code"]
            except Exception as e:  # noqa: BLE001 — 예정 노드를 못 구해도 목록은 낸다
                self.log("자동 예정 노드를 구하지 못함: %s" % e)
        # 기능마다 쓸 수 있는 모델(선택지) — 런처가 「기능 → 모델」 로 고르게 한다
        fnopts = {n: list(g.get("options") or []) for n, g in sorted(contract.items())
                  if g["pattern"] in fnc.PLACEABLE}
        # 그 선택지가 어떤 것인가 — 성능(장당 ms) · GPU 어림 · 무엇을 하는 것인지. 레지스트리 하나에서 온다
        info = {}
        for key, o in sorted((self.reg.get("options") or {}).items()):
            cue = o.get("cue_models")                 # 사전({근거: [모델]})이거나 글 하나(규칙만 쓰는 선택지)
            if cue == "targeted":                     # targeted 는 cues 구역 전체가 근거 목록 — 탐지기 + 근거가 쓰는 모델
                cue = {k: list(c.get("needs") or []) + list(c.get("detector") or [])
                       for k, c in (self.reg.get("cues") or {}).items() if isinstance(c, dict)}
            cue = cue if isinstance(cue, dict) else {}
            # domain: 도메인 전용 파이프라인(targeted · river) — 모델이 아니라 기능이다. 런처는 모델 칸에 이름 대신 쓰는 모델을 적는다
            info[key] = {"shown": o.get("shown") or key, "kind": o.get("kind"), "gpu_mb": o.get("gpu_mb"),
                         "domain": bool(o.get("domain")),
                         "p90_ms": o.get("p90_ms") or {},
                         "models": sorted({m for ms in cue.values() for m in (ms or [])}),
                         "prompt": ("free" if key == "yoloe" else ("text" if key in ("targeted",) else None))}
            for m in info[key]["models"]:
                spec = (self.reg.get("models") or {}).get(m) or {}
                info[key].setdefault("notes", {})[m] = {"note": spec.get("note"), "device": spec.get("device"),
                                                        "files": [str(x) for x in (spec.get("files") or [])],
                                                        "packages": list(spec.get("packages") or [])}
        return {"ok": ok, "why": why, "mode": "k3s" if ok else "local", "function_options": fnopts, "model_info": info,
                # source = 그 노드가 맡은 기기 이름(ds.source · nodes.json). 사람이 노드 이름보다 이것으로 알아본다 —
                # 이 기기는 data_stream 서버다. 화면에 붙일 말은 읽는 쪽(런처)이 고른다(언어가 있으니)
                "nodes": [dict({"name": n["name"], "role": n.get("role"), "ready": bool(n.get("ready")),
                                "stale": bool(n.get("stale")), "limited": not placement.server_node(n, self.local),
                                "source": (n.get("source") or ("data_stream" if n["name"] == self.local else None)),
                                "model": (n.get("caps") or {}).get("model"),
                                "caps": n.get("caps") or {}, "thermal": n.get("thermal") or {}},
                               **self._node_room(n), **self._node_gpu(n)) for n in snap],
                "units": units}

    @staticmethod
    def with_limits(nodes, limits):
        """노드마다 걸린 한도를 스냅숏에 얹는다 — {노드: {headroom_cpu, headroom_mem, cores_reserved, temp_max,
        gpu_spare_mb, fps}}. 런처에서 고친다. **서버에도 걸 수 있다** (기본은 「없음」).

        placement.limit_of 가 이 사전을 읽는다. 적지 않은 값은 그 노드의 기본값(상수 · 라벨)이 그대로 쓰인다."""
        if not limits:
            return nodes
        out = []
        for n in nodes:
            lim = {k: v for k, v in (limits.get(n["name"]) or {}).items() if v is not None}
            out.append(dict(n, limits=dict(n.get("limits") or {}, **lim)) if lim else n)
        return out

    @staticmethod
    def _node_gpu(n):
        """그 노드의 GPU 메모리 (지금 nvidia-smi 가 본 값) — GPU 가 없으면 칸이 없다.
        k3s 쪽 노드는 GPU 를 알리지 않는다(metrics-server 가 GPU 를 모른다) — data_stream 기기만 채워진다."""
        gs = n.get("gpus") or []
        if not gs:
            return {}
        g = gs[0] or {}
        try:
            return {"gpu": {"total_mb": round(float(g.get("mem_total_mb") or 0)),
                            "used_mb": round(float(g.get("mem_used_mb") or 0)),
                            "committed": round(float(n.get("gpu_util_committed") or 0), 3)}}
        except (TypeError, ValueError):
            return {}

    def _node_room(self, n):
        """그 노드의 전체 · 지금 쓸 수 있는 몫 (화면에 「자원」으로 보여 준다).
        한도를 거는 노드는 한도를 뺀 몫, 서버는 한도가 없으니 물리적으로 남은 몫이다 — 배치가 보는 숫자와 같다."""
        try:
            waive = not (n["name"] != self.local and n.get("role") != "gpu")
            cpu, mem = placement._cpu_room(n, waive=waive)
            return {"alloc": {"cpu_m": round(float(n.get("cpu_alloc_m") or 0)),
                              "mem_mb": round(float(n.get("mem_alloc_mb") or 0))},
                    "room": {"cpu_m": round(cpu), "mem_mb": round(mem)}}
        except (KeyError, TypeError, ValueError):       # 자원 칸이 없는 노드(옛 플러그인) — 숫자 없이 둔다
            return {}

    def prepare(self, sources, models, profile=None, target_fps=None, cpu_tasks=None, task_specs=None, request_id=None,
                functions=None, assign=None, manual=None, limits=None):
        """클러스터 확인 → 입력 영상 → 스냅숏 → placement/3. 아무것도 띄우지 않고 기록도 하지 않는다(미리보기와 같은 계산).

        functions: 런처가 고른 **기능 이름들**. 단계로 펼치는 것은 여기서 한다 — 상주(resident)와 설정 · 배포
        흐름은 배치 목록에서 빠지고 따로 돌려준다(놓는 게 아니라 있는지 보고 알릴 몫이다).
        assign:    {소스: [모델...]} 기기별 모델 매칭 (없으면 고른 모델을 모든 기기에).
        manual:    {단위: 노드 | "none"} 수동 배치 — 배치 계산기가 사람이 고른 자리를 지킨다."""
        ok, why, caps = check_cluster(self.cluster)
        nodes = []
        if ok:
            try:
                nodes = self.cluster.nodes()
            except Exception as e:  # noqa: BLE001
                ok, why, caps = False, "k3s 확인 실패: %s" % e, {}
        if ok and caps.get("cpu_tasks") is False:       # 셋째 상태 — 노드 정보는 두고 그 노드들의 CPU 배치만 막는다
            nodes = [dict(n, cpu_tasks=False, cpu_why=caps.get("why") or "") for n in nodes]
        mode, basis = ("k3s", "live") if ok else ("local", "saved")
        infos, waiting = {}, []
        for s in sorted(sources):
            info = source_info(self.r, s, int(self.clock() * 1000))
            if info is None or info["age_s"] > INPUT_FRESH_S:
                # 입력 대기 — 빼지 않는다. 로봇이 켜지면 같은 프로세스가 바로 처리하도록 예산을 미리 잡는다.
                # 해상도는 Redis 에 남은 옛 프레임에서, 없으면 보수적으로 크게
                w, h = (info["w"], info["h"]) if info and info.get("w") else WAIT_RES
                infos[s] = {"age_s": None, "w": w, "h": h, "fps": WAIT_FPS, "waiting": True}
                waiting.append(s)
            else:
                infos[s] = info
        snapshot = {"nodes": self.with_limits(
                        [self.local_node_snapshot()] + [n for n in nodes if n.get("name") != self.local], limits),
                    "costs": {}}
        groups, vision_off = (self.vision_groups(models, assign, infos) if (infos and models) else ([], []))
        for g in groups:
            snapshot["costs"]["+".join(sorted(g["models"]))] = self.costs(
                g["models"], {s: infos[s] for s in g["sources"]}, profile)
        fps = {s: float((target_fps or {}).get(s) or max(1.0, infos[s]["fps"] or 10.0)) for s in infos}
        specs = self.cpu_specs(cpu_tasks, task_specs)
        if mode != "k3s" or self.cpu_ex is None:        # k3s 가 없으면 CPU 작업도 data_stream 기기에만
            specs = [dict(t, roles=[r for r in t.get("roles", []) if r == "gpu"]) for t in specs]
        stages, residents, others = [], [], []
        if functions:
            # 기능은 소스마다 다를 수 있다 — {소스: [기능...]} 이면 그대로, 목록이면 고른 소스 전부에 같은 기능.
            # 카메라를 보는 기기 단계만 소스 수만큼 늘어난다 (functions.expand)
            srcs = None if isinstance(functions, dict) else sorted(infos)
            stages, residents, others = fnc.expand(self.function_contract(), functions, sources=srcs)
            if mode != "k3s" or self.cpu_ex is None:     # k3s 가 없으면 기능 단계도 data_stream 기기에만
                stages = [dict(s, roles=[r for r in (s.get("roles") or []) if r == "gpu"]) for s in stages]
        req = {"request_id": request_id,
               "gpu": [{"models": g["models"], "sources": g["sources"],
                        "target_fps": {s: fps[s] for s in g["sources"]},
                        "waiting": sorted(set(g["sources"]) & set(waiting))} for g in groups],
               "cpu": specs}
        if stages:
            req["functions"] = stages
        if manual:
            req["manual"] = dict(manual)
        self._specs = {t["task"]: t for t in specs}
        plan = placement.plan(req, snapshot, self.reg_hash, self.local)
        placed = {s for u in plan["gpu_units"] for s in u["sources"]}
        plan["waiting"] = [reason_item("input:%s" % s, "input_wait", "입력 대기 — 영상 없음. 들어오면 바로 처리")
                           for s in waiting if s in placed or not models] + list(plan.get("waiting") or [])
        return {"ok": ok, "why": why, "caps": caps, "mode": mode, "basis": basis, "plan": plan, "snapshot": snapshot,
                "residents": residents, "others": others, "vision_off": vision_off}

    def preview(self, sources, models, profile=None, target_fps=None, cpu_tasks=None, functions=None,
                assign=None, manual=None, limits=None):
        """런처 「배치 보기」 — prepare + GPU 단위마다 지금 들일 수 있는지(Executor admit, 읽기만)"""
        p = self.prepare(sources, models, profile, target_fps, cpu_tasks, functions=functions, assign=assign,
                         manual=manual, limits=limits)
        for u in p["plan"]["gpu_units"]:
            c = p["snapshot"]["costs"]["+".join(u["models"])]
            ok, why, calc = self.ex.admit(u["need_mb"], c["basis"], c["estimate_mb"])
            u["admit"] = {"ok": ok, "why": why, **calc}
            u["p90_ms"], u["p90_basis"] = c["p90_ms"], c["p90_basis"]
        return p

    def submit(self, sources, models, profile=None, profile_path=None, target_fps=None, cpu_tasks=None,
               task_specs=None, functions=None, assign=None, manual=None, limits=None):
        rid = "req-" + uuid.uuid4().hex[:10]
        request = {"sources": sorted(sources), "models": sorted(models), "target_fps": target_fps or {},
                   "cpu_tasks": sorted(cpu_tasks or []),
                   "functions": ({s: sorted(v) for s, v in sorted(functions.items())}
                                 if isinstance(functions, dict) else sorted(functions or []))}
        if assign:                                      # 기기별 모델 매칭 · 수동 배치는 있을 때만 적는다(요청 기록도 추가만)
            request["assign"] = {s: sorted(ms) for s, ms in sorted(assign.items())}
        if manual:
            request["manual"] = dict(sorted(manual.items()))
        self._resubmit[rid] = dict(sources=sources, models=models, profile=profile, profile_path=profile_path,
                                   target_fps=target_fps, cpu_tasks=cpu_tasks, task_specs=task_specs,
                                   functions=functions, assign=assign, manual=manual, limits=limits)
        self.state.put_request(rid, dict(request, status="requested", at=self.clock()))
        self.emit("requested", rid, request=request)
        p = self.prepare(sources, models, profile, target_fps, cpu_tasks, task_specs, request_id=rid,
                         functions=functions, assign=assign, manual=manual, limits=limits)
        ok, why, mode, basis, plan, snapshot = p["ok"], p["why"], p["mode"], p["basis"], p["plan"], p["snapshot"]
        caps = p.get("caps") or {}
        self.emit("cluster_checked", rid, reason=why or caps.get("why") or "",
                  detail=dict({"ok": ok}, **({"caps": caps} if caps else {})))
        if not ok:
            self.emit("fallback", rid, mode="local", basis="saved", reason=why)
        self.emit("planned", rid, mode=mode, basis=basis, plan=plan)
        xid = "exe-" + uuid.uuid4().hex[:10]
        refused = list(plan["refused"])                 # 요청 기록의 거절 = 배치표 거절 + 실행 직전 거절 (같은 코드)
        req_rec = dict(request, status="planned", plan=plan, mode=mode, execution_id=xid, at=self.clock(), refused=refused)
        self.state.put_request(rid, req_rec)
        for item in plan["degraded"] + plan.get("waiting", []):
            self.emit("degraded", rid, execution_id=xid, mode=mode, reason=item["reason"], detail=item,
                      **_stream_tags(item))
        for item in plan["refused"]:
            self.emit("failed", rid, execution_id=xid, mode=mode, reason=item["reason"], detail=item,
                      **_stream_tags(item))
        late = []
        for u in plan["gpu_units"]:
            if u["node"] != self.local:                 # GPU 노드가 늘면 그 노드의 Executor 로 (지금은 data_stream 기기 하나)
                item = reason_item(u["unit"], "remote_gpu", "원격 GPU Executor 없음: %s" % u["node"])
                late.append(item)
                self.emit("failed", rid, execution_id=xid, mode=mode, reason=item["reason"], code=item["code"])
                continue
            c = snapshot["costs"]["+".join(u["models"])]
            okk, why2, calc = self.ex.admit(u["need_mb"], c["basis"], c["estimate_mb"])
            if not okk:
                item = reason_item(u["unit"], calc.get("code") or "gpu_mem", why2, stage="admit")
                late.append(item)
                self.emit("failed", rid, execution_id=xid, mode=mode, reason=why2, code=item["code"], detail=calc)
                continue
            uid = self.ex.launch({"devices": u["sources"], "models": u["models"], "profile_path": profile_path,
                                  "fps_cap": u["fps_cap"], "gpu_util": u["gpu_util"], "request_id": rid,
                                  **({"limit_mode": u["limit_mode"]} if u.get("limit_mode") else {}),
                                  "execution_id": xid}, u["need_mb"], calc)
            where = self.gpu_where(u["node"])
            self.emit("applied", rid, execution_id=xid, mode=mode,
                      reason="실행: %s, %s, %s (pid %s)%s" % (
                          "+".join(u["models"]), _fps_text(u["fps_cap"]), where,
                          (self.state.units().get(uid) or {}).get("pid", "?"), (" — " + why2) if why2 else ""),
                      detail={"unit_id": uid, "admit": calc, "fps_cap": u["fps_cap"], "node": u["node"], "where": where})
        if late:
            self.state.put_request(rid, dict(req_rec, refused=refused + late))
        qual = {n: g.get("quality") for n, g in (plan.get("functions") or {}).items()}
        for u in plan["cpu_units"]:
            if u.get("function") and qual.get(u["function"]):   # 쌍이 깨져 낮춰 도는 중이면 기록에 품질을 남긴다
                u = dict(u, quality=qual[u["function"]])
            self.launch_cpu(rid, xid, mode, u)
        return rid

    # ------------------------------------------------------------ CPU 작업
    def launch_cpu(self, rid, xid, mode, u, replaces=None):
        uid = "cpu-%s-%s" % (u["task"], uuid.uuid4().hex[:6])
        rec = {"unit_id": uid, "kind": "cpu", "task": u["task"], "node": u["node"], "requests": u["requests"],
               "limits": u["limits"], "priority_class": u["priority_class"], "request_id": rid, "execution_id": xid,
               "status": "starting", "started_at": self.clock(), "replaces": replaces, **_stream_tags(u)}
        if u["node"] == self.local or self.cpu_ex is None:
            rec.update(status="local", where="data_stream 기기 (런처가 띄우는 기존 방식)")
            self.state.put_unit(uid, rec)
            self.emit("applied", rid, execution_id=xid, mode=mode, reason="실행: CPU 작업 %s, %s (data_stream 기기)" % (u["task"], self.local),
                      detail={"unit_id": uid, "node": u["node"], "note": u.get("reason")}, **_stream_tags(u))
            return uid
        try:
            self.cpu_ex.launch(rec)
        except Exception as e:  # noqa: BLE001
            rec.update(status="failed", reason=str(e)[:200])
            self.state.put_unit(uid, rec)
            self.emit("failed", rid, execution_id=xid, mode=mode, reason="CPU 작업을 못 띄움: %s" % e,
                      code="cpu_launch_failed", detail={"unit_id": uid}, **_stream_tags(u))
            return uid
        self.state.put_unit(uid, rec)
        self.emit("applied", rid, execution_id=xid, mode=mode, reason="실행: CPU 작업 %s, %s (k3s 파드)" % (u["task"], u["node"]),
                  detail={"unit_id": uid, "node": u["node"], "requests": u["requests"], "limits": u["limits"],
                          "priority_class": u["priority_class"]}, **_stream_tags(u))
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
            ref = (out["refused"] or [{}])[0]
            return self.emit("failed", u["request_id"], execution_id=u["execution_id"], mode="k3s",
                             reason="%s — 옮길 노드 없음" % reason, code=ref.get("code") or "no_node",
                             detail={"unit_id": u["unit_id"], "refused": out["refused"]})
        new = out["cpu_units"][0]
        nid = self.launch_cpu(u["request_id"], u["execution_id"], "k3s", new, replaces=u["unit_id"])
        return self.emit("replanned", u["request_id"], execution_id=u["execution_id"], mode="k3s",
                         reason="%s: %s 에서 %s 로" % (reason, u["node"], new["node"]),
                         detail={"from": u["unit_id"], "to": nid, "from_node": u["node"], "to_node": new["node"]})

    def poll_cpu(self):
        units = {k: u for k, u in self.state.units().items() if u.get("kind") == "cpu"
                 and u.get("status") in ("starting", "running")}
        if not units or self.cpu_ex is None:
            return []
        events = []
        ok, why, caps = check_cluster(self.cluster)
        try:
            nodes = self.cluster.nodes() if ok else []
        except Exception as e:  # noqa: BLE001
            ok, why, nodes = False, str(e), []
        if ok and caps.get("cpu_tasks") is False:       # 옮길 때도 CPU 작업을 못 올리는 노드는 빼고 고른다
            nodes = [dict(n, cpu_tasks=False, cpu_why=caps.get("why") or "") for n in nodes]
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
                                            reason="처리 중: CPU 작업 %s @ %s (k3s 파드)" % (u.get("task"), u["node"]),
                                            detail={"unit_id": uid, "node": u["node"]}))
                elif now - u["started_at"] > ge.LOAD_TIMEOUT_S:
                    events.append(self.replace_cpu(u, "%d초 안에 하트비트 없음" % ge.LOAD_TIMEOUT_S, nodes))
            elif age is None or age > CPU_REPLACE_AFTER_S:
                events.append(self.replace_cpu(u, "하트비트 %s초 없음" % ("?" if age is None else int(age)), nodes))
        return events

    # ------------------------------------------------------------ 감시
    def gpu_where(self, node):
        """사람이 읽는 실행 위치 — 「book-4vg1edpvi5 GPU (NVIDIA GeForce RTX 4050 Laptop GPU)」"""
        if node != self.local:
            return "%s GPU" % node
        if not hasattr(self, "_gpu_name"):
            try:
                g = self.gpus()
                self._gpu_name = g[0]["name"] if g else ""
            except Exception:  # noqa: BLE001
                self._gpu_name = ""
        here = "이 기기 · WSL" if _is_wsl() else "이 기기"
        return "%s GPU%s — %s" % (node, (" (%s)" % self._gpu_name) if self._gpu_name else "", here)

    def unbudgeted_up(self, rid):
        """예산이 모자라 배치표에서 뺀 입력 대기 소스 중 지금 영상이 들어오는 것"""
        req = self.state.data["requests"].get(rid) or {}
        srcs = [x["item"].split(":", 1)[1] for x in (req.get("plan") or {}).get("waiting", [])
                if x.get("code") == "input_wait_unbudgeted"]
        now_ms = int(self.clock() * 1000)
        return [s for s in srcs if (source_info(self.r, s, now_ms) or {}).get("age_s", 1e9) <= INPUT_FRESH_S]

    def pending_sources(self, rid):
        """예산이 모자라 빠진 채 영상을 기다리는 소스 — 돌 것이 없어도 끝내지 않고 기다릴 이유"""
        req = self.state.data["requests"].get(rid) or {}
        return [x["item"].split(":", 1)[1] for x in (req.get("plan") or {}).get("waiting", [])
                if x.get("code") == "input_wait_unbudgeted"]

    def replan_if_needed(self, rid):
        """빠졌던 입력 대기 소스에 영상이 들어오면 같은 요청을 다시 배치한다. 새 request_id 를 돌려준다(없으면 None).
        영상이 오는 소스끼리 GPU 예산을 다시 나누므로 지금 돌던 단위를 멈추고 새로 띄운다."""
        args = self._resubmit.get(rid)
        came = self.unbudgeted_up(rid) if args else []
        if not came:
            return None
        self.emit("replanned", rid, reason="입력 들어옴: %s — 예산을 다시 나눔" % ", ".join(came), detail={"sources_up": came})
        self.stop(rid, "다시 배치")
        new = self.submit(**args)
        self._resubmit.pop(rid, None)
        return new

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
            age = {s: (source_info(self.r, s, now_ms) or {}).get("age_s") for s in u["devices"]}
            age = {s: (1e9 if a is None else a) for s, a in age.items()}
            frames = sum(int(x.get("frames") or 0) for x in streams.values())
            prev_up = set(u.get("sources_up") or [])
            up = {s for s, a in age.items() if a <= INPUT_FRESH_S}
            if not u.get("running_emitted"):
                if frames > 0 and up:
                    self.state.update_unit(uid, running_emitted=True, running_at=now, sources_up=sorted(up))
                    fps_now = {s: (streams.get(s) or {}).get("fps") for s in sorted(up)}
                    wait = sorted(set(u["devices"]) - up)
                    events.append(self.emit("running", rid, execution_id=xid, mode="local",
                                            reason="처리 중: %s · %s @ %s%s" % (
                                                "+".join(u.get("models") or []), _fps_text(fps_now),
                                                self.gpu_where(u.get("node") or self.local),
                                                (" · 영상 기다림: " + ", ".join(wait)) if wait else ""),
                                            detail={"unit_id": uid, "sources_up": sorted(up), "node": u.get("node") or self.local,
                                                    "waiting": sorted(set(u["devices"]) - up),
                                                    "fps": {s: x.get("fps") for s, x in streams.items()}}))
                continue
            down = {s for s in prev_up if age[s] > INPUT_DOWN_S}
            came = up - prev_up
            for s in sorted(came):                       # 로봇이 켜졌다 — 이미 띄운 프로세스가 바로 처리한다
                events.append(self.emit("running", rid, execution_id=xid, mode="local",
                                        reason="입력 들어옴: %s — %s 에서 처리 시작" % (s, self.gpu_where(u.get("node") or self.local)),
                                        detail={"unit_id": uid, "source_up": s}))
            for s in sorted(down):
                events.append(self.emit("degraded", rid, execution_id=xid, mode="local",
                                        reason="입력 끊김: %s — 들어오면 바로 다시 처리" % s,
                                        detail={"unit_id": uid, "source_down": s}))
            if came or down:
                self.state.update_unit(uid, sources_up=sorted((prev_up | came) - down))
            live = (prev_up | came) - down
            warm = now - float(u.get("running_at") or now) < WARMUP_S     # 적재 직후 첫 장들은 느리다 — 저하로 보지 않는다
            # 처리량 예산이 없는 서버 단위(limit_mode none)의 fps 상한은 입력 fps 일 뿐 약속이 아니다 — 저하로 보지 않는다
            low = [] if warm or u.get("limit_mode") == "none" else [s for s, cap in (u.get("fps_cap") or {}).items()
                   if s in live and (streams.get(s) or {}).get("fps") is not None and streams[s]["fps"] < DEGRADED_RATIO * cap]
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
                if u.get("degraded_emitted"):                # 저하를 알렸으면 회복도 알린다 — 읽는 쪽의 「마지막 상태」가 degraded 로 남지 않게
                    events.append(self.emit("running", rid, execution_id=xid, mode="local",
                                            reason="fps 회복: %s @ %s" % (
                                                _fps_text({s: (streams.get(s) or {}).get("fps") for s in sorted(live)}),
                                                self.gpu_where(u.get("node") or self.local)),
                                            detail={"unit_id": uid, "recovered": True}))
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
