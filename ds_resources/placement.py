"""배치 계산기 placement/1 — 요청 + 자원 스냅숏 + 비용표 → 배치표 (core, 실행 방식과 무관, 2026-11-24).

순수 계산이다(입출력 없음). 같은 입력이면 같은 배치표가 나온다 — 모든 순회는 이름순, 배치표에 입력 해시를 적는다.

GPU 작업  선택지 묶음 하나 = 프로세스 하나(그 요청의 소스를 모두 맡는다). role=gpu · ready · stale 아닌 노드에만.
          메모리   need_mb ≤ 여유(total − used − reserved) − SPARE_MB       (Executor 가 실행 직전에 다시 잰다)
          처리량   Σ 소스(목표 fps × p90 ms / 1000) ≤ GPU_BUDGET (GPU 하나에 도는 모든 프로세스 합)
                   넘치면 소스별 fps 를 같은 비율로 낮추고(FPS_FLOOR 아래로는 안 낮춤), 그래도 넘치면
                   소스를 이름 역순으로 빼며 이유를 남긴다. GPU 노드가 없으면 파이로 옮기지 않고 거절한다.
CPU 작업  role 이 허용된 · ready · stale 아닌 노드. 쓸 수 있는 몫 = 할당 가능량 − max(서비스 p95, 순간 부하)
          − 할당 가능량 × CPU_HEADROOM. 네트워크: 장당 전송 시간이 장당 처리 시간보다 길면 그 노드에 두지 않는다,
          테일넷 중계(relay)면 벌점. requests = 선언값, limits = requests × LIMIT_FACTOR, PriorityClass 낮음.
          placement: measured 인 작업(cam360_slice)은 노드별 비교 측정이 있어야 옮긴다 — 없으면 data_stream 기기에.
          노드에 cpu_tasks: False 가 있으면(k3s 클러스터 객체 없음 — 컨트롤러가 check() caps 로 단다) 그 노드는 뺀다.
거절 · 낮춤  {item, code, reason} — code 는 streams.REASON_CODES. CPU 거절은 노드마다 뺀 이유(skipped)도 남기고,
          가장 앞선 이유를 code 로 낸다(CPU_SKIP_ORDER: 기다리거나 고치면 놓일 수 있는 것부터).
"""
import hashlib
import json
import math

from ds_resources.streams import reason_item

ALGORITHM = "placement/1"
SPARE_MB = 300.0
GPU_BUDGET = 0.85
FPS_FLOOR = 1.0
CPU_HEADROOM = 0.20
LIMIT_FACTOR = 2.0
#: 로봇 서비스가 실제로 돌 때의 부하를 아직 못 잰 로봇 노드 — 그 몫을 모르니 이만큼 비워 둔다(CPU · 메모리 비율).
#: 로봇이 돌기 시작하면 기록기가 active p95 를 바로 재기 시작하고, 표본 10개 뒤부터는 실측으로 바뀐다
ROBOT_UNMEASURED_CPU = 0.5
ROBOT_UNMEASURED_MEM = 0.3
RELAY_PENALTY = 0.5
PRIORITY_CLASS = "ds-vision-low"
#: CPU 작업이 어느 노드에도 못 놓였을 때 거절 코드로 낼 순서 — 앞의 것일수록 '기다리거나 고치면 놓인다'
CPU_SKIP_ORDER = ("no_cpu_capability", "unknown_bandwidth", "bandwidth_unsupported", "no_bandwidth", "no_node")
CPU_SKIP_TEXT = {"no_cpu_capability": "CPU 작업을 올릴 수 없는 k3s", "unknown_bandwidth": "대역폭 아직 모름",
                 "bandwidth_unsupported": "대역폭을 잴 수 없음", "no_bandwidth": "대역폭 부족",
                 "no_node": "CPU · 메모리 부족"}


def digest(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def _gpu_nodes(nodes):
    return sorted((n for n in nodes if n.get("role") == "gpu" and n.get("ready") and not n.get("stale")
                   and n.get("gpus")), key=lambda n: n["name"])


def _room(node, reserved):
    g = node["gpus"][0]
    return float(g["mem_total_mb"]) - float(g["mem_used_mb"]) - reserved - SPARE_MB


def place_gpu(request, nodes, costs, out):
    """request: {models, sources: [이름], target_fps: {소스: fps}}
    costs: {"need_mb": MB, "basis": ..., "p90_ms": {소스: ms}}  — 선택지 묶음 하나의 비용"""
    models = sorted(request["models"])
    srcs = sorted(request["sources"])
    gnodes = _gpu_nodes(nodes)
    if not gnodes:
        for s in srcs:
            out["refused"].append(reason_item("gpu:%s:%s" % ("+".join(models), s), "no_gpu_node",
                                              "쓸 수 있는 GPU 노드가 없음 — GPU 작업은 파이로 옮기지 않는다"))
        return
    need = float(costs["need_mb"])
    ledger = out.setdefault("_ledger", {})               # 이 배치표 안에서 앞서 놓은 묶음의 몫 (노드 → [MB, 사용률])
    for n in gnodes:
        ledger.setdefault(n["name"], [0.0, float(n.get("gpu_util_committed") or 0.0)])
    reserved = {k: v[0] for k, v in ledger.items()}
    util = {k: v[1] for k, v in ledger.items()}
    fits = [n for n in gnodes if need <= _room(n, reserved[n["name"]])]
    exclusive = False
    if not fits and costs.get("basis", "estimate") == "estimate" and costs.get("estimate_mb") is not None:
        # 실측이 없어 여유분(×1.2)까지는 안 들어가지만 추정치 자체는 들어간다 — GPU Executor 의 단독 측정(exclusive)과
        # 같은 규칙으로 배치표에도 놓는다. 그 노드에 이 배치표 · 이미 도는 몫이 없을 때만 (2026-11-25 실측: 5580 > 5298)
        est = float(costs["estimate_mb"])
        fits = [n for n in gnodes if not reserved[n["name"]] and not util[n["name"]]
                and est <= _room(n, reserved[n["name"]])]
        exclusive = bool(fits)
    if not fits:
        best = max(gnodes, key=lambda n: (_room(n, reserved[n["name"]]), n["name"]))
        out["refused"].append(reason_item("gpu:%s" % "+".join(models), "gpu_mem",
                                          "GPU 메모리 부족: 필요 %d MB > 여유 %d MB (%s)" % (
                                              need, _room(best, reserved[best["name"]]), best["name"])))
        return
    node = max(fits, key=lambda n: (_room(n, reserved[n["name"]]) - util[n["name"]] * 1000, n["name"]))
    budget = max(0.0, GPU_BUDGET - util[node["name"]])
    p90 = costs["p90_ms"]
    waiting = set(request.get("waiting") or ())
    target = {s: float(request.get("target_fps", {}).get(s, 10.0)) for s in srcs}
    load = lambda f: sum(f[s] * p90[s] / 1000.0 for s in f)  # noqa: E731
    keep = list(srcs)
    # 넘치면 빼는 순서: 입력 대기 소스 먼저(영상이 오는 소스를 지킨다), 그다음 p90 이 큰 것 — 뺀 뒤에는 남은 것끼리 다시 나눈다
    drop_order = sorted(srcs, key=lambda s: (s not in waiting, -p90[s], s))
    while True:
        fps = {s: target[s] for s in keep}
        degraded = []
        if fps and load(fps) > budget:
            scale = budget / load(fps)
            # 내림 — 반올림하면 6.296 → 6.30 처럼 올라가 부하가 예산을 살짝 넘고 소스가 통째로 빠진다(2026-11-24 실측 버그)
            fps = {s: max(FPS_FLOOR, math.floor(fps[s] * scale * 100) / 100.0) for s in fps}
            degraded = sorted(fps)
        if not fps or load(fps) <= budget + 1e-9:
            break
        drop = next(s for s in drop_order if s in keep)
        keep.remove(drop)
        if drop in waiting:
            out.setdefault("waiting", []).append(reason_item(
                "input:%s" % drop, "input_wait_unbudgeted",
                "입력 대기 — 지금 예산으로는 미리 잡지 못함 (p90 %.0f ms), 영상이 들어오면 다시 배치" % p90[drop]))
        else:
            out["refused"].append(reason_item("gpu:%s:%s" % ("+".join(models), drop), "gpu_throughput",
                                              "GPU 처리량 부족: fps %.1f 까지 낮춰도 p90 합이 예산 %.2f 를 넘음" % (
                                                  FPS_FLOOR, budget)))
    if not fps:
        return
    ledger[node["name"]][0] += need
    ledger[node["name"]][1] += load(fps)
    out["gpu_units"].append({"unit": "gpu:%s" % "+".join(models), "node": node["name"], "models": models,
                             "sources": sorted(fps), "fps_cap": fps, "need_mb": round(need),
                             "need_basis": costs.get("basis", "estimate"), "gpu_util": round(load(fps), 3),
                             **({"exclusive": True} if exclusive else {})})
    for s in degraded:
        out["degraded"].append(reason_item("gpu:%s:%s" % ("+".join(models), s), "fps_lowered",
                                           "fps %.1f → %.2f (p90 %.0f ms 기준 예산)" % (
                                               request.get("target_fps", {}).get(s, 10.0), fps[s], p90[s])))


def _cpu_room(n):
    alloc_cpu, alloc_mem = float(n["cpu_alloc_m"]), float(n["mem_alloc_mb"])
    share = n.get("service") or {}
    used_cpu = max(float(share.get("cpu_p95_m") or 0), float(n.get("cpu_now_m") or 0))
    used_mem = max(float(share.get("mem_p95_mb") or 0), float(n.get("mem_now_mb") or 0))
    if n.get("role") == "robot" and share.get("active_measured") is False:
        used_cpu = max(used_cpu, alloc_cpu * ROBOT_UNMEASURED_CPU)
        used_mem = max(used_mem, alloc_mem * ROBOT_UNMEASURED_MEM)
    return alloc_cpu - used_cpu - alloc_cpu * CPU_HEADROOM, alloc_mem - used_mem - alloc_mem * CPU_HEADROOM


def place_cpu(task, nodes, out, local_node):
    """task: {task, cpu_m, mem_mb, roles: [허용 role], frame_kb, fps, proc_ms, placement: any|measured, measured: {노드: ms}}"""
    name = task["task"]
    if task.get("placement") == "measured" and not task.get("measured"):
        out["cpu_units"].append({"task": name, "node": local_node, "reason": "비교 측정 전 — data_stream 기기에 둔다",
                                 **_resources(task)})
        return
    cands, skipped = [], {}
    taken = out.setdefault("_cpu_ledger", {})            # 이 배치표 안에서 앞서 놓은 CPU 작업 (노드 → [m, MB])
    for n in sorted(nodes, key=lambda n: n["name"]):
        if not n.get("ready") or n.get("stale") or n.get("role") not in task.get("roles", []):
            continue
        if n.get("cpu_tasks") is False:                  # k3s 가 읽히지만 CPU 작업은 못 올림 (check() caps)
            skipped[n["name"]] = "no_cpu_capability"
            continue
        cpu, mem = _cpu_room(n)
        cpu -= taken.get(n["name"], [0, 0])[0]
        mem -= taken.get(n["name"], [0, 0])[1]
        if cpu < task["cpu_m"] or mem < task["mem_mb"]:
            skipped[n["name"]] = "no_node"
            continue
        net = n.get("net") or {}
        per_frame_ms = 0.0
        if n["name"] != local_node and task.get("frame_kb"):
            bw = float(net.get("bw_mbps") or 0)
            if bw <= 0:                                  # 모름 — 잴 수 없는 설치인지, 아직 못 잰 것인지 가른다
                skipped[n["name"]] = "bandwidth_unsupported" if net.get("net_status") == "unsupported" \
                    else "unknown_bandwidth"
                continue
            per_frame_ms = task["frame_kb"] * 8.0 / bw                       # KB → kbit ÷ Mbit/s = ms
            proc = (task.get("measured") or {}).get(n["name"], task.get("proc_ms") or 0)
            if per_frame_ms > proc:
                skipped[n["name"]] = "no_bandwidth"
                continue
        score = cpu / float(n["cpu_alloc_m"]) - (RELAY_PENALTY if net.get("relay") else 0.0)
        if task.get("measured"):
            score -= (task["measured"].get(n["name"], 1e9)) / 1000.0
        cands.append((score, n["name"], per_frame_ms))
    if not cands:
        code = next((c for c in CPU_SKIP_ORDER if c in skipped.values()), "no_node")
        cap_why = next((n.get("cpu_why") for n in nodes if n.get("cpu_tasks") is False and n.get("cpu_why")), None)
        why = {"no_cpu_capability": "k3s 에 CPU 작업을 올릴 수 없음" + (" (%s)" % cap_why if cap_why else ""),
               "unknown_bandwidth": "대역폭 아직 모름 — 측정되면 다시 배치할 수 있음",
               "bandwidth_unsupported": "대역폭을 잴 수 없는 설치 (테일넷 주소 없음 등)",
               "no_bandwidth": "대역폭 부족 (장당 전송이 장당 처리보다 김)"}.get(
            code, "CPU · 메모리 · 네트워크 조건을 맞는 노드가 없음")
        detail = ", ".join("%s %s" % (k, CPU_SKIP_TEXT[v]) for k, v in sorted(skipped.items()))
        out["refused"].append(reason_item("cpu:%s" % name, code, why + ((" — " + detail) if detail else ""),
                                          skipped=skipped))
        return
    score, node, net_ms = max(cands)
    t = taken.setdefault(node, [0.0, 0.0])
    t[0] += task["cpu_m"]
    t[1] += task["mem_mb"]
    out["cpu_units"].append({"task": name, "node": node, "net_ms_per_frame": round(net_ms, 2), **_resources(task)})


def _resources(task):
    return {"requests": {"cpu_m": task["cpu_m"], "mem_mb": task["mem_mb"]},
            "limits": {"cpu_m": int(task["cpu_m"] * LIMIT_FACTOR), "mem_mb": int(task["mem_mb"] * LIMIT_FACTOR)},
            "priority_class": PRIORITY_CLASS}


def plan(request, snapshot, registry_hash, local_node):
    """request: {request_id, gpu: [{models, sources, target_fps}], cpu: [task...]}
    snapshot: {nodes: [...], costs: {"+".join(sorted(models)): {need_mb, basis, p90_ms}}}"""
    out = {"algorithm": ALGORITHM, "request_id": request.get("request_id"),
           "inputs": {"request": digest({k: v for k, v in request.items() if k != "request_id"}),
                      "snapshot": digest(snapshot), "registry": registry_hash},
           "gpu_units": [], "cpu_units": [], "refused": [], "degraded": []}
    nodes = snapshot.get("nodes") or []
    for g in sorted(request.get("gpu") or [], key=lambda g: "+".join(sorted(g["models"]))):
        key = "+".join(sorted(g["models"]))
        if key not in snapshot.get("costs", {}):
            out["refused"].append(reason_item("gpu:%s" % key, "no_cost", "비용표에 이 조합이 없음"))
            continue
        place_gpu(g, nodes, snapshot["costs"][key], out)
    for t in sorted(request.get("cpu") or [], key=lambda t: t["task"]):
        place_cpu(t, nodes, out, local_node)
    out.pop("_ledger", None)
    out.pop("_cpu_ledger", None)
    return out
