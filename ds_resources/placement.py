"""배치 계산기 placement/2 — 요청 + 자원 스냅숏 + 비용표 + 기능 계약 → 배치표 (core, 실행 방식과 무관, 2026-11-25).

순수 계산이다(입출력 없음). 같은 입력이면 같은 배치표가 나온다 — 모든 순회는 이름순, 배치표에 입력 해시를 적는다.

GPU 작업  선택지 묶음 하나 = 프로세스 하나(그 요청의 소스를 모두 맡는다). role=gpu · ready · stale 아닌 노드에만.
          메모리   need_mb ≤ 여유(total − used − reserved) − SPARE_MB       (Executor 가 실행 직전에 다시 잰다)
          처리량   Σ 소스(목표 fps × p90 ms / 1000) ≤ GPU_BUDGET (GPU 하나에 도는 모든 프로세스 합)
                   넘치면 소스별 fps 를 같은 비율로 낮추고(FPS_FLOOR 아래로는 안 낮춤), 그래도 넘치면
                   소스를 이름 역순으로 빼며 이유를 남긴다. GPU 노드가 없으면 파이로 옮기지 않고 거절한다.
CPU 작업  둘로 갈린다 — eligible(하드 조건: 어기면 후보가 아니다, 코드가 남는다) 과 score(소프트: 클수록 좋다).
          하드   role 이 허용된 · ready · stale 아닌 노드 · cpu_tasks 가 꺼지지 않은 노드. pin(기기 전용 · 서버 전용),
                 작업 특성 needs(arch · sensors · accel), 온도 가드(THERMAL_MAX_C · throttle), CPU · 메모리
                 (쓸 수 있는 몫 = 할당 가능량 − max(서비스 p95, 순간 부하) − caps.cores_reserved
                 − 할당 가능량 × CPU_HEADROOM), 대역폭(장당 전송 > 장당 처리면 뺀다), 마감(전송 + 처리 > deadline_ms).
          소프트 여유 비율 − 중계(relay) 벌점 − 노드별 비교 측정 ms + sticky 보너스 + 온도 여유.
          **모르는 값(None)은 「없음」이 아니다** — 노드 특성 · 온도를 모르면 통과시키고 사유에 「모름」만 남긴다.
                 「확인했는데 없음」(sensors: [])은 거부다. 모른다고 못 놓으면 placement/1 보다 퇴행한다.
          requests = 선언값, limits = requests × LIMIT_FACTOR, PriorityClass 는 작업의 priority_class(기능 계약의
          criticality 로 고른 것) 또는 낮음.
          placement: measured 인 작업(cam360_slice)은 노드별 비교 측정이 있어야 옮긴다 — 없으면 data_stream 기기에.
          노드에 cpu_tasks: False 가 있으면(k3s 클러스터 객체 없음 — 컨트롤러가 check() caps 로 단다) 그 노드는 뺀다.
기능 단위  request["functions"] 는 functions.expand() 가 낸 배치 단계들이다. 단계마다 CPU 작업처럼 놓고,
          다 놓은 뒤 functions.pairs() 로 기능 단위 판정을 받는다 — rejected 면 그 기능의 놓인 단계까지 취소하고
          (split 은 한쪽만으로는 뜻이 없다), degraded 면 그대로 두고 품질만 낮춘다(cascade 의 기기 단계는 optional).
          out["functions"] = {기능: {verdict, quality, reason, placed, missing}}. 옛 경로(request["cpu"])는 그대로다.
거절 · 낮춤  {item, code, reason} — code 는 streams.REASON_CODES. CPU 거절은 노드마다 뺀 이유(skipped)도 남기고,
          가장 앞선 이유를 code 로 낸다(CPU_SKIP_ORDER: 기다리거나 고치면 놓일 수 있는 것부터).
"""
import hashlib
import json
import math

from ds_resources import functions
from ds_resources.streams import reason_item

ALGORITHM = "placement/2"
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
#: 온도 가드 — 이 온도를 넘은 노드에는 새 일을 두지 않는다. 파이 5 는 80~85℃ 에서 클럭을 조이기 시작하므로
#: 그 전에 뺀다는 뜻이다(조여진 뒤에 빼면 이미 제어 · 영상이 밀린다). throttled 가 참이면 온도와 무관하게 뺀다
THERMAL_MAX_C = 75.0
#: sticky 단계가 지금 도는 노드에 주는 보너스 — 옮기면 상태를 잃으므로 같은 자리를 선호한다.
#: 여유 비율(0~1) 보다 작게 둔다 — 자리가 정말 모자라면 보너스로 버티지 않고 옮겨야 한다
STICKY_BONUS = 0.25
#: 온도 여유 가중 — 같은 조건이면 시원한 노드로. 아주 작게 둔다(여유 · 측정 ms 를 뒤집을 만큼은 아니다)
THERMAL_WEIGHT = 0.1
#: CPU 작업이 어느 노드에도 못 놓였을 때 거절 코드로 낼 순서 — 앞의 것일수록 '기다리거나 고치면 놓인다'.
#: 앞의 것(k3s 객체 · 대역폭 측정 · 온도 · 마감 · 혼잡)은 기다리거나 식으면 풀리고,
#: 뒤의 것(센서 · 가속기 · arch)은 하드웨어가 바뀌어야 풀린다 — 사람에게 먼저 보여야 할 쪽이 앞이다
CPU_SKIP_ORDER = ("no_cpu_capability", "unknown_bandwidth", "thermal_guard", "deadline_miss", "no_bandwidth",
                  "bandwidth_unsupported", "no_sensor", "no_accel", "arch_mismatch", "no_node")
CPU_SKIP_TEXT = {"no_cpu_capability": "CPU 작업을 올릴 수 없는 k3s", "unknown_bandwidth": "대역폭 아직 모름",
                 "bandwidth_unsupported": "대역폭을 잴 수 없음", "no_bandwidth": "대역폭 부족",
                 "thermal_guard": "뜨겁거나 throttle 중", "deadline_miss": "전송 + 처리가 마감을 넘음",
                 "no_sensor": "필요한 센서 없음", "no_accel": "필요한 가속기 없음",
                 "arch_mismatch": "이미지 아키텍처 다름", "no_node": "CPU · 메모리 부족"}
#: 빠진 단계 → 지금 도는 품질 (streams.QUALITIES 의 값). 둘 다 놓이면 full, 거절이면 품질이 없다
STAGE_QUALITY = {"device": "server_only", "server": "device_only"}
#: 기다려도 안 풀리는 조건 — 하드웨어가 바뀌어야 한다. 「비교 측정 전 지름길」을 막는 것은 이것뿐이다
HARDWARE_CODES = ("no_sensor", "no_accel", "arch_mismatch")


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
                "입력 대기 — 영상 없음. 들어오면 다시 배치", p90_ms=p90[drop]))
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
    cores = (n.get("caps") or {}).get("cores_reserved")   # 그 노드가 제 일(제어 등)에 쓰라고 비워 둔 코어 수
    used_cpu += 0.0 if cores is None else float(cores) * 1000.0      # 모르면 0 — 모른다고 못 놓으면 퇴행이다
    return alloc_cpu - used_cpu - alloc_cpu * CPU_HEADROOM, alloc_mem - used_mem - alloc_mem * CPU_HEADROOM


def _temp(node):
    """그 노드가 아는 가장 높은 온도 (℃) 또는 None(모름) — 지금 값과 최근 최고값 중 큰 쪽"""
    th = node.get("thermal") or {}
    temps = [float(th[k]) for k in ("temp_c", "temp_max_c") if th.get(k) is not None]
    return max(temps) if temps else None


def _proc_ms(task, node_name):
    """그 노드에서의 장당 처리 시간 — 노드별 비교 측정이 있으면 그 값, 없으면 선언값"""
    return float((task.get("measured") or {}).get(node_name, task.get("proc_ms") or 0) or 0)


def _frames_local(task, node):
    """받을 장이 이미 그 노드에 있나 — 작업이 카메라를 요구하고 그 노드에 카메라가 있으면 그렇다.
    센서 목록을 모르면(None) 단정하지 않는다(전송이 있다고 본다 — 보수적인 쪽)."""
    need = set(((task.get("needs") or {}).get("sensors")) or ())
    have = (node.get("caps") or {}).get("sensors")
    return "camera" in need and have is not None and "camera" in have


def _net(task, node, local_node):
    """(장당 전송 ms, 받고 내보내는 전송 ms, 대역폭 거절 코드 | None) — KB × 8 ÷ Mbit/s = ms.

    장이 어디서 오는지로 갈린다:
      * 같은 노드(data_stream 기기) — 전송이 없다
      * **카메라가 그 노드에 달렸고 작업이 그 카메라를 요구한다 — 받을 장이 이미 거기 있다(0).
        내보내는 것만 전송한다.** 캐스케이드 1차 거름의 요점이 이것이다 — 원본을 받지 않고 결과만 보낸다
        (이 구분이 없으면 카메라가 달린 파이에 거름을 둘 때 원본을 네트워크로 받아 온다고 잘못 센다)
      * 그 밖 — 받을 장 + 내보낼 것 모두 전송

    대역폭을 모르면 전송 ms 는 None(모름)이고, **받을 장이 있을 때만** 코드가 붙는다 — 내보내는 양이
    수십 바이트인 거름 단계가 대역폭 측정을 기다리게 하지 않으려는 것이다."""
    if node["name"] == local_node:
        return 0.0, 0.0, None
    net = node.get("net") or {}
    bw = float(net.get("bw_mbps") or 0)
    frame = 0.0 if _frames_local(task, node) else float(task.get("frame_kb") or 0)
    if bw <= 0:                                          # 모름 — 잴 수 없는 설치인지, 아직 못 잰 것인지 가른다
        if not frame:
            return 0.0, None, None                       # 보낼 장이 없으면 대역폭을 몰라도 그대로 (placement/1 과 같다)
        return None, None, ("bandwidth_unsupported" if net.get("net_status") == "unsupported" else "unknown_bandwidth")
    egress = float(task.get("egress_kb") or 0)
    return frame * 8.0 / bw, (frame + egress) * 8.0 / bw, None


def eligible(task, node, local_node, taken=None):
    """하드 조건 — (ok, code, 사람이 읽는 사유). 하나라도 어기면 그 노드는 후보에서 빠지고 코드가 남는다.

    task   place_cpu 가 받는 작업 사전. 기능 계약에서 온 칸(pin · needs · deadline_ms · egress_kb)이 있으면 함께 본다
    taken  {노드: [m, MB]} 이 배치표 안에서 앞서 놓은 몫

    모르는 값(None)은 「없음」이 아니다 — 노드 특성 · 온도를 모르면 통과시키고 사유에 「모름」을 남긴다
    (모른다고 못 놓으면 placement/1 보다 퇴행한다). 다만 sensors: [] 는 「확인했는데 없음」이라 거부다."""
    name = node["name"]
    local = name == local_node
    if not node.get("ready") or node.get("stale") or node.get("role") not in task.get("roles", []):
        return False, "no_node", "역할 · 상태가 맞지 않음"
    if node.get("cpu_tasks") is False:                   # k3s 가 읽히지만 CPU 작업은 못 올림 (check() caps)
        return False, "no_cpu_capability", CPU_SKIP_TEXT["no_cpu_capability"]
    unknown = []
    pin = task.get("pin") or "none"
    if pin == "device" and local:
        return False, "no_node", "pin: device — 기기 안에서만 도는 단계다 (data_stream 기기는 후보가 아니다)"
    if pin == "server" and not local:
        return False, "no_node", "pin: server — 서버(data_stream 기기)에서만 도는 단계다"
    caps = node.get("caps") or {}
    needs = task.get("needs") or {}
    arch = needs.get("arch") or "any"
    if arch != "any":
        if caps.get("arch") is None:
            unknown.append("arch 모름")
        elif caps["arch"] != arch:
            return False, "arch_mismatch", "arch %s 가 필요한데 노드는 %s" % (arch, caps["arch"])
    want = sorted(needs.get("sensors") or [])
    if want:
        have = caps.get("sensors")
        if have is None:                                 # 모름 — 센서를 안 올린 노드다
            unknown.append("센서 모름")
        else:
            miss = [s for s in want if s not in have]
            if miss:
                return False, "no_sensor", "센서 %s 가 노드에 없음" % " · ".join(miss)
    accel = needs.get("accel") or "none"
    if accel != "none":
        if caps.get("accel") is None:
            unknown.append("가속기 모름")
        elif caps["accel"] != accel:
            return False, "no_accel", "가속기 %s 가 필요한데 노드는 %s" % (accel, caps["accel"])
    if (node.get("thermal") or {}).get("throttled"):
        return False, "thermal_guard", "노드가 throttle 중 — 식을 때까지 새 일을 두지 않는다"
    temp = _temp(node)
    if temp is None:
        unknown.append("온도 모름")
    elif temp > THERMAL_MAX_C:
        return False, "thermal_guard", "온도 %.1f℃ > 한계 %.1f℃" % (temp, THERMAL_MAX_C)
    cpu, mem = _cpu_room(node)
    held = (taken or {}).get(name) or (0.0, 0.0)
    cpu, mem = cpu - held[0], mem - held[1]
    if cpu < task["cpu_m"] or mem < task["mem_mb"]:
        return False, "no_node", "CPU · 메모리 부족 (쓸 수 있는 몫 %d m · %d MB)" % (cpu, mem)
    per_frame, net_ms, code = _net(task, node, local_node)
    if code:
        return False, code, CPU_SKIP_TEXT[code]
    proc = _proc_ms(task, name)
    if per_frame and per_frame > proc:
        return False, "no_bandwidth", "장당 전송 %.0f ms > 장당 처리 %.0f ms" % (per_frame, proc)
    deadline = task.get("deadline_ms")
    if deadline is not None and net_ms is not None and net_ms + proc > float(deadline):
        return False, "deadline_miss", "전송 %.0f + 처리 %.0f ms > 마감 %.0f ms" % (net_ms, proc, float(deadline))
    return True, None, "조건 맞음" + ((" (%s)" % " · ".join(unknown)) if unknown else "")


def score(task, node, taken=None):
    """소프트 — 클수록 좋다. 여유 비율 − 중계 벌점 − 노드별 측정 ms + sticky 보너스 + 온도 여유"""
    cpu, _mem = _cpu_room(node)
    cpu -= ((taken or {}).get(node["name"]) or (0.0, 0.0))[0]
    out = cpu / float(node["cpu_alloc_m"]) - (RELAY_PENALTY if (node.get("net") or {}).get("relay") else 0.0)
    if task.get("measured"):
        out -= (task["measured"].get(node["name"], 1e9)) / 1000.0
    if task.get("state") == "sticky" and task.get("current_node") == node["name"]:
        out += STICKY_BONUS                              # 옮기면 상태를 잃으므로 같은 자리를 선호한다
    temp = _temp(node)
    if temp is not None:                                 # 모르면 0 — 온도를 모르는 노드가 불리해지지 않게
        out += THERMAL_WEIGHT * (THERMAL_MAX_C - temp) / THERMAL_MAX_C
    return out


def place_cpu(task, nodes, out, local_node):
    """task: {task, cpu_m, mem_mb, roles: [허용 role], frame_kb, fps, proc_ms, placement: any|measured,
    measured: {노드: ms}} + (기능 계약에서 왔으면) {pin, needs, deadline_ms, egress_kb, state, current_node, ...}.

    eligible(하드) 로 후보를 걸러 score(소프트) 가 가장 높은 노드에 둔다 — 조건과 점수가 섞이지 않게 갈라 두었다."""
    name = task["task"]
    # 비교 측정 전에는 옮기지 않는다 — 단 둘은 예외다.
    #   (1) pin: device 인 단계는 애초에 서버로 갈 수 없으니 그냥 노드를 고른다
    #   (2) data_stream 기기가 **하드웨어 조건**(센서 · 가속기 · arch)을 어기면 이 지름길을 타지 않는다 —
    #       카메라를 요구하는 1차 거름을 카메라 없는 서버에 얹어 버리던 구멍이다. 보통 경로로 내려보내
    #       이유 코드(no_sensor 등)가 남게 한다. 자리·대역폭 부족은 예전처럼 지름길을 막지 않는다
    if task.get("placement") == "measured" and not task.get("measured") and (task.get("pin") or "none") != "device":
        here = next((n for n in nodes if n["name"] == local_node), None)
        blocked = None
        if here is not None:
            ok, code, _why = eligible(task, here, local_node)
            blocked = code if (not ok and code in HARDWARE_CODES) else None
        if blocked is None:
            out["cpu_units"].append({"task": name, "node": local_node, "reason": "비교 측정 전 — data_stream 기기에 둔다",
                                     **_tags(task), **_resources(task)})
            return
    cands, skipped, why_node = [], {}, {}
    taken = out.setdefault("_cpu_ledger", {})            # 이 배치표 안에서 앞서 놓은 CPU 작업 (노드 → [m, MB])
    for n in sorted(nodes, key=lambda n: n["name"]):
        if not n.get("ready") or n.get("stale") or n.get("role") not in task.get("roles", []):
            continue                                     # 애초에 후보가 아닌 노드는 이유를 남기지 않는다
        ok, code, why = eligible(task, n, local_node, taken)
        if not ok:
            skipped[n["name"]], why_node[n["name"]] = code, why
            continue
        cands.append((score(task, n, taken), n["name"], _net(task, n, local_node)[0] or 0.0))
    if not cands:
        code = next((c for c in CPU_SKIP_ORDER if c in skipped.values()), "no_node")
        cap_why = next((n.get("cpu_why") for n in nodes if n.get("cpu_tasks") is False and n.get("cpu_why")), None)
        why = {"no_cpu_capability": "k3s 에 CPU 작업을 올릴 수 없음" + (" (%s)" % cap_why if cap_why else ""),
               "unknown_bandwidth": "대역폭 아직 모름 — 측정되면 다시 배치할 수 있음",
               "bandwidth_unsupported": "대역폭을 잴 수 없는 설치 (테일넷 주소 없음 등)",
               "no_bandwidth": "대역폭 부족 (장당 전송이 장당 처리보다 김)",
               "thermal_guard": "노드가 뜨겁거나 throttle 중 — 식으면 다시 배치할 수 있음",
               "deadline_miss": "전송 + 처리 시간이 작업 마감을 넘음",
               "no_sensor": "그 작업이 필요한 센서를 가진 노드가 없음",
               "no_accel": "그 작업이 필요한 가속기(GPU · NPU)를 가진 노드가 없음",
               "arch_mismatch": "이미지 아키텍처가 맞는 노드가 없음 (arm64 · amd64)"}.get(
            code, "CPU · 메모리 · 네트워크 조건을 맞는 노드가 없음")
        # 노드마다 뺀 까닭은 eligible 이 낸 사람이 읽는 사유를 그대로 쓴다 (코드만으로는 pin · 센서 · 온도를 가릴 수 없다)
        detail = ", ".join("%s %s" % (k, why_node.get(k) or CPU_SKIP_TEXT.get(v, v)) for k, v in sorted(skipped.items()))
        out["refused"].append(reason_item("cpu:%s" % name, code, why + ((" — " + detail) if detail else ""),
                                          skipped=skipped))
        return
    _score, node, net_ms = max(cands)
    t = taken.setdefault(node, [0.0, 0.0])
    t[0] += task["cpu_m"]
    t[1] += task["mem_mb"]
    out["cpu_units"].append({"task": name, "node": node, "net_ms_per_frame": round(net_ms, 2),
                             **_tags(task), **_resources(task)})


def _tags(task):
    """기능 계약에서 온 작업이면 기능 · 단계를 배치표에 함께 적는다 (읽는 쪽이 기능 단위로 묶어 본다)"""
    return {k: task[k] for k in ("function", "stage") if task.get(k)}


def _resources(task):
    return {"requests": {"cpu_m": task["cpu_m"], "mem_mb": task["mem_mb"]},
            "limits": {"cpu_m": int(task["cpu_m"] * LIMIT_FACTOR), "mem_mb": int(task["mem_mb"] * LIMIT_FACTOR)},
            # 기능 계약은 criticality 로 PriorityClass 를 고른다(functions.PRIORITY_CLASSES). 옛 경로는 그대로 낮음
            "priority_class": task.get("priority_class") or PRIORITY_CLASS}


def stage_task(stage, measured=None):
    """기능 계약의 배치 단계 → place_cpu 가 받는 작업 사전.

    자원 칸은 functions.as_cpu_task 가 낸다(옛 cpu_tasks 와 같은 이름 · 같은 뜻). 거기에는 없는 계약 칸
    (pin · needs · deadline_ms · egress_kb · state · priority_class · 기능 · 단계)을 하드 조건 · 소프트 점수가
    보도록 그대로 얹는다. current_node(지금 도는 노드)는 부르는 쪽이 단계에 달아 두면 sticky 보너스에 쓰인다."""
    task = functions.as_cpu_task(stage, measured=measured)
    for k in ("function", "stage", "pin", "needs", "deadline_ms", "egress_kb", "state", "criticality",
              "priority_class", "current_node"):
        if stage.get(k) is not None:
            task[k] = stage[k]
    return task


def _quality(g):
    """functions.pairs 판정 → 지금 도는 품질 (streams.QUALITIES). 거절이면 품질이 없다(None)"""
    if g["verdict"] == "rejected":
        return None
    for side in ("device", "server"):
        if side in g["missing"]:
            return STAGE_QUALITY[side]
    return "full"


def place_functions(stages, nodes, out, local_node, measured=None):
    """기능 계약의 배치 단계들을 놓고, 놓은 뒤 기능 단위 쌍 규칙(functions.pairs)을 배치표에 반영한다.

    stages   functions.expand() 가 낸 배치 단계 사전들
    measured {단계 unit: {노드: ms}} 노드별 비교 측정 (placement: measured 인 단계가 쓴다)"""
    if not stages:
        return
    measured = measured or {}
    stages = sorted(stages, key=lambda s: (s["function"], s["stage"]))
    by_unit = {s["unit"]: s for s in stages}
    for s in stages:
        place_cpu(stage_task(s, measured.get(s["unit"])), nodes, out, local_node)
    placed = [(by_unit[u["task"]]["function"], by_unit[u["task"]]["stage"])
              for u in out["cpu_units"] if u["task"] in by_unit]
    out["functions"] = {}
    for fn, g in functions.pairs(stages, placed=placed).items():
        quality = _quality(g)
        out["functions"][fn] = dict({"verdict": g["verdict"], "reason": g["reason"],
                                     "placed": list(g["placed"]), "missing": list(g["missing"])},
                                    **({"quality": quality} if quality else {}))
        if g["verdict"] == "rejected":
            # 한쪽만으로는 뜻이 없다 — 이미 놓은 단계까지 되돌린다 (split 의 반쪽은 자원만 먹는다)
            cancel = [u for u in out["cpu_units"] if by_unit.get(u["task"], {}).get("function") == fn]
            for u in cancel:
                out["cpu_units"].remove(u)
            out["refused"].append(reason_item("fn:%s" % fn, "pair_incomplete", g["reason"], function=fn,
                                              missing=list(g["missing"]),
                                              cancelled=sorted(u["task"] for u in cancel)))
        elif g["verdict"] == "degraded":
            out["degraded"].append(reason_item("fn:%s" % fn, "pair_incomplete", g["reason"], function=fn,
                                               missing=list(g["missing"]), quality=quality))


def plan(request, snapshot, registry_hash, local_node):
    """request: {request_id, gpu: [{models, sources, target_fps}], cpu: [task...], functions: [배치 단계...]}
    snapshot: {nodes: [...], costs: {"+".join(sorted(models)): {need_mb, basis, p90_ms}},
               measured: {단계 unit: {노드: ms}}}
    request["gpu"] · request["cpu"] 는 placement/1 과 똑같이 돈다 — functions 는 그 뒤에 더해진다(옛 경로 호환)."""
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
    place_functions(request.get("functions") or [], nodes, out, local_node, snapshot.get("measured"))
    out.pop("_ledger", None)
    out.pop("_cpu_ledger", None)
    return out
