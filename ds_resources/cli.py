"""비전 오케스트레이터 명령 — 런처가 부른다 (core, 2026-11-24).

  python -m ds_resources.cli plan   --sources pi13 a2 --models yoloe [--profile 파일] [--cpu-tasks ...] [--json]
      배치표만 계산한다(띄우지 않고 스트림에 쓰지 않는다). 런처 「배치 보기」.
  python -m ds_resources.cli run    --sources ... --models ... [--profile 파일]
      요청을 내고 감시한다. Ctrl+C · SIGTERM(런처 ds_stop.sh)이면 맡은 작업을 멈추고 끝낸다.
      시작할 때 지난 실행이 남긴 작업을 정리한다(GPU 를 두 번 쓰지 않게).
  python -m ds_resources.cli status [--json]
      실행 상태 파일을 읽기만 한다.

k3s: --cluster auto(기본)는 kubectl 과 kubeconfig 가 있을 때만 orchestrator.k3s_source 를 경로 문자열로 불러온다.
k3s 가 없는 사용자는 아무것도 안 해도 local(data_stream 기기 GPU)로 돈다. --cluster none 이면 k3s 를 보지 않는다.
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ds_resources import controller as ctl  # noqa: E402
from ds_resources import gpu_executor as ge  # noqa: E402
from ds_resources import memory as mm  # noqa: E402
from ds_resources.backends import resolve_cluster  # noqa: E402
from ds_resources.recorder import load_source  # noqa: E402
from ds_resources.state import StateFile  # noqa: E402
from ds_resources.streams import item_codes  # noqa: E402

STATE = REPO / "state" / "orchestrator_state.json"
RETRY_GPU = 6                 # run: GPU 메모리 부족으로 하나도 못 띄웠으면 이만큼 다시 요청 (앞 프로세스가 반납 중일 수 있다)
RETRY_CODES = ("gpu_mem",)    # 다시 요청할 이유 코드 — 배치표 거절이든 실행 직전 들이기(admit) 거절이든 (문구는 보지 않는다)
RETRY_GAP_S = 5
MEMORY = REPO / "state" / "resource_memory.json"


def build(args, writer):
    import redis
    r = redis.Redis.from_url(args.redis, socket_connect_timeout=2, socket_timeout=5)
    st = StateFile(STATE, writer=writer)
    if writer and not st.writer:
        raise SystemExit("다른 오케스트레이터가 이미 돌고 있습니다 (%s.lock) — 런처에서 비전을 껐다 켜세요" % STATE)
    mem = mm.ResourceMemory(MEMORY, writer=False)
    ex = ge.GpuExecutor(st, r, log=log)
    src_path, cpu_path = resolve_cluster(args.cluster)
    cluster = cpu = None
    if src_path:
        try:
            cluster = load_source(src_path, memory=mem, log=log, probe_net=False)
            import yaml
            tasks = (yaml.safe_load(open(REPO / "vision" / "model_registry.yaml", encoding="utf-8")) or {}).get("cpu_tasks") or {}
            cpu = load_source(cpu_path, tasks=tasks, redis_url=args.redis_public or args.redis)
        except Exception as e:  # noqa: BLE001 — k3s 쪽이 안 되면 local 로
            log("k3s 백엔드를 못 불러옴 (%s) — local 로" % e)
            cluster = cpu = None
    return ctl.Controller(r, st, ex, mem, cluster=cluster, cpu_executor=cpu, log=log)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_profile(path):
    if not path or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except ValueError:
        return None


def brief(p):
    """사람이 읽는 배치 요약 (런처 대화 상자)"""
    plan = p["plan"]
    lines = ["방식: %s (%s)%s" % (p["mode"], "지금 잰 자원" if p["basis"] == "live" else "기억한 자원",
                                  "" if p["ok"] else " — " + p["why"])]
    caps = p.get("caps") or {}
    if caps.get("cpu_tasks") is False:                  # k3s 는 읽히지만 파이에 CPU 작업을 못 올리는 상태
        lines.append("파이 CPU 작업: 막힘 — %s" % (caps.get("why") or "이유 모름"))
    for u in plan["gpu_units"]:
        a = u.get("admit")
        lines.append("GPU %s · %s → %s" % ("+".join(u["models"]), " ".join(u["sources"]), u["node"]))
        need = "  필요 %d MB (%s)" % (u["need_mb"], "실측" if u["need_basis"] == "measured" else "레지스트리 추정")
        if a is not None:                               # 미리보기에만 있다 (run 은 실행 직전에 따로 확인한다)
            need += " · 지금 여유 %s MB → %s" % (a.get("room_mb", "?"), "들일 수 있음" if a.get("ok")
                                                 else ("안 됨: " + (a.get("why") or "")))
        lines.append(need)
        lines.append("  fps 상한 " + ", ".join("%s %.1f" % kv for kv in sorted(u["fps_cap"].items()))
                     + " · GPU 사용률 %.0f%%" % (100 * u["gpu_util"]))
    for u in plan["cpu_units"]:
        lines.append("CPU %s → %s%s" % (u["task"], u["node"], (" (%s)" % u["reason"]) if u.get("reason") else ""))
    for d in plan["degraded"]:
        lines.append("낮춤: %s — %s" % (d["item"], d["reason"]))
    for d in plan.get("waiting", []):
        lines.append("대기: %s — %s" % (d["item"], d["reason"]))
    for d in plan["refused"]:
        lines.append("못 함: %s — %s" % (d["item"], d["reason"]))
    return "\n".join(lines)


def refused_codes(req):
    """요청 기록의 거절 이유 코드들 — requests.<id>.refused (배치표 + admit), 없으면(옛 기록) plan.refused"""
    items = req.get("refused")
    if items is None:
        items = (req.get("plan") or {}).get("refused")
    return item_codes(items)


def should_retry(req):
    return bool(refused_codes(req) & set(RETRY_CODES))


def submit_with_retry(k, submit, nothing_running, stopped=lambda: False, sleep=time.sleep, retries=None, gap=None):
    """submit() → 요청 id. 하나도 못 띄웠고 거절 코드가 RETRY_CODES 면 gap 초 뒤 다시 낸다(앞 프로세스가 GPU 를 반납 중)."""
    rid = submit()
    for attempt in range(RETRY_GPU if retries is None else retries):
        req = k.state.data["requests"].get(rid) or {}
        if stopped() or not nothing_running(rid) or not should_retry(req):
            break
        wait = RETRY_GAP_S if gap is None else gap
        log("GPU 메모리 부족 — %d초 뒤 다시 요청 (%d/%d)" % (wait, attempt + 1, RETRY_GPU if retries is None else retries))
        sleep(wait)
        rid = submit()
    return rid


def main(argv=None):
    ap = argparse.ArgumentParser(description="자원 기반 비전 오케스트레이터")
    ap.add_argument("cmd", choices=("plan", "run", "status"))
    ap.add_argument("--sources", nargs="*", default=[])
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--cpu-tasks", nargs="*", default=[])
    ap.add_argument("--profile")
    ap.add_argument("--cluster", default=os.environ.get("DS_CLUSTER", "auto"), help="auto | none | 모듈:클래스")
    ap.add_argument("--redis", default="redis://127.0.0.1:6380/0")
    ap.add_argument("--redis-public", help="파이 파드가 닿는 Redis 주소 (기본: 이 기기 테일넷 주소 :6380)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--poll", type=float, default=3.0)
    a = ap.parse_args(argv)
    if not a.redis_public:
        import subprocess
        try:
            ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3).stdout.split()[0]
            a.redis_public = "redis://%s:6380/0" % ip
        except (OSError, subprocess.SubprocessError, IndexError):
            a.redis_public = a.redis

    if a.cmd == "status":
        st = StateFile(STATE, writer=False)
        units = st.units()
        if a.json:
            print(json.dumps(units, ensure_ascii=False, indent=1))
        else:
            for uid, u in sorted(units.items(), key=lambda kv: kv[1].get("started_at") or 0):
                print("%-28s %-6s %-9s %s %s" % (uid, u.get("kind"), u.get("status"), u.get("node", ""),
                                                 u.get("reason", "")))
        return 0

    profile = load_profile(a.profile)
    if a.cmd == "plan":
        k = build(a, writer=False)
        p = k.preview(a.sources, a.models, profile, cpu_tasks=a.cpu_tasks)
        print(json.dumps(p["plan"] | {"mode": p["mode"], "basis": p["basis"], "why": p["why"], "caps": p.get("caps") or {}},
                         ensure_ascii=False)
              if a.json else brief(p))
        return 0

    k = build(a, writer=True)
    n = k.cleanup_previous()
    if n:
        log("지난 실행이 남긴 작업 %d개를 정리함" % n)
    stop = {"flag": False}

    def on_signal(*_):
        stop["flag"] = True
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    def nothing_running(rid):
        return not k.ex.active_units() and not any(
            u.get("kind") == "cpu" and u.get("status") in ("starting", "running") for u in k.state.units().values()) \
            and not any(u.get("status") == "local" for u in k.state.units().values() if u.get("request_id") == rid)

    # 바로 앞 프로세스가 GPU 메모리를 아직 반납하는 중일 수 있다 — 배치표든 admit 이든 gpu_mem 이면 다시
    rid = submit_with_retry(k, lambda: k.submit(a.sources, a.models, profile, a.profile, cpu_tasks=a.cpu_tasks),
                            nothing_running, stopped=lambda: stop["flag"])
    req = k.state.data["requests"].get(rid) or {}
    log("요청 %s — %s" % (rid, (req.get("mode") or "")))
    print(brief({"plan": req.get("plan") or {}, "mode": req.get("mode"), "basis": "live" if req.get("mode") == "k3s"
                 else "saved", "ok": req.get("mode") == "k3s", "why": ""}) if req.get("plan") else "", flush=True)
    code = 0
    try:
        while not stop["flag"]:
            for e in k.poll():
                log("%-9s %s" % (e["event"], e["reason"]))
            new = k.replan_if_needed(rid)                # 빠졌던 입력 대기 소스에 영상이 들어옴 → 다시 배치
            if new:
                log("다시 배치: %s → %s" % (rid, new))
                rid = new
                continue
            if nothing_running(rid) and not k.pending_sources(rid):
                req = k.state.data["requests"].get(rid) or {}
                why = [x["reason"] for x in (req.get("refused") if req.get("refused") is not None
                                              else (req.get("plan") or {}).get("refused", []))]
                log("돌고 있는 작업이 없어 끝냄%s" % ((" — " + " / ".join(why)) if why else ""))
                code = 2
                break
            time.sleep(a.poll)
    finally:
        k.stop(rid, "오케스트레이터 끝남")
        k.flush_outbox()
        log("끝 — %s" % rid)
    return code                                         # 0 이 아니면 ds_run 창이 닫히지 않고 이유를 보여 준다


if __name__ == "__main__":
    sys.exit(main())
