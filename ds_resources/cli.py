"""비전 오케스트레이터 명령 — 런처가 부른다 (core, 2026-11-24).

  python -m ds_resources.cli plan   --sources pi13 a2 --models yoloe [--profile 파일] [--cpu-tasks ...] [--json]
      배치표만 계산한다(띄우지 않고 스트림에 쓰지 않는다). 런처 「배치 보기」.
  python -m ds_resources.cli run    --sources ... --models ... [--profile 파일]
      요청을 내고 감시한다. Ctrl+C · SIGTERM(런처 ds_stop.sh)이면 맡은 작업을 멈추고 끝낸다.
      시작할 때 지난 실행이 남긴 작업을 정리한다(GPU 를 두 번 쓰지 않게).
  python -m ds_resources.cli status [--json]
      실행 상태 파일을 읽기만 한다.
  python -m ds_resources.cli places [--json]
      수동 배치가 고를 수 있는 것 — 지금 보이는 노드와 놓을 수 있는 작업 단위. 읽기만 한다.
  python -m ds_resources.cli preflight [--json]
      이 기기에서 무엇이 되고 무엇이 왜 안 되는지 — 자원 · 둘레(docker · k3s · Redis) · 가중치를 읽고
      기능마다 된다/안 된다와 고치는 길을 적는다. 아무것도 깔거나 띄우지 않는다.
      issues: 기기 단위로 지금 고칠 것 [{id, level, title, detail, fix}] (런처 시작 점검 배너).
  python -m ds_resources.cli fix <id> [--dry-run]
      다시 점검해서 그 id 의 고치기 명령(fix.argv)을 저장소 루트에서 돌리고 결과 JSON 을 찍는다
      (KUBECONFIG 가 없으면 ~/.kube/config). 밖에서 받은 명령은 돌리지 않는다.

기기별 모델 매칭 · 수동 배치 (2026-11-26)
  --assign pi13=yoloe a2=yoloe,river robot1=       기기마다 올릴 모델 (적지 않은 기기는 --models 전부,
                                                  빈 값은 그 기기에 비전을 올리지 않음)
  --pin object_detect.device=pi1 relay=none        작업 단위를 사람이 고른 노드에 (none 은 올리지 않음).
                                                  고른 자리는 한도를 면제하지만 센서 · 가속기 · arch 는 지킨다

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


def kv_pairs(items, what):
    """["pi13=yoloe,river", "a2="] → {"pi13": ["yoloe", "river"], "a2": []}. 틀린 모양이면 SystemExit"""
    out = {}
    for x in items or []:
        if "=" not in x:
            raise SystemExit("%s 는 기기=값 모양으로 적으세요 (받은 것: %r)" % (what, x))
        k, _, v = x.partition("=")
        k = k.strip()
        if not k:
            raise SystemExit("%s 에 이름이 없습니다 (받은 것: %r)" % (what, x))
        out[k] = [y.strip() for y in v.split(",") if y.strip()]
    return out


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
        lines.append("GPU %s, %s: %s" % ("+".join(u["models"]), " ".join(u["sources"]), u["node"]))
        need = "  필요 %d MB (%s)" % (u["need_mb"], "실측" if u["need_basis"] == "measured" else "레지스트리 추정")
        if a is not None:                               # 미리보기에만 있다 (run 은 실행 직전에 따로 확인한다)
            need += ", 지금 여유 %s MB: %s" % (a.get("room_mb", "?"), "들일 수 있음" if a.get("ok")
                                                 else ("안 됨: " + (a.get("why") or "")))
        lines.append(need)
        lines.append("  fps 상한 " + ", ".join("%s %.1f" % kv for kv in sorted(u["fps_cap"].items()))
                     + (" (입력 fps, 한도 면제: 서버는 처리량 예산 없음)" if u.get("limit_mode") == "none"
                        else ", GPU 사용률 %.0f%%" % (100 * u["gpu_util"])))
    for s in p.get("vision_off") or []:
        lines.append("비전 없음: %s — 기기별 모델 매칭에서 비워 둠" % s)
    for u in plan["cpu_units"]:
        mark = []
        if u.get("placement_by") == "manual":
            mark.append("수동")
        if u.get("limit_mode") == "none":
            mark.append("한도 없음" + (" (%s)" % ", ".join(u["waived"]) if u.get("waived") else ""))
        lines.append("CPU %s: %s%s%s" % (u["task"], u["node"], (" [%s]" % ", ".join(mark)) if mark else "",
                                          (" (%s)" % u["reason"]) if u.get("reason") else ""))
    for d in plan["degraded"]:
        lines.append("낮춤: %s — %s" % (d["item"], d["reason"]))
    for d in plan.get("waiting", []):
        lines.append("대기: %s — %s" % (d["item"], d["reason"]))
    for d in plan["refused"]:
        lines.append("못 함: %s — %s" % (d["item"], d["reason"]))
    return "\n".join(lines)


def plan_json(p):
    """기계가 읽는 배치 미리보기 (런처가 창 안에 그린다) — 배치표 칸(gpu_units · cpu_units · refused · degraded ·
    waiting · functions, GPU 단위의 admit) 그대로에 방식 · 근거와 스냅숏 노드 요약을 **더하기만** 한다.
    옛 칸(mode · basis · why · caps)은 이름도 자리도 그대로다."""
    plan = p["plan"]
    nodes = []
    for n in (p.get("snapshot") or {}).get("nodes") or []:
        one = {"name": n.get("name"), "role": n.get("role"), "ready": bool(n.get("ready")),
               "stale": bool(n.get("stale")), "source": n.get("source")}
        one.update(ctl.Controller._node_gpu(n))
        nodes.append(one)
    return dict(plan, mode=p["mode"], basis=p["basis"], why=p["why"], caps=p.get("caps") or {},
                ok=bool(p.get("ok")), vision_off=list(p.get("vision_off") or []), nodes=nodes,
                waiting=list(plan.get("waiting") or []), functions=plan.get("functions") or {})


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


def per_source_models(funcs, models, profile=None):
    """{소스: [기능]} → {소스: [모델]} — 그 소스의 기능들이 쓰는 서버 주 모델(프로필 슬롯이 이기고 없으면 기본값) 중
    --models 안의 것. 주 모델이 없는 기능(물체 추적 · 재식별 등)은 다른 기능의 탐지를 받아 쓰므로 더하지 않는다.
    어느 소스든 모델을 하나도 못 정하면 None (예전처럼 --models 전부)."""
    from ds_resources import functions as fnc
    from ds_resources import slots as slt
    reg = fnc._registry(None)
    table = fnc.load(reg)
    picks = slt.check_picks(reg, table, ((profile or {}).get("functions") or {}))
    want = set(models)
    out = {}
    for src, fns in funcs.items():
        got = []
        for fn in fns:
            if fn not in table:
                continue
            sel = slt.effective(reg, table, picks, {fn})
            slots_ = ((table[fn]["stages"].get("server") or {}).get("slots")) or []
            if not slots_:
                continue
            v = ((sel.get(fn) or {}).get("server") or {}).get(slots_[0]["role"])
            v = v[0] if isinstance(v, list) and v else v
            if v in want and v not in got:
                got.append(v)
        if not got:
            return None
        out[src] = sorted(got)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="자원 기반 비전 오케스트레이터")
    ap.add_argument("cmd", choices=("plan", "run", "status", "places", "preflight", "fix"))
    ap.add_argument("target", nargs="?", help="fix: 고칠 점검 id (preflight issues[].id)")
    ap.add_argument("--dry-run", action="store_true", help="fix: 돌리지 않고 무엇을 돌릴지만")
    ap.add_argument("--sources", nargs="*", default=[])
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--cpu-tasks", nargs="*", default=[])
    ap.add_argument("--profile")
    ap.add_argument("--functions", nargs="*", default=[],
                    help="기능 이름들 (vision/model_registry.yaml functions). 단계로 펼쳐 기기와 서버에 나눠 놓는다. "
                         "소스마다 다르게 주려면 pi12=object_detect,single_cam_track 처럼 적는다")
    ap.add_argument("--function-params", metavar="JSON",
                    help='기능 매개변수 (프로필 function_params 와 같은 칸 — 여기 준 것이 이긴다). 예: '
                         '\'{"reid": {"scope": "group", "group_sources": ["pi12", "a2"]}, "object_track": {"input": "targeted_detect"}}\'')
    ap.add_argument("--assign", nargs="*", default=[], metavar="기기=모델[,모델]",
                    help="기기별 모델 매칭 (적지 않은 기기는 --models 전부, 빈 값은 비전 없음)")
    ap.add_argument("--pin", nargs="*", default=[], metavar="작업=노드",
                    help="수동 배치 — 그 작업을 고른 노드에 (none 은 올리지 않음)")
    ap.add_argument("--limit", nargs="*", default=[], metavar="노드=cpu:0.2,mem:0.1,cores:1,temp:80,gpu:300",
                    help="노드별 한도 — 비워 둘 비율(cpu, mem), 비울 코어(cores), 온도 한계(temp), "
                         "GPU 예비 MB(gpu), fps 상한(fps). 서버에도 걸 수 있다(기본은 없음)")
    ap.add_argument("--cluster", default=os.environ.get("DS_CLUSTER", "auto"), help="auto | none | 모듈:클래스")
    ap.add_argument("--redis", default="redis://127.0.0.1:6380/0")
    ap.add_argument("--redis-public", help="파이 파드가 닿는 Redis 주소 (기본: 이 기기 테일넷 주소 :6380)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--poll", type=float, default=3.0)
    a = ap.parse_args(argv)
    if a.cmd == "fix":
        from ds_resources import preflight as pf
        if not a.target:
            ap.error("fix 에는 id 가 필요합니다 (preflight --json 의 issues[].id)")
        res = pf.run_fix(a.target, dry_run=a.dry_run)
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0 if res["ok"] else 1
    if not a.redis_public:
        import subprocess
        try:
            ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3).stdout.split()[0]
            a.redis_public = "redis://%s:6380/0" % ip
        except (OSError, subprocess.SubprocessError, IndexError):
            a.redis_public = a.redis

    if a.cmd == "preflight":
        from ds_resources import preflight as pf
        rep = pf.report()
        print(json.dumps(rep, ensure_ascii=False, indent=1) if a.json else pf.text(rep))
        return 0 if rep["summary"]["usable"] else 1

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

    limits = {}
    for node, bits in kv_pairs(a.limit, "--limit").items():
        one = {}
        for b in bits:
            k, _, v = b.partition(":")
            key = {"cpu": "headroom_cpu", "headroom": "headroom_cpu", "headroom_cpu": "headroom_cpu",
                   "mem": "headroom_mem", "headroom_mem": "headroom_mem",
                   "cores": "cores_reserved", "cores_reserved": "cores_reserved",
                   "temp": "temp_max", "temp_max": "temp_max",
                   "gpu": "gpu_spare_mb", "gpu_spare_mb": "gpu_spare_mb", "fps": "fps"}.get(k.strip())
            if not key:
                raise SystemExit("--limit 는 cpu:0.2, mem:0.1, cores:1, temp:80, gpu:300, fps:10 중에서 적으세요 "
                                 "(받은 것: %r)" % b)
            try:
                one[key] = int(float(v)) if key == "cores_reserved" else float(v)
            except ValueError:
                raise SystemExit("--limit %s 의 값이 수가 아닙니다: %r" % (key, v))
        if one:
            limits[node] = one
    limits = limits or None
    # --functions 는 평평한 목록이거나 소스별(pi12=a,b)이다 — 섞어 쓰지 않는다
    funcs = a.functions
    if any("=" in x for x in funcs):
        if not all("=" in x for x in funcs):
            raise SystemExit("--functions 는 모두 소스별(pi12=object_detect)이거나 모두 기능 이름이어야 합니다")
        funcs = kv_pairs(funcs, "--functions")
    assign = kv_pairs(a.assign, "--assign") or None
    manual = {k: (v[0] if v else "none") for k, v in kv_pairs(a.pin, "--pin").items()} or None

    fparams = None
    if a.function_params:
        try:
            fparams = json.loads(a.function_params)
        except ValueError as e:
            raise SystemExit("--function-params 는 JSON 이어야 합니다: %s" % e)
        if not isinstance(fparams, dict):
            raise SystemExit("--function-params 는 {기능: {이름: 값}} 사전이어야 합니다")

    # 소스별 기능(pi12=a,b)을 받았으면 — 요청하지 않은 소스에 같은 일을 하지 않게 (2026-11-27 실측: robot1 에만 켠
    # 거리 추정 · 물체 추적이 pi12 에서도 돌았다)
    #   ① --assign 이 없으면 소스마다 그 소스의 기능이 쓰는 서버 주 모델만 올린다 (--models 안에서)
    #   ② 소스를 가리는 기능(object_track · reid)에는 매개변수 sources 로 켠 소스를 실어 보낸다
    if isinstance(funcs, dict) and funcs:
        by_fn = {}
        for src, fns in funcs.items():
            for fn in fns:
                by_fn.setdefault(fn, []).append(src)
        if assign is None and a.models:
            auto = per_source_models(funcs, a.models, load_profile(a.profile))
            if auto:
                assign = auto
        fparams = dict(fparams or {})
        for fn in ("object_track", "reid", "multi_cam_track"):
            if fn in by_fn and "sources" not in (fparams.get(fn) or {}):
                fparams[fn] = dict(fparams.get(fn) or {}, sources=sorted(by_fn[fn]))

    def with_params(prof):
        """프로필 function_params 위에 --function-params 를 덮는다 (기능마다)"""
        if not fparams:
            return prof
        prof = dict(prof or {})
        merged = {k: dict(v or {}) for k, v in (prof.get("function_params") or {}).items()}
        for fn, vals in fparams.items():
            merged.setdefault(fn, {}).update(vals or {})
        prof["function_params"] = merged
        return prof

    if a.cmd == "places":
        k = build(a, writer=False)
        pl = k.places(limits, functions=funcs or None, sources=a.sources, assign=assign, manual=manual,
                      profile=with_params(load_profile(a.profile)))
        if a.json:
            print(json.dumps(pl, ensure_ascii=False, indent=1))
        else:
            print("방식: %s%s" % (pl["mode"], (" — " + pl["why"]) if pl["why"] else ""))
            for n in pl["nodes"]:
                print("노드 %-22s %-7s %-12s 한도 %s" % (n["name"], n["role"] or "",
                      "쓸 수 있음" if (n["ready"] and not n["stale"]) else ("stale" if n["stale"] else "안 됨"),
                      "있음" if n["limited"] else "없음 (서버)"))
            for u in pl["units"]:
                print("작업 %-28s %-12s %-11s %s%s" % (u["unit"], u["pattern"], u["criticality"],
                      ("pin " + u["pin"]) if u["pin"] != "none" else (", ".join(u["roles"]) or "아무 곳"),
                      ", 한도 면제" if u["must_device"] else ""))
        return 0

    profile = with_params(load_profile(a.profile))
    if a.cmd == "plan":
        k = build(a, writer=False)
        p = k.preview(a.sources, a.models, profile, cpu_tasks=a.cpu_tasks, functions=funcs,
                      assign=assign, manual=manual, limits=limits)
        print(json.dumps(plan_json(p), ensure_ascii=False) if a.json else brief(p))
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
    rid = submit_with_retry(k, lambda: k.submit(a.sources, a.models, profile, a.profile, cpu_tasks=a.cpu_tasks,
                                                       functions=funcs, assign=assign, manual=manual,
                                                       limits=limits),
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
                log("다시 배치: %s 에서 %s 로" % (rid, new))
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
