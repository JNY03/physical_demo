#!/usr/bin/env bash
# 엣지를 띄운다. **torch가 있는 파이썬을 골라 준다.**
#
# 왜 이 스크립트가 있나: `./venv/bin/python3 server.py`로 띄우면 조용히 축소
# 모드로 돈다 — 그 venv에는 torch도 ultralytics도 없어서 YOLOE와 depth가 전부
# "없음"으로 빠지고, 화면에는 원본만 나온다. 로그를 안 보면 왜 비었는지 알 수
# 없다. 여기서 먼저 확인하고, 없으면 **뜨지 않고 이유를 말한다**(AI-O-02).
#
# **WSL에 들어가지 않은 Windows 터미널에서 그냥 실행해도 된다.** Git Bash로
# 이 파일을 실행하면 스스로 WSL로 넘어가고, 이 셸은 감시자로 남는다. 감시자가
# 있어야 Ctrl-C가 실제로 먹는다 — wsl.exe는 콘솔 Ctrl-C를 자기가 삼키고 리눅스
# 쪽으로 넘기지 않는 경우가 있어서(2026-09-22 실측), 신호 전달만 믿으면 창은
# 닫혔는데 서버가 포트를 쥔 채 살아 있다. 그래서 감시자가 **pid와 포트를 직접
# 보고 죽인다**.
#
#   bash run.sh               # PowerShell/cmd/Git Bash 어디서나
#   bash run.sh --port 9000
#   bash run.sh stop          # 띄우지 않고 남은 엣지를 정리만 한다
#   ./run.sh                  # WSL 안에서라면 그대로
#   EDGE_PYTHON=/path/to/python ./run.sh
#   EDGE_VERIFY=1 ./run.sh    # 기동·연동·스트리밍을 확인한 뒤 계속 실행
#   EDGE_WSL_DISTRO=Ubuntu    # 넘길 배포판(기본 Ubuntu)
#   EDGE_NO_RECLAIM=1         # 기동 전 포트 회수를 하지 않는다
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 인자에서 포트를 읽는다. 없으면 config의 server.port.
arg_port() {
  local prev="" a
  for a in "$@"; do
    case "$prev" in --port|-p) echo "$a"; return 0 ;; esac
    case "$a" in --port=*) echo "${a#--port=}"; return 0 ;; esac
    prev="$a"
  done
  return 1
}

# ───────────────────────────────────────────────────────────────────────────
# Windows(Git Bash/MSYS) 쪽. 여기서는 서버를 직접 띄우지 않는다 — WSL로 넘기고
# 감시자로 남아서 Ctrl-C를 **확실한 kill로 번역한다**.
# ───────────────────────────────────────────────────────────────────────────
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)

  # MSYS는 `/mnt/c/...`처럼 생긴 인자를 Windows 경로로 멋대로 바꾼다. 꺼야
  # wsl.exe에 리눅스 경로를 그대로 넘길 수 있다.
  export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'

  WSL="$(command -v wsl.exe || echo /c/Windows/System32/wsl.exe)"
  [ -x "$WSL" ] || { echo "wsl.exe를 찾지 못했다: $WSL" >&2; exit 1; }
  DISTRO="${EDGE_WSL_DISTRO:-Ubuntu}"

  # wsl.exe 출력은 UTF-16 잔재(NUL)와 CR이 섞여 나온다. 쓰기 전에 턴다.
  wsl_out() { "$WSL" -d "$DISTRO" -- "$@" 2>/dev/null | tr -d '\r\000'; }

  # 배포판부터 확인한다. **wsl.exe는 "그런 배포판 없다"도 stdout에 UTF-16으로
  # 뱉는다** — NUL만 털면 깨진 글자가 남아서 "비어 있나?" 검사를 그대로
  # 통과한다(2026-09-22 실측: WSL_E_DISTRO_NOT_FOUND가 경로인 척 흘러갔다).
  # 그래서 아래 검사들은 전부 **정확히 일치하는지**로 본다.
  if [ "$(wsl_out echo EDGE_WSL_OK)" != "EDGE_WSL_OK" ]; then
    echo "WSL 배포판 '$DISTRO'에 닿지 못했다. wsl.exe가 말하는 이유:" >&2
    # 그 메시지도 UTF-16LE다. iconv로 풀어야 읽히고, 없으면 NUL만 턴다.
    { "$WSL" -d "$DISTRO" -- true 2>&1 \
        | { iconv -f UTF-16LE -t UTF-8 2>/dev/null || tr -d '\r\000'; } \
        | sed 's/^/  /' >&2; } || true
    echo "  (배포판 목록: wsl.exe -l -v / 다른 이름이면 EDGE_WSL_DISTRO=이름)" >&2
    exit 1
  fi

  # cygpath -m(정슬래시)로 넘긴다 — -w의 역슬래시는 wsl.exe에 인자로 실릴 때
  # 이스케이프로 먹혀서 `C:UsersAdmin...`이 된다(2026-09-22 실측).
  LINUX_HERE="$(wsl_out wslpath -a "$(cygpath -m "$HERE")" || true)"
  case "$LINUX_HERE" in
    /mnt/*) ;;
    *) LINUX_HERE="/mnt${HERE}" ;;      # 자동마운트 기본값으로 되돌린다
  esac
  [ "$(wsl_out ls "$LINUX_HERE/server.py")" = "$LINUX_HERE/server.py" ] \
    || { echo "WSL($DISTRO)에서 엣지 경로를 찾지 못했다: $LINUX_HERE" >&2; exit 1; }

  PIDFILE="/tmp/edge-server.pid"
  PORT="$(arg_port "$@" || true)"
  if [ -z "$PORT" ]; then
    PORT="$(wsl_out python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["server"]["port"])' "$LINUX_HERE/config.json")"
  fi
  case "$PORT" in ''|*[!0-9]*) PORT=8891 ;; esac

  # 정리는 **WSL 안의 run.sh stop**에게 맡긴다. 여기서 `wsl.exe -- bash -c` 로
  # 스크립트를 밀어 넣으면 안 된다 — wsl.exe는 `--` 뒤를 한 줄로 이어 붙여
  # 바깥 셸에 다시 먹이므로, 따옴표 안의 $1이며 환경변수가 **거기서 먼저
  # 펼쳐져 빈 값이 된다**(2026-09-22 실측: pidfile=, port= 로 도착). 스크립트
  # 파일로 부르면 인자와 env가 그대로 간다.
  reap() {
    "$WSL" -d "$DISTRO" -- env EDGE_PIDFILE="$PIDFILE"       bash "$LINUX_HERE/run.sh" stop --port "$PORT" 2>&1 | tr -d '\r\000'
  }
  # fuser는 포트를 쥔 pid를 stdout에 뱉는다. 비어 있지 않으면 누가 쥐고 있다.
  port_busy() { [ -n "$(wsl_out fuser "$PORT/tcp")" ]; }

  # `bash run.sh stop` — 띄우지 않고 정리만 한다.
  if [ "${1:-}" = "stop" ]; then
    echo "== 정리: pid($PIDFILE) + 포트 $PORT"
    reap
    echo "정리 완료"
    exit 0
  fi

  CHILD=""
  SHUTTING=0
  cleanup() {
    [ "$SHUTTING" = 1 ] && return 0
    SHUTTING=1
    trap '' INT TERM        # 정리 중 두 번째 Ctrl-C에 흔들리지 않는다
    echo
    echo "== 중지 요청 — WSL 쪽 엣지를 정리한다 (포트 $PORT)"
    reap
    if [ -n "$CHILD" ]; then
      kill "$CHILD" 2>/dev/null || true
      wait "$CHILD" 2>/dev/null || true
    fi
    # reap도 파이프라 종료코드를 못 믿는다. **포트로 확인한다** — 여기서
    # 조용히 넘어가면 다음 실행이 옛 서버 화면을 보게 된다.
    if port_busy; then
      echo "== 경고: 포트 $PORT가 아직 잡혀 있다. 'bash run.sh stop'으로 다시 확인한다." >&2
    else
      echo "== 정리 완료 — 포트 $PORT 반환"
    fi
  }
  trap cleanup INT TERM HUP EXIT

  # 기동 전 회수. 지난 실행이 뭔가 남겼으면 여기서 푼다 — 안 그러면 새 서버가
  # "Address already in use"로 조용히 죽고, 화면은 옛 서버를 계속 보여 준다.
  if [ "${EDGE_NO_RECLAIM:-0}" != "1" ] && port_busy; then
    echo "== 포트 $PORT를 쥔 이전 실행이 있다 — 회수한다"
    reap
  fi

  echo "== WSL($DISTRO)로 넘긴다: $LINUX_HERE"
  echo "== 중지: 이 창에서 Ctrl-C (전체 종료)"
  "$WSL" -d "$DISTRO" -- env \
    EDGE_PIDFILE="$PIDFILE" \
    ${EDGE_PYTHON:+EDGE_PYTHON="$EDGE_PYTHON"} \
    ${EDGE_VERIFY:+EDGE_VERIFY="$EDGE_VERIFY"} \
    bash "$LINUX_HERE/run.sh" "$@" &
  CHILD=$!
  set +e
  wait "$CHILD"
  RC=$?
  set -e
  cleanup
  trap - EXIT
  exit "$RC"
  ;;
esac

# ───────────────────────────────────────────────────────────────────────────
# 여기부터는 리눅스(WSL 안). 실제로 서버를 띄운다.
# ───────────────────────────────────────────────────────────────────────────

PIDFILE="${EDGE_PIDFILE:-/tmp/edge-server.pid}"

# `./run.sh stop` — WSL 안에서도 정리만 할 수 있어야 한다.
if [ "${1:-}" = "stop" ]; then
  PORT="$(arg_port "$@" || python3 -c 'import json;print(json.load(open("config.json"))["server"]["port"])')"
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  # pid 파일이 낡았거나 파이썬이 자식을 남겼어도 **포트는 반드시 푼다**.
  if command -v fuser >/dev/null 2>&1 && fuser "$PORT/tcp" >/dev/null 2>&1; then
    fuser -k -TERM "$PORT/tcp" >/dev/null 2>&1 || true
    for _ in $(seq 1 40); do fuser "$PORT/tcp" >/dev/null 2>&1 || break; sleep 0.25; done
    fuser -k -KILL "$PORT/tcp" >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do fuser "$PORT/tcp" >/dev/null 2>&1 || break; sleep 0.25; done
  fi
  rm -f "$PIDFILE"
  if command -v fuser >/dev/null 2>&1 && fuser "$PORT/tcp" >/dev/null 2>&1; then
    echo "포트 $PORT가 아직 잡혀 있다 — 누가 쥐고 있는지 확인이 필요하다." >&2
    exit 1
  fi
  echo "정리 완료 — 포트 $PORT 반환"
  exit 0
fi

# 후보를 순서대로 본다. 앞의 것이 이긴다.
CANDIDATES=(
  "${EDGE_PYTHON:-}"
  "$HOME/physical-demo-edge-venv/bin/python"
  "/home/ny/physical-demo-edge-venv/bin/python"
  "$HERE/venv310/bin/python"
  "$HERE/venv312/bin/python"
  "$HERE/venv/bin/python"
  "$HOME/Physical-Project-mk2-jny-backup/demo/venv/bin/python"
  "$HOME/Physical-Project-mk2-jny-backup/research/oln_training/venv/bin/python"
)

PY=""
for c in "${CANDIDATES[@]}"; do
  [ -n "$c" ] && [ -x "$c" ] || continue
  if "$c" -c "import torch, ultralytics" >/dev/null 2>&1; then PY="$c"; break; fi
done

if [ -z "$PY" ]; then
  echo "torch + ultralytics가 있는 파이썬을 찾지 못했다." >&2
  echo "확인한 곳:" >&2
  for c in "${CANDIDATES[@]}"; do [ -n "$c" ] && echo "  $c" >&2; done
  echo >&2
  echo "고치는 법 중 하나:" >&2
  echo "  EDGE_PYTHON=/경로/python ./run.sh" >&2
  echo "  또는  python3.12 -m venv venv312 && ./venv312/bin/pip install \\" >&2
  echo "          torch torchvision --index-url https://download.pytorch.org/whl/cu123 \\" >&2
  echo "        && ./venv312/bin/pip install -r requirements.txt" >&2
  exit 1
fi

echo "== 파이썬: $PY"
"$PY" - <<'PYEOF'
import torch, ultralytics
print(f"   torch {torch.__version__}  cuda {torch.cuda.is_available()}"
      + (f"  {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
print(f"   ultralytics {ultralytics.__version__}")
PYEOF

# MoGe2-Aerial은 peft/moge가 필요하고, 그 둘은 pylibs/에 따로 깔려 있다.
export PYTHONPATH="$HERE/pylibs:$HERE/third_party/utils3d:$HERE/third_party/AerialMetric/MoGe${PYTHONPATH:+:$PYTHONPATH}"

# 가중치 점검 — 없으면 해당 단계만 빠지므로 미리 알려 준다.
"$PY" - <<'PYEOF'
import json, os
cfg = json.load(open("config.json"))
def chk(label, rel):
    if not rel: return
    ok = os.path.exists(rel)
    print(f"   {label:24s} {'OK' if ok else '없음'}  {rel}")
for k in ("ovd", "ovd2"):
    if cfg.get(k, {}).get("enabled"):
        chk(f"{k} weights", cfg[k].get("weights"))
d = cfg.get("depth", {})
if d.get("enabled"):
    prov = d.get("provider", "moge2_aerial")
    chk(f"depth[{prov}]", d.get("zipdepth_model") if prov == "zipdepth"
        else d.get("checkpoint"))
PYEOF

if [ "${EDGE_VERIFY:-0}" = "1" ]; then
  "$PY" -u server.py "$@" &
  SERVER_PID=$!
  echo "$SERVER_PID" > "$PIDFILE"
  # INT/TERM을 서버에 그대로 넘긴다. server.py가 SIGTERM을 KeyboardInterrupt로
  # 받아 리슨 소켓까지 반환하므로, 여기서 끊지 말고 **전달만** 한다.
  trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true' INT TERM HUP
  trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true; rm -f "$PIDFILE"' EXIT

  echo "== 서버 기동 및 검증 대기"
  for _ in $(seq 1 180); do
    if curl -fsS --max-time 2 http://127.0.0.1:8891/api/health >/tmp/edge-health.json 2>/dev/null; then
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "서버가 초기화 중 종료됐다." >&2
      wait "$SERVER_PID"
      exit 1
    fi
    sleep 1
  done
  test -s /tmp/edge-health.json || { echo "health 응답 timeout" >&2; exit 1; }

  echo "== Pi3 프레임 검증"
  curl -fsS --max-time 20 -o /tmp/pi3-frame \
    "http://100.85.243.54:8890/api/frame?camera_id=0"
  test -s /tmp/pi3-frame || { echo "Pi3 프레임이 비어 있다" >&2; exit 1; }

  echo "== 실시간 게이트 활성화"
  "$PY" server.py ctl infer-on --port 8891
  "$PY" server.py ctl push-on --port 8891
  "$PY" server.py ctl pull-on --port 8891
  echo "== 상태"
  "$PY" server.py ctl status --port 8891
  echo "== live API"
  curl -fsS --max-time 10 http://127.0.0.1:8891/api/live
  echo
  echo "== 원본 스트림 포트 확인"
  curl -fsS --max-time 3 -o /tmp/edge-original-stream \
    http://127.0.0.1:8891/original?camera=0
  test -s /tmp/edge-original-stream || { echo "원본 스트림이 비어 있다" >&2; exit 1; }
  echo "검증 완료: http://127.0.0.1:8891/"
  wait "$SERVER_PID"
else
  # exec이라 pid가 그대로 파이썬의 것이 된다 — 감시자가 이 pid로 죽인다.
  echo $$ > "$PIDFILE"
  exec "$PY" -u server.py "$@"
fi
