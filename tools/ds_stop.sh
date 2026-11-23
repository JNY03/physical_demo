#!/usr/bin/env bash
# ds_stop.sh <이름>... — ds_run.sh 로 띄운 묶음을 끈다. SIGINT(Ctrl+C 와 같다) → 기다림 → SIGTERM → SIGKILL.
# 끄기 전에 그 번호의 명령줄이 이름표에 적힌 명령과 같은지 본다 — 번호가 재사용된 엉뚱한 프로세스를 끄지 않는다.
set -u
dir=/tmp/ds_launcher
wait_s="${DS_STOP_WAIT:-8}"
for name in "$@"; do
    f="$dir/$name.pid"
    [ -f "$f" ] || { echo "$name: 없음"; continue; }
    pid="$(cut -f1 "$f")"; cmd="$(cut -f2- "$f")"
    now="$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null | sed 's/ $//')"
    if [ -z "$now" ]; then echo "$name: 이미 끝남"; rm -f "$f"; continue; fi
    if [ "$now" != "$cmd" ]; then echo "$name: 번호 $pid 가 다른 프로세스 — 건드리지 않음"; rm -f "$f"; continue; fi
    touch "$dir/$name.stopped"
    kill -INT -- -"$pid" 2>/dev/null
    for _ in $(seq 1 $((wait_s * 5))); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
    if kill -0 "$pid" 2>/dev/null; then kill -TERM -- -"$pid" 2>/dev/null; sleep 2; fi
    if kill -0 "$pid" 2>/dev/null; then kill -KILL -- -"$pid" 2>/dev/null; fi
    echo "$name: 끔"
done
