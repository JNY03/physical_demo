#!/usr/bin/env bash
# ds_run.sh <이름> <명령...> — 런처(launcher.ps1)가 띄우는 창 하나를 **이름표**와 함께 돌린다 (2026-11-23).
#
# * 명령은 setsid 로 새 프로세스 묶음에서 돈다. 이름표 /tmp/ds_launcher/<이름>.pid 에 묶음 번호와 명령줄을 적는다
#   — ds_stop.sh 가 그 묶음만 끄고, 런처를 다시 열면 ds_status.sh 로 이어받는다.
# * 창에서 Ctrl+C 를 누르면 묶음에 SIGINT 를 넘긴다(예전과 같이 '지금 장까지 저장하고 끝').
# * 런처가 끈 것이면(<이름>.stopped) 바로 닫히고, 스스로 끝났으면 로그를 보도록 키를 기다린다.
set -u
name="$1"; shift
dir=/tmp/ds_launcher
mkdir -p "$dir"
rm -f "$dir/$name.stopped"
setsid "$@" &
pid=$!
printf '%s\t%s\n' "$pid" "$*" > "$dir/$name.pid"
trap 'kill -INT -- -"$pid" 2>/dev/null' INT
trap 'kill -TERM -- -"$pid" 2>/dev/null' TERM HUP
while kill -0 "$pid" 2>/dev/null; do
    wait "$pid"
done
wait "$pid" 2>/dev/null
rc=$?
# 이름표가 아직 이 프로세스 것이면 지운다(같은 이름으로 새로 띄운 것을 지우지 않게)
if [ "$(cut -f1 "$dir/$name.pid" 2>/dev/null)" = "$pid" ]; then rm -f "$dir/$name.pid"; fi
if [ -f "$dir/$name.stopped" ]; then
    rm -f "$dir/$name.stopped"
    exit 0
fi
echo
echo "끝남 (코드 $rc)"
read -n1 -rsp '아무 키나 누르면 창이 닫힙니다'
exit "$rc"
