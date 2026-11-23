#!/usr/bin/env bash
# ds_status.sh — ds_run.sh 로 띄워 지금 도는 것: 한 줄에 '이름<TAB>번호<TAB>명령'. 죽은 이름표는 지운다.
dir=/tmp/ds_launcher
[ -d "$dir" ] || exit 0
for f in "$dir"/*.pid; do
    [ -f "$f" ] || continue
    name="$(basename "$f" .pid)"; pid="$(cut -f1 "$f")"; cmd="$(cut -f2- "$f")"
    now="$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null | sed 's/ $//')"
    if [ -n "$now" ] && [ "$now" = "$cmd" ]; then printf '%s\t%s\t%s\n' "$name" "$pid" "$cmd"; else rm -f "$f"; fi
done
