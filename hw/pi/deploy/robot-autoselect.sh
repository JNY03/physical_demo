#!/bin/bash
# 붙어 있는 로봇에 컨테이너를 맞춘다 (reconcile)
# ================================================
# "Go1 이냐 EP 냐"를 택일로 보지 않는다. 둘 다 붙어 있으면 둘 다 띄운다 —
# HW-interface/multi-robot-contract.md 시나리오 04(pi1 + Go1 + EP 혼재)가 그것이고,
# 두 컨테이너는 entity_id 와 spool 볼륨이 갈라져 있어 서로 간섭하지 않는다.
#
# 감지 근거는 기종마다 하나씩이고, 둘 다 "그 로봇이 실제로 응답한다"는 사실이다.
# USB 가 꽂혀 있다거나 인터페이스가 떴다는 것만으로는 부족하다 — 전원이 꺼진 로봇에
# 케이블만 꽂혀 있어도 그것들은 참이 된다.
#
#   EP  : RNDIS 192.168.42.2   (USB 케이블)
#   Go1 : 내부망 192.168.123.161 (랜선)
#
# 설치:
#   sudo cp pi/deploy/robot-autoselect.{service,timer} /etc/systemd/system/
#   sudo cp pi/deploy/99-hw-robot.rules /etc/udev/rules.d/
#   sudo systemctl daemon-reload && sudo udevadm control --reload
#   sudo systemctl enable --now robot-autoselect.timer
#
# 타이머(10초)가 주기적으로 맞추고, udev 가 USB 착탈 순간에 즉시 한 번 더 부른다.
set -u

COMPOSE_FILE="${COMPOSE_FILE:-/home/physical/hw/pi/deploy/docker-compose.yml}"
EP_HOST="${EP_HOST:-192.168.42.2}"
GO1_HOST="${GO1_HOST:-192.168.123.161}"
STATE_DIR="${STATE_DIR:-/run/hw-robot}"
# 응답이 몇 번 연속으로 없어야 내리는가. 1 로 두면 핑 한 번 놓친 것으로 컨테이너가
# 내려가고, 다음 주기에 다시 뜬다 — 무선·USB 는 그 정도로 흔들린다.
MISS_LIMIT="${MISS_LIMIT:-3}"

mkdir -p "$STATE_DIR"

alive() { ping -c1 -W1 "$1" >/dev/null 2>&1; }

running() {
    [ -n "$(docker compose -f "$COMPOSE_FILE" ps -q "$1" 2>/dev/null)" ]
}

reconcile() {
    # 한 local 문에 몰아 쓰면 안 된다 — bash 는 local 의 인자를 대입 **전에** 전부
    # 전개하므로 같은 줄에서 $svc 를 참조하면 set -u 가 unbound 로 잡는다.
    local svc="$1"
    local host="$2"
    local miss_file="$STATE_DIR/$svc.miss"
    local miss

    if alive "$host"; then
        echo 0 > "$miss_file"
        if running "$svc"; then
            return
        fi
        echo "[autoselect] $svc 응답($host) — 띄운다"
        # 서비스 이름을 명시하면 compose v2 가 그 서비스의 profile 을 알아서 켠다.
        docker compose -f "$COMPOSE_FILE" up -d "$svc"
        return
    fi

    # `|| return` 은 직전 명령의 실패 상태(1)를 그대로 반환한다. 그게 스크립트의
    # 마지막 문장이 되면(= Go1 이 안 붙어 있는 평소) 서비스가 실패로 끝난다.
    running "$svc" || return 0        # 원래 안 떠 있으면 셀 것도 없다

    miss=$(( $(cat "$miss_file" 2>/dev/null || echo 0) + 1 ))
    echo "$miss" > "$miss_file"
    if [ "$miss" -lt "$MISS_LIMIT" ]; then
        echo "[autoselect] $svc 무응답 $miss/$MISS_LIMIT — 아직 기다린다"
        return
    fi
    echo "[autoselect] $svc 무응답 $miss 회 — 내린다"
    # rm -sf : 멈추고 지운다. 볼륨은 남으므로 미전송 spool 은 다음에 살아 돌아온다.
    docker compose -f "$COMPOSE_FILE" rm -sf "$svc"
    echo 0 > "$miss_file"
}

reconcile ep  "$EP_HOST"
reconcile go1 "$GO1_HOST"

# 로봇이 하나도 없거나, 있는 그대로 두는 것도 정상 결과다. 마지막 판정의 종료 코드가
# 서비스 상태로 새어 나가지 않게 명시적으로 끝낸다.
exit 0
