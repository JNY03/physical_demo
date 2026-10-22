#!/usr/bin/env bash
# pi7 단말 노드를 pi7 자체 브로커(127.0.0.1)로 되돌린다.
# 실행: sudo bash fix_broker_host.sh
set -euo pipefail

echo "== 변경 전 =="
grep '^HW_BROKER_HOST' /etc/hw-node.env

cp -a /etc/hw-node.env "/etc/hw-node.env.bak.$(date +%y%m%d-%H%M%S)"
sed -i 's|^HW_BROKER_HOST=.*|HW_BROKER_HOST=127.0.0.1|' /etc/hw-node.env

echo "== 변경 후 =="
grep '^HW_BROKER_HOST' /etc/hw-node.env

systemctl restart robot-node sensor-node
sleep 4

echo "== 서비스 상태 =="
systemctl is-active robot-node sensor-node

echo "== 접속 로그 =="
journalctl -u robot-node -u sensor-node -n 12 --no-pager | grep -E '접속|규약|버퍼' || true

echo "== 브로커에 흐르는지 (8초) =="
mosquitto_sub -h localhost -t 'zoneA/#' -t 'terminal/+/uplink' -v -W 8 2>&1 | cut -c1-140 | head -20
