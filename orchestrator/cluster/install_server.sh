#!/usr/bin/env bash
# install_server.sh — data_stream 기기(WSL)에 k3s 서버를 설치한다. **sudo 로 한 번** 실행한다.
#
#   sudo bash orchestrator/cluster/install_server.sh
#
# * 테일넷 주소(tailscale0)로 묶는다 — 파이들은 같은 와이파이여도 테일넷으로 붙는다(WSL 은 NAT 뒤).
# * traefik · servicelb 는 끈다(쓰지 않는다). metrics-server 는 남긴다(노드 사용량).
# * 끝나면 실행한 사용자의 ~/.kube/config 와 ~/.kube/k3s-node-token(600)을 만든다 —
#   cluster_up.sh 가 sudo 없이 이것을 읽는다.
# * 끝에서 orchestrator/k3s/ 의 클러스터 객체(PriorityClass ds-vision-low …)를 폴더째 apply 한다 — 객체가 늘면
#   그 폴더에 파일만 더한다. 실행 중에는 아무도 만들지 않는다(없으면 k3s 원천 check() 가 알리고 CPU 배치만 막는다).
# * k3s 는 자원 기반 비전 오케스트레이션에만 필요하다. 없어도 data_stream · 비전 · 런처는 돈다.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "sudo 로 실행하세요: sudo bash $0" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
USER_NAME="${SUDO_USER:-$(logname)}"
USER_HOME="$(getent passwd "$USER_NAME" | cut -d: -f6)"
VERSION="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["k3s_version"])' "$HERE/nodes.json")"
NODE="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["server"]["name"])' "$HERE/nodes.json")"
TS_IP="$(tailscale ip -4 | head -1)"
[ -n "$TS_IP" ] || { echo "tailscale 주소가 없습니다 (tailscale up 먼저)" >&2; exit 1; }

sysctl -qw net.ipv4.ip_forward=1
echo "net.ipv4.ip_forward=1" > /etc/sysctl.d/90-k3s.conf

curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="$VERSION" INSTALL_K3S_EXEC="server \
  --node-name $NODE --node-ip $TS_IP --advertise-address $TS_IP --tls-san $TS_IP \
  --flannel-iface tailscale0 --disable traefik --disable servicelb \
  --write-kubeconfig-mode 600 --node-label ds.role=gpu" sh -

for i in $(seq 60); do [ -s /var/lib/rancher/k3s/server/node-token ] && break; sleep 2; done
install -d -m 700 -o "$USER_NAME" -g "$USER_NAME" "$USER_HOME/.kube"
install -m 600 -o "$USER_NAME" -g "$USER_NAME" /var/lib/rancher/k3s/server/node-token "$USER_HOME/.kube/k3s-node-token"
sed "s#https://127.0.0.1:6443#https://$TS_IP:6443#" /etc/rancher/k3s/k3s.yaml > "$USER_HOME/.kube/config"
chown "$USER_NAME:$USER_NAME" "$USER_HOME/.kube/config"; chmod 600 "$USER_HOME/.kube/config"

# 클러스터 범위 객체 — API 가 받을 때까지 기다렸다가 폴더째 (다시 실행해도 같은 결과: apply)
K="k3s kubectl --kubeconfig /etc/rancher/k3s/k3s.yaml"
for i in $(seq 60); do $K get --raw /readyz >/dev/null 2>&1 && break; sleep 2; done
$K apply -f "$HERE/../k3s/"
echo "k3s 서버 설치 끝 — $NODE @ $TS_IP. 다음: bash orchestrator/cluster/cluster_up.sh"
