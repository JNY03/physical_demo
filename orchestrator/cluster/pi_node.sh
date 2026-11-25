#!/usr/bin/env bash
# pi_node.sh — 파이 한 대에서 root 로 돈다. cluster_up.sh 가 ssh 로 보내 실행한다.
#
#   check                       cgroup 메모리 · 기존 k3s 연결 주소 · boot_id
#   remove-old <server_url>     연결 주소가 <server_url> 이 아닌 k3s agent 를 백업 후 지운다
#   cgroup                      부팅 줄에 cgroup_memory=1 cgroup_enable=memory 를 더한다 (재부팅 필요하면 "changed")
#   join <ver> <url> <name> <ip> <role> <taint 0|1>   k3s agent 설치 — 토큰은 표준입력 첫 줄
#   registry <host:port>        사설 레지스트리(http)를 containerd 미러로 등록. 바뀌었을 때만 k3s-agent 재시작 ("changed" | "same")
set -euo pipefail
cmd="${1:-}"; shift || true
SVC_ENV=/etc/systemd/system/k3s-agent.service.env

current_url() { grep -h '^K3S_URL' "$SVC_ENV" 2>/dev/null | cut -d= -f2- | tr -d "'\"" || true; }

case "$cmd" in
registry)
    reg="$1"; f=/etc/rancher/k3s/registries.yaml
    want="# data_stream 사설 레지스트리 (테일넷 http) — Docker 계획 D8. cluster_up.sh registry 가 쓴다
mirrors:
  \"$reg\":
    endpoint:
      - \"http://$reg\""
    if [ -f "$f" ] && [ "$(cat "$f")" = "$want" ]; then echo "same"; exit 0; fi
    mkdir -p /etc/rancher/k3s
    [ -f "$f" ] && cp "$f" "$f.bak_$(date +%y%m%d_%H%M%S)"
    printf '%s\n' "$want" > "$f"
    systemctl restart k3s-agent
    echo "changed"
    ;;
check)
    echo "cgroup_mem=$(grep -qw memory /sys/fs/cgroup/cgroup.controllers 2>/dev/null && echo yes || echo no)"
    echo "k3s_url=$(current_url)"
    echo "k3s_active=$(systemctl is-active k3s-agent 2>/dev/null || true)"
    echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
    ;;
remove-old)
    want="$1"; have="$(current_url)"
    if [ -z "$have" ]; then echo "no-old"; exit 0; fi
    if [ "$have" = "$want" ]; then echo "same-cluster"; exit 0; fi
    b=/root/k3s_old_backup_$(date +%y%m%d_%H%M%S)
    mkdir -p "$b"; chmod 700 "$b"
    cp -a /etc/systemd/system/k3s-agent.service* "$b/" 2>/dev/null || true
    cp -a /etc/rancher/k3s "$b/etc_rancher_k3s" 2>/dev/null || true
    if [ -x /usr/local/bin/k3s-agent-uninstall.sh ]; then /usr/local/bin/k3s-agent-uninstall.sh >/dev/null 2>&1 || true; fi
    echo "removed old=$have backup=$b"
    ;;
cgroup)
    f=/boot/firmware/cmdline.txt; [ -f "$f" ] || f=/boot/cmdline.txt
    if grep -q 'cgroup_enable=memory' "$f" && grep -q 'cgroup_memory=1' "$f"; then echo "unchanged"; exit 0; fi
    cp -n "$f" "$f.bak_k3s"
    sed -i '1 s/$/ cgroup_memory=1 cgroup_enable=memory/' "$f"
    echo "changed"
    ;;
join)
    ver="$1"; url="$2"; name="$3"; ip="$4"; role="$5"; taint="$6"
    read -r token
    [ -n "$token" ] || { echo "토큰이 없습니다" >&2; exit 1; }
    extra="--node-label ds.role=$role"
    [ "$taint" = 1 ] && extra="$extra --node-taint ds.role=$role:NoSchedule"
    curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="$ver" K3S_URL="$url" K3S_TOKEN="$token" \
        INSTALL_K3S_EXEC="agent --node-name $name --node-ip $ip --flannel-iface tailscale0 $extra" sh - >/dev/null
    systemctl is-active k3s-agent
    ;;
*)
    echo "사용법: $0 check | remove-old <url> | cgroup | join <ver> <url> <name> <ip> <role> <taint>" >&2; exit 2;;
esac
