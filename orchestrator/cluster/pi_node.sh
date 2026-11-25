#!/usr/bin/env bash
# pi_node.sh — 파이 한 대에서 root 로 돈다. cluster_up.sh 가 ssh 로 보내 실행한다.
#
#   check                       cgroup 메모리 · 기존 k3s 연결 주소 · boot_id
#   remove-old <server_url>     연결 주소가 <server_url> 이 아닌 k3s agent 를 백업 후 지운다
#   cgroup                      부팅 줄에 cgroup_memory=1 cgroup_enable=memory 를 더한다 (재부팅 필요하면 "changed")
#   join <ver> <url> <name> <ip> <role> <taint 0|1>   k3s agent 설치 — 토큰은 표준입력 첫 줄
#   registry <host:port>        사설 레지스트리(http)를 containerd 미러로 등록. 바뀌었을 때만 k3s-agent 재시작 ("changed" | "same")
#   caps                        이 파이의 특성 · 온도를 key=value 로 낸다 (읽기만 — 아무것도 고치지 않는다).
#                               kubectl 은 서버에만 있으니 cluster_up.sh caps 가 이 값을 라벨 · 애너테이션으로 올린다.
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
caps)
    # 이 파이의 특성 · 온도를 key=value 로 낸다 — 못 읽은 칸은 아예 내지 않는다(모르는 것을 짐작하지 않는다).
    a="$(uname -m)"; case "$a" in x86_64|amd64) a=amd64;; aarch64|arm64) a=arm64;; esac
    echo "arch=$a"
    m=""
    [ -r /proc/device-tree/model ] && m="$(tr -d '\0' < /proc/device-tree/model)"
    [ -z "$m" ] && [ -r /sys/devices/virtual/dmi/id/product_name ] && m="$(cat /sys/devices/virtual/dmi/id/product_name)"
    [ -n "$m" ] && echo "model=$m"
    s=""
    compgen -G "/dev/i2c-*" >/dev/null 2>&1 && s="i2c"
    if compgen -G "/dev/ttyACM*" >/dev/null 2>&1 || compgen -G "/dev/ttyUSB*" >/dev/null 2>&1; then s="${s:+$s,}serial"; fi
    # 카메라: /dev/video0~9 중 **V4L2 캡처 장치**만 센다. 파이 5 의 /dev/video19 이상은 ISP · 코덱
    # (메모리↔메모리) 노드라 카메라가 없어도 생긴다 — 번호만 보고 세면 안 된다. 캡처 비트(0x1001)가 있고
    # M2M 비트(0xC000)가 없어야 카메라다. v4l2-ctl 이 없으면 확인할 수 없어 세지 않는다.
    # CSI 카메라는 libcamera 쪽을 봐야 알 수 있어 여기서는 가리지 않는다 (모르면 모른다).
    cam=no
    if command -v v4l2-ctl >/dev/null 2>&1; then
        for d in /dev/video[0-9]; do
            [ -e "$d" ] || continue
            ch="$(v4l2-ctl --device="$d" --info 2>/dev/null |
                  sed -n 's/^[[:space:]]*Device Caps[[:space:]]*:[[:space:]]*\(0x[0-9a-fA-F]*\).*/\1/p' | head -1)"
            [ -n "$ch" ] || continue
            if [ $(( ch & 0x1001 )) -ne 0 ] && [ $(( ch & 0xC000 )) -eq 0 ]; then cam=yes; fi
        done
    fi
    [ "$cam" = yes ] && s="${s:+$s,}camera"
    [ -n "$s" ] && echo "sensors=$s"
    ac=none
    compgen -G "/dev/hailo*" >/dev/null 2>&1 && ac=npu
    if [ "$ac" = none ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then ac=gpu; fi
    echo "accel=$ac"
    t=""; thr=""
    if command -v vcgencmd >/dev/null 2>&1; then
        t="$(timeout 3 vcgencmd measure_temp 2>/dev/null | sed -n "s/^temp=\([0-9.]*\).*/\1/p")"
        thr="$(timeout 3 vcgencmd get_throttled 2>/dev/null | sed -n 's/^throttled=//p')"
    fi
    if [ -z "$t" ]; then
        z="$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1 || true)"
        if [ -n "$z" ]; then t="$(awk -v v="$z" 'BEGIN{printf "%.1f", v/1000}')"; fi
    fi
    [ -n "$t" ] && echo "temp_c=$t"
    [ -n "$thr" ] && echo "throttled=$thr"
    exit 0
    ;;
*)
    echo "사용법: $0 check | remove-old <url> | cgroup | join <ver> <url> <name> <ip> <role> <taint> | caps" >&2; exit 2;;
esac
