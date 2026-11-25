#!/usr/bin/env bash
# cluster_up.sh — nodes.json 의 파이들을 data_stream k3s 클러스터에 붙인다. WSL 에서 **sudo 없이** 실행한다.
#
#   bash orchestrator/cluster/cluster_up.sh prep    기존(다른) k3s 제거 · cgroup 설정 · 필요하면 재부팅
#   bash orchestrator/cluster/cluster_up.sh join    agent 설치 · 라벨 · taint (서버 설치 뒤)
#   bash orchestrator/cluster/cluster_up.sh registry 사설 레지스트리(이 기기 테일넷 :REGISTRY_PORT, 기본 5000)를 파이 containerd 에 등록
#   bash orchestrator/cluster/cluster_up.sh caps    파이의 특성(센서 · 가속기 · 기기 이름) · 온도를 읽어 라벨 · 애너테이션으로 올린다
#   bash orchestrator/cluster/cluster_up.sh all     prep · join · registry · caps
#   ... <명령> pi7 pi10                              일부 노드만
#
# 준비: 파이에 SSH 키 로그인(physical@<이름>). 파이 sudo 에 비밀번호가 필요하면 PI_SUDO_PASS 환경변수로 준다
#       (파일에 적지 않는다). join 은 install_server.sh 가 만든 ~/.kube/k3s-node-token 을 읽는다.
# 다시 돌려도 된다 — 이미 된 단계는 건너뛴다.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NODES="$HERE/nodes.json"
PI_USER="${PI_USER:-physical}"
step="${1:-all}"; shift || true
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10)
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"     # k3s 의 kubectl 은 없으면 root 전용 /etc/rancher/k3s/k3s.yaml 을 읽는다

jq_py() { python3 -c "import json,sys; d=json.load(open('$NODES')); $1" "${@:2}"; }
VERSION="$(jq_py 'print(d["k3s_version"])')"
TS_IP="$(tailscale ip -4 | head -1)"
URL="https://$TS_IP:6443"
REGISTRY="${DS_REGISTRY:-$TS_IP:$(jq_py 'print(d.get("registry_port", 5000))')}"
mapfile -t ALL < <(jq_py 'print("\n".join("%s %s %d %s" % (a["name"], a["role"], a["role"] in d["taint_roles"], a.get("source", "-")) for a in d["agents"]))')
want=("$@")

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
declare -A NEEDPASS=()
needpass() {  # 비밀번호 없는 sudo 면 비밀번호 줄을 보내지 않는다 — 보내면 그 줄이 join 의 토큰 자리로 들어간다
    local h="$1"
    if [ -z "${NEEDPASS[$h]:-}" ]; then
        if "${SSH[@]}" "$PI_USER@$h" "sudo -n true" >/dev/null 2>&1; then NEEDPASS[$h]=0; else NEEDPASS[$h]=1; fi
    fi
    [ "${NEEDPASS[$h]}" = 1 ]
}
pass_line() { if needpass "$1"; then [ -n "${PI_SUDO_PASS:-}" ] || { echo "$1: sudo 비밀번호 필요 (PI_SUDO_PASS)" >&2; return 1; }; printf '%s\n' "$PI_SUDO_PASS"; fi; }
rsudo() {  # rsudo <host> <args...> — 원격 root 로 pi_node.sh 실행. 명령이 표준입력으로 받을 줄은 STDIN_EXTRA
    local h="$1"; shift
    { pass_line "$h"; [ -n "${STDIN_EXTRA:-}" ] && printf '%s\n' "$STDIN_EXTRA"; true; } |
        "${SSH[@]}" "$PI_USER@$h" "sudo -S -p '' bash /tmp/pi_node.sh $*"
}
wait_ssh() {  # 재부팅 뒤 다시 붙을 때까지 (최대 240초)
    local h="$1" old="$2"
    for i in $(seq 48); do
        sleep 5
        local now; now="$("${SSH[@]}" "$PI_USER@$h" cat /proc/sys/kernel/random/boot_id 2>/dev/null || true)"
        [ -n "$now" ] && [ "$now" != "$old" ] && return 0
    done
    return 1
}

for line in "${ALL[@]}"; do
    read -r name role taint source <<<"$line"
    if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " =~ " $name " ]]; then continue; fi
    if ! scp -q -o BatchMode=yes -o ConnectTimeout=10 "$HERE/pi_node.sh" "$PI_USER@$name:/tmp/pi_node.sh"; then
        log "$name: SSH 안 됨 — 건너뜀"; continue
    fi
    if [ "$step" = prep ] || [ "$step" = all ]; then
        log "$name: $(rsudo "$name" remove-old "$URL")"
        res="$(rsudo "$name" cgroup)"
        if [ "$res" = changed ]; then
            boot="$("${SSH[@]}" "$PI_USER@$name" cat /proc/sys/kernel/random/boot_id)"
            log "$name: cgroup 설정 바꿈 — 재부팅"
            { pass_line "$name"; true; } |
                "${SSH[@]}" "$PI_USER@$name" "sudo -S -p '' systemctl reboot" >/dev/null 2>&1 || true
            if wait_ssh "$name" "$boot"; then log "$name: 다시 붙음"; else log "$name: 재부팅 뒤 안 돌아옴 — 확인 필요"; continue; fi
            scp -q -o BatchMode=yes "$HERE/pi_node.sh" "$PI_USER@$name:/tmp/pi_node.sh"
        fi
        log "$name: $(rsudo "$name" check | tr '\n' ' ')"
    fi
    if [ "$step" = join ] || [ "$step" = all ]; then
        TOKEN_FILE="$HOME/.kube/k3s-node-token"
        [ -s "$TOKEN_FILE" ] || { log "토큰 파일이 없습니다 — sudo bash $HERE/install_server.sh 먼저"; exit 1; }
        if [ "$(kubectl get node "$name" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)" = True ]; then
            log "$name: 이미 Ready"
        else
            ip="$("${SSH[@]}" "$PI_USER@$name" "tailscale ip -4 | head -1")"
            log "$name: agent 설치 ($ip, role=$role)"
            STDIN_EXTRA="$(cat "$TOKEN_FILE")" rsudo "$name" join "$VERSION" "$URL" "$name" "$ip" "$role" "$taint" | sed "s/^/  $name: /"
        fi
        kubectl label node "$name" "ds.role=$role" --overwrite >/dev/null 2>&1 || true
        # ds.source — 이 노드에 붙은 data_stream 소스. 자원 기록기가 로봇 서비스가 실제로 도는지(그 소스 영상이 오는지) 가른다
        [ "$source" != "-" ] && kubectl label node "$name" "ds.source=$source" --overwrite >/dev/null 2>&1 || true
        if [ "$taint" = 1 ]; then kubectl taint node "$name" "ds.role=$role:NoSchedule" --overwrite >/dev/null 2>&1 || true; fi
    fi
    if [ "$step" = registry ] || [ "$step" = all ]; then
        log "$name: 레지스트리 $REGISTRY — $(rsudo "$name" registry "$REGISTRY")"
    fi
    if [ "$step" = caps ] || [ "$step" = all ]; then
        # 파이에서 특성 · 온도를 읽어(읽기만) 이 기기의 kubectl 로 올린다 — 파이에는 kubectl 이 없다.
        # 라벨 값에는 공백 · 쉼표를 쓸 수 없어(k8s 규칙) 가속기만 라벨, 기기 이름 · 센서 목록 · 온도는 애너테이션.
        # --overwrite 와 「없으면 지우기」라서 다시 돌려도 같은 결과다(멱등).
        caps_out="$("${SSH[@]}" "$PI_USER@$name" "bash /tmp/pi_node.sh caps" 2>/dev/null || true)"
        c_arch=""; c_model=""; c_sensors=""; c_accel=""; c_temp=""; c_thr=""
        while IFS='=' read -r k v; do
            case "$k" in
            arch) c_arch="$v";; model) c_model="$v";; sensors) c_sensors="$v";;
            accel) c_accel="$v";; temp_c) c_temp="$v";; throttled) c_thr="$v";;
            esac
        done <<<"$caps_out"
        if [ -z "$c_arch" ]; then
            log "$name: 특성을 못 읽음 (SSH · pi_node.sh caps) — 건너뜀"
        else
            kset() {  # kset <키> <값> — 값이 비면 지운다 (옛 값이 남아 배치를 속이지 않게)
                local kind="$1" key="$2" val="$3"
                if [ -n "$val" ]; then kubectl "$kind" node "$name" "$key=$val" --overwrite >/dev/null 2>&1 || true
                else kubectl "$kind" node "$name" "$key-" >/dev/null 2>&1 || true; fi
            }
            kset label "ds.accel" "$c_accel"
            kset annotate "ds.model" "$c_model"
            kset annotate "ds.sensors" "$c_sensors"
            kset annotate "ds.thermal/temp_c" "$c_temp"
            kset annotate "ds.thermal/throttled" "$c_thr"
            log "$name: 특성 arch=$c_arch accel=${c_accel:--} sensors=${c_sensors:--} temp=${c_temp:--} throttled=${c_thr:--} model=${c_model:--}"
        fi
    fi
done
if [ "$step" != prep ]; then kubectl get nodes -L ds.role -L ds.accel -o wide 2>/dev/null || true; fi
