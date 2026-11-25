#!/usr/bin/env bash
# install_docker_wsl.sh — WSL(Ubuntu, systemd) 에 docker-ce + buildx · compose + NVIDIA Container Toolkit + arm64 빌드(QEMU)를 설치한다.
#   sudo bash tools/docker/install_docker_wsl.sh [사용자]
# 다시 돌려도 된다. 사설 레지스트리(테일넷 :5000)를 http 로 믿도록 daemon.json 에 넣는다.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "sudo 로 실행하세요"; exit 1; }
USER_NAME="${1:-${SUDO_USER:-}}"
. /etc/os-release
CODENAME="${UBUNTU_CODENAME:-$VERSION_CODENAME}"

apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg jq
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
if ! curl -fsI "https://download.docker.com/linux/ubuntu/dists/$CODENAME/Release" >/dev/null; then CODENAME=noble; fi
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $CODENAME stable" \
  > /etc/apt/sources.list.d/docker.list

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /etc/apt/keyrings/nvidia-container-toolkit.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update -qq
# arm64 에뮬레이션: Ubuntu 25.10+ 는 qemu-user(정적) + qemu-user-binfmt, 그 전은 qemu-user-static
QEMU=qemu-user-static
apt-cache policy qemu-user-static 2>/dev/null | grep -q 'Candidate: (none)' && QEMU="qemu-user qemu-user-binfmt"
apt-cache show qemu-user-static >/dev/null 2>&1 || QEMU="qemu-user qemu-user-binfmt"
export DEBIAN_FRONTEND=noninteractive
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit $QEMU

# 테일넷 사설 레지스트리(이 기기 :5000)를 http 로 허용 + NVIDIA 런타임 등록
TS_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
mkdir -p /etc/docker
cfg=/etc/docker/daemon.json; [ -s "$cfg" ] || echo '{}' > "$cfg"
regs='["localhost:5000","127.0.0.1:5000"'"${TS_IP:+,\"$TS_IP:5000\"}"']'
tmp=$(mktemp); jq --argjson r "$regs" '."insecure-registries" = ((."insecure-registries" // []) + $r | unique)' "$cfg" > "$tmp" && mv "$tmp" "$cfg"
nvidia-ctk runtime configure --runtime=docker >/dev/null

systemctl enable --now docker
systemctl restart docker
[ -n "$USER_NAME" ] && usermod -aG docker "$USER_NAME"

echo "--- 확인"
docker version --format 'docker {{.Server.Version}}'
docker compose version
docker buildx version
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || echo "GPU 컨테이너 시험 실패 — 이 출력과 함께 알려 주세요"
docker run --rm --platform linux/arm64 busybox:1.36 uname -m || echo "arm64 에뮬레이션 실패"
echo "완료 — 사용자 '$USER_NAME' 는 WSL 을 다시 열거나 'newgrp docker' 뒤에 sudo 없이 docker 를 쓸 수 있다"
