#!/bin/sh
# ds-core · ds-vision-* 컨테이너의 첫 화면 (Docker 계획 §3 · D4)
#
# 처음 띄우는 사람이 제일 먼저 묻는 것은 「내 환경에서 이게 돌아갑니까」다. 그 답을 찾아다니게 두지 않고
# 컨테이너가 뜰 때 preflight 한 장(ds_resources.cli preflight)을 로그에 그대로 찍는다 —
# 자원 · 둘레(GPU · NVIDIA 툴킷 · k3s · Redis) · 가중치를 읽어 기능마다 된다/안 된다와 고치는 길을 적는다.
# 아무것도 깔거나 띄우지 않는다(읽기만 한다). 그 뒤 받은 명령을 exec 한다 — 신호(SIGTERM)가 그대로 간다.
#
#   DS_PREFLIGHT=0   이 안내를 건너뛴다 (로그를 짧게 하고 싶을 때. 몇 초를 아낀다)
set -u

if [ "${DS_PREFLIGHT:-1}" != "0" ]; then
    echo "=============================================================="
    echo " 이 컨테이너에서 무엇이 되는지 (preflight — 읽기만 한다)"
    echo "=============================================================="
    # preflight 는 「쓸 수 있는 기능이 없음」이면 코드 1 로 끝난다. 안내는 안내일 뿐이니 그것으로 컨테이너를 죽이지 않는다.
    python -m ds_resources.cli preflight || echo "(preflight 를 끝까지 못 읽었습니다 — 위 줄까지가 읽은 것입니다)"
    cat <<'EOF'

컨테이너에서 더 읽을 것:
  - 가중치가 없다고 나오면 **호스트에서** 받으세요: python tools/fetch_models.py --models <선택지>
    (models/ 는 컨테이너에 읽기 전용으로 들어옵니다 — 컨테이너 안에서는 받을 수 없습니다)
  - GPU 가 「없음」이면 이 컨테이너는 CPU 로 되는 것만 합니다. GPU 를 쓰려면 호스트에 NVIDIA 컨테이너 툴킷이
    있어야 하고 vision-gpu 프로필로 띄워야 합니다 (docker/README.md).
  - 기기(파이) 주소가 없다고 하면 config/hosts.json 을 채워 마운트하세요 (보기: /ds/config.example/).
  - 자세한 순서는 docker/README.md.
--------------------------------------------------------------
EOF
fi

echo "시작: $*"
exec "$@"
