#!/usr/bin/env bash
# MoGe2-Aerial 체크포인트가 준비되면 depth provider를 바꾼다.
#
# 받는 데 오래 걸려(6 Mbps 회선에서 ~35분) 그동안 zipdepth로 돌려 두었다.
# zipdepth는 상대 역depth라 **미터를 내지 않는다** — 바꿔야 거리가 나온다.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CKPT="models/Moge2-Aerial.pt"
if [ ! -f "$CKPT" ]; then
  echo "아직 없다: $CKPT"
  INC=$(find "$HOME/.cache/huggingface" -path '*Aerial*' -name '*.incomplete' 2>/dev/null | head -1)
  if [ -n "$INC" ]; then
    SZ=$(stat -c%s "$INC")
    printf "  받는 중: %.0f MB / 1480 MB (%.0f%%)\n" "$((SZ/1048576))" "$((SZ*100/1552000000))"
  else
    echo "  다운로드 프로세스도 없다. 다시 받으려면:  ./fetch_models.sh"
  fi
  exit 1
fi

# MoGe 포크와 peft가 있어야 로드된다 — 없으면 바꿔 봐야 "없음"으로 빠진다.
[ -d third_party/AerialMetric/MoGe ] || { echo "third_party/AerialMetric 없음 — ./fetch_models.sh"; exit 1; }

python3 - <<'PY'
import json
from pathlib import Path
p = Path("config.json"); c = json.loads(p.read_text(encoding="utf-8"))
c["depth"]["provider"] = "moge2_aerial"
p.write_text(json.dumps(c, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("depth.provider = moge2_aerial")
PY
echo "이제 ./run.sh 로 다시 띄운다. 지연을 꼭 재라 —"
echo "  209ms를 넘으면 depth.mode 를 \"async\"로 돌려야 한다(config 주석 참고)."
