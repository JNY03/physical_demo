#!/usr/bin/env bash
# 엣지를 띄운다. **torch가 있는 파이썬을 골라 준다.**
#
# 왜 이 스크립트가 있나: `./venv/bin/python server.py`로 띄우면 조용히 축소
# 모드로 돈다 — 그 venv에는 torch도 ultralytics도 없어서 YOLOE와 depth가 전부
# "없음"으로 빠지고, 화면에는 원본만 나온다. 로그를 안 보면 왜 비었는지 알 수
# 없다. 여기서 먼저 확인하고, 없으면 **뜨지 않고 이유를 말한다**(AI-O-02).
#
#   ./run.sh                  # 기본 포트(config server.port)
#   ./run.sh --port 9000
#   EDGE_PYTHON=/path/to/python ./run.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 후보를 순서대로 본다. 앞의 것이 이긴다.
CANDIDATES=(
  "${EDGE_PYTHON:-}"
  "$HERE/venv312/bin/python"
  "$HERE/venv/bin/python"
  "$HOME/Physical-Project-mk2-jny-backup/demo/venv/bin/python"
  "$HOME/Physical-Project-mk2-jny-backup/research/oln_training/venv/bin/python"
)

PY=""
for c in "${CANDIDATES[@]}"; do
  [ -n "$c" ] && [ -x "$c" ] || continue
  if "$c" -c "import torch, ultralytics" >/dev/null 2>&1; then PY="$c"; break; fi
done

if [ -z "$PY" ]; then
  echo "torch + ultralytics가 있는 파이썬을 찾지 못했다." >&2
  echo "확인한 곳:" >&2
  for c in "${CANDIDATES[@]}"; do [ -n "$c" ] && echo "  $c" >&2; done
  echo >&2
  echo "고치는 법 중 하나:" >&2
  echo "  EDGE_PYTHON=/경로/python ./run.sh" >&2
  echo "  또는  python3.12 -m venv venv312 && ./venv312/bin/pip install \\" >&2
  echo "          torch torchvision --index-url https://download.pytorch.org/whl/cu124 \\" >&2
  echo "        && ./venv312/bin/pip install -r requirements.txt" >&2
  exit 1
fi

echo "== 파이썬: $PY"
"$PY" - <<'PYEOF'
import torch, ultralytics
print(f"   torch {torch.__version__}  cuda {torch.cuda.is_available()}"
      + (f"  {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
print(f"   ultralytics {ultralytics.__version__}")
PYEOF

# MoGe2-Aerial은 peft/moge가 필요하고, 그 둘은 pylibs/에 따로 깔려 있다.
[ -d "$HERE/pylibs" ] && export PYTHONPATH="$HERE/pylibs${PYTHONPATH:+:$PYTHONPATH}"

# 가중치 점검 — 없으면 해당 단계만 빠지므로 미리 알려 준다.
"$PY" - <<'PYEOF'
import json, os
cfg = json.load(open("config.json"))
def chk(label, rel):
    if not rel: return
    ok = os.path.exists(rel)
    print(f"   {label:24s} {'OK' if ok else '없음'}  {rel}")
for k in ("ovd", "ovd2"):
    if cfg.get(k, {}).get("enabled"):
        chk(f"{k} weights", cfg[k].get("weights"))
d = cfg.get("depth", {})
if d.get("enabled"):
    prov = d.get("provider", "moge2_aerial")
    chk(f"depth[{prov}]", d.get("zipdepth_model") if prov == "zipdepth"
        else d.get("checkpoint"))
PYEOF

exec "$PY" -u server.py "$@"
