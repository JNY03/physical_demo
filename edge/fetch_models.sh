#!/usr/bin/env bash
# 가중치와 서드파티 코드를 내려받는다. 재실행 안전(멱등).
#
# **저장소에 가중치를 커밋하지 않는 이유**: MoGe2-Aerial 체크포인트만 1.48 GB다.
# 엣지 노드를 다른 기기로 옮길 때 git clone이 몇 GB가 되면 옮기기가 싫어진다.
# 코드는 git으로, 가중치는 원본에서 — 그래야 이전이 가볍다.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="$HERE/models"
THIRD="$HERE/third_party"
mkdir -p "$MODELS" "$THIRD"

need() { command -v "$1" >/dev/null || { echo "필요: $1" >&2; exit 1; }; }

echo "== 1/4  AerialMetric 저장소 (MoGe 포크 + LoRA 설정)"
# 본가 microsoft/MoGe가 아니라 **이쪽 포크**를 써야 한다. LoRA 체크포인트의 키가
# 이 포크 기준이라 본가로 로드하면 조용히 어긋난 가중치가 붙는다.
need git
if [ -d "$THIRD/AerialMetric/.git" ]; then
  git -C "$THIRD/AerialMetric" pull --ff-only -q || echo "   (pull 실패 — 기존 것 사용)"
else
  git clone --depth 1 -q https://github.com/kuieless/AerialMetric.git \
      "$THIRD/AerialMetric"
fi
test -f "$THIRD/AerialMetric/MoGe/configs/train/v2.json" \
  && echo "   lora_config OK" \
  || echo "   ⚠ MoGe/configs/train/v2.json 없음 — config.json의 lora_config를 확인한다"

echo "== 2/4  MoGe2-Aerial 체크포인트 (1.48 GB)"
CKPT="$MODELS/Moge2-Aerial.pt"
if [ -f "$CKPT" ]; then
  echo "   이미 있음 — 건너뜀 ($(du -h "$CKPT" | cut -f1))"
else
  need python3
  MODELS="$MODELS" python3 - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
src = hf_hub_download(repo_id="Kuiee/AerialMetric-ECCV2026", repo_type="dataset",
                      filename="weights/Moge2-Aerial.pt")
dst = os.path.join(os.environ["MODELS"], "Moge2-Aerial.pt")
# HF 캐시에서 복사한다. 심볼릭 링크로 두면 캐시를 비울 때 조용히 끊긴다.
shutil.copyfile(src, dst)
print(f"   받음 → {dst}  ({os.path.getsize(dst)/1e9:.2f} GB)")
PY
fi

echo "== 3/4  MoGe 패키지 설치 (AerialMetric 포크)"
# -e로 깔아 두면 저장소를 갱신해도 다시 안 깔아도 된다.
pip install -q -e "$THIRD/AerialMetric/MoGe" 2>&1 | tail -2 || \
  echo "   ⚠ 설치 실패 — third_party/AerialMetric/MoGe를 직접 확인한다"
pip install -q "utils3d @ git+https://github.com/EasternJournalist/utils3d.git" \
  2>&1 | tail -2 || echo "   ⚠ utils3d 설치 실패"

echo "== 4/4  OVD 가중치 (config에서 켜진 것만)"
# **켜진 단계의 가중치만 받는다.** 대역이 좁은 현장에서 안 쓸 25MB를 받는 것은
# 그 자체로 비용이고, models/에 남아 "이게 왜 있지"가 된다.
MODELS="$MODELS" python3 - <<'PY'
import json, os, shutil
from pathlib import Path
cfg = json.loads(Path("config.json").read_text(encoding="utf-8"))
models = Path(os.environ["MODELS"]); models.mkdir(exist_ok=True)
want = [(k, Path(cfg[k]["weights"]).name)
        for k in ("ovd", "ovd2")
        if cfg.get(k, {}).get("enabled") and cfg[k].get("weights")]
if not want:
    print("   켜진 OVD 없음 — 건너뜀")
else:
    try:
        from ultralytics import YOLO
        for key, name in want:
            dst = models / name
            if dst.exists():
                print(f"   {name} 이미 있음"); continue
            YOLO(name)                       # 받아서 캐시에 둔다
            for c in (Path.cwd() / name, Path.home() / ".cache/ultralytics" / name):
                if c.exists():
                    shutil.move(str(c), dst); print(f"   {name} → {dst}"); break
            else:
                print(f"   ⚠ {name} 위치를 못 찾았다 — 수동으로 models/에 둔다")
    except Exception as exc:
        print(f"   ⚠ ultralytics 준비 실패: {exc!r}")
PY

echo
echo "== 완료"
du -sh "$MODELS" "$THIRD" 2>/dev/null || true
echo "확인:  python3 -c \"import moge, peft; print('moge OK')\""
