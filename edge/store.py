"""엣지 저장과 보존 — 무엇을 남기고 무엇을 지울 것인가.

implements: AI-L-01, AI-L-02, AI-O-03, AI-B-10

**왜 필요했나**(2026-09-21): `store/unknown_crops`에 **46만 장(2.9 GB)** 이 쌓였다.
저장 조건에 중복 방지가 전혀 없어서 배경이 아닌 미지 후보를 **매 프레임 전부** 다시
썼기 때문이다. 정지 장면에서는 같은 물체가 초당 2장씩 영원히 쌓인다.

AI-L-01은 "모든 입력 데이터를 학습용으로 저장하는 것을 기본으로 하지 않고, 후보가 된
이유와 함께 선별"하라고 요구한다. 같은 물체의 1000번째 crop은 새 이유를 담고 있지
않으므로 후보가 아니다. 그래서 세 층으로 막는다.

1. **정체성 기준 게이트**(CropGate) — 이미 본 자리와의 IoU로 슬롯을 찾아 슬롯당
   장수와 간격을 제한한다.
2. **파일 수 상한**(Rotation) — 정체성이 없는 경우에도 무한히 자라지 않게.
3. **바이트 상한**(Rotation) — 2026-09-21에 새로 넣었다. 아래 실측 참고.

── 왜 파일 수만으로는 부족했나 ─────────────────────────────────────────────
상한은 지켜지고 있었는데 저장소가 **5.9 GB**였다.

    records       14,199 files  3.3 G   ← 232 KB/파일
    overlay        3,026 files  1.2 G   ← 400 KB/파일
    meta          14,178 files  1.1 G   ←  77 KB/파일

record 한 장(448 KB)을 열어 보니 `unknown` 항목 300개가 244 KB로 55%를 먹고 있었다
— 후보당 815 B, CLIP ranking 전체를 매 프레임 그대로 쓰고 있었다. `records` 상한이
20000이니 가득 차면 4.6 GB다. **개수는 비용이 아니다.** 비용은 바이트이므로 상한도
바이트로 걸어야 한다. 둘을 AND로 걸어 먼저 걸리는 쪽이 이긴다.

삭제는 재현 가능성과 맞바꾸는 값이므로(AI-O-03) `records`처럼 감사·재현에 쓰이는
것은 넉넉히, `overlay`처럼 눈으로만 보는 것은 짧게 둔다. 어느 쪽이든 값은 설정에
드러나 있어야 한다.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


class CropGate:
    """같은 것을 반복 저장하지 않는 게이트.

    **track_id로는 묶이지 않는다**(2026-09-21 실측). 미지 crop에 붙은 track_id는
    RPN 후보의 것인데, 후보는 프레임마다 500~600개가 새로 나오고 id가 계속 갈린다 —
    이 게이트를 track_id로 걸었더니 90초에 2166장이 그대로 쌓였다. 화면 정체성이
    RPN에서 흩어진 것과 같은 원인이다.

    그래서 **위치로 묶는다.** 정지한 장면에서 같은 물체는 같은 자리에 있으므로,
    박스를 격자로 양자화한 서명이 사실상 그 물체의 임시 식별자가 된다. 카메라가
    움직이면 서명이 바뀌어 새로 받는데, 그건 실제로 새 관측이므로 맞는 동작이다.

    격자를 쓰는 대가: 서로 다른 두 물체가 거의 같은 자리·크기면 하나로 본다.
    학습 후보 선별에서 그 정도 손실은 46만 장이 쌓이는 것보다 낫다(AI-L-01:
    모든 입력을 저장하지 않고 이유가 있는 것만 선별한다).
    """

    def __init__(self, cfg_store: dict):
        p = cfg_store.get("unknown_crops_policy", {})
        self.max_per_slot = int(p.get("max_per_slot", 2))
        self.min_interval_s = float(p.get("min_interval_s", 20.0))
        self.match_iou = float(p.get("match_iou", 0.5))
        self.max_per_frame = int(p.get("max_per_frame", 4))
        self.ttl_s = float(p.get("slot_ttl_s", 120.0))
        # key → (box, n, last_t, verdict)
        self._seen: dict[tuple, tuple] = {}
        self._next = 0
        self.skipped = 0
        self.saved = 0

    def _find_slot(self, cam: int, box):
        """이미 본 자리인가 — **겹침으로 찾는다. 고정 격자를 쓰지 않는다.**

        격자로 양자화하면 경계에 걸친 박스가 프레임마다 두 슬롯을 오가며 둘 다
        저장을 받는다(2026-09-21 실측: 격자 48px에서 초당 7장이 계속 저장됐다).
        이 저장소에서 같은 함정에 두 번 빠졌다 — 노출 기하 양자화에서도 같은 이유로
        불감대로 바꿨다. 기준을 고정 격자가 아니라 **이미 있는 것**으로 잡으면
        경계가 생기지 않는다.
        """
        best, best_v = None, self.match_iou
        for k, (b, _, _, _) in self._seen.items():
            if k[0] != cam:
                continue
            v = _iou(box, b)
            if v >= best_v:
                best, best_v = k, v
        return best

    def allow(self, cam: int, box, verdict: str, now: float | None = None) -> bool:
        """이 crop을 저장할 것인가.

        같은 자리에서 같은 판정이면 `max_per_slot`장까지, 그 사이 최소
        `min_interval_s` 간격을 둔다. 판정이 바뀌면(배경↔미지) 새 이유이므로 한 장
        더 받는다 — AI-L-01이 요구하는 것은 '후보가 된 이유'이지 장수가 아니다.

        거절할 때도 슬롯의 박스는 갱신한다. 물체가 천천히 움직이면 박스가 조금씩
        옮겨가는데, 갱신하지 않으면 원래 자리와의 겹침이 끊겨 새 슬롯이 생긴다.
        """
        now = now or time.time()
        box = tuple(float(v) for v in box)
        key = self._find_slot(cam, box)
        if key is None:
            self._next += 1
            key = (cam, self._next)
            n, last, prev = 0, 0.0, ""
        else:
            _, n, last, prev = self._seen[key]

        if verdict == prev:
            if n >= self.max_per_slot or (now - last) < self.min_interval_s:
                self._seen[key] = (box, n, last, prev)
                self.skipped += 1
                return False
        elif n >= self.max_per_slot + 1:
            self._seen[key] = (box, n, last, prev)
            self.skipped += 1
            return False

        self._seen[key] = (box, n + 1, now, verdict)
        self.saved += 1
        return True

    def sweep(self, now: float | None = None) -> None:
        """오래된 슬롯을 버린다 — 장면이 바뀌면 옛 자리 기록은 쓸모가 없다."""
        now = now or time.time()
        for k in [k for k, (_, _, t, _) in self._seen.items() if now - t > self.ttl_s]:
            self._seen.pop(k, None)

    def stats(self) -> dict:
        return {"slots": len(self._seen), "saved": self.saved,
                "skipped": self.skipped}


def prune_dir(path: Path, max_files: int = 0, max_bytes: int = 0) -> tuple[int, int]:
    """오래된 파일부터 지워 **개수와 바이트를 동시에** 상한 이하로 만든다.

    둘은 AND다 — 먼저 걸리는 쪽이 이긴다. 개수만 걸면 record 한 장이 448 KB일 때
    20000개가 4.6 GB가 되고, 바이트만 걸면 작은 파일이 수십만 개 쌓여 파일시스템이
    느려진다. 실제로 둘 다 겪어서 둘 다 건다.

    mtime 기준이라 같은 관측의 .jpg/.json 쌍이 갈릴 수 있지만, 둘은 거의 같은
    시각에 쓰이므로 경계에서만 어긋나고 그 한 쌍은 다음 호출에서 정리된다.

    Returns: (지운 개수, 확보한 바이트)
    """
    if not path.is_dir() or (max_files <= 0 and max_bytes <= 0):
        return 0, 0
    try:
        entries = [(e.stat().st_mtime, e.stat().st_size, e.path)
                   for e in os.scandir(path) if e.is_file()]
    except OSError:
        return 0, 0

    total_bytes = sum(sz for _, sz, _ in entries)
    over_count = len(entries) - max_files if max_files > 0 else 0
    if over_count <= 0 and (max_bytes <= 0 or total_bytes <= max_bytes):
        return 0, 0

    entries.sort()                       # 오래된 것부터
    removed = freed = 0
    for i, (_, size, fp) in enumerate(entries):
        need_count = max_files > 0 and (len(entries) - removed) > max_files
        need_bytes = max_bytes > 0 and (total_bytes - freed) > max_bytes
        if not (need_count or need_bytes):
            break
        try:
            os.unlink(fp)
            removed += 1
            freed += size
        except OSError:
            pass
    return removed, freed


class Rotation:
    """디렉터리별 개수·바이트 상한. 매번 훑으면 비싸므로 간격을 두고 검사한다."""

    def __init__(self, root: Path, cfg_store: dict):
        self.root = Path(root)

        def caps(key):
            # 설정에는 `$comment` 같은 주석 키가 섞여 있다 — 숫자만 받는다.
            return {k: int(v) for k, v in (cfg_store.get(key) or {}).items()
                    if not k.startswith("$") and isinstance(v, (int, float))}

        self.file_caps = caps("max_files")
        # MB로 적게 한다 — 바이트로 적으면 0이 몇 개인지 세게 된다.
        self.byte_caps = {k: v * 1024 * 1024 for k, v in caps("max_mb").items()}
        self.every_s = float(cfg_store.get("prune_interval_s", 30.0))
        self._last = 0.0
        self.removed_total = 0
        self.freed_total = 0

    def maybe_prune(self, now: float | None = None) -> dict:
        now = now or time.time()
        if now - self._last < self.every_s:
            return {}
        self._last = now
        out = {}
        for name in set(self.file_caps) | set(self.byte_caps):
            n, freed = prune_dir(self.root / name,
                                 self.file_caps.get(name, 0),
                                 self.byte_caps.get(name, 0))
            if n:
                out[name] = {"removed": n, "freed_mb": round(freed / 1048576, 1)}
                self.removed_total += n
                self.freed_total += freed
        return out

    def usage(self) -> dict:
        """디렉터리별 현재 사용량. 상한이 실제로 무엇을 막고 있는지 헬스에 드러낸다."""
        out = {}
        for name in set(self.file_caps) | set(self.byte_caps):
            d = self.root / name
            if not d.is_dir():
                continue
            try:
                sizes = [e.stat().st_size for e in os.scandir(d) if e.is_file()]
            except OSError:
                continue
            out[name] = {"files": len(sizes),
                         "mb": round(sum(sizes) / 1048576, 1),
                         "cap_files": self.file_caps.get(name, 0),
                         "cap_mb": round(self.byte_caps.get(name, 0) / 1048576)}
        return out
