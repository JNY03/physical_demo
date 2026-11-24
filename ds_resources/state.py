"""실행 상태 파일 — 오케스트레이션 실행에 꼭 필요한 상태 (core, 2026-11-24).

Redis 는 보여 주기용이다. 실행 단위 · 예약 · 배치표는 여기(state/orchestrator_state.json)에 둔다.
Redis 가 죽어도 이 파일로 실행을 이어 간다.

단일 Writer: 파일 잠금(<파일>.lock, fcntl)을 잡은 프로세스만 쓴다. 못 잡으면 읽기 전용(writer=False)으로 열리고
쓰기를 부르면 PermissionError. 저장은 원자적(임시 파일 → 이름 바꾸기)이다.
"""
import copy
import fcntl
import json
import os
import tempfile
import threading
from pathlib import Path


class StateFile:
    def __init__(self, path, writer=True):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.data = {"version": 1, "units": {}, "requests": {}}
        self._lockf = None
        self.writer = False
        if writer:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            f = open(str(self.path) + ".lock", "a+")
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._lockf, self.writer = f, True
            except OSError:
                f.close()
        self.load()

    def load(self):
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(d, dict) and d.get("version") == 1:
                d.setdefault("units", {})
                d.setdefault("requests", {})
                self.data = d
        except (OSError, ValueError):
            pass

    def _save(self):
        if not self.writer:
            raise PermissionError("%s 는 다른 프로세스가 쓰고 있다 (읽기 전용으로 열림)" % self.path)
        fd, tmp = tempfile.mkstemp(prefix=".state-", dir=str(self.path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    def units(self):
        with self.lock:
            if not self.writer:
                self.load()
            return copy.deepcopy(self.data["units"])

    def put_unit(self, uid, rec):
        with self.lock:
            self.data["units"][uid] = copy.deepcopy(rec)
            self._save()

    def update_unit(self, uid, **kw):
        with self.lock:
            self.data["units"].setdefault(uid, {}).update(kw)
            self._save()

    def put_request(self, rid, rec):
        with self.lock:
            self.data["requests"][rid] = copy.deepcopy(rec)
            self._save()

    def close(self):
        if self._lockf is not None:
            try:
                fcntl.flock(self._lockf.fileno(), fcntl.LOCK_UN)
            finally:
                self._lockf.close()
                self._lockf, self.writer = None, False
