"""tools/fetch_models.py — 가중치 받기 · 검증 (2026-11-25, Docker 계획 D3).

네트워크는 쓰지 않는다: 받기는 이 테스트가 띄우는 로컬 http.server, git 은 임시 저장소로 시험한다.
"""
import hashlib
import http.server
import re
import shutil
import subprocess
import sys
import threading
from functools import partial
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_models as fm  # noqa: E402

cc = fm.cc
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- 해시
def test_sha256_file(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    assert fm.sha256_file(f) == sha(b"abc")


def test_manifest_hash_is_sorted_relpath_and_file_hash(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.txt").write_bytes(b"B")
    (tmp_path / "sub" / "a.txt").write_bytes(b"A")
    want = sha(("b.txt\t%s\nsub/a.txt\t%s\n" % (sha(b"B"), sha(b"A"))).encode())
    assert fm.manifest_hash(tmp_path) == want
    assert fm.manifest_hash(tmp_path, ["sub/a.txt", "b.txt"]) == want          # include 순서와 무관
    # .cache (HF 내려받기 흔적) 는 세지 않는다
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "x.lock").write_bytes(b"")
    assert fm.manifest_hash(tmp_path) == want
    # include 로 일부만
    assert fm.manifest_hash(tmp_path, ["b.txt"]) == sha(("b.txt\t%s\n" % sha(b"B")).encode())
    # 빠진 파일이 있으면 None
    assert fm.manifest_hash(tmp_path, ["b.txt", "없음.txt"]) is None
    (tmp_path / "b.txt").write_bytes(b"changed")
    assert fm.manifest_hash(tmp_path) != want


def test_resolve_path(tmp_path):
    home = tmp_path / "ds"
    assert fm.resolve_path("models/x.pt", home) == home / "models" / "x.pt"
    assert fm.resolve_path("models/a/b.onnx", home, dest=tmp_path / "w") == tmp_path / "w" / "a" / "b.onnx"
    assert fm.resolve_path("/old/repo/models/x.pt", home, dest=tmp_path / "w") == tmp_path / "w" / "x.pt"
    assert fm.resolve_path("/old/repo/models/x.pt", home) == Path("/old/repo/models/x.pt")
    assert fm.resolve_path("external/UniDepth", home, dest=tmp_path / "w") == home / "external" / "UniDepth"


def test_hf_names():
    assert fm.hf_repo_dir("lpiccinelli/unidepth-v2-vitb14") == "models--lpiccinelli--unidepth-v2-vitb14"
    assert fm.hf_repo_dir("datasets/Kuiee/AerialMetric-ECCV2026") == "datasets--Kuiee--AerialMetric-ECCV2026"
    assert fm.hf_file_url("datasets/o/n", "r1", "w/x.pt").endswith("/datasets/o/n/resolve/r1/w/x.pt")
    assert fm.hf_file_url("o/n", "r1", "x").endswith("/o/n/resolve/r1/x")


# ---------------------------------------------------------------- 레지스트리
def test_registry_sources_are_pinned():
    """files 가 있는 모델은 모두 source 를 갖고, 받는 항목은 고정 판 · 해시를 갖는다."""
    for m, spec in cc._REG["models"].items():
        if spec.get("files"):
            assert spec.get("source"), "%s 에 source 가 없다" % m
        for s in spec.get("source") or []:
            if "repo" in s:
                assert HEX40.match(s["commit"]) and s["path"].startswith("external/"), (m, s)
                continue
            if "manual" in s:
                assert "sha256" not in s or HEX64.match(s["sha256"]), (m, s)
                continue
            assert HEX64.match(str(s.get("sha256"))), (m, s)
            if "hf" in s:
                assert HEX40.match(str(s["rev"])), (m, s)           # 브랜치 이름이 아니라 커밋
                assert ("file" in s) != ("include" in s), (m, s)
            else:
                assert str(s["url"]).startswith("https://"), (m, s)
            assert s.get("cache") or not str(s["path"]).startswith("/"), (m, s)   # DS_HOME 기준 상대 경로


def test_items_cover_all_models():
    items = fm.items_for(set(cc._REG["models"]), home=Path("/ds"))
    assert items[0]["kind"] == "git"                      # 외부 저장소가 먼저 (그 안에 받을 가중치가 있다)
    wheres = [i["where"] for i in items]
    assert wheres.count("/ds/external/UniDepth") == 1     # depth · unidepth_l 이 함께 쓰지만 한 번만
    assert {i["model"] for i in items} >= {"yoloe_text", "clip", "reid", "depth", "moge2_aerial"}


# ---------------------------------------------------------------- 필요한 모델
def test_needed_options():
    assert fm.needed(["yoloe"])[0] == {"yoloe_pf_l"}
    assert fm.needed(["river"])[0] == {"yoloe_text", "yoloe_free", "clip_fp32", "depth"}
    assert fm.needed(["unidepth"])[0] == {"unidepth_l"}
    assert fm.needed(["clip"])[0] == {"clip"}             # 모델 키를 바로 줘도 된다
    with pytest.raises(SystemExit):
        fm.needed(["없는선택지"])


def test_needed_targeted_follows_cue_needs(monkeypatch):
    """빈 models/ 처럼 아무 가중치도 없어도 단서 needs (clip · depth · OVD) 를 따라 받을 목록이 나온다."""
    monkeypatch.setattr(cc, "have", lambda key, files: False)
    prof = {"models": {"targeted": {"targets": [{"name": "traffic cone", "size_mm": [300, 300, 700],
                                                 "colours": ["orange"]}]}}}
    need, trained = fm.needed([], prof, {})
    assert {"yoloe_text", "yoloe_free", "clip", "depth"} <= need     # free_arm · semantic · size_veto
    assert "yolo_world" in need                                       # 탐지기 근거의 OVD 확신
    assert "grounding_dino" not in need                               # 그 근거를 선언한 대상이 아니다
    assert trained == []
    assert cc.have("x", {}) is False                                  # 판정 함수는 원래대로 돌아온다
    # 거리를 끄면 depth 가 빠진다
    prof["models"]["targeted"]["distance"] = False
    assert "depth" not in fm.needed([], prof, {})[0]


def test_needed_trained_target_and_reid():
    need, trained = fm.needed(["targeted"], {"models": {"targeted": {"targets": ["demo_box"]}},
                                             "features": {"reid": True}}, {})
    assert {"trained", "reid"} <= need
    assert [n for n, _ in trained] == ["demo_box", "demo_box"]
    assert any(str(p).endswith(".pt") for _, p in trained)
    need, _ = fm.needed(["targeted"], {"models": {"targeted": {"targets": ["demo_door"]}}}, {})
    assert "grounding_dino" in need                       # demo_door 는 GDINO 근거를 선언했다


# ---------------------------------------------------------------- 검사 (--check)
@pytest.fixture
def fake_models(monkeypatch, tmp_path):
    """레지스트리에 시험용 모델을 끼운다 → (등록 함수, models 폴더)"""
    dest = tmp_path / "models"

    def add(key, source, files=("models/x",)):
        monkeypatch.setitem(cc._REG["models"], key, {"files": list(files), "source": source})
    return add, dest


def _rows(models, dest, fetch=False, hf_cache=None):
    rows, code = fm.run(fm.items_for(models, home=dest.parent, dest=dest, hf_cache=hf_cache), fetch, log=lambda *a: None)
    return [r["status"] for r in rows], code


def test_check_ok_missing_mismatch_manual(fake_models):
    add, dest = fake_models
    dest.mkdir()
    (dest / "good.pt").write_bytes(b"good")
    (dest / "bad.pt").write_bytes(b"tampered")
    (dest / "d").mkdir()
    (dest / "d" / "c.json").write_bytes(b"{}")
    add("t_good", [{"path": "models/good.pt", "url": "https://x/good.pt", "sha256": sha(b"good")}])
    add("t_bad", [{"path": "models/bad.pt", "url": "https://x/bad.pt", "sha256": sha(b"good")}])
    add("t_miss", [{"path": "models/none.pt", "url": "https://x/none.pt", "sha256": sha(b"x")}])
    add("t_dir", [{"path": "models/d", "hf": "o/n", "rev": "0" * 40, "include": ["c.json"],
                   "sha256": fm.manifest_hash(dest / "d", ["c.json"])}])
    add("t_manual", [{"path": "models/hand.json", "manual": "직접 복사"}])
    add("t_nohash", [{"path": "models/good.pt", "url": "https://x/good.pt"}])
    assert _rows({"t_good", "t_dir"}, dest) == (["ok", "ok"], 0)
    assert _rows({"t_bad"}, dest) == (["mismatch"], 1)
    assert _rows({"t_miss"}, dest) == (["missing"], 1)
    assert _rows({"t_manual"}, dest) == (["manual"], 1)          # 필요한 것이 없으면 0 이 아니다
    assert _rows({"t_nohash"}, dest) == (["unverified"], 0)
    add("t_opt", [{"path": "models/hand.json", "manual": "선택", "optional": True}])
    assert _rows({"t_opt"}, dest) == (["manual"], 0)


def test_check_mode_never_writes(fake_models):
    add, dest = fake_models
    add("t_miss", [{"path": "models/none.pt", "url": "http://127.0.0.1:9/none.pt", "sha256": sha(b"x")}])
    assert _rows({"t_miss"}, dest, fetch=False) == (["missing"], 1)
    assert not dest.exists()


# ---------------------------------------------------------------- 받기 (로컬 http.server)
@pytest.fixture
def server(tmp_path):
    root = tmp_path / "srv"
    root.mkdir()
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield root, "http://127.0.0.1:%d" % httpd.server_address[1]
    httpd.shutdown()


def _leftovers(d):
    return [p for p in Path(d).rglob("*") if p.name.endswith(".part") or p.name.startswith(".fetch-")]


def test_download_url_verifies_and_is_atomic(fake_models, server):
    add, dest = fake_models
    root, url = server
    (root / "w.pt").write_bytes(b"weights" * 1000)
    add("t_url", [{"path": "models/w.pt", "url": url + "/w.pt", "sha256": sha(b"weights" * 1000)}])
    assert _rows({"t_url"}, dest, fetch=True) == (["ok"], 0)
    assert (dest / "w.pt").read_bytes() == b"weights" * 1000
    assert not _leftovers(dest)
    # 이미 맞으면 다시 받지 않는다
    (root / "w.pt").unlink()
    assert _rows({"t_url"}, dest, fetch=True) == (["ok"], 0)


def test_download_hash_mismatch_is_rejected(fake_models, server):
    add, dest = fake_models
    root, url = server
    (root / "evil.pt").write_bytes(b"not what the registry says")
    add("t_evil", [{"path": "models/evil.pt", "url": url + "/evil.pt", "sha256": sha(b"expected")}])
    statuses, code = _rows({"t_evil"}, dest, fetch=True)
    assert statuses == ["failed"] and code == 1
    assert not (dest / "evil.pt").exists()                       # 틀린 파일은 제자리에 오지 않는다
    assert not _leftovers(dest)
    # 이미 있던 (틀린) 파일도 틀린 받기로 덮어쓰지 않는다
    (dest / "evil.pt").write_bytes(b"old")
    assert _rows({"t_evil"}, dest, fetch=True)[0] == ["failed"]
    assert (dest / "evil.pt").read_bytes() == b"old"


def test_download_hf_dir_and_cache(fake_models, server, monkeypatch, tmp_path):
    add, dest = fake_models
    root, url = server
    monkeypatch.setattr(fm, "HF_URL", url)
    rev = "a" * 40
    files = {"config.json": b'{"a": 1}', "sub/model.safetensors": b"\0" * 4096}
    for n, b in files.items():
        p = root / "org" / "m" / "resolve" / rev / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    good = sha("".join("%s\t%s\n" % (n, sha(files[n])) for n in sorted(files)).encode())
    add("t_hf", [{"path": "models/m", "hf": "org/m", "rev": rev, "include": sorted(files), "sha256": good}])
    assert _rows({"t_hf"}, dest, fetch=True) == (["ok"], 0)
    assert (dest / "m" / "sub" / "model.safetensors").read_bytes() == files["sub/model.safetensors"]
    assert not _leftovers(dest)
    # HF 캐시 배치 — snapshots/<rev>/ 와 refs/main
    cache = tmp_path / "hf"
    add("t_cache", [{"hf": "org/m", "rev": rev, "cache": True, "include": sorted(files), "sha256": good}])
    assert _rows({"t_cache"}, dest, fetch=True, hf_cache=cache) == (["ok"], 0)
    snap = cache / "models--org--m" / "snapshots" / rev
    assert (snap / "config.json").is_file()
    assert (cache / "models--org--m" / "refs" / "main").read_text() == rev
    # refs/main 이 다른 판을 가리키면 mismatch
    (cache / "models--org--m" / "refs" / "main").write_text("b" * 40)
    assert _rows({"t_cache"}, dest, hf_cache=cache) == (["mismatch"], 1)
    # manifest 가 틀리면 아무것도 옮기지 않는다
    add("t_hf_bad", [{"path": "models/m2", "hf": "org/m", "rev": rev, "include": sorted(files), "sha256": "0" * 64}])
    assert _rows({"t_hf_bad"}, dest, fetch=True)[0] == ["failed"]
    assert not any(p.is_file() for p in (dest / "m2").rglob("*"))


def test_hf_single_file(fake_models, server, monkeypatch):
    add, dest = fake_models
    root, url = server
    monkeypatch.setattr(fm, "HF_URL", url)
    p = root / "datasets" / "o" / "d" / "resolve" / ("c" * 40) / "weights" / "x.pt"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"lora")
    add("t_ds", [{"path": "external/A/weights/x.pt", "hf": "datasets/o/d", "rev": "c" * 40,
                  "file": "weights/x.pt", "sha256": sha(b"lora")}])
    assert _rows({"t_ds"}, dest, fetch=True) == (["ok"], 0)
    assert (dest.parent / "external" / "A" / "weights" / "x.pt").read_bytes() == b"lora"


# ---------------------------------------------------------------- git
@pytest.mark.skipif(shutil.which("git") is None, reason="git 없음")
def test_git_clone_pins_commit(fake_models, tmp_path):
    add, dest = fake_models
    up = tmp_path / "upstream"
    up.mkdir()
    env = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    run = partial(subprocess.run, check=True, capture_output=True, cwd=str(up))
    run(["git", "init", "-q", "-b", "main"])
    (up / "f.txt").write_text("one")
    run(["git", "add", "f.txt"])
    run(["git"] + env + ["commit", "-qm", "1"])
    first = run(["git", "rev-parse", "HEAD"], text=True).stdout.strip()
    (up / "f.txt").write_text("two")
    run(["git"] + env + ["commit", "-qam", "2"])
    add("t_git", [{"path": "external/Up", "repo": str(up), "commit": first}])
    assert _rows({"t_git"}, dest) == (["missing"], 1)
    assert _rows({"t_git"}, dest, fetch=True) == (["ok"], 0)
    clone = dest.parent / "external" / "Up"
    assert (clone / "f.txt").read_text() == "one"                 # 브랜치 끝이 아니라 고정한 커밋
    assert fm.git_head(clone) == first
    # 있는 저장소가 다른 커밋이면 손대지 않고 알린다
    add("t_git2", [{"path": "external/Up", "repo": str(up), "commit": "f" * 40}])
    statuses, code = _rows({"t_git2"}, dest, fetch=True)
    assert statuses == ["failed"] and code == 1
    assert fm.git_head(clone) == first


# ---------------------------------------------------------------- 명령줄
def test_main_hash_and_check(tmp_path, capsys, fake_models):
    f = tmp_path / "w.pt"
    f.write_bytes(b"w")
    assert fm.main(["--hash", str(f)]) == 0
    assert capsys.readouterr().out.strip() == sha(b"w")
    add, dest = fake_models
    add("t_main", [{"path": "models/none.pt", "url": "https://x/none.pt", "sha256": sha(b"x")}])
    assert fm.main(["--check", "--models", "t_main", "--dest", str(dest), "--json"]) == 1
    out = capsys.readouterr().out
    assert '"missing"' in out and '"ok": false' in out


def test_does_not_import_torch():
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, %r); import fetch_models; "
                        "print('torch' in sys.modules)" % str(HERE)], capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "False"
