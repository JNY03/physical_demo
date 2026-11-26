"""기준 사진 근거(ref_match) — 사람이 올린 사진과 후보가 얼마나 닮았나를 점수로 (2026-11-26).

가중치 없이 돈다: 임베딩은 파일에서 읽는 것뿐이고, 후보 쪽 임베딩은 시험이 직접 만들어 넣는다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
ta = pytest.importorskip("target_assess")


def _refs(tmp_path, name, vecs, encoder="clip_b32", state="ready"):
    """대상 파일과 임베딩 묶음을 만든다 (cue_catalog 가 저장하는 모양 그대로)"""
    arr = np.asarray(vecs, dtype="float32")
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    npz = tmp_path / ("%s.refs.npz" % name)
    np.savez(npz, **{"%s/%s" % (encoder, name): arr})
    f = tmp_path / ("%s.json" % name)
    f.write_text(json.dumps({"name": name}), encoding="utf-8")
    return {"name": name, "_file": str(f), "cues": {"ref_match": "score"},
            "reference_images": {"dir": "%s.refs" % name, "files": ["01.png"], "count": len(arr),
                                 "encoder": encoder, "state": state, "embeddings": npz.name}}


def test_embeddings_are_read_and_normalised(tmp_path):
    t = _refs(tmp_path, "vest", [[3.0, 0.0], [0.0, 4.0]])
    out = ta.load_reference_embeddings(t)
    assert out.shape == (2, 2)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)       # 길이 1 로 맞춰 둔다


def test_unknown_and_missing_are_not_a_pass(tmp_path):
    """모름과 없음을 가른다 — 못 읽으면 None 이고, None 은 통과가 아니다"""
    t = _refs(tmp_path, "vest", [[1.0, 0.0]], state="pending")
    assert ta.load_reference_embeddings(t) is None            # 가중치가 없어 아직 못 만든 것
    t2 = _refs(tmp_path, "cone", [[1.0, 0.0]])
    Path(t2["_file"]).parent.joinpath(t2["reference_images"]["embeddings"]).unlink()
    assert ta.load_reference_embeddings(t2) is None           # 파일이 사라진 것
    t3 = _refs(tmp_path, "box", [[1.0, 0.0]], encoder="other") # 키가 다른 인코더
    t3["reference_images"]["encoder"] = "clip_b32"
    assert ta.load_reference_embeddings(t3) is None


def _row(prep, emb):
    """_one 을 돌려 ref_match 한 줄만 떼어 본다 — (%, 설명, 통과여부)"""
    ev = ta.TargetEvidence.__new__(ta.TargetEvidence)
    ev.prep = {prep.name: prep}
    ev.targets = {prep.name: prep.t}
    ev.clip = None
    ev.parts = None
    d = {"name": prep.name, "xyxy": [0.0, 0.0, 100.0, 100.0], "_crop": None if emb is None else object()}
    out = ta.TargetEvidence._one(ev, d, emb, None, 640, 480)
    v = (out.get("cue_values") or {}).get("ref_match")
    card = next((r for r in (out["cards"][prep.name]["rows"]) if r[0] == "ref_match"), ["", None, ""])
    return (None if v is None else v["pct"]), card[2], (None if v is None else v["passed"])


def test_score_follows_the_distance_from_the_threshold(tmp_path):
    """하한에서 50%, +0.10 에서 100%, −0.10 아래는 0% — 특징 사전과 같은 자"""
    t = _refs(tmp_path, "vest", [[1.0, 0.0]])
    t["ref_sim_min"] = 0.70
    prep = ta._Prepared(t, None)
    assert prep.refs is not None
    def at(cos):
        emb = np.array([cos, np.sqrt(max(0.0, 1.0 - cos * cos))], dtype="float32")
        return _row(prep, emb)
    assert at(0.70)[0] == 50.0 and at(0.70)[2] is True
    assert at(0.80)[0] == 100.0
    assert at(0.60)[0] == 0.0 and at(0.60)[2] is False        # 하한 아래는 떨어진다(gate 로 쓸 수 있게)
    assert 50.0 < at(0.75)[0] < 100.0


def test_small_crop_is_not_measured_rather_than_zero(tmp_path):
    """작은 조각이라 임베딩이 없으면 0점이 아니라 '못 잼'이다 — 0점이면 멀쩡한 판정을 끌어내린다"""
    t = _refs(tmp_path, "vest", [[1.0, 0.0]])
    prep = ta._Prepared(t, None)
    pct, text, passed = _row(prep, None)
    assert pct is None and "not measured" in text and passed is None


def test_best_of_several_photos_is_used(tmp_path):
    """사진이 여럿이면 가장 닮은 몇 장의 평균 — 한 장만 맞아도 어느 정도는 오른다"""
    t = _refs(tmp_path, "vest", [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    t["ref_sim_min"] = 0.70
    prep = ta._Prepared(t, None)
    pct, text, _passed = _row(prep, np.array([1.0, 0.0], dtype="float32"))
    assert pct is not None and "3 photos" in text and "best 1.0000" in text
