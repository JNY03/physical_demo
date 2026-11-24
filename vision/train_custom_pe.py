"""train_custom_pe.py — 박스 친 이미지로 YOLOE 새 클래스 임베딩을 만든다 (가중치는 건드리지 않음).

YOLOE 는 클래스마다 임베딩 벡터 하나로 찾는다. 텍스트 모드에서는 MobileCLIP 이 문구를 벡터로
바꾸지만 'go1' · 'insta 360' 같은 고유명은 텍스트 인코더가 모른다. 그래서 정답 박스마다
시각 프롬프트 임베딩(VPE)을 뽑아 클래스별로 평균한 것을 그 클래스의 벡터로 쓴다.
가중치를 다시 학습하지 않으므로 기존 어휘(person · boat …)의 성능은 그대로다.

**원본 장면에서 뽑는다 — 크롭에서 뽑지 않는다** (2026-11-24 측정). YOLOE VPE 는 박스 안
특징만이 아니라 주변 맥락까지 섞는다. 크롭을 회색 캔버스에 붙여 뽑은 VPE 는 같은 물체를
장면에서 뽑은 VPE 와 코사인 평균 0.16 이었고, 그 평균으로는 Insta360 을 하나도 못 찾았다
(realsense 세션 교차검증 0/40). 원본 프레임 + 박스로 뽑으면 40/40 이다.
그래서 입력은 `{class, image, box}` 목록(train_list.json)이다 — sample 폴더의 크롭은
사람이 보는 용도, 학습은 그 크롭을 잘라낸 원본과 박스로 한다(manifest.csv 에 둘 다 있다).

좌우 반전을 한 번 더 넣어 박스당 VPE 2개 → 정규화 → 클래스별 평균 → 다시 정규화.
k-means 로 대표 벡터를 여럿 두는 것도 재 봤으나 평균보다 낫지 않았다.

  python vision/train_custom_pe.py --list models/custom_pe/train_list.json \
      --out models/custom_pe/yoloe-11l_custom.npz

결과 NPZ 는 ultralytics YOLOE.load_prompt_embeddings 형식이고, 만든 가중치 계열(yoloe-11l)에만
맞는다. vision_config.yaml models.yoloe.prompt.learned_pe 에 경로를 적으면 classes 에 그 이름이
있을 때 텍스트 임베딩 대신 이 벡터를 쓴다.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def load_bgr(path):
    """jpg/png(투명 배경 포함) → BGR. 투명한 곳은 흰색으로 채운다."""
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return np.array(bg.convert("RGB"))[:, :, ::-1].copy()


def box_vpes(model, bgr, box):
    """원본 장면 + 박스 → VPE 2개(원본 · 좌우 반전)."""
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    W = bgr.shape[1]
    x1, y1, x2, y2 = box
    out = []
    for img, b in ((bgr, [x1, y1, x2, y2]), (cv2.flip(bgr, 1), [W - x2, y1, W - x1, y2])):
        model.predict(img, refer_image=img, predictor=YOLOEVPSegPredictor, verbose=False,
                      visual_prompts=dict(bboxes=np.array([b], dtype=float), cls=np.array([0])))
        out.append(F.normalize(model.model.pe[0, 0].float(), dim=-1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weight", default="models/yoloe-11l-seg.pt", help="-pf 가 아닌 YOLOE 가중치")
    ap.add_argument("--list", default="models/custom_pe/train_list.json",
                    help="[{class, image, box:[x1,y1,x2,y2]}] — 상대 경로는 작업 폴더 기준")
    ap.add_argument("--out", default="models/custom_pe/yoloe-11l_custom.npz")
    a = ap.parse_args()

    from ultralytics import YOLOE
    model = YOLOE(a.weight)
    vecs = defaultdict(list)
    for item in json.load(open(a.list)):
        vecs[item["class"].strip()] += box_vpes(model, load_bgr(item["image"]), item["box"])

    names = list(vecs)
    embs = []
    for n in names:
        embs.append(F.normalize(torch.stack(vecs[n]).mean(0), dim=-1))
        print("%-12s 박스 %3d개 → VPE %3d개 평균" % (n, len(vecs[n]) // 2, len(vecs[n])))

    model = YOLOE(a.weight)                    # VP 예측기가 바꿔 둔 상태를 버리고 새로 싣는다
    model.set_classes(names, torch.stack(embs)[None].to(next(model.model.parameters()).device))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    print("저장:", model.save_prompt_embeddings(a.out))


if __name__ == "__main__":
    main()
