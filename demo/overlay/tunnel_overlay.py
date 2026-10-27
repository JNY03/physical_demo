"""드론 영상 -> tunnel/goal 오버레이. (--src 로 영상을 준다)

파란 구조물 전체 = tunnel(파란 bbox), U자 안쪽 개구부 = goal(빨간 bbox).
거리는 MoGe2-Aerial 미터 깊이를 **드론 고도 5m 기준으로 보정**해 쓴다.

두 번 훑는다:
  A. 샘플 패스 — N프레임마다 depth를 재서 보정계수 k와 거리 시계열을 만든다.
  B. 렌더 패스 — 전 프레임을 HSV로 검출하고, 샘플 사이는 박스 높이 비로
     거리를 옮겨 그린다(말단 Range.at_box와 같은 원리).
"""
import argparse, os, sys, time
import cv2
import numpy as np

# edge/ 안에서 돌아야 한다 — models.py 가 config.json 의 상대경로(models/, third_party/)를
# 그 디렉터리 기준으로 푼다. 절대경로를 박지 않고 이 파일 위치에서 찾는다.
EDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "edge")
os.chdir(os.path.normpath(EDGE))
sys.path.insert(0, ".")
import io, json
from models import AerialDepth

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--cut", required=True, help="잘라낸 원본 출력")
ap.add_argument("--out", required=True, help="오버레이 출력")
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--depth-every", type=int, default=15)
ap.add_argument("--altitude-m", type=float, default=5.0)
ap.add_argument("--no-depth", action="store_true")
A = ap.parse_args()

LO = np.array([95, 90, 40]); HI = np.array([130, 255, 255])
K9 = np.ones((9, 9), np.uint8); K5 = np.ones((5, 5), np.uint8)
K15 = np.ones((15, 15), np.uint8)


def detect(img):
    """(tunnel_bbox, goal_bbox) — 없으면 None. bbox = (x, y, w, h)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, LO, HI)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K9)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K5)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None, None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[i]
    if area < 5000:                       # 4K에서 이보다 작으면 잡음
        return None, None
    comp = (lab == i).astype(np.uint8)

    goal = None
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        hm = np.zeros_like(comp); cv2.drawContours(hm, [hull], -1, 1, -1)
        hole = ((hm == 1) & (comp == 0)).astype(np.uint8)
        hole = cv2.morphologyEx(hole, cv2.MORPH_OPEN, K15)
        hn, _, hs, _ = cv2.connectedComponentsWithStats(hole, 8)
        if hn > 1:
            j = 1 + int(np.argmax(hs[1:, cv2.CC_STAT_AREA]))
            # **안 보이면 그리지 않는다.** 드론이 U자 뒤로 돌면 개구부가 막힌다.
            if hs[j, cv2.CC_STAT_AREA] > 0.03 * area:
                goal = tuple(int(v) for v in hs[j][:4])
    return (int(x), int(y), int(w), int(h)), goal


def box_depth(dm, box, pct):
    x, y, w, h = box
    x2, y2 = min(dm.shape[1], x + w), min(dm.shape[0], y + h)
    x, y = max(0, x), max(0, y)
    if x2 <= x or y2 <= y:
        return None
    p = dm[y:y2, x:x2]
    v = p[np.isfinite(p) & (p > 0)]
    return float(np.percentile(v, pct)) if v.size else None


cap = cv2.VideoCapture(A.src)
fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
n_want = min(total, int(round(A.seconds * fps)))
print(f"입력 {W}x{H} {fps:.2f}fps  전체 {total}프레임 → 처리 {n_want}프레임 ({n_want/fps:.1f}s)")

# ── 패스 A: depth 샘플 ───────────────────────────────────────────────────────
samples = {}          # frame_idx -> (raw_depth_m, tunnel_h_px)
if not A.no_depth:
    cfg = json.load(io.open("config.json", encoding="utf-8"))
    cfg["_root"] = os.getcwd()
    pct = float(cfg["depth"].get("box_percentile", 25.0))
    dep = AerialDepth(cfg)
    if not dep.available():
        raise SystemExit("depth 모델을 못 올렸다")
    t0 = time.perf_counter()
    idxs = list(range(0, n_want, A.depth_every))
    for k, f in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        tb, _ = detect(img)
        if tb is None:
            continue
        dm = np.asarray(dep.depth_map(img, hfov_deg=None), dtype=np.float32)
        if dm.shape[:2] != img.shape[:2]:
            dm = cv2.resize(dm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        d = box_depth(dm, tb, pct)
        if d:
            samples[f] = (d, tb[3])
        if k % 10 == 0:
            print(f"  depth 샘플 {k+1}/{len(idxs)}  f{f}  d={d}", flush=True)
    print(f"  샘플 {len(samples)}개, {time.perf_counter()-t0:.0f}s")

# ── 보정계수: 고도 5m 기준 ───────────────────────────────────────────────────
raw = [v[0] for v in samples.values()]
if raw:
    med = float(np.median(raw))
    kcal = A.altitude_m / med
    print(f"보정: MoGe 원값 중앙값 {med:.2f}m → 고도 {A.altitude_m}m 기준 계수 k={kcal:.3f}")
else:
    med, kcal = None, 1.0
    print("보정: depth 샘플이 없어 원값 그대로 쓴다")

skeys = sorted(samples)


def dist_at(fi, box_h):
    """가장 가까운 샘플의 거리를 박스 높이 비로 현재 프레임에 옮긴다."""
    if not skeys or not box_h:
        return None
    j = min(skeys, key=lambda s: abs(s - fi))
    d_raw, h_ref = samples[j]
    if not h_ref:
        return None
    return d_raw * kcal * (h_ref / float(box_h))


# ── 패스 B: 렌더 ─────────────────────────────────────────────────────────────
BLUE = (255, 176, 48)      # BGR — 물체보다 밝은 하늘색 계열이라 겹쳐도 보인다
RED = (40, 40, 255)
th = max(3, W // 640)
fs = W / 1400.0


def put(img, txt, org, color):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)


def writer(path):
    for cc in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, (W, H))
        if vw.isOpened():
            print(f"  인코더 {cc} → {os.path.basename(path)}")
            return vw
        vw.release()
    raise SystemExit("VideoWriter를 열지 못했다")


cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
w_cut, w_out = writer(A.cut), writer(A.out)
ema_t = ema_g = None
alpha = 0.4
n_t = n_g = 0
t0 = time.perf_counter()
for fi in range(n_want):
    ok, img = cap.read()
    if not ok:
        break
    w_cut.write(img)
    tb, gb = detect(img)
    vis = img
    if tb is not None:
        ema_t = tb if ema_t is None else tuple(
            int(alpha * b + (1 - alpha) * a) for a, b in zip(ema_t, tb))
        x, y, w, h = ema_t
        d = dist_at(fi, h)
        cv2.rectangle(vis, (x, y), (x + w, y + h), BLUE, th)
        put(vis, f"tunnel  {d:.1f}m" if d else "tunnel", (x, max(36, y - 16)), BLUE)
        n_t += 1
        if gb is not None:
            ema_g = gb if ema_g is None else tuple(
                int(alpha * b + (1 - alpha) * a) for a, b in zip(ema_g, gb))
            gx, gy, gw, gh = ema_g
            # goal 거리는 tunnel과 같은 강체라 같은 값을 쓴다. 개구부 안쪽의
            # depth를 재면 그 너머 **지면**이 잡혀 더 멀게 나온다 — 통과해야 할
            # 문의 거리로는 틀린 값이다.
            dg = d
            cv2.rectangle(vis, (gx, gy), (gx + gw, gy + gh), RED, th)
            # tunnel 라벨은 박스 위, goal 라벨은 **개구부 아래**에 둔다. 둘 다
            # 위에 두면 개구부가 구조물 상단에 붙어 있어 글자가 겹친다.
            put(vis, f"goal  {dg:.1f}m" if dg else "goal",
                (gx, min(H - 12, gy + gh + int(46 * fs))), RED)
            n_g += 1
        else:
            ema_g = None            # 안 보이면 그리지 않고 상태도 버린다
    w_out.write(vis)
    if fi % 150 == 0:
        el = time.perf_counter() - t0
        print(f"  렌더 {fi}/{n_want}  {el:.0f}s  ({fi/max(el,1e-6):.1f} fps)", flush=True)

cap.release(); w_cut.release(); w_out.release()
print(f"\n완료 {n_want}프레임  {time.perf_counter()-t0:.0f}s")
print(f"  tunnel 그린 프레임 {n_t} ({n_t/max(n_want,1)*100:.1f}%)")
print(f"  goal   그린 프레임 {n_g} ({n_g/max(n_want,1)*100:.1f}%)")
print(f"  저장: {A.cut}")
print(f"  저장: {A.out}")
