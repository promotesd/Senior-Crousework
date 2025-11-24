import numpy as np
import math
import cv2

def TransformKernel(kernel):
    transform_kernel = kernel.copy()
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            transform_kernel[i][j] = kernel[kernel.shape[0]-i-1][kernel.shape[1]-j-1]
    return transform_kernel

def GetPaddedImage(image):
    imagePadded = np.asarray([[0 for x in range(0, image.shape[1]+2)] for y in range(0, image.shape[0]+2)], dtype=np.uint8)
    imagePadded[1:(imagePadded.shape[0]-1), 1:(imagePadded.shape[1]-1)] = image
    return imagePadded

def Convolution(image, kernel):
    kernel = TransformKernel(kernel)
    imagePadded = GetPaddedImage(image)
    imageConvolution = np.zeros(image.shape, dtype=np.float32)
    for i in range(1, imagePadded.shape[0]-1):
        for j in range(1, imagePadded.shape[1]-1):
            s = 0.0
            for m in range(kernel.shape[0]):
                for n in range(kernel.shape[1]):
                    s += kernel[m][n] * imagePadded[i+m-1][j+n-1]
            imageConvolution[i-1][j-1] = abs(s)
    mx = float(imageConvolution.max())
    if mx > 1e-8:
        imageConvolution /= mx
    return imageConvolution

def PerformSobel(image):
    img = image.copy()
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=60, sigmaSpace=60)
    img = cv2.medianBlur(img, ksize=5)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    kernelY = np.asarray([[ 1, 2, 1],
                          [ 0, 0, 0],
                          [-1,-2,-1]])
    kernelX = np.asarray([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
    gradientY = cv2.filter2D(img, cv2.CV_32F, kernelY)
    gradientX = cv2.filter2D(img, cv2.CV_32F, kernelX)

    theta = np.arctan2(gradientY, gradientX)
    mag   = np.sqrt(gradientX**2 + gradientY**2)
    m = float(mag.max())
    if m > 1e-8:
        mag = mag / m
    return mag, theta

def remove_small_components(edges_bin: np.ndarray, min_size: int = None, connectivity: int = 8) -> np.ndarray:
    edges = edges_bin.astype(np.uint8)
    if edges.max() == 1:
        edges *= 255
    H, W = edges.shape[:2]
    if min_size is None:
        min_size = max(16, int(0.001 * H * W))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=connectivity)
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            keep[i] = True
    return np.where(keep[labels], 255, 0).astype(np.uint8)

def local_max_mask(sl: np.ndarray) -> np.ndarray:
    if sl.max() <= 0:
        return np.zeros_like(sl, dtype=np.uint8)
    sl8 = (sl / sl.max() * 255.0).astype(np.uint8)
    dil = cv2.dilate(sl8, np.ones((3,3), np.uint8), 1)
    return (sl8 == dil).astype(np.uint8)

def save_hough_gray(acc2d: np.ndarray, path: str):
    m = float(acc2d.max())
    vis = np.zeros_like(acc2d, dtype=np.uint8) if m < 1e-8 else (acc2d / m * 255.0).astype(np.uint8)
    cv2.imwrite(path, vis)

_EDGE01_FOR_VERIFY = None
_DIST_FOR_VERIFY  = None

def circle_support_ratio(yc: float, xc: float, r: float, n_samples: int = 180, tol: float = 2.0) -> float:
    global _DIST_FOR_VERIFY
    if _DIST_FOR_VERIFY is None or r <= 0:
        return 0.0
    H, W = _DIST_FOR_VERIFY.shape
    thetas = np.linspace(0.0, 2.0*np.pi, n_samples, endpoint=False)
    ys = np.round(yc + r*np.sin(thetas)).astype(int)
    xs = np.round(xc + r*np.cos(thetas)).astype(int)
    mask = (ys>=0) & (ys<H) & (xs>=0) & (xs<W)
    if not mask.any():
        return 0.0
    d = _DIST_FOR_VERIFY[ys[mask], xs[mask]]
    return float((d <= tol).sum()) / float(mask.sum())


def _polarity_vote_allowed(img: np.ndarray, y:int, x:int, st:float, ct:float, rr:int,
                           toward_sign:int, expect_brighter_inside:bool=True) -> bool:
    H, W = img.shape
    rin  = max(2, int(round(0.30 * rr)))   # inside ~ 0.30r
    rout = max(2, int(round(0.06 * rr)))   # outside ~ 0.06r

    yin  = y + int(round(toward_sign * rin  * st))
    xin  = x + int(round(toward_sign * rin  * ct))
    yout = y - int(round(toward_sign * rout * st))
    xout = x - int(round(toward_sign * rout * ct))

    # clamp
    yin  = min(max(yin,  0), H-1); xin  = min(max(xin,  0), W-1)
    yout = min(max(yout, 0), H-1); xout = min(max(xout, 0), W-1)

    I_in  = float(img[yin,  xin])
    I_out = float(img[yout, xout])
    return (I_in >= I_out) if expect_brighter_inside else (I_in <= I_out)

def bilinear_splat(acc, y, x, r_idx, w):
    H, W, _ = acc.shape
    y0 = int(math.floor(y)); x0 = int(math.floor(x))
    if y0 < 0 or x0 < 0 or y0 >= H or x0 >= W:
        return
    dy = y - y0; dx = x - x0
    w00 = (1-dy) * (1-dx)
    w01 = (1-dy) * dx
    w10 = dy * (1-dx)
    w11 = dy * dx
    if y0+0 < H and x0+0 < W: acc[y0+0, x0+0, r_idx] += w * w00
    if y0+0 < H and x0+1 < W: acc[y0+0, x0+1, r_idx] += w * w01
    if y0+1 < H and x0+0 < W: acc[y0+1, x0+0, r_idx] += w * w10
    if y0+1 < H and x0+1 < W: acc[y0+1, x0+1, r_idx] += w * w11

def HoughCircles(image, edge_mask, expect_brighter_inside=True, r_scale=(0.065, 0.20)):
    rows, cols = image.shape[:2]
    mag, theta_map = PerformSobel(image)

    r_min = max(10, int(r_scale[0] * min(rows, cols)))
    r_max = max(r_min+1, int(r_scale[1] * min(rows, cols)))
    radius = list(range(r_min, r_max+1))
    R = len(radius)

    accumulator = np.zeros((rows, cols, R), dtype=np.float32)
    edge01 = (edge_mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(edge01)
    img_gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for idx_r, rr in enumerate(radius):
        for y, x in zip(ys, xs):
            base = float(theta_map[y, x])
            st, ct = math.sin(base), math.cos(base)
            w = float(mag[y, x]) + 1e-6

            a1f = y + (rr * st); b1f = x + (rr * ct)
            if 0.0 <= a1f < rows-1 and 0.0 <= b1f < cols-1:
                if _polarity_vote_allowed(img_gray, y, x, st, ct, rr, +1, expect_brighter_inside):
                    bilinear_splat(accumulator, a1f, b1f, idx_r, w)

            a2f = y - (rr * st); b2f = x - (rr * ct)
            if 0.0 <= a2f < rows-1 and 0.0 <= b2f < cols-1:
                if _polarity_vote_allowed(img_gray, y, x, st, ct, rr, -1, expect_brighter_inside):
                    bilinear_splat(accumulator, a2f, b2f, idx_r, w)

    return accumulator, radius

def refine_peak(accumulator, radius, y, x, r_idx, wy=1, wx=1, wr=1):
    H, W, R = accumulator.shape
    y0 = max(0, y - wy); y1 = min(H, y + wy + 1)
    x0 = max(0, x - wx); x1 = min(W, x + wx + 1)
    r0 = max(0, r_idx - wr); r1 = min(R, r_idx + wr + 1)
    block = accumulator[y0:y1, x0:x1, r0:r1]
    if block.size == 0 or block.max() <= 0:
        return float(y), float(x), float(radius[r_idx])
    wsum = block.sum()
    if wsum <= 0:
        return float(y), float(x), float(radius[r_idx])
    ys = np.arange(y0, y1)[:, None, None]
    xs = np.arange(x0, x1)[None, :, None]
    rs = np.array(radius[r0:r1])[None, None, :]
    yc = float((ys * block).sum() / wsum)
    xc = float((xs * block).sum() / wsum)
    rc = float((rs * block).sum() / wsum)
    return yc, xc, rc

def FilterCircles(accumulator, radius, t_rel=0.30, min_abs=2,
                  dr_frac=0.50,           
                  keep_top=8,
                  tol_frac=0.055, cov_thr=0.20,
                  center_min_dist=28,    
                  center_merge_frac=0.80, 
                  debug=False):
    H, W, R = accumulator.shape
    cand = []
    for r_idx, r in enumerate(radius):
        sl = accumulator[:, :, r_idx]
        m = float(sl.max())
        if m <= 0: 
            continue
        t = max(t_rel * m, min_abs)
        lmax = local_max_mask(sl)
        ys, xs = np.where((sl >= t) & (lmax > 0))
        for y, x in zip(ys, xs):
            cand.append((float(sl[y, x]), int(y), int(x), int(r_idx)))

    cand.sort(reverse=True, key=lambda z: z[0])

    circles = []
    circle_scores = []   
    taken = np.zeros_like(accumulator, dtype=np.uint8)

    for score, y, x, r_idx in cand:
        if taken[y, x, r_idx]:
            continue

        yc_f, xc_f, rc_f = refine_peak(accumulator, radius, y, x, r_idx)

        cov = circle_support_ratio(yc_f, xc_f, rc_f, n_samples=180, tol=max(2.5, tol_frac * rc_f))
        if cov < cov_thr:
            continue

        merged = False
        for k, (py, px, pr) in enumerate(circles):
            d = ((yc_f - py)**2 + (xc_f - px)**2) ** 0.5
            if d < center_merge_frac * max(pr, rc_f):
                if score > circle_scores[k]:
                    circles[k] = (yc_f, xc_f, rc_f)
                    circle_scores[k] = score
                merged = True
                break
        if merged:
            continue

        circles.append((yc_f, xc_f, rc_f))
        circle_scores.append(score)

        yy = int(round(yc_f)); xx = int(round(xc_f)); rr_pix = int(round(rc_f))
        y0 = max(0, yy - int(0.60 * rr_pix)); y1 = min(H, yy + int(0.60 * rr_pix) + 1)
        x0 = max(0, xx - int(0.60 * rr_pix)); x1 = min(W, xx + int(0.60 * rr_pix) + 1)
        r0 = max(0, r_idx - max(1, int(dr_frac * rr_pix)))
        r1 = min(R, r_idx + max(1, int(dr_frac * rr_pix)) + 1)
        taken[y0:y1, x0:x1, r0:r1] = 1

        if len(circles) >= keep_top:
            break


    pruned = []
    for cy, cx, r in circles:
        ok = True
        for py, px, pr in pruned:
            if ((cy - py)**2 + (cx - px)**2) ** 0.5 < max(center_min_dist, 0.30 * pr):
                ok = False
                break
        if ok:
            pruned.append((cy, cx, r))

    return pruned


def DetectCircles(img_gray, edges, r_scale=(0.065, 0.20), expect_brighter_inside=True, **filter_kwargs):
    global _EDGE01_FOR_VERIFY, _DIST_FOR_VERIFY

    edge01 = (edges > 0).astype(np.uint8)
    inv = (1 - edge01) * 255
    _EDGE01_FOR_VERIFY = edge01
    _DIST_FOR_VERIFY = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    accumulator, radius = HoughCircles(img_gray, edges, expect_brighter_inside, r_scale=r_scale)

    save_hough_gray(accumulator.max(axis=2), 'hough_space_max.png')
    save_hough_gray(accumulator.sum(axis=2), 'hough_space_sum.png')

    circles = FilterCircles(accumulator, radius, **filter_kwargs)
    return circles

if __name__ == "__main__":
    img = cv2.imread('./coins.png', 0)

    mag, theta = PerformSobel(img)
    nz  = mag[mag > 0]
    thr = np.quantile(nz, 0.93) if nz.size else 1.0
    edges01 = (mag >= thr).astype(np.uint8)
    edges = remove_small_components(edges01, min_size=600, connectivity=8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    cv2.imwrite('edges.png', edges)

    found = DetectCircles(
        img, edges,
        r_scale=(0.060, 0.22),
        t_rel=0.33, min_abs=3,
        tol_frac=0.040, cov_thr=0.26,
        keep_top=4, dr_frac=0.28,
        center_min_dist=40,
        debug=True
    )

    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if found:
        print(f"[INFO] Found {len(found)} circle(s):")
        for (cy, cx, r) in found:
            print(f"  center=({cx:.2f},{cy:.2f}), radius={r:.2f}")
            cv2.circle(result, (int(round(cx)), int(round(cy))), int(round(r)), (0, 0, 255), 2)
            cv2.circle(result, (int(round(cx)), int(round(cy))), 2, (0, 255, 0), -1)
    else:
        print("[WARN] No circles.")
    cv2.imwrite('houghCoin.png', result)
    print("[DONE] Saved: edges.png, hough_space_max.png, hough_space_sum.png, houghCoin.png")
