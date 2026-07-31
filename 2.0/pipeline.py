import io
import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2lab, lab2rgb, deltaE_cie76
from sklearn.mixture import GaussianMixture

# ============================================================
# Optional rembg background removal.
# Real, class-agnostic salient-object segmentation (U2Net/ISNet).
# NOT an "eyewear parts" model -- it does not know frame/temple/
# bridge/lens as separate things, it just gives foreground-vs-
# background. That's genuinely useful here and something the
# original heuristic couldn't do (irregular/patterned backgrounds).
# If it isn't installed, or the model can't be fetched (no internet
# at runtime), we fall back to corner-sampling automatically.
# ============================================================
try:
    from rembg import remove as rembg_remove, new_session as rembg_new_session
    _REMBG_SESSION = None
    _REMBG_AVAILABLE = True
except Exception:
    _REMBG_AVAILABLE = False


def rgb_to_hex(rgb) -> str:
    r, g, b = np.clip(np.round(rgb), 0, 255).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


def fetch_image_bytes_to_pil(raw_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(raw_bytes))
    has_alpha = (
        img.mode in ("RGBA", "LA")
        or (img.mode == "P" and "transparency" in img.info)
    )
    if has_alpha:
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


# ------------------------------------------------------------
# Background removal: rembg if available, else corner-sampling
# ------------------------------------------------------------

def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        _REMBG_SESSION = rembg_new_session("isnet-general-use")
    return _REMBG_SESSION


def rembg_foreground_mask(pil_img: Image.Image):
    """
    Returns a boolean foreground mask via rembg, or None if rembg
    is unavailable / fails (e.g. no internet to fetch model weights).
    Caller must fall back to detect_background_color/make_foreground_mask.
    """
    if not _REMBG_AVAILABLE:
        return None
    try:
        session = _get_rembg_session()
        result = rembg_remove(pil_img, session=session)  # RGBA
        alpha = np.array(result)[:, :, 3]
        return alpha > 127
    except Exception:
        return None


def detect_background_color(img_np: np.ndarray) -> np.ndarray:
    """Corner-sampling fallback background estimate."""
    h, w, _ = img_np.shape
    patch = max(2, min(h, w) // 20)
    corners = np.vstack([
        img_np[0:patch, 0:patch].reshape(-1, 3),
        img_np[0:patch, w - patch:w].reshape(-1, 3),
        img_np[h - patch:h, 0:patch].reshape(-1, 3),
        img_np[h - patch:h, w - patch:w].reshape(-1, 3),
    ])
    return np.median(corners, axis=0)


def make_foreground_mask(img_np: np.ndarray, bg_color: np.ndarray, threshold: float = 25.0) -> np.ndarray:
    dist = np.linalg.norm(img_np - bg_color[None, None, :], axis=2)
    mask = dist > threshold
    if not mask.any():
        mask = dist > (threshold * 0.5)
    return mask


def get_foreground_mask(pil_img_resized: Image.Image, img_np: np.ndarray) -> np.ndarray:
    """
    Union of rembg (if available) and corner-distance fallback.

    IMPORTANT, TESTED FINDING: rembg is a general salient-object model.
    On thin, disconnected pieces (like glasses temples sticking out from
    the main frame) it sometimes decides they're "not the subject" and
    drops them entirely -- verified on a synthetic test image where
    rembg's mask cut off both temple arms while the plain corner-distance
    method caught them correctly. So we don't trust rembg alone; we
    union it with the distance-based mask so temples aren't lost, while
    still getting rembg's cleaner handling of irregular/patterned
    backgrounds for the main frame body.
    """
    bg_color = detect_background_color(img_np)
    fallback_mask = make_foreground_mask(img_np, bg_color)

    rembg_mask = rembg_foreground_mask(pil_img_resized)
    if rembg_mask is not None and rembg_mask.any():
        return rembg_mask | fallback_mask
    return fallback_mask


# ------------------------------------------------------------
# Reflection / specular highlight stripping (real, measurable rule)
# ------------------------------------------------------------

def strip_reflections(img_np: np.ndarray, mask: np.ndarray,
                       v_thresh: float = 235.0, s_thresh: float = 30.0) -> np.ndarray:
    """
    Removes near-white, low-saturation glare pixels from a mask.
    Rule: HSV Value high AND Saturation low -> specular highlight,
    not the object's real material color.
    """
    bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    # OpenCV: H in 0-179, S in 0-255, V in 0-255
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    glare = (v > v_thresh) & (s < s_thresh)
    return mask & (~glare)


# ------------------------------------------------------------
# Lens separation via morphological opening
# (debugged approach: require TWO comparably-sized blobs that
#  together make up most of the object, else fall back to none)
# ------------------------------------------------------------

def separate_lens_regions(obj_mask: np.ndarray):
    """
    obj_mask: boolean mask of the glasses object (cropped to bbox).
    Returns (lens_mask, non_lens_mask). lens_mask may be all-False
    if no confident two-blob lens split is found (e.g. rimless
    frames, or a shot angle where lenses aren't visually distinct).
    """
    h, w = obj_mask.shape
    u8 = (obj_mask.astype(np.uint8)) * 255
    total_area = u8.sum() / 255.0
    if total_area == 0:
        return np.zeros_like(obj_mask), obj_mask.copy()

    best_lens_mask = None
    # search a range of opening kernel sizes to erode away thin
    # rims/bridge/temples and leave the two thicker lens blobs
    for frac in np.linspace(0.06, 0.30, 25):
        k = max(3, int(round(min(h, w) * frac)))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel)

        n_labels, labels = cv2.connectedComponents(opened)
        if n_labels <= 1:
            continue

        areas = [(labels == i).sum() for i in range(1, n_labels)]
        if len(areas) < 2:
            continue

        order = np.argsort(areas)[::-1]
        a1, a2 = areas[order[0]], areas[order[1]]
        if a2 == 0:
            continue

        # the two largest blobs must be comparably sized (true lens
        # pair), and together must cover a large majority of the
        # object -- rules out two small temple-remnant blobs
        size_ratio = a2 / a1
        combined_frac = (a1 + a2) / total_area

        if size_ratio > 0.45 and combined_frac > 0.35:
            lens_labels = {order[0] + 1, order[1] + 1}
            candidate = np.isin(labels, list(lens_labels))
            best_lens_mask = candidate
            break

    if best_lens_mask is None:
        return np.zeros_like(obj_mask), obj_mask.copy()

    # slightly dilate the lens mask back out so we don't leave a
    # thin rind of lens-edge pixels in the "non-lens" set
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lens_dilated = cv2.dilate(best_lens_mask.astype(np.uint8), dil_kernel).astype(bool)
    lens_dilated &= obj_mask

    non_lens = obj_mask & (~lens_dilated)
    return lens_dilated, non_lens


# ------------------------------------------------------------
# Color clustering: GMM with BIC-selected component count,
# then merge visually-identical clusters in LAB space.
# ------------------------------------------------------------

def extract_dominant_colors_gmm(pixels_rgb: np.ndarray, max_k: int = 4, merge_delta_e: float = 14.0):
    """
    Returns list of (rgb, percentage, tightness) sorted by percentage desc.
    tightness = mean distance of points to their cluster centroid in LAB
    (lower = more uniform region, e.g. solid black frame; higher = more
    mixed, e.g. tortoise pattern). This is a real computed metric, not
    a fabricated confidence score.
    """
    n = len(pixels_rgb)
    if n == 0:
        return []
    if n < 8:
        return [(np.mean(pixels_rgb, axis=0), 100.0, 0.0)]

    lab = rgb2lab(pixels_rgb / 255.0)

    n_unique = len(np.unique(np.round(lab, 1), axis=0))
    k_cap = max(1, min(max_k, n_unique, n // 4))

    best_model = None
    best_bic = np.inf
    for k in range(1, k_cap + 1):
        try:
            gm = GaussianMixture(n_components=k, covariance_type="diag",
                                  random_state=42, max_iter=100)
            gm.fit(lab)
            bic = gm.bic(lab)
            if bic < best_bic:
                best_bic = bic
                best_model = gm
        except Exception:
            continue

    if best_model is None:
        return [(np.mean(pixels_rgb, axis=0), 100.0, 0.0)]

    labels = best_model.predict(lab)
    k_final = best_model.n_components

    clusters = []
    for i in range(k_final):
        pts = lab[labels == i]
        if len(pts) == 0:
            continue
        centroid = pts.mean(axis=0)
        tightness = float(np.mean(np.linalg.norm(pts - centroid, axis=1)))
        clusters.append({"centroid": centroid, "count": len(pts), "tightness": tightness})

    clusters.sort(key=lambda c: -c["count"])

    # merge visually-identical clusters (deltaE) so near-duplicate
    # centroids don't get reported as separate colors
    merged = []
    for c in clusters:
        dup = None
        for m in merged:
            de = deltaE_cie76(c["centroid"][None, :], m["centroid"][None, :])[0]
            if de < merge_delta_e:
                dup = m
                break
        if dup is not None:
            total = dup["count"] + c["count"]
            dup["centroid"] = (dup["centroid"] * dup["count"] + c["centroid"] * c["count"]) / total
            dup["tightness"] = (dup["tightness"] * dup["count"] + c["tightness"] * c["count"]) / total
            dup["count"] = total
        else:
            merged.append(dict(c))

    merged.sort(key=lambda c: -c["count"])
    total_n = sum(m["count"] for m in merged) or 1

    out = []
    for m in merged:
        rgb = lab2rgb(m["centroid"][None, None, :])[0, 0] * 255.0
        pct = m["count"] / total_n * 100.0
        out.append((rgb, pct, round(m["tightness"], 2)))
    return out
