"""
Eyeglass Color Extraction Pipeline

Detects:
    - Primary frame/rim color
    - Secondary/multi-color frame colors
    - Bridge color
    - Left temple color
    - Right temple color
    - Temple accent/pattern colors
    - Lens/tint color

Pipeline:

    Image
      |
      +--> rembg foreground segmentation
      |
      +--> YOLO eyewear component segmentation
      |       |
      |       +--> frame
      |       +--> lens
      |       +--> bridge
      |       +--> temple_left
      |       +--> temple_right
      |
      +--> reflection/highlight removal
      |
      +--> color-space filtering
      |
      +--> GMM color clustering
      |
      +--> representative actual-pixel color
      |
      +--> HEX

IMPORTANT:
A generic YOLO model is NOT enough.

For maximum accuracy, train a custom segmentation model with:

    frame
    lens
    bridge
    temple_left
    temple_right

The fallback geometry is provided only so the application can work
without a custom model.
"""

import io
import os
import traceback
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from PIL import Image, ImageOps

# ============================================================
# OPTIONAL / REQUIRED LIBRARIES
# ============================================================

# -----------------------------
# rembg
# -----------------------------

_REMBG_AVAILABLE = False
_REMBG_SESSION = None

try:
    from rembg import remove as _rembg_remove
    from rembg import new_session as _rembg_new_session

    try:
        _REMBG_SESSION = _rembg_new_session(
            os.environ.get("REMBG_MODEL", "u2net")
        )
        _REMBG_AVAILABLE = True
    except Exception:
        traceback.print_exc()
        _REMBG_AVAILABLE = False

except ImportError:
    _REMBG_AVAILABLE = False


# -----------------------------
# YOLO / Torch
# -----------------------------

_YOLO_AVAILABLE = False
_YOLO_MODEL_PATH = os.environ.get(
    "EYEGLASS_SEG_MODEL",
    ""
).strip()

_YOLO_MODEL = None

try:
    import torch
    import torchvision

    from ultralytics import YOLO

    _YOLO_AVAILABLE = True

except ImportError:
    YOLO = None
    torch = None
    torchvision = None
    _YOLO_AVAILABLE = False


# -----------------------------
# scikit-learn GMM
# -----------------------------

_SKLEARN_AVAILABLE = False

try:
    from sklearn.mixture import GaussianMixture

    _SKLEARN_AVAILABLE = True

except ImportError:
    GaussianMixture = None


# -----------------------------
# scipy
# -----------------------------

_SCIPY_AVAILABLE = False

try:
    from scipy import ndimage as _ndi

    _SCIPY_AVAILABLE = True

except ImportError:
    _ndi = None


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 900

YOLO_CONFIDENCE = float(
    os.environ.get("EYEGLASS_YOLO_CONF", "0.35")
)

YOLO_IOU = float(
    os.environ.get("EYEGLASS_YOLO_IOU", "0.50")
)

MAX_COLOR_SAMPLES = 30000

MIN_COMPONENT_PIXELS = 100

REFLECTION_VALUE = 242.0

REFLECTION_SATURATION = 0.10

FRAME_EXTRA_MIN_PERCENT = 6.0

FRAME_EXTRA_MIN_DISTANCE = 22.0

COLOR_CONFIDENCE_GOOD = 0.90

COLOR_CONFIDENCE_REVIEW = 0.75


# ============================================================
# IMAGE LOADING
# ============================================================

def fetch_image_bytes_to_pil(data: bytes) -> Image.Image:
    """
    Convert downloaded/uploaded bytes into a clean RGB PIL image.
    """

    img = Image.open(io.BytesIO(data))

    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def prepare_image(
    img: Image.Image,
    max_size: int = IMAGE_SIZE
) -> Image.Image:
    """
    Resize while preserving aspect ratio.
    """

    img = img.copy()

    img.thumbnail(
        (max_size, max_size),
        Image.Resampling.LANCZOS
    )

    return img


# ============================================================
# FOREGROUND SEGMENTATION - REMBG
# ============================================================

def fallback_foreground_mask(
    img_np: np.ndarray,
    corner_fraction: float = 0.05,
    threshold: float = 25.0,
) -> np.ndarray:

    h, w, _ = img_np.shape

    ch = max(1, int(h * corner_fraction))
    cw = max(1, int(w * corner_fraction))

    corners = np.concatenate(
        [
            img_np[:ch, :cw].reshape(-1, 3),
            img_np[:ch, -cw:].reshape(-1, 3),
            img_np[-ch:, :cw].reshape(-1, 3),
            img_np[-ch:, -cw:].reshape(-1, 3),
        ],
        axis=0,
    )

    bg = np.median(corners, axis=0)

    distance = np.linalg.norm(
        img_np - bg[None, None, :],
        axis=-1,
    )

    mask = distance > threshold

    mask = mask.astype(np.uint8)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return mask.astype(bool)


def get_foreground_mask(
    img: Image.Image,
    img_np: np.ndarray,
) -> np.ndarray:

    if _REMBG_AVAILABLE:

        try:

            result = _rembg_remove(
                img,
                session=_REMBG_SESSION,
            )

            result_np = np.array(result)

            if (
                result_np.ndim == 3
                and result_np.shape[-1] == 4
            ):

                alpha = result_np[..., 3]

                mask = alpha > 30

                if mask.any():

                    return mask

        except Exception:

            traceback.print_exc()

    return fallback_foreground_mask(img_np)


# ============================================================
# MASK CLEANING
# ============================================================

def clean_mask(mask: np.ndarray) -> np.ndarray:

    mask = mask.astype(np.uint8)

    kernel_small = np.ones((3, 3), np.uint8)

    kernel_medium = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_medium,
    )

    return mask.astype(bool)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:

    mask_uint8 = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8,
    )

    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    return labels == largest_label


# ============================================================
# YOLO MODEL
# ============================================================

def get_yolo_model():

    global _YOLO_MODEL

    if not _YOLO_AVAILABLE:
        return None

    if not _YOLO_MODEL_PATH:
        return None

    if not os.path.exists(_YOLO_MODEL_PATH):

        print(
            f"[WARNING] YOLO model not found: "
            f"{_YOLO_MODEL_PATH}"
        )

        return None

    if _YOLO_MODEL is None:

        try:

            _YOLO_MODEL = YOLO(
                _YOLO_MODEL_PATH
            )

            print(
                f"[INFO] Loaded YOLO model: "
                f"{_YOLO_MODEL_PATH}"
            )

        except Exception:

            traceback.print_exc()

            _YOLO_MODEL = None

    return _YOLO_MODEL


# ============================================================
# YOLO COMPONENT SEGMENTATION
# ============================================================

def yolo_component_masks(
    img_np: np.ndarray,
    foreground_mask: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:

    model = get_yolo_model()

    if model is None:
        return None

    try:

        results = model.predict(
            source=img_np,
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU,
            verbose=False,
            retina_masks=True,
        )

        if not results:
            return None

        result = results[0]

        if result.masks is None:
            return None

        names = result.names

        masks = result.masks.data.cpu().numpy()

        if result.boxes is None:
            return None

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        component_masks = {}

        h, w = foreground_mask.shape

        for mask_data, class_id in zip(
            masks,
            classes,
        ):

            class_name = str(
                names.get(int(class_id), "")
            ).lower().strip()

            if not class_name:
                continue

            mask_resized = cv2.resize(
                mask_data.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            mask_resized &= foreground_mask

            mask_resized = clean_mask(
                mask_resized
            )

            if mask_resized.sum() < MIN_COMPONENT_PIXELS:
                continue

            if class_name in (
                "frame",
                "rim",
            ):

                key = "frame"

            elif "lens" in class_name:

                key = "lens"

            elif "bridge" in class_name:

                key = "bridge"

            elif (
                "temple_left" in class_name
                or "left_temple" in class_name
                or class_name == "left_temple"
            ):

                key = "temple_left"

            elif (
                "temple_right" in class_name
                or "right_temple" in class_name
                or class_name == "right_temple"
            ):

                key = "temple_right"

            elif (
                "temple" in class_name
                or "leg" in class_name
            ):

                # If the model only has one temple class,
                # keep it as generic temple.
                key = "temple"

            else:

                continue

            if key in component_masks:

                component_masks[key] |= mask_resized

            else:

                component_masks[key] = mask_resized

        if not component_masks:
            return None

        return component_masks

    except Exception:

        traceback.print_exc()

        return None


# ============================================================
# FALLBACK LENS SEGMENTATION
# ============================================================

def heuristic_lens_mask(
    foreground_mask: np.ndarray,
) -> np.ndarray:

    h, w = foreground_mask.shape

    if not foreground_mask.any():

        return np.zeros_like(
            foreground_mask
        )

    if _SCIPY_AVAILABLE:

        distance = _ndi.distance_transform_edt(
            foreground_mask
        )

    else:

        distance = foreground_mask.astype(
            np.float32
        )

    inside = distance[foreground_mask]

    if len(inside) == 0:

        return np.zeros_like(
            foreground_mask
        )

    threshold = max(
        np.percentile(inside, 65),
        2.0,
    )

    thick_regions = distance > threshold

    # Central region
    x = np.arange(w)

    central = (
        (x >= w * 0.08)
        & (x <= w * 0.92)
    )

    candidate = (
        thick_regions
        & central[None, :]
    )

    # Avoid extreme upper/lower areas
    y = np.arange(h)

    vertical = (
        (y >= h * 0.20)
        & (y <= h * 0.80)
    )

    candidate &= vertical[:, None]

    candidate &= foreground_mask

    candidate = clean_mask(candidate)

    return candidate


# ============================================================
# COMPONENT MASKS
# ============================================================

def get_component_masks(
    img_np: np.ndarray,
    foreground_mask: np.ndarray,
) -> Dict[str, np.ndarray]:

    yolo_masks = yolo_component_masks(
        img_np,
        foreground_mask,
    )

    if yolo_masks:

        # Make sure every mask is limited to foreground
        for key in yolo_masks:

            yolo_masks[key] &= foreground_mask

            yolo_masks[key] = clean_mask(
                yolo_masks[key]
            )

        if "lens" not in yolo_masks:

            yolo_masks["lens"] = heuristic_lens_mask(
                foreground_mask
            )

        if "frame" not in yolo_masks:

            lens = yolo_masks["lens"]

            yolo_masks["frame"] = (
                foreground_mask & ~lens
            )

        return yolo_masks

    # --------------------------------------------------------
    # No custom YOLO model
    # --------------------------------------------------------

    lens = heuristic_lens_mask(
        foreground_mask
    )

    non_lens = (
        foreground_mask & ~lens
    )

    return {
        "frame": non_lens,
        "lens": lens,
    }


# ============================================================
# REFLECTION / GLARE FILTER
# ============================================================

def reflection_mask(
    img_np: np.ndarray,
) -> np.ndarray:

    hsv = cv2.cvtColor(
        img_np.astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    )

    saturation = hsv[..., 1]

    value = hsv[..., 2]

    reflection = (
        (value >= REFLECTION_VALUE)
        & (
            saturation
            <= REFLECTION_SATURATION * 255
        )
    )

    return reflection


def remove_reflections(
    mask: np.ndarray,
    img_np: np.ndarray,
) -> np.ndarray:

    glare = reflection_mask(
        img_np
    )

    result = mask & ~glare

    return result


# ============================================================
# PIXEL QUALITY FILTER
# ============================================================

def get_clean_pixels(
    img_np: np.ndarray,
    mask: np.ndarray,
    max_samples: int = MAX_COLOR_SAMPLES,
) -> np.ndarray:

    mask = mask.astype(bool)

    if not mask.any():

        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    pixels = img_np[mask].astype(
        np.float64
    )

    # --------------------------------------------------------
    # Remove extreme highlight pixels
    # --------------------------------------------------------

    max_channel = pixels.max(axis=1)

    min_channel = pixels.min(axis=1)

    saturation = (
        max_channel - min_channel
    ) / np.clip(
        max_channel,
        1.0,
        None,
    )

    keep = ~(
        (max_channel > 245)
        & (saturation < 0.08)
    )

    pixels = pixels[keep]

    if len(pixels) == 0:

        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # Remove extreme black noise
    # --------------------------------------------------------

    # Don't remove normal black frames.
    # Only remove absolute sensor/processing artifacts.
    pixels = pixels[
        np.max(pixels, axis=1) >= 2
    ]

    if len(pixels) == 0:

        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    if len(pixels) > max_samples:

        rng = np.random.default_rng(42)

        indexes = rng.choice(
            len(pixels),
            max_samples,
            replace=False,
        )

        pixels = pixels[indexes]

    return pixels


# ============================================================
# GMM COLOR CLUSTERING
# ============================================================

def rgb_cluster_colors(
    pixels: np.ndarray,
    max_k: int = 4,
) -> List[Tuple[np.ndarray, float, float]]:

    if len(pixels) == 0:

        return []

    pixels = np.asarray(
        pixels,
        dtype=np.float64,
    )

    if not _SKLEARN_AVAILABLE:

        median = np.median(
            pixels,
            axis=0,
        )

        tightness = float(
            np.mean(
                np.std(
                    pixels,
                    axis=0,
                )
            )
        )

        return [
            (
                median,
                100.0,
                tightness,
            )
        ]

    sample = pixels

    max_k = max(
        1,
        min(
            max_k,
            len(sample),
        ),
    )

    best_model = None

    best_bic = np.inf

    # --------------------------------------------------------
    # Find best K using BIC
    # --------------------------------------------------------

    for k in range(
        1,
        max_k + 1,
    ):

        try:

            model = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=42,
                n_init=3,
                reg_covar=1e-3,
            )

            model.fit(sample)

            bic = model.bic(sample)

            if bic < best_bic:

                best_bic = bic

                best_model = model

        except Exception:

            traceback.print_exc()

    if best_model is None:

        median = np.median(
            pixels,
            axis=0,
        )

        return [
            (
                median,
                100.0,
                float(
                    np.mean(
                        np.std(
                            pixels,
                            axis=0,
                        )
                    )
                ),
            )
        ]

    labels = best_model.predict(
        sample
    )

    results = []

    for cluster_id in range(
        best_model.n_components
    ):

        cluster_pixels = sample[
            labels == cluster_id
        ]

        if len(cluster_pixels) == 0:
            continue

        percentage = (
            100.0
            * len(cluster_pixels)
            / len(sample)
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Use MEDIAN actual pixels instead of only GMM mean.
        # This produces a representative real image color.
        # ----------------------------------------------------

        representative = np.median(
            cluster_pixels,
            axis=0,
        )

        tightness = float(
            np.mean(
                np.std(
                    cluster_pixels,
                    axis=0,
                )
            )
        )

        results.append(
            (
                representative,
                percentage,
                tightness,
            )
        )

    results.sort(
        key=lambda x: -x[1]
    )

    return results


# ============================================================
# HEX
# ============================================================

def rgb_to_hex(rgb) -> str:

    rgb = np.asarray(
        rgb,
        dtype=np.float64,
    )

    rgb = np.clip(
        rgb,
        0,
        255,
    )

    r, g, b = [
        int(round(float(x)))
        for x in rgb[:3]
    ]

    return "#{:02X}{:02X}{:02X}".format(
        r,
        g,
        b,
    )


def rgb_to_string(rgb) -> str:

    rgb = np.asarray(
        rgb,
        dtype=np.float64,
    )

    return "rgb({}, {}, {})".format(
        int(round(rgb[0])),
        int(round(rgb[1])),
        int(round(rgb[2])),
    )


# ============================================================
# COLOR CONFIDENCE
# ============================================================

def color_confidence(
    percentage: float,
    tightness: float,
    pixel_count: int,
) -> float:

    # Cluster percentage
    pct_score = min(
        percentage / 60.0,
        1.0,
    )

    # Color purity
    tight_score = max(
        0.0,
        min(
            1.0,
            1.0 - (
                tightness / 80.0
            ),
        ),
    )

    # Number of usable pixels
    pixel_score = min(
        pixel_count / 5000.0,
        1.0,
    )

    confidence = (
        pct_score * 0.40
        + tight_score * 0.35
        + pixel_score * 0.25
    )

    return round(
        float(
            max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            )
        ),
        3,
    )


# ============================================================
# EXTRACT COLORS FROM MASK
# ============================================================

def extract_colors_from_mask(
    img_np: np.ndarray,
    mask: np.ndarray,
    max_k: int = 4,
) -> List[Dict]:

    if mask is None or not mask.any():

        return []

    clean_mask = remove_reflections(
        mask,
        img_np,
    )

    pixels = get_clean_pixels(
        img_np,
        clean_mask,
    )

    if len(pixels) == 0:

        return []

    clusters = rgb_cluster_colors(
        pixels,
        max_k=max_k,
    )

    results = []

    for rgb, percentage, tightness in clusters:

        confidence = color_confidence(
            percentage,
            tightness,
            len(pixels),
        )

        results.append(
            {
                "rgb": rgb_to_string(rgb),
                "hex": rgb_to_hex(rgb),
                "percentage": round(
                    float(percentage),
                    2,
                ),
                "tightness": round(
                    float(tightness),
                    2,
                ),
                "confidence": confidence,
            }
        )

    return results


# ============================================================
# MULTI-COLOR FILTER
# ============================================================

def select_multi_colors(
    colors: List[Dict],
) -> List[Dict]:

    if not colors:

        return []

    primary_rgb = np.array(
        [
            int(x)
            for x in (
                colors[0]["rgb"]
                .replace("rgb(", "")
                .replace(")", "")
                .split(",")
            )
        ],
        dtype=float,
    )

    output = [
        colors[0]
    ]

    for color in colors[1:]:

        rgb = np.array(
            [
                int(x)
                for x in (
                    color["rgb"]
                    .replace("rgb(", "")
                    .replace(")", "")
                    .split(",")
                )
            ],
            dtype=float,
        )

        distance = np.linalg.norm(
            rgb - primary_rgb
        )

        if (
            color["percentage"]
            >= FRAME_EXTRA_MIN_PERCENT
            and distance
            >= FRAME_EXTRA_MIN_DISTANCE
        ):

            output.append(color)

    return output


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_colors_from_image(
    img: Image.Image,
) -> Tuple[List[Dict], Tuple[int, int], Dict]:

    original_size = img.size

    img = prepare_image(img)

    img_np = np.asarray(
        img,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # 1. Foreground
    # --------------------------------------------------------

    foreground = get_foreground_mask(
        img,
        img_np,
    )

    foreground = clean_mask(
        foreground
    )

    foreground = keep_largest_component(
        foreground
    )

    if not foreground.any():

        raise ValueError(
            "Could not detect eyeglass foreground."
        )

    # --------------------------------------------------------
    # 2. Components
    # --------------------------------------------------------

    components = get_component_masks(
        img_np,
        foreground,
    )

    frame_mask = components.get(
        "frame"
    )

    lens_mask = components.get(
        "lens"
    )

    bridge_mask = components.get(
        "bridge"
    )

    temple_left_mask = components.get(
        "temple_left"
    )

    temple_right_mask = components.get(
        "temple_right"
    )

    temple_mask = components.get(
        "temple"
    )

    # --------------------------------------------------------
    # If YOLO didn't provide frame,
    # use foreground minus lens.
    # --------------------------------------------------------

    if (
        frame_mask is None
        or not frame_mask.any()
    ):

        frame_mask = (
            foreground
            & ~lens_mask
        )

    # --------------------------------------------------------
    # 3. Frame colors
    # --------------------------------------------------------

    frame_colors = extract_colors_from_mask(
        img_np,
        frame_mask,
        max_k=4,
    )

    frame_colors = select_multi_colors(
        frame_colors
    )

    # --------------------------------------------------------
    # 4. Bridge
    # --------------------------------------------------------

    bridge_colors = []

    if (
        bridge_mask is not None
        and bridge_mask.any()
    ):

        bridge_colors = extract_colors_from_mask(
            img_np,
            bridge_mask,
            max_k=2,
        )

    # --------------------------------------------------------
    # 5. Temple
    # --------------------------------------------------------

    temple_results = []

    if (
        temple_left_mask is not None
        or temple_right_mask is not None
    ):

        if (
            temple_left_mask is not None
            and temple_left_mask.any()
        ):

            left_colors = extract_colors_from_mask(
                img_np,
                temple_left_mask,
                max_k=3,
            )

            if left_colors:

                temple_results.append(
                    {
                        "side": "left",
                        "colors": left_colors,
                    }
                )

        if (
            temple_right_mask is not None
            and temple_right_mask.any()
        ):

            right_colors = extract_colors_from_mask(
                img_np,
                temple_right_mask,
                max_k=3,
            )

            if right_colors:

                temple_results.append(
                    {
                        "side": "right",
                        "colors": right_colors,
                    }
                )

    elif (
        temple_mask is not None
        and temple_mask.any()
    ):

        generic_temple = extract_colors_from_mask(
            img_np,
            temple_mask,
            max_k=3,
        )

        if generic_temple:

            temple_results.append(
                {
                    "side": "unknown",
                    "colors": generic_temple,
                }
            )

    # --------------------------------------------------------
    # 6. Lens / tint
    # --------------------------------------------------------

    lens_colors = []

    if (
        lens_mask is not None
        and lens_mask.any()
    ):

        lens_colors = extract_colors_from_mask(
            img_np,
            lens_mask,
            max_k=2,
        )

    # --------------------------------------------------------
    # 7. Build output
    # --------------------------------------------------------

    output = []

    # Primary frame

    if frame_colors:

        primary = frame_colors[0]

        output.append(
            {
                "label": "Primary Frame Color",
                **primary,
            }
        )

        for index, color in enumerate(
            frame_colors[1:],
            start=2,
        ):

            output.append(
                {
                    "label": (
                        f"Frame Color {index} "
                        "(Multi-Color Frame)"
                    ),
                    **color,
                }
            )

    # Bridge

    if bridge_colors:

        output.append(
            {
                "label": "Bridge Color",
                **bridge_colors[0],
            }
        )

    # Temples

    for temple in temple_results:

        side = temple["side"]

        colors = temple["colors"]

        if not colors:
            continue

        output.append(
            {
                "label": (
                    f"Temple Color ({side})"
                ),
                **colors[0],
            }
        )

        for index, color in enumerate(
            colors[1:],
            start=2,
        ):

            output.append(
                {
                    "label": (
                        f"Temple Accent "
                        f"({side}) {index}"
                    ),
                    **color,
                }
            )

    # Lens

    if lens_colors:

        output.append(
            {
                "label": "Lens / Tint Color",
                **lens_colors[0],
            }
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    meta = {
        "rembg_available": _REMBG_AVAILABLE,
        "yolo_available": _YOLO_AVAILABLE,
        "yolo_model_configured": bool(
            _YOLO_MODEL_PATH
        ),
        "yolo_model_path": _YOLO_MODEL_PATH or None,
        "scikit_learn_available": (
            _SKLEARN_AVAILABLE
        ),
        "scipy_available": (
            _SCIPY_AVAILABLE
        ),
        "opencv_version": cv2.__version__,
        "foreground_pixels": int(
            foreground.sum()
        ),
        "frame_detected": bool(
            frame_mask is not None
            and frame_mask.any()
        ),
        "lens_detected": bool(
            lens_mask is not None
            and lens_mask.any()
        ),
        "bridge_detected": bool(
            bridge_mask is not None
            and bridge_mask.any()
        ),
        "temple_detected": bool(
            temple_results
        ),
        "segmentation_method": (
            "YOLO"
            if (
                _YOLO_MODEL is not None
                and components
            )
            else "REMBG + heuristic"
        ),
    }

    return (
        output,
        original_size,
        meta,
    )


# ============================================================
# URL HELPER
# ============================================================

def extract_eyeglass_colors(
    url: str,
) -> Tuple[List[Dict], Tuple[int, int], Dict]:

    import requests

    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64)"
            )
        },
        timeout=30,
    )

    response.raise_for_status()

    img = fetch_image_bytes_to_pil(
        response.content
    )

    return extract_colors_from_image(
        img
    )


# ============================================================
# DEBUG / TEST
# ============================================================

if __name__ == "__main__":

    import json
    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python pipeline.py image.jpg"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    image = Image.open(
        image_path
    )

    colors, dimensions, meta = (
        extract_colors_from_image(
            image
        )
    )

    print(
        json.dumps(
            {
                "dimensions": dimensions,
                "colors": colors,
                "meta": meta,
            },
            indent=2,
        )
    )
