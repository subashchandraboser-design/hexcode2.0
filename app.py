import numpy as np
import requests
from PIL import Image
from flask import Flask, render_template, request, jsonify

import pipeline as P

app = Flask(__name__)


def fetch_image(url: str) -> Image.Image:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return P.fetch_image_bytes_to_pil(resp.content)


def zone(obj, obj_mask, y0, y1, x0, x1):
    oh, ow, _ = obj.shape
    crop = obj[int(oh * y0):int(oh * y1), int(ow * x0):int(ow * x1)]
    crop_mask = obj_mask[int(oh * y0):int(oh * y1), int(ow * x0):int(ow * x1)]
    return crop[crop_mask]


def bounding_box(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    return top, bottom + 1, left, right + 1


def extract_colors_from_image(img: Image.Image):
    """
    Core pipeline. Takes any PIL Image (loaded from a URL, an uploaded
    file, or a bulk-import loop) and returns (results, size, meta).
    Kept separate from URL-fetching so upload and bulk paths don't
    duplicate the pipeline logic.
    """
    img_resized = img.resize((500, 250), Image.Resampling.LANCZOS)
    img_np = np.array(img_resized, dtype=np.float64)

    # 1. Foreground mask: rembg (if available) unioned with corner-distance
    #    fallback, so thin temples aren't dropped by the salient-object model.
    fg_mask = P.get_foreground_mask(img_resized, img_np)

    # 2. Strip specular highlights/glare before any color sampling.
    fg_mask = P.strip_reflections(img_np, fg_mask)

    fg_pixels = img_np[fg_mask]
    if len(fg_pixels) == 0:
        raise ValueError("No eyeglass pixels found in the provided image.")

    # 3. Crop to bounding box, work in object-relative coordinates.
    top, bottom, left, right = bounding_box(fg_mask)
    obj = img_np[top:bottom, left:right]
    obj_mask = fg_mask[top:bottom, left:right]

    # 4. Separate lens pixels from frame/temple pixels using morphological
    #    opening. lens_mask may be empty (rimless frames, no clean split
    #    found) -- that's a real, disclosed limitation, not silently faked.
    lens_mask, non_lens_mask = P.separate_lens_regions(obj_mask)

    # -----------------------------------------------------------
    # Frame / rim color: center-bridge region, lens pixels excluded
    # -----------------------------------------------------------
    front_rim_region = zone(obj, non_lens_mask, 0.20, 0.65, 0.30, 0.70)
    frame_colors = P.extract_dominant_colors_gmm(front_rim_region, max_k=3)
    if len(frame_colors) == 0:
        # fall back to the whole non-lens region if the narrow zone missed
        front_rim_region = obj[non_lens_mask]
        frame_colors = P.extract_dominant_colors_gmm(front_rim_region, max_k=3)
    primary_frame_rgb = frame_colors[0][0] if frame_colors else fg_pixels[0]

    # -----------------------------------------------------------
    # Temple / leg color: far left & right, lens pixels excluded
    # -----------------------------------------------------------
    left_leg = zone(obj, non_lens_mask, 0.15, 0.85, 0.0, 0.30)
    right_leg = zone(obj, non_lens_mask, 0.15, 0.85, 0.70, 1.0)
    legs_parts = [p for p in (left_leg, right_leg) if len(p) > 0]
    legs_combined = np.vstack(legs_parts) if legs_parts else np.empty((0, 3))

    leg_colors = P.extract_dominant_colors_gmm(legs_combined, max_k=4)

    temple_rgb = leg_colors[0][0] if leg_colors else primary_frame_rgb
    accent_rgb, accent_pct, accent_tight = None, None, None
    for rgb, pct, tight in leg_colors[1:]:
        if np.linalg.norm(rgb - temple_rgb) > 35:
            accent_rgb, accent_pct, accent_tight = rgb, pct, tight
            break

    # -----------------------------------------------------------
    # Lens / tint color: only from pixels the lens-separator actually
    # identified as lens. If no confident split was found, we say so
    # instead of guessing "Clear".
    # -----------------------------------------------------------
    lens_detected = bool(lens_mask.any())
    tint_hex = None
    tint_rgb_str = None
    tint_pct = 0.0
    tint_tightness = None

    if lens_detected:
        lens_pixels = obj[lens_mask]
        lens_colors = P.extract_dominant_colors_gmm(lens_pixels, max_k=2)
        if lens_colors:
            l_rgb, l_pct, l_tight = lens_colors[0]
            tint_hex = P.rgb_to_hex(l_rgb)
            tint_rgb_str = f"rgb({int(l_rgb[0])}, {int(l_rgb[1])}, {int(l_rgb[2])})"
            tint_pct = round(float(l_pct), 1)
            tint_tightness = l_tight

    # -----------------------------------------------------------
    # Assemble results. Percentages are real pixel-count fractions.
    # "tightness" is a real computed dispersion metric (lower = more
    # uniform color region), not a fabricated confidence score.
    # -----------------------------------------------------------
    frame_pct = frame_colors[0][1] if frame_colors else 0.0
    frame_tight = frame_colors[0][2] if frame_colors else None
    temple_pct = leg_colors[0][1] if leg_colors else 0.0
    temple_tight = leg_colors[0][2] if leg_colors else None

    results = [
        {
            "label": "Primary Frame Color",
            "hex": P.rgb_to_hex(primary_frame_rgb),
            "rgb": f"rgb({int(primary_frame_rgb[0])}, {int(primary_frame_rgb[1])}, {int(primary_frame_rgb[2])})",
            "percentage": round(float(frame_pct), 1),
            "cluster_tightness": frame_tight,
        },
        {
            "label": "Temple Color (Legs)",
            "hex": P.rgb_to_hex(temple_rgb),
            "rgb": f"rgb({int(temple_rgb[0])}, {int(temple_rgb[1])}, {int(temple_rgb[2])})",
            "percentage": round(float(temple_pct), 1),
            "cluster_tightness": temple_tight,
        },
    ]

    if accent_rgb is not None:
        results.append({
            "label": "Temple Accent / Pattern Color",
            "hex": P.rgb_to_hex(accent_rgb),
            "rgb": f"rgb({int(accent_rgb[0])}, {int(accent_rgb[1])}, {int(accent_rgb[2])})",
            "percentage": round(float(accent_pct), 1),
            "cluster_tightness": accent_tight,
        })

    if lens_detected and tint_hex:
        results.append({
            "label": "Tint Color (Lens / Clip-on)",
            "hex": tint_hex,
            "rgb": tint_rgb_str,
            "percentage": tint_pct,
            "cluster_tightness": tint_tightness,
        })
    else:
        results.append({
            "label": "Tint Color (Lens / Clip-on)",
            "hex": None,
            "rgb": None,
            "percentage": 0.0,
            "note": "No confident lens region found (rimless frame, extreme "
                    "angle, or lenses not visually distinct from the rim). "
                    "Reporting as undetected rather than guessing 'Clear'.",
        })

    return results, img.size, {
        "rembg_used": P._REMBG_AVAILABLE,
        "lens_region_detected": lens_detected,
    }


def extract_eyeglass_colors(url: str):
    """URL entry point: download then run the shared core pipeline."""
    img = fetch_image(url)
    return extract_colors_from_image(img)


def colors_to_row(url: str, colors, dimensions, meta, error: str = None) -> dict:
    """
    Flattens one image's result into a single flat dict -- one row,
    hex codes as columns -- for the bulk table / CSV export.
    """
    row = {
        "url": url,
        "dimensions": f"{dimensions[0]} x {dimensions[1]}" if dimensions else "",
        "primary_frame_hex": "",
        "primary_frame_rgb": "",
        "primary_frame_pct": "",
        "temple_hex": "",
        "temple_rgb": "",
        "temple_pct": "",
        "accent_hex": "",
        "accent_rgb": "",
        "accent_pct": "",
        "tint_hex": "",
        "tint_rgb": "",
        "tint_pct": "",
        "lens_detected": meta.get("lens_region_detected") if meta else "",
        "rembg_used": meta.get("rembg_used") if meta else "",
        "error": error or "",
    }
    if error:
        return row

    label_to_prefix = {
        "Primary Frame Color": "primary_frame",
        "Temple Color (Legs)": "temple",
        "Temple Accent / Pattern Color": "accent",
        "Tint Color (Lens / Clip-on)": "tint",
    }
    for c in colors:
        prefix = label_to_prefix.get(c["label"])
        if not prefix:
            continue
        row[f"{prefix}_hex"] = c.get("hex") or ""
        row[f"{prefix}_rgb"] = c.get("rgb") or ""
        row[f"{prefix}_pct"] = c.get("percentage", "")
    return row


def parse_url_list_file(raw_bytes: bytes) -> list:
    """
    Accepts a .csv or .txt file: one URL per line, or a CSV where any
    field on the line contains an http(s) URL. Lines with no URL-like
    field (e.g. a header row) are skipped rather than erroring out.
    """
    text = raw_bytes.decode("utf-8", errors="ignore")
    urls = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = [f.strip().strip('"') for f in line.split(",")]
        found = next((f for f in fields if f.lower().startswith("http")), None)
        if found:
            urls.append(found)
    return urls


CSV_COLUMNS = [
    "url", "dimensions",
    "primary_frame_hex", "primary_frame_rgb", "primary_frame_pct",
    "temple_hex", "temple_rgb", "temple_pct",
    "accent_hex", "accent_rgb", "accent_pct",
    "tint_hex", "tint_rgb", "tint_pct",
    "lens_detected", "rembg_used", "error",
]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract():
    try:
        data = request.get_json()
        url = data.get("url", "").strip() if data else ""
        if not url:
            return jsonify({"error": "Please enter a valid image URL"}), 400

        colors, dimensions, meta = extract_eyeglass_colors(url)
        return jsonify({
            "success": True,
            "url": url,
            "dimensions": f"{dimensions[0]} x {dimensions[1]}",
            "colors": colors,
            "meta": meta,
        })
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Could not download image: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/extract_file", methods=["POST"])
def extract_file():
    """Single-image upload (no URL needed)."""
    try:
        if "image" not in request.files or request.files["image"].filename == "":
            return jsonify({"error": "Please choose an image file"}), 400

        file = request.files["image"]
        img = P.fetch_image_bytes_to_pil(file.read())
        colors, dimensions, meta = extract_colors_from_image(img)

        return jsonify({
            "success": True,
            "url": file.filename,
            "dimensions": f"{dimensions[0]} x {dimensions[1]}",
            "colors": colors,
            "meta": meta,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# In-memory store for the most recent bulk run, keyed by nothing fancy --
# single-user dev tool, so we just keep the last batch for /export to
# reuse. For multi-user production use, store per-session instead.
_LAST_BULK_ROWS = []


@app.route("/bulk_extract", methods=["POST"])
def bulk_extract():
    """
    Bulk import: upload a .csv or .txt file containing a list of CDN
    image URLs. Each URL becomes one row; hex codes for each part land
    in their own column (see CSV_COLUMNS). One failing URL doesn't stop
    the batch -- its row just carries an 'error' field instead.
    """
    global _LAST_BULK_ROWS
    try:
        if "urls_file" not in request.files or request.files["urls_file"].filename == "":
            return jsonify({"error": "Please choose a .csv or .txt file listing image URLs"}), 400

        raw = request.files["urls_file"].read()
        urls = parse_url_list_file(raw)

        if not urls:
            return jsonify({"error": "No URLs found in that file"}), 400

        MAX_ROWS = 40  # keep a single request from running indefinitely
        truncated = len(urls) > MAX_ROWS
        urls = urls[:MAX_ROWS]

        rows = []
        for url in urls:
            try:
                colors, dimensions, meta = extract_eyeglass_colors(url)
                rows.append(colors_to_row(url, colors, dimensions, meta))
            except Exception as e:
                rows.append(colors_to_row(url, None, None, {}, error=str(e)))

        _LAST_BULK_ROWS = rows
        return jsonify({"success": True, "rows": rows, "truncated": truncated, "count": len(rows)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/export", methods=["POST"])
def export():
    """
    Exports rows to a downloadable CSV. Accepts either an explicit
    {"rows": [...]} body (e.g. from the frontend table), or, if no body
    rows are given, falls back to the last bulk run held in memory.
    """
    import csv
    import io as _io
    from flask import Response

    data = request.get_json(silent=True) or {}
    rows = data.get("rows") or _LAST_BULK_ROWS

    if not rows:
        return jsonify({"error": "No results to export yet -- run a bulk extract first"}), 400

    buf = _io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=eyeglass_colors_export.csv"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
