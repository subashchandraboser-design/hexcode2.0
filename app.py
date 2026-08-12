import csv
import io
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response

import pipeline as P

app = Flask(__name__)

# ------------------------------------------------------------------
# In-memory job store for bulk processing.
#
# WHY THIS CHANGED:
# The old /bulk_extract endpoint processed every URL one-by-one
# INSIDE a single HTTP request/response cycle. For large lists
# (hundreds of URLs), each needing a network fetch + rembg
# inference + GMM clustering, that easily takes many minutes.
# gunicorn's --timeout kills the worker long before that finishes,
# so the request dies and the browser gets a truncated/failed
# response -- which is why you were only ever seeing ~50 rows.
#
# Now /bulk_extract/start kicks off a background job and returns
# immediately. The frontend polls /bulk_extract/status/<job_id>
# for progress, and /export/<job_id> streams the CSV once done
# (or even partially done). No single HTTP request stays open
# longer than a second, so gunicorn's timeout is irrelevant to it.
#
# CAVEAT: this dict lives in one process's memory. It works with
# multiple THREADS in one worker, but NOT with multiple gunicorn
# WORKER PROCESSES (each has separate memory, so a poll could hit
# a worker that never started the job). Run with a single worker
# and multiple threads, e.g.:
#
#   gunicorn --workers 1 --threads 8 --timeout 120 app:app
#
# If you need multiple worker processes for other traffic, swap
# this dict for Redis (or a small SQLite table) as the job store.
# ------------------------------------------------------------------
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 60 * 60  # drop finished jobs after 1 hour

MAX_WORKERS = int(os.environ.get("BULK_MAX_WORKERS", "6"))
REQUEST_TIMEOUT = 15


def fetch_image(url: str) -> Image.Image:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return P.fetch_image_bytes_to_pil(resp.content)


def zone(obj, obj_mask, y0, y1, x0, x1):
    oh, ow, _ = obj.shape
    crop = obj[int(oh * y0):int(oh * y1), int(ow * x0):int(ow * x1)]
    crop_mask = obj_mask[int(oh * y0):int(oh * y1), int(ow * x0):int(ow * x1)]
    return crop[crop_mask]


def bounding_box(mask):
    import numpy as np
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    return top, bottom + 1, left, right + 1


FRAME_MULTI_COLOR_MIN_PCT = 8.0
FRAME_MULTI_COLOR_MIN_DIST = 30.0


def extract_colors_from_image(img: Image.Image):
    import numpy as np

    img_resized = img.resize((500, 250), Image.Resampling.LANCZOS)
    img_np = np.array(img_resized, dtype=np.float64)

    fg_mask = P.get_foreground_mask(img_resized, img_np)
    fg_mask = P.strip_reflections(img_np, fg_mask)

    fg_pixels = img_np[fg_mask]
    if len(fg_pixels) == 0:
        raise ValueError("No eyeglass pixels found in the provided image.")

    top, bottom, left, right = bounding_box(fg_mask)
    obj = img_np[top:bottom, left:right]
    obj_mask = fg_mask[top:bottom, left:right]

    lens_mask, non_lens_mask = P.separate_lens_regions(obj_mask, obj)

    # Frame / rim color
    front_rim_region = zone(obj, non_lens_mask, 0.20, 0.65, 0.30, 0.70)
    frame_colors = P.extract_dominant_colors_gmm(front_rim_region, max_k=3)
    if len(frame_colors) == 0:
        front_rim_region = obj[non_lens_mask]
        frame_colors = P.extract_dominant_colors_gmm(front_rim_region, max_k=3)
    primary_frame_rgb = frame_colors[0][0] if frame_colors else fg_pixels[0]

    extra_frame_colors = []
    for rgb, pct, tight in frame_colors[1:]:
        if pct >= FRAME_MULTI_COLOR_MIN_PCT and np.linalg.norm(rgb - primary_frame_rgb) > FRAME_MULTI_COLOR_MIN_DIST:
            extra_frame_colors.append((rgb, pct, tight))

    # Temple / leg color
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

    # Lens / tint color
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
    ]

    for i, (rgb, pct, tight) in enumerate(extra_frame_colors, start=2):
        results.append({
            "label": f"Frame Color {i} (Multi-Color Frame)",
            "hex": P.rgb_to_hex(rgb),
            "rgb": f"rgb({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})",
            "percentage": round(float(pct), 1),
            "cluster_tightness": tight,
        })

    results.append({
        "label": "Temple Color (Legs)",
        "hex": P.rgb_to_hex(temple_rgb),
        "rgb": f"rgb({int(temple_rgb[0])}, {int(temple_rgb[1])}, {int(temple_rgb[2])})",
        "percentage": round(float(temple_pct), 1),
        "cluster_tightness": temple_tight,
    })

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
            "note": "No confident lens region found.",
        })

    return results, img.size, {
        "rembg_used": P._REMBG_AVAILABLE,
        "lens_region_detected": lens_detected,
        "multi_color_frame": bool(extra_frame_colors),
    }


def extract_eyeglass_colors(url: str):
    img = fetch_image(url)
    return extract_colors_from_image(img)


def colors_to_row(url: str, colors, dimensions, meta, error: str = None) -> dict:
    row = {
        "url": url,
        "dimensions": f"{dimensions[0]} x {dimensions[1]}" if dimensions else "",
        "primary_frame_hex": "",
        "primary_frame_rgb": "",
        "primary_frame_pct": "",
        "frame_hex": "",
        "frame_rgb": "",
        "frame_pct": "",
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
        "multi_color_frame": meta.get("multi_color_frame") if meta else "",
        "rembg_used": meta.get("rembg_used") if meta else "",
        "error": error or "",
    }
    if error:
        return row

    frame_hexes, frame_rgbs, frame_pcts = [], [], []
    for c in colors:
        label = c["label"]
        if label == "Primary Frame Color" or label.startswith("Frame Color "):
            frame_hexes.append(c.get("hex") or "")
            frame_rgbs.append(c.get("rgb") or "")
            frame_pcts.append(str(c.get("percentage", "")))
            if label == "Primary Frame Color":
                row["primary_frame_hex"] = c.get("hex") or ""
                row["primary_frame_rgb"] = c.get("rgb") or ""
                row["primary_frame_pct"] = c.get("percentage", "")
        elif label == "Temple Color (Legs)":
            row["temple_hex"] = c.get("hex") or ""
            row["temple_rgb"] = c.get("rgb") or ""
            row["temple_pct"] = c.get("percentage", "")
        elif label == "Temple Accent / Pattern Color":
            row["accent_hex"] = c.get("hex") or ""
            row["accent_rgb"] = c.get("rgb") or ""
            row["accent_pct"] = c.get("percentage", "")
        elif label == "Tint Color (Lens / Clip-on)":
            row["tint_hex"] = c.get("hex") or ""
            row["tint_rgb"] = c.get("rgb") or ""
            row["tint_pct"] = c.get("percentage", "")

    row["frame_hex"] = " | ".join(frame_hexes)
    row["frame_rgb"] = " | ".join(frame_rgbs)
    row["frame_pct"] = " | ".join(frame_pcts)
    return row


def parse_url_list_file(file_storage) -> list:
    """Reads the URL list file as a stream (doesn't load huge files fully into RAM)."""
    seen = set()
    urls = []

    text_stream = io.TextIOWrapper(file_storage.stream, encoding="utf-8", errors="ignore")

    first_line = text_stream.readline()
    if not first_line:
        return urls

    text_stream.seek(0)

    if "Image Src" in first_line or "Variant Image" in first_line:
        reader = csv.DictReader(text_stream)
        for row in reader:
            for col in ("Image Src", "Variant Image"):
                u = (row.get(col) or "").strip()
                if u.lower().startswith("http") and u not in seen:
                    seen.add(u)
                    urls.append(u)
    else:
        reader = csv.reader(text_stream)
        for row in reader:
            found = next((f.strip().strip('"') for f in row if f.strip().lower().startswith("http")), None)
            if found and found not in seen:
                seen.add(found)
                urls.append(found)

    return urls


CSV_COLUMNS = [
    "url", "dimensions",
    "primary_frame_hex", "primary_frame_rgb", "primary_frame_pct",
    "frame_hex", "frame_rgb", "frame_pct",
    "temple_hex", "temple_rgb", "temple_pct",
    "accent_hex", "accent_rgb", "accent_pct",
    "tint_hex", "tint_rgb", "tint_pct",
    "lens_detected", "multi_color_frame", "rembg_used", "error",
]


# ------------------------------------------------------------------
# Background job runner
# ------------------------------------------------------------------

def _process_one(url):
    try:
        colors, dimensions, meta = extract_eyeglass_colors(url)
        return colors_to_row(url, colors, dimensions, meta)
    except Exception as e:
        return colors_to_row(url, None, None, {}, error=str(e))


def _run_bulk_job(job_id, urls):
    job = JOBS[job_id]
    n = len(urls)
    job["rows"] = [None] * n

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {executor.submit(_process_one, u): i for i, u in enumerate(urls)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                row = future.result()
                with JOBS_LOCK:
                    job["rows"][idx] = row
                    job["done"] += 1
    finally:
        with JOBS_LOCK:
            job["status"] = "finished"
            job["finished_at"] = time.time()


def _cleanup_old_jobs():
    now = time.time()
    with JOBS_LOCK:
        stale = [
            jid for jid, j in JOBS.items()
            if j.get("status") == "finished" and now - j.get("finished_at", now) > JOB_TTL_SECONDS
        ]
        for jid in stale:
            del JOBS[jid]


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

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


@app.route("/bulk_extract/start", methods=["POST"])
def bulk_extract_start():
    """Kicks off background processing and returns immediately with a job_id.
    No 500-image request ever sits open long enough to hit a server timeout."""
    _cleanup_old_jobs()

    if "urls_file" not in request.files or request.files["urls_file"].filename == "":
        return jsonify({"error": "Please choose a .csv or .txt file listing image URLs"}), 400

    file = request.files["urls_file"]
    urls = parse_url_list_file(file)

    if not urls:
        return jsonify({"error": "No URLs found in that file"}), 400

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "total": len(urls),
            "done": 0,
            "rows": [],
            "created_at": time.time(),
            "finished_at": None,
        }

    thread = threading.Thread(target=_run_bulk_job, args=(job_id, urls), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "total": len(urls)})


@app.route("/bulk_extract/status/<job_id>", methods=["GET"])
def bulk_extract_status(job_id):
    """Frontend polls this every couple seconds for live progress + rows so far."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Unknown or expired job id"}), 404
        rows_so_far = [r for r in job["rows"] if r is not None]
        return jsonify({
            "status": job["status"],
            "total": job["total"],
            "done": job["done"],
            "rows": rows_so_far,
        })


@app.route("/export/<job_id>", methods=["GET"])
def export_job(job_id):
    """Streams the CSV straight from the job store -- works even while a
    job is still partially running, so you never lose completed rows."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Unknown or expired job id"}), 404
        rows = [r for r in job["rows"] if r is not None]

    if not rows:
        return jsonify({"error": "No results to export yet"}), 400

    buf = io.StringIO()
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
