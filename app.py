import csv
import io
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# response.
#
# /bulk_extract/start kicks off a background job and returns
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
#   gunicorn --workers 1 --threads 8 --timeout 180 app:app
#
# If you need multiple worker processes for other traffic, swap
# this dict for Redis (or a small SQLite table) as the job store.
# ------------------------------------------------------------------
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 60 * 60  # drop finished jobs after 1 hour

MAX_WORKERS = int(os.environ.get("BULK_MAX_WORKERS", "6"))


# ------------------------------------------------------------------
# NOTE ON THIS REWRITE:
# The previous app.py reimplemented color extraction itself, calling
# P.strip_reflections / P.separate_lens_regions / P.extract_dominant_colors_gmm
# -- none of which exist in the current pipeline.py (it now does its
# OWN foreground/component segmentation + GMM clustering internally
# and exposes ready-to-use extract_colors_from_image() /
# extract_eyeglass_colors()). That mismatch is why nothing was coming
# back. This version just calls the pipeline directly and adapts the
# row-building logic to the pipeline's actual output schema:
#
#   each color dict looks like:
#     {"label": ..., "hex": "#RRGGBB", "rgb": "rgb(r, g, b)",
#      "percentage": float, "tightness": float, "confidence": float}
#
#   labels seen in practice:
#     "Primary Frame Color"
#     "Frame Color 2 (Multi-Color Frame)", "Frame Color 3 ...", etc.
#     "Bridge Color"
#     "Temple Color (left)" / "Temple Color (right)" / "Temple Color (unknown)"
#     "Temple Accent (left) 2", "Temple Accent (right) 2", etc.
#     "Lens / Tint Color"
# ------------------------------------------------------------------

def colors_to_row(url: str, colors, dimensions, meta, error: str = None) -> dict:
    row = {
        "url": url,
        "dimensions": f"{dimensions[0]} x {dimensions[1]}" if dimensions else "",

        "primary_frame_hex": "", "primary_frame_rgb": "",
        "primary_frame_pct": "", "primary_frame_confidence": "",

        "frame_extra_colors": "",  # JSON list of any additional frame colors

        "bridge_hex": "", "bridge_rgb": "",
        "bridge_pct": "", "bridge_confidence": "",

        "temple_left_hex": "", "temple_left_rgb": "",
        "temple_left_pct": "", "temple_left_confidence": "",

        "temple_right_hex": "", "temple_right_rgb": "",
        "temple_right_pct": "", "temple_right_confidence": "",

        "temple_unknown_hex": "", "temple_unknown_rgb": "",
        "temple_unknown_pct": "", "temple_unknown_confidence": "",

        "temple_accents": "",  # JSON list of any temple accent/pattern colors

        "tint_hex": "", "tint_rgb": "",
        "tint_pct": "", "tint_confidence": "",

        "frame_detected": meta.get("frame_detected") if meta else "",
        "lens_detected": meta.get("lens_detected") if meta else "",
        "bridge_detected": meta.get("bridge_detected") if meta else "",
        "temple_detected": meta.get("temple_detected") if meta else "",
        "segmentation_method": meta.get("segmentation_method") if meta else "",
        "overall_confidence": meta.get("overall_confidence") if meta else "",

        "error": error or "",
    }
    if error:
        return row

    frame_extras = []
    temple_accents = []

    for c in colors:
        label = c.get("label", "")
        hexv = c.get("hex") or ""
        rgbv = c.get("rgb") or ""
        pct = c.get("percentage", "")
        conf = c.get("confidence", "")

        if label == "Primary Frame Color":
            row["primary_frame_hex"] = hexv
            row["primary_frame_rgb"] = rgbv
            row["primary_frame_pct"] = pct
            row["primary_frame_confidence"] = conf

        elif label.startswith("Frame Color "):
            frame_extras.append({"label": label, "hex": hexv, "rgb": rgbv,
                                  "percentage": pct, "confidence": conf})

        elif label == "Bridge Color":
            row["bridge_hex"] = hexv
            row["bridge_rgb"] = rgbv
            row["bridge_pct"] = pct
            row["bridge_confidence"] = conf

        elif label == "Temple Color (left)":
            row["temple_left_hex"] = hexv
            row["temple_left_rgb"] = rgbv
            row["temple_left_pct"] = pct
            row["temple_left_confidence"] = conf

        elif label == "Temple Color (right)":
            row["temple_right_hex"] = hexv
            row["temple_right_rgb"] = rgbv
            row["temple_right_pct"] = pct
            row["temple_right_confidence"] = conf

        elif label == "Temple Color (unknown)":
            row["temple_unknown_hex"] = hexv
            row["temple_unknown_rgb"] = rgbv
            row["temple_unknown_pct"] = pct
            row["temple_unknown_confidence"] = conf

        elif label.startswith("Temple Accent "):
            temple_accents.append({"label": label, "hex": hexv, "rgb": rgbv,
                                    "percentage": pct, "confidence": conf})

        elif label == "Lens / Tint Color":
            row["tint_hex"] = hexv
            row["tint_rgb"] = rgbv
            row["tint_pct"] = pct
            row["tint_confidence"] = conf

    row["frame_extra_colors"] = json.dumps(frame_extras) if frame_extras else ""
    row["temple_accents"] = json.dumps(temple_accents) if temple_accents else ""

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
    "primary_frame_hex", "primary_frame_rgb", "primary_frame_pct", "primary_frame_confidence",
    "frame_extra_colors",
    "bridge_hex", "bridge_rgb", "bridge_pct", "bridge_confidence",
    "temple_left_hex", "temple_left_rgb", "temple_left_pct", "temple_left_confidence",
    "temple_right_hex", "temple_right_rgb", "temple_right_pct", "temple_right_confidence",
    "temple_unknown_hex", "temple_unknown_rgb", "temple_unknown_pct", "temple_unknown_confidence",
    "temple_accents",
    "tint_hex", "tint_rgb", "tint_pct", "tint_confidence",
    "frame_detected", "lens_detected", "bridge_detected", "temple_detected",
    "segmentation_method", "overall_confidence",
    "error",
]


# ------------------------------------------------------------------
# Background job runner
# ------------------------------------------------------------------

def _process_one(url):
    try:
        colors, dimensions, meta = P.extract_eyeglass_colors(url)
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

        colors, dimensions, meta = P.extract_eyeglass_colors(url)
        return jsonify({
            "success": True,
            "url": url,
            "dimensions": f"{dimensions[0]} x {dimensions[1]}",
            "colors": colors,
            "meta": meta,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/extract_file", methods=["POST"])
def extract_file():
    try:
        if "image" not in request.files or request.files["image"].filename == "":
            return jsonify({"error": "Please choose an image file"}), 400

        file = request.files["image"]
        img = P.fetch_image_bytes_to_pil(file.read())
        colors, dimensions, meta = P.extract_colors_from_image(img)

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
