import csv
import io


def parse_url_list_file(raw_bytes: bytes) -> list:
    """
    Accepts:
      - A Shopify product export CSV (has an 'Image Src' header) -- pulls
        only the 'Image Src' and 'Variant Image' columns, which is where
        real image URLs live. All 130+ metafield columns are ignored.
      - A plain .csv/.txt file with one URL per line, or a URL somewhere
        on each line.
    Always deduplicates, so repeated variant rows pointing at the same
    image don't get processed twice.
    """
    text = raw_bytes.decode("utf-8", errors="ignore")
    seen = set()
    urls = []

    # Try as a proper CSV first (handles quoted commas correctly).
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return urls

    header = [h.strip() for h in rows[0]]
    image_cols = [i for i, h in enumerate(header)
                  if h in ("Image Src", "Variant Image")]

    if image_cols:
        # Shopify-style export: read only the known image columns.
        for row in rows[1:]:
            for i in image_cols:
                if i < len(row):
                    u = row[i].strip()
                    if u.lower().startswith("http") and u not in seen:
                        seen.add(u)
                        urls.append(u)
        return urls

    # Fallback: generic file, one URL per line (or first URL-like field).
    for row in rows:
        found = next((f.strip().strip('"') for f in row
                      if f.strip().lower().startswith("http")), None)
        if found and found not in seen:
            seen.add(found)
            urls.append(found)

    return urls
