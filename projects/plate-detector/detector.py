#!/usr/bin/env python3
"""
Guatemalan license plate detector.

Usage:
  python detector.py <image_path>           # single image
  python detector.py <folder>               # all JPG/PNG images in folder
  python detector.py <image> --debug        # save intermediate debug images
"""

import cv2
import numpy as np
import easyocr
import re
import sys
import argparse
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

# GT car plate: 30 cm × 15 cm → aspect ratio 2.0 (tolerance ±0.6)
ASPECT_MIN = 1.4
ASPECT_MAX = 5.5   # allow raw text-band detections (ratio ~3-5); expanded later

# Plate must be between 0.12 % and 11 % of total image area
AREA_MIN = 0.0012
AREA_MAX = 0.11

# Only search in the lower portion and horizontal center of the image
# (all images are frontal shots; plate is always in lower center)
SEARCH_Y_FROM = 0.38
SEARCH_Y_TO   = 0.97
SEARCH_X_FROM = 0.04
SEARCH_X_TO   = 0.96

# GT plate regex: 1 letter + 3 digits + 2-3 letters  (e.g. P757JGT)
_GT_RE = re.compile(r'^[A-Z]\d{3}[A-Z]{2,3}$')

# Common OCR substitution corrections, applied per position in the plate:
#   positions 0, 4, 5, 6  → expected LETTER  (digit → letter map)
#   positions 1, 2, 3     → expected DIGIT   (letter → digit map)
_D2L = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '7': 'Z', '6': 'G', '4': 'A', '3': 'E', '2': 'Z'}
_L2D = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'Z': '7', 'G': '6',
        'A': '4', 'T': '7', 'D': '0', 'E': '3', 'Q': '0', 'C': '0', 'J': '1'}

_reader = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _reader


# ── Plate text helpers ────────────────────────────────────────────────────────

def _normalize_7(text: str) -> str | None:
    """
    Try to coerce a 7-char alphanumeric string into GT plate format
    [Letter][Digit][Digit][Digit][Letter][Letter][Letter].
    Returns normalized string if it matches, else None.
    """
    t = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(t) != 7:
        return None
    out = []
    for i, c in enumerate(t):
        if i == 0 or i >= 4:      # letter positions
            out.append(_D2L.get(c, c) if c.isdigit() else c)
        else:                      # digit positions 1–3
            out.append(_L2D.get(c, c) if c.isalpha() else c)
    result = ''.join(out)
    return result if _GT_RE.match(result) else None


def _extract_plate_text(raw: str) -> str | None:
    """
    Slide a 7-char window over the cleaned OCR output and return the first
    7-char chunk that normalizes to a valid GT plate.
    """
    clean = re.sub(r'[^A-Z0-9]', '', raw.upper())

    # All modern GT plates start with 'P'.  OCR may misread P as R, F, D, or 0.
    # First pass: force the candidate's first character to 'P' if it's a P-look-alike.
    _P_VARIANTS = frozenset('PRFD0O')
    for i in range(max(0, len(clean) - 6)):
        if clean[i] not in _P_VARIANTS:
            continue
        chunk = 'P' + clean[i + 1:i + 7]   # force first char → P
        candidate = _normalize_7(chunk)
        if candidate:
            return candidate

    # Second pass: allow any starting letter (older plates / severe OCR error)
    for i in range(max(0, len(clean) - 6)):
        candidate = _normalize_7(clean[i:i + 7])
        if candidate:
            return candidate
    return None


# ── Detection helpers ─────────────────────────────────────────────────────────

def _get_roi(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Return (ROI sub-image, (x_offset, y_offset)) restricted to plate search area."""
    h, w = img.shape[:2]
    x1 = int(w * SEARCH_X_FROM)
    x2 = int(w * SEARCH_X_TO)
    y1 = int(h * SEARCH_Y_FROM)
    y2 = int(h * SEARCH_Y_TO)
    return img[y1:y2, x1:x2], (x1, y1)


def _detect_sobel_morph(img: np.ndarray) -> list[tuple]:
    """
    Find text-block candidates via Sobel X + horizontal morphological closing.

    Key fix: use only the TOP 15 % of Sobel magnitudes (instead of Otsu) so
    that only strong character-edge pixels survive, preventing the closing
    from merging the entire ROI into one giant blob.
    """
    roi, (ox, oy) = _get_roi(img)
    rh, rw = roi.shape[:2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)

    # Keep only the strongest 15 % of edge pixels to avoid connecting
    # everything in complex backgrounds.
    nonzero = sobelx[sobelx > 0]
    if nonzero.size == 0:
        return []
    thresh_val = float(np.percentile(nonzero, 85))
    _, thresh = cv2.threshold(sobelx, max(thresh_val, 30), 255, cv2.THRESH_BINARY)

    # Horizontal closing: connect characters within a plate row.
    # kw ≈ 1/40 of ROI width keeps it tightly scoped to character gaps.
    kw = max(12, rw // 40)
    kh = max(2,  rh // 70)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close)

    # Skip erosion — it destroys text-band contours that are already thin
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        # Expand narrow text-band boxes vertically to approximate full plate height.
        # A text band (ratio ~3-5) needs height ≈ width/2 to reach plate ratio ~2.0.
        ratio = bw / bh if bh > 0 else 0
        if ratio > 2.9:
            target_h = max(bh, bw // 2)
            extra    = (target_h - bh) // 2
            by = max(0, by - extra)
            bh = target_h
        boxes.append((bx + ox, by + oy, bw, bh))
    return boxes


def _detect_color(img: np.ndarray) -> list[tuple]:
    """
    Find plate candidates using the blue strips as anchor.

    Strategy: use blue (the specific GT plate header/footer color) as the
    primary signal — it is far more discriminative than white because the
    car body is often also white.  Expand the blue region vertically to
    cover the full plate height, then intersect with white to confirm the
    plate body is present.  This avoids the "entire white car body merges
    into one giant contour" problem.
    """
    roi, (ox, oy) = _get_roi(img)
    rh, rw = roi.shape[:2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # GT plate blue (header strip "GUATEMALA" + footer "CENTRO AMERICA")
    blue = cv2.inRange(hsv, np.array([85, 35, 20]), np.array([145, 255, 255]))

    # Expand blue vertically to cover the full plate height (~half plate width)
    # and slightly horizontally to fill the strip width.
    plate_h_est = max(40, rh // 7)
    k_anchor = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(25, rw // 28), plate_h_est))
    blue_expanded = cv2.dilate(blue, k_anchor)

    # Plate body must be white — confirm the expanded blue region contains white
    white = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 75, 255]))
    candidate = cv2.bitwise_and(blue_expanded, white)

    # Close to fill any gaps within the plate rectangle
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 15))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [(cv2.boundingRect(c)[0] + ox, cv2.boundingRect(c)[1] + oy,
             cv2.boundingRect(c)[2], cv2.boundingRect(c)[3])
            for c in contours]


def _filter_candidates(boxes: list[tuple], img_shape: tuple) -> list[tuple]:
    """Keep only boxes whose geometry matches a GT car plate."""
    h, w = img_shape[:2]
    total = w * h
    valid = []
    seen: set[tuple] = set()
    for box in boxes:
        bx, by, bw, bh = box
        if bh < 10 or (bx, by, bw, bh) in seen:
            continue
        seen.add((bx, by, bw, bh))
        if not (total * AREA_MIN <= bw * bh <= total * AREA_MAX):
            continue
        ratio = bw / bh
        if not (ASPECT_MIN <= ratio <= ASPECT_MAX):
            continue
        valid.append((bx, by, bw, bh))
    return valid


def _nms(boxes: list[tuple], iou_thresh: float = 0.35) -> list[tuple]:
    """Non-maximum suppression — remove overlapping boxes, keep the larger one."""
    boxes = sorted({tuple(b) for b in boxes}, key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple] = []
    for b1 in boxes:
        x1, y1, w1, h1 = b1
        skip = False
        for x2, y2, w2, h2 in kept:
            ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter = ix * iy
            union = w1 * h1 + w2 * h2 - inter
            if union > 0 and inter / union > iou_thresh:
                skip = True
                break
        if not skip:
            kept.append(b1)
    return kept


# ── OCR ───────────────────────────────────────────────────────────────────────

def _ocr_versions(crop: np.ndarray) -> list[np.ndarray]:
    """
    Return up to 3 preprocessed versions of the plate crop for OCR.
    Multiple versions improve robustness across different lighting conditions.
    """
    th = 90
    h, w = crop.shape[:2]
    tw = max(100, int(w * th / h))
    versions = []

    # 1 — Original, upscaled
    v0 = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_CUBIC)
    versions.append(v0)

    # 2 — CLAHE-enhanced (helps dark / backlit plates)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    v1 = cv2.resize(enhanced, (tw, th), interpolation=cv2.INTER_CUBIC)
    versions.append(v1)

    # 3 — Grayscale + Otsu binarization (maximizes contrast for clean plates)
    gray = cv2.cvtColor(v1, cv2.COLOR_BGR2GRAY)
    _, v2_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(cv2.cvtColor(v2_gray, cv2.COLOR_GRAY2BGR))

    return versions


def _run_ocr(crop: np.ndarray, debug: bool = False) -> tuple[str | None, float]:
    """
    Run EasyOCR on multiple preprocessed versions of the crop.
    Returns (plate_text, confidence_score).

    Uses a sliding-window normalizer to recover plate text even when
    EasyOCR makes common digit↔letter substitutions.
    """
    reader = _get_reader()
    best_text: str | None = None
    best_conf: float = 0.0

    for version in _ocr_versions(crop):
        results = reader.readtext(
            version,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            paragraph=False,
        )
        raw = ''.join(t.upper().replace(' ', '') for _, t, c in results if c > 0.05)

        if debug:
            print(f"      raw_ocr='{raw}'")

        text = _extract_plate_text(raw)
        if text:
            avg_conf = float(np.mean([c for _, _, c in results])) if results else 0.5
            score = 0.55 + 0.45 * avg_conf   # boost: pattern matched
            if score > best_conf:
                best_text = text
                best_conf = score
            break  # pattern found — no need to try other versions

        # No pattern match: keep highest-confidence raw result as fallback
        if results:
            avg_conf = float(np.mean([c for _, _, c in results]))
            if avg_conf > best_conf:
                best_conf = avg_conf
                best_text = raw[:10] if raw else None

    return best_text, best_conf


# ── Crop helpers ─────────────────────────────────────────────────────────────

def _tight_crop(img: np.ndarray, x: int, y: int, w: int, h: int,
                pad: int = 6) -> np.ndarray:
    """
    Return a plate crop from the candidate box.

    For oversized boxes (produced by the blue-anchor color method), shrink to
    the tightest bounding rectangle around the white content inside the box.
    This prevents the OCR from reading GUATEMALA / CENTRO AMERICA strips or
    surrounding bumper/body areas.
    """
    ih, iw = img.shape[:2]
    img_area = iw * ih

    # Expected plate area ≈ 3 % of image; flag anything > 2× that as oversized
    if w * h > img_area * 0.05:
        # Find white pixels inside the box and tighten
        x1c = max(0, x);  y1c = max(0, y)
        x2c = min(iw, x + w); y2c = min(ih, y + h)
        region = img[y1c:y2c, x1c:x2c]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # Use white+blue so the tightened box includes the plate header strip
        white = cv2.inRange(hsv, np.array([0,  0, 90]),  np.array([180, 80, 255]))
        blue  = cv2.inRange(hsv, np.array([85, 35, 20]), np.array([145, 255, 255]))
        plate_mask = cv2.bitwise_or(white, blue)
        pts = cv2.findNonZero(plate_mask)
        if pts is not None:
            rx, ry, rw, rh = cv2.boundingRect(pts)
            # Accept any tightened box that is at least 20 px tall
            if rh > 20:
                x = x1c + rx;  y = y1c + ry
                w = rw;        h = rh

    x1 = max(0, x - pad);   y1 = max(0, y - pad)
    x2 = min(iw, x + w + pad); y2 = min(ih, y + h + pad)
    return img[y1:y2, x1:x2]


# ── Main detection ────────────────────────────────────────────────────────────

def detect_plate(
    image_path: str | Path,
    debug: bool = False,
    debug_dir: Path | None = None,
) -> tuple[str | None, np.ndarray]:
    """
    Detect and read the license plate in *image_path*.
    Returns (plate_text, annotated_image).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    stem = Path(image_path).stem

    # 1. Gather candidates from both methods
    sobel_boxes = _filter_candidates(_detect_sobel_morph(img), img.shape)
    color_boxes  = _filter_candidates(_detect_color(img),       img.shape)
    candidates   = _nms(sobel_boxes + color_boxes)

    if debug and debug_dir:
        vis = img.copy()
        for i, (x, y, w, h) in enumerate(candidates):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis, str(i), (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / f"{stem}_cands.jpg"), vis)

    if not candidates:
        return None, img.copy()

    # 2. Evaluate candidates; sort by closeness to ideal plate ratio (2.0)
    candidates.sort(key=lambda b: abs(b[2] / b[3] - 2.0))

    annotated  = img.copy()
    best_text: str | None = None
    best_conf: float      = 0.0
    best_box: tuple | None = None

    for box in candidates[:8]:
        x, y, w, h = box
        crop = _tight_crop(img, x, y, w, h)

        text, conf = _run_ocr(crop, debug=debug)

        if debug:
            print(f"    box=({x},{y},{w},{h}) ratio={w/h:.2f} "
                  f"text='{text}' conf={conf:.2f}")

        # Prefer P-starting plates (all modern GT plates start with 'P').
        # Give them a 30 % confidence bonus so they beat non-P false positives
        # (e.g. "CHEVROLET" text which also matches the 7-char letter-digit pattern).
        effective_conf = conf * 1.3 if (text and text.startswith('P')) else conf

        if effective_conf > best_conf:
            best_text = text
            best_conf = effective_conf
            best_box  = box

        # Stop early only when we have a high-confidence P-starting plate
        if effective_conf >= 0.72 and text and text.startswith('P'):
            break

    # 3. Annotate result
    if best_box:
        x, y, w, h = best_box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 0), 3)
        label = best_text or ''
        cv2.putText(annotated, label, (x, max(y - 12, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 0), 3)

    return best_text, annotated


# ── GUI refactor: importable entry point used by app.py ───────────────────────

def detect(image_path: str) -> tuple[np.ndarray, str | None]:
    """Returns (annotated_image_bgr, plate_text_or_None).

    Thin wrapper around detect_plate() that swaps the return order so
    app.py can call a stable, documented API without touching the core pipeline.
    """
    # GUI refactor: detect_plate returns (text, annotated); we expose (annotated, text)
    text, annotated = detect_plate(image_path)
    return annotated, text


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Guatemalan license plate detector')
    parser.add_argument('input',          help='Image file or folder')
    parser.add_argument('--output', '-o', default='output',
                        help='Output folder (default: output)')
    parser.add_argument('--debug',  '-d', action='store_true',
                        help='Save debug candidate images and print OCR details')
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)

    debug_dir: Path | None = None
    if args.debug:
        debug_dir = output_path / 'debug'
        debug_dir.mkdir(exist_ok=True)

    if input_path.is_dir():
        images = sorted(input_path.glob('*.jpg')) + sorted(input_path.glob('*.png'))
    elif input_path.is_file():
        images = [input_path]
    else:
        print(f"ERROR: '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    if not images:
        print("No images found.")
        sys.exit(0)

    print(f"Processing {len(images)} image(s)…\n")
    detected = 0

    for img_path in images:
        print(f"  {img_path.name}")
        try:
            text, annotated = detect_plate(
                img_path, debug=args.debug, debug_dir=debug_dir)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            continue

        cv2.imwrite(str(output_path / img_path.name), annotated)

        if text:
            print(f"    → Placa: {text}")
            detected += 1
        else:
            print(f"    → No detectada")

    print(f"\nResultado: {detected}/{len(images)} placas detectadas.")
    print(f"Imágenes anotadas guardadas en: {output_path}/")


if __name__ == '__main__':
    main()
