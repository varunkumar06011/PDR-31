"""Pixel-to-mm calibration using a circular reference marker."""

import logging

import cv2
import numpy as np
import PIL
from PIL import Image

logger = logging.getLogger(__name__)

MARKER_REAL_DIAMETER_MM = 20.0


def detect_marker(image: PIL.Image.Image) -> float:
    """Find a circular reference marker in the image using Hough Circle Transform.

    Returns the marker diameter in pixels. Raises ValueError if no circle
    is confidently detected.

    Uses strict parameters + edge verification to avoid false positives from
    skin folds, cloth wrinkles, or shadows.
    """
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)

    height, width = gray.shape
    min_radius = int(min(height, width) * 0.03)
    max_radius = int(min(height, width) * 0.12)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_radius * 3,
        param1=100,
        param2=50,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        raise ValueError(
            "Calibration marker not detected — ensure a clear circular marker "
            "(e.g. a coin or printed dot, ~20mm diameter) is placed flat in the photo."
        )

    circles = np.round(circles[0]).astype(int)
    logger.info("detect_marker: HoughCircles found %d candidate(s)", len(circles))

    edges = cv2.Canny(gray, 50, 150)

    best_circle = None
    best_score = 0.0

    for idx, (cx, cy, r) in enumerate(circles):
        if r <= 0:
            continue

        h_grid, w_grid = np.ogrid[:height, :width]
        dist = np.sqrt((w_grid - cx) ** 2 + (h_grid - cy) ** 2)
        ring_mask = (dist >= r - 3) & (dist <= r + 3)
        inner_mask = dist < r - 5

        ring_pixels = int(np.count_nonzero(ring_mask))
        if ring_pixels == 0:
            logger.info("  candidate %d: r=%d — skipped (no ring pixels)", idx, r)
            continue

        edge_density = float(np.count_nonzero(edges[ring_mask])) / ring_pixels
        inner_std = float(gray[inner_mask].std()) if inner_mask.any() else 0.0

        logger.info(
            "  candidate %d: cx=%d cy=%d r=%d | edge_density=%.3f | inner_std=%.1f",
            idx, cx, cy, r, edge_density, inner_std,
        )

        if edge_density < 0.15:
            logger.info("    → REJECTED (edge_density < 0.15)")
            continue

        if inner_std > 40:
            logger.info("    → REJECTED (inner_std > 40)")
            continue

        score = edge_density * (1.0 - inner_std / 100.0)
        logger.info("    → score=%.3f", score)
        if score > best_score:
            best_score = score
            best_circle = (cx, cy, r)

    if best_circle is None:
        raise ValueError(
            "Calibration marker not detected — found circular shapes but none "
            "passed edge verification. Use a high-contrast circular marker "
            "(e.g. a coin on a plain background) placed flat in the photo."
        )

    diameter_px = float(best_circle[2] * 2)
    return diameter_px


def pixels_to_mm2(
    mask: np.ndarray,
    marker_diameter_px: float,
    marker_real_diameter_mm: float = MARKER_REAL_DIAMETER_MM,
) -> float:
    """Convert wound mask pixel area to real-world mm².

    Args:
        mask: Binary mask array (H, W), values 0 or 255.
        marker_diameter_px: Diameter of the reference marker in pixels.
        marker_real_diameter_mm: Real-world diameter of the marker in mm.

    Returns:
        Wound area in mm².
    """
    if marker_diameter_px <= 0:
        raise ValueError("marker_diameter_px must be positive")

    pixels_per_mm = marker_diameter_px / marker_real_diameter_mm
    pixel_area_mm2 = 1.0 / (pixels_per_mm ** 2)

    wound_pixels = int(np.count_nonzero(mask))
    area_mm2 = wound_pixels * pixel_area_mm2
    return round(area_mm2, 1)
