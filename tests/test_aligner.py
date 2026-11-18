"""Checks the aligner against fake scans made by distorting the blank rubric.

Because each fake scan is built with a known distortion, the test knows
exactly where every point on the page should end up and can measure how far
off the aligner is.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aligner import AlignmentError, align, load_template  # noqa: E402
from rasterizer import rasterize_pdf  # noqa: E402

DPI = 150
SCALE = DPI / 72
# Bubble centres on the blank rubric, in PDF points: a few in the table and
# the participation grid down to the bottom row, where errors are largest.
CHECK_POINTS = np.array(
    [[231, 153], [741, 153], [231, 299], [741, 299], [212, 381], [419, 381], [212, 550], [419, 550]],
    dtype=np.float32,
) * SCALE


@pytest.fixture(scope="module")
def template():
    return load_template(dpi=DPI)


def make_scan(template, angle, jitter=0.0, fill=True, dark_surround=False, seed=0):
    """Return (scan image, true matrix mapping template pixels -> scan pixels)."""
    rng = np.random.default_rng(seed)
    page = cv2.cvtColor(template.gray, cv2.COLOR_GRAY2BGR)
    if fill:  # pencil in some bubbles and scribble across the table border
        for x, y in CHECK_POINTS[::2]:
            cv2.circle(page, (int(x), int(y)), 20, (40, 40, 40), -1)
        cv2.line(page, (1500, 600), (1850, 720), (30, 30, 30), 4)

    w, h = template.size
    side = int(1.3 * np.hypot(w, h))
    centre = np.array([side / 2, side / 2])
    radians = np.deg2rad(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    page_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    scan_corners = (page_corners - [w / 2, h / 2]) @ rotation.T * 1.1 + centre
    scan_corners += rng.uniform(-jitter, jitter, scan_corners.shape)  # perspective wobble
    matrix = cv2.getPerspectiveTransform(page_corners, scan_corners.astype(np.float32))

    background = (60, 60, 60) if dark_surround else (255, 255, 255)
    scan = cv2.warpPerspective(page, matrix, (side, side), borderValue=background)
    noise = rng.normal(0, 6, scan.shape)
    return np.clip(scan + noise, 0, 255).astype(np.uint8), matrix


def worst_error(result, true_matrix):
    """Largest distance (pixels) between where a point lands and where it should."""
    round_trip = result.homography @ true_matrix
    landed = cv2.perspectiveTransform(CHECK_POINTS.reshape(-1, 1, 2), round_trip).reshape(-1, 2)
    return float(np.abs(landed - CHECK_POINTS).max())


@pytest.mark.parametrize("angle", [0, 3, -5, 90, 87, 180, 184, 270, -93])
def test_rotated_scan_is_straightened(template, angle):
    scan, matrix = make_scan(template, angle)
    result = align(scan, template)
    assert result.image.shape[:2] == template.gray.shape
    assert worst_error(result, matrix) < 2.0  # bubbles are ~46 px wide at this DPI


@pytest.mark.parametrize("seed", range(5))
def test_perspective_and_dark_surround(template, seed):
    scan, matrix = make_scan(template, angle=4, jitter=40, dark_surround=True, seed=seed)
    assert worst_error(align(scan, template), matrix) < 2.0


def test_blank_template_aligns_to_itself(template):
    result = align(template.gray, template)
    assert result.score > 0.99
    assert np.allclose(result.homography / result.homography[2, 2], np.eye(3), atol=1e-3)


def test_page_without_rubric_is_rejected(template):
    blank = np.full((1240, 1875, 3), 255, np.uint8)
    with pytest.raises(AlignmentError):
        align(blank, template)
    boxed = blank.copy()  # right shape of rectangle, but nothing else matches
    cv2.rectangle(boxed, (100, 300), (1700, 730), (0, 0, 0), 3)
    with pytest.raises(AlignmentError):
        align(boxed, template)


def test_scanned_pdf_round_trip(template, tmp_path):
    """The full path: an image-only PDF, like a scanner produces, goes in."""
    scan, matrix = make_scan(template, angle=178)
    png = tmp_path / "scan.png"
    cv2.imwrite(str(png), scan)
    pdf = tmp_path / "scan.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width=scan.shape[1] / SCALE, height=scan.shape[0] / SCALE)
        page.insert_image(page.rect, filename=str(png))
        doc.save(pdf)

    pages = rasterize_pdf(pdf, DPI)
    assert len(pages) == 1
    assert worst_error(align(pages[0], template), matrix) < 2.0
