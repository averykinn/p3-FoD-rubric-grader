# Straighten a scanned rubric so it lines up with the blank template.
# 
# The rubric has no dedicated registration marks, so the anchor is the outer
# border of the scoring table: a large black rectangle whose four corners are
# easy to find. The steps are:
# 
# 1. Threshold the page and find the table border (the biggest rectangular
#    contour with the right aspect ratio).
# 2. Fit a straight line to each side of the border and intersect them to get
#    the four corners precisely.
# 3. Warp the page so those corners land where they are on the blank template.
#    Four corners cannot tell which way up the page is, so every rotation is
#    tried and the one that looks most like the template wins.
# 
# The result has the same pixel size as the template rendered at ``dpi``, so a
# point at (x, y) PDF points on the blank rubric is at (x, y) * dpi / 72 in the
# aligned image.
# 
# Usage:
#     python aligner.py scans.pdf -o aligned/ --debug

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rasterizer import DEFAULT_DPI, rasterize_pdf

TEMPLATE_PDF = Path(__file__).parent / "templates" / "FoD_rubric_blank.pdf"

# A table candidate must cover at least this fraction of the image.
MIN_AREA_FRACTION = 0.02
# How much of its convex hull a candidate must fill (1.0 = no dents at all).
MIN_SOLIDITY = 0.90
# How far the area of the fitted four-corner shape may be from the contour's area.
QUAD_AREA_TOLERANCE = 0.10
# Allowed relative difference from the template table's width/height ratio.
ASPECT_TOLERANCE = 0.30
# Below this similarity to the template, the page is reported as not a rubric.
MIN_MATCH_SCORE = 0.35


class AlignmentError(Exception):
    """The page could not be matched to the rubric template."""


@dataclass
class Template:
    gray: np.ndarray  # blank rubric, grayscale
    corners: np.ndarray  # table corners: top-left, top-right, bottom-right, bottom-left

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        return self.gray.shape[1], self.gray.shape[0]

    @property
    def table_aspect(self) -> float:
        tl, tr, br, bl = self.corners
        return float(np.linalg.norm(tr - tl) / np.linalg.norm(bl - tl))


@dataclass
class Alignment:
    image: np.ndarray  # the straightened page, same size as the template
    homography: np.ndarray  # 3x3 matrix mapping scan pixels -> template pixels
    corners: np.ndarray  # table corners found in the scan, matching Template.corners order
    score: float  # similarity to the template, 1.0 = identical


def load_template(pdf_path: str | Path = TEMPLATE_PDF, dpi: int = DEFAULT_DPI) -> Template:
    """Render the blank rubric and locate its table corners."""
    gray = _to_gray(rasterize_pdf(pdf_path, dpi)[0])
    corners = find_table_corners(gray)
    # The template is upright, so the corners can be named by position:
    # top-left has the smallest x+y, bottom-right the largest, and so on.
    total = corners.sum(axis=1)
    diff = corners[:, 0] - corners[:, 1]
    ordered = np.array(
        [corners[total.argmin()], corners[diff.argmax()], corners[total.argmax()], corners[diff.argmin()]],
        dtype=np.float32,
    )
    return Template(gray=gray, corners=ordered)


def find_table_corners(image: np.ndarray, expected_aspect: float | None = None) -> np.ndarray:
    """Find the four corners of the table border.

    Returns a 4x2 float32 array going clockwise around the table. Which corner
    comes first is arbitrary. If ``expected_aspect`` (long side / short side)
    is given, rectangles of a different shape are ignored.
    """
    gray = _to_gray(image)
    binary = _binarize(gray)
    # RETR_LIST rather than RETR_EXTERNAL: on a scan with a dark surround the
    # whole page is one outer contour and the table would be hidden inside it.
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    min_area = MIN_AREA_FRACTION * gray.shape[0] * gray.shape[1]
    best, best_area = None, 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area <= best_area:
            continue
        # Cheap check first: a table border has no big dents in its outline.
        if area / cv2.contourArea(cv2.convexHull(contour)) < MIN_SOLIDITY:
            continue
        try:
            corners = _refine_corners(contour)
        except np.linalg.LinAlgError:  # two fitted sides came out parallel: not a usable rectangle
            continue
        # The four corners must describe the same shape as the contour itself.
        if abs(cv2.contourArea(corners) / area - 1) > QUAD_AREA_TOLERANCE:
            continue
        if expected_aspect is not None:
            sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
            long_side, short_side = sorted([sides[0] + sides[2], sides[1] + sides[3]], reverse=True)
            if abs(long_side / short_side - expected_aspect) / expected_aspect > ASPECT_TOLERANCE:
                continue
        best, best_area = corners, area

    if best is None:
        raise AlignmentError("Could not find the rubric table border on this page.")
    return best


def align(image: np.ndarray, template: Template) -> Alignment:
    """Warp a scanned page so it sits exactly on top of the template."""
    gray = _to_gray(image)
    aspect = max(template.table_aspect, 1 / template.table_aspect)
    corners = find_table_corners(gray, expected_aspect=aspect)

    best: Alignment | None = None
    for shift in range(4):
        candidate = np.roll(corners, shift, axis=0)
        homography = cv2.getPerspectiveTransform(candidate, template.corners)
        warped_gray = cv2.warpPerspective(gray, homography, template.size, borderValue=255)
        score = _similarity(warped_gray, template.gray)
        if best is None or score > best.score:
            best = Alignment(warped_gray, homography, candidate, score)

    if best.score < MIN_MATCH_SCORE:
        raise AlignmentError(
            f"Found a table, but the page does not look like the rubric (match score {best.score:.2f})."
        )
    border = (255, 255, 255) if image.ndim == 3 else 255
    best.image = cv2.warpPerspective(image, best.homography, template.size, borderValue=border)
    return best


def draw_debug(image: np.ndarray, alignment: Alignment) -> np.ndarray:
    """Draw the detected table corners on the original scan."""
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    points = alignment.corners.astype(int)
    cv2.polylines(out, [points], isClosed=True, color=(0, 200, 0), thickness=3)
    for name, (x, y) in zip(("TL", "TR", "BR", "BL"), points):
        cv2.circle(out, (x, y), 12, (0, 0, 255), 3)
        cv2.putText(out, name, (x + 16, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return out


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Ink becomes white (255), paper becomes black (0)."""
    # Even out shadows and uneven lighting first, by dividing by an estimate
    # of the paper's brightness (the page with every mark smaller than about
    # 1/30 of its width wiped out). Otherwise one threshold for the whole page
    # turns a shadowed area into "ink" and the table border gets lost in it.
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    size = max(small.shape) // 30 | 1
    paper = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    paper = cv2.resize(paper, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    gray = cv2.divide(gray, np.maximum(paper, 1), scale=255)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Close 1-pixel gaps so a faintly scanned border stays in one piece.
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _refine_corners(contour: np.ndarray) -> np.ndarray:
    """Turn a roughly rectangular contour into four precise corners.

    A rough quadrilateral gives the approximate sides. A line is then fitted
    to the contour points along each side, and neighbouring lines are
    intersected. This is far steadier than picking single corner pixels, and
    pen strokes that cross the border are outvoted by the rest of the side.
    """
    hull = cv2.convexHull(contour)
    rough = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(rough) != 4:
        rough = cv2.boxPoints(cv2.minAreaRect(contour))
    rough = rough.reshape(4, 2).astype(np.float32)

    # Put the corners in clockwise order (as seen on screen, y pointing down).
    centre = rough.mean(axis=0)
    angles = np.arctan2(rough[:, 1] - centre[1], rough[:, 0] - centre[0])
    rough = rough[np.argsort(angles)]

    points = contour.reshape(-1, 2).astype(np.float32)
    # The rough corners can be pulled off by a pen stroke sticking out of the
    # border, so fit twice: loosely around the rough sides, then tightly
    # around the first fit.
    corners = _fit_sides(points, rough, band=0.02)
    return _fit_sides(points, corners, band=0.004)


def _fit_sides(points: np.ndarray, corners: np.ndarray, band: float) -> np.ndarray:
    """Fit a line to the points near each side and return where the lines cross.

    ``band`` is how far from a side (as a fraction of its length) a point may
    be and still count as belonging to it.
    """
    lines = []
    for i in range(4):
        start, end = corners[i], corners[(i + 1) % 4]
        direction = end - start
        length = np.linalg.norm(direction)
        direction = direction / length
        offset = points - start
        along = offset @ direction
        across = np.abs(offset[:, 0] * direction[1] - offset[:, 1] * direction[0])
        # Ignore the ends of the side, where the corner is slightly rounded.
        near = (across < max(2.0, band * length)) & (along > 0.05 * length) & (along < 0.95 * length)
        if near.sum() < 2:
            lines.append((start, direction))
            continue
        vx, vy, x0, y0 = cv2.fitLine(points[near], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))

    # Corner i is where side i-1 meets side i.
    return np.array([_intersect(lines[i - 1], lines[i]) for i in range(4)], dtype=np.float32)


def _intersect(line_a, line_b) -> np.ndarray:
    (p, d), (q, e) = line_a, line_b
    t = np.linalg.solve(np.column_stack([d, -e]), q - p)[0]
    return p + t * d


def _similarity(image: np.ndarray, reference: np.ndarray) -> float:
    """Correlation between two same-sized grayscale images, from -1 to 1."""
    scale = 400 / reference.shape[1]
    small_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_reference = cv2.resize(reference, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.matchTemplate(small_image, small_reference, cv2.TM_CCOEFF_NORMED)[0, 0])


def _save_png(path: Path, image: np.ndarray) -> bool:
    # cv2.imwrite can't handle non-English characters in Windows paths, so
    # encode in memory and let numpy write the file.
    ok, data = cv2.imencode(".png", image)
    if not ok:
        return False
    try:
        data.tofile(str(path))
    except OSError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Straighten scanned rubric pages.")
    parser.add_argument("pdf", type=Path, help="PDF of scanned rubrics (one rubric per page)")
    parser.add_argument("-o", "--output", type=Path, default=Path("aligned"), help="folder for the aligned PNGs")
    parser.add_argument("--template", type=Path, default=TEMPLATE_PDF, help="blank rubric PDF")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="resolution of the aligned output")
    parser.add_argument("--debug", action="store_true", help="also save each scan with the detected corners drawn on")
    args = parser.parse_args()

    template = load_template(args.template, args.dpi)
    args.output.mkdir(parents=True, exist_ok=True)
    failures = 0
    for number, page in enumerate(rasterize_pdf(args.pdf, args.dpi), start=1):
        name = f"{args.pdf.stem}_page{number:03d}"
        try:
            result = align(page, template)
        except AlignmentError as error:
            failures += 1
            print(f"page {number}: FAILED - {error}")
            continue
        saved = _save_png(args.output / f"{name}.png", result.image)
        if args.debug:
            saved &= _save_png(args.output / f"{name}_debug.png", draw_debug(page, result))
        if not saved:
            failures += 1
            print(f"page {number}: FAILED - couldn't write the PNG")
            continue
        print(f"page {number}: aligned (match score {result.score:.2f})")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
