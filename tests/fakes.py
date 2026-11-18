# Fake rubric scans for the tests: the blank rubric with bubbles filled in,
# names written on, and the page tilted the way a scanner or camera would.

import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).parent.parent))

from grader import QUESTIONS  # noqa: E402
from rasterizer import DEFAULT_DPI  # noqa: E402

SCALE = DEFAULT_DPI / 72  # pixels per PDF point in a page at the default DPI
QUESTIONS_BY_KEY = {question.key: question for question in QUESTIONS}

PEN = 35  # grey level of a solid pen mark
PENCIL = 115  # a firm pencil mark
FAINT_PENCIL = 195  # a very light mark, or one that was rubbed out


def blank_page(template) -> np.ndarray:
    return cv2.cvtColor(template.gray, cv2.COLOR_GRAY2BGR)


def filled_page(template, answers: dict[str, int], shade: int = PEN) -> np.ndarray:
    # The blank rubric with one bubble filled in per question, e.g. {"quality": 7}.
    page = blank_page(template)
    for key, score in answers.items():
        mark(page, key, score, shade)
    return page


def mark(page: np.ndarray, key: str, score: int, shade: int = PEN, size: float = 0.9) -> None:
    # Fill in the bubble for `score`. size is the mark's radius as a fraction of the bubble's.
    question = QUESTIONS_BY_KEY[key]
    x, y = question.centres[question.scores.index(score)]
    radius = round(11 * size * SCALE)
    cv2.circle(page, (round(x * SCALE), round(y * SCALE)), radius, (shade, shade, shade), -1, cv2.LINE_AA)


def cross(page: np.ndarray, key: str, score: int, shade: int = PEN) -> None:
    # Put an X through a bubble instead of filling it in.
    question = QUESTIONS_BY_KEY[key]
    x, y = (round(v * SCALE) for v in question.centres[question.scores.index(score)])
    r = round(9 * SCALE)
    cv2.line(page, (x - r, y - r), (x + r, y + r), (shade, shade, shade), 4, cv2.LINE_AA)
    cv2.line(page, (x - r, y + r), (x + r, y - r), (shade, shade, shade), 4, cv2.LINE_AA)


def tick(page: np.ndarray, key: str, score: int, width: int = 3, shade: int = PEN) -> None:
    # A check mark through a bubble. width is the pen line in pixels (3 px is about 0.5 mm at 150 dpi).
    question = QUESTIONS_BY_KEY[key]
    x, y = (v * SCALE for v in question.centres[question.scores.index(score)])
    points = np.array([(x - 7 * SCALE, y), (x - 2 * SCALE, y + 6 * SCALE), (x + 8 * SCALE, y - 8 * SCALE)], np.int32)
    cv2.polylines(page, [points], False, (shade, shade, shade), width, cv2.LINE_AA)


def write_name(page: np.ndarray, member: int, name: str = "Student", shade: int = 50) -> None:
    # Handwriting sits just above the printed name line (9 pt below the bubbles' centre).
    y = QUESTIONS_BY_KEY[f"member_{member}"].centres[0][1]
    baseline = (round(50 * SCALE), round((y + 6) * SCALE))
    cv2.putText(page, name, baseline, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.3, (shade, shade, shade), 2, cv2.LINE_AA)


def tilt(page: np.ndarray, angle: float, seed: int = 0, background: int = 255) -> np.ndarray:
    # Rotate the page and warp it a little, like a crooked scan or a phone photo.
    rng = np.random.default_rng(seed)
    height, width = page.shape[:2]
    side = int(1.3 * np.hypot(width, height))
    radians = np.deg2rad(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    moved = (corners - [width / 2, height / 2]) @ rotation.T * 1.1 + side / 2
    moved += rng.uniform(-30, 30, moved.shape)
    matrix = cv2.getPerspectiveTransform(corners, moved.astype(np.float32))
    shade = (background, background, background)
    scan = cv2.warpPerspective(page, matrix, (side, side), borderValue=shade)
    noise = rng.normal(0, 5, scan.shape)
    return np.clip(scan + noise, 0, 255).astype(np.uint8)


def save_pdf(images: list[np.ndarray], path: Path, dpi: int = DEFAULT_DPI) -> Path:
    # Save images as an image-only PDF, one per page, like a scanner does.
    with pymupdf.open() as doc:
        for image in images:
            ok, png = cv2.imencode(".png", image)
            assert ok
            height, width = image.shape[:2]
            page = doc.new_page(width=width * 72 / dpi, height=height * 72 / dpi)
            page.insert_image(page.rect, stream=png.tobytes())
        doc.save(path)
    return path
