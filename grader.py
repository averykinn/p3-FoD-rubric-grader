# Read the filled-in bubbles on a straightened rubric page.
#
# Works on the output of aligner.align(): the page is the same size as the
# blank template, so every bubble sits at a known spot. Positions below are in
# PDF points on the blank rubric (templates/FoD_rubric_blank.pdf); multiply by
# dpi / 72 to get pixels.
#
# For each question (one row of bubbles) the reader measures how dark the
# inside of every bubble is, picks the darkest one, and adds a note whenever
# the answer is doubtful (nothing filled in, a faint mark, two bubbles filled).
# The review screen highlights those notes so a person can check them.

from dataclasses import dataclass

import cv2
import numpy as np

from aligner import Template
from rasterizer import DEFAULT_DPI

# "Underachieved (score = 0-4)" only has four bubbles. They are read as 1-4,
# so a score of 0 has to be typed in on the review screen.
TOPIC_SCORES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
PARTICIPATION_SCORES = (0, 1, 2, 3, 4, 5)

# Bubble centres for each topic row, left to right. The rows aren't printed
# perfectly in line, so every bubble is listed.
PRESENTATION_BUBBLES = ((231, 153), (269, 153), (308, 153), (345, 153), (429, 153),
                        (466, 153), (504, 153), (542, 153), (580, 153), (741, 153))
QUALITY_BUBBLES = ((231, 205), (269, 205), (308, 205), (345, 205), (429, 205),
                   (466, 205), (504, 205), (543, 205), (580, 205), (741, 205))
DELIVERY_BUBBLES = ((231, 253), (269, 253), (308, 253), (345, 251), (429, 253),
                    (466, 253), (504, 253), (543, 253), (580, 253), (741, 253))
QA_BUBBLES = ((231, 299), (269, 299), (308, 299), (345, 299), (428, 299),
              (465, 299), (503, 299), (541, 299), (579, 299), (741, 299))

# The participation grid is regular: one row per team member, scores 0-5.
PARTICIPATION_X = (212, 254, 294, 336, 378, 419)
PARTICIPATION_Y = (381, 414, 448, 482, 516, 550)

# Bubbles are rings: 11 pt outer radius, 10 pt inner radius.
BUBBLE_RADIUS = 11
# Only the middle of a bubble is measured, so the printed ring and small
# alignment errors don't count as pencil.
SAMPLE_RADIUS = 6.5
# How far (in points) a row of bubbles may be nudged to line up with the scan.
SNAP_DISTANCE = 6

# Darkness of a bubble's middle: 0 = blank paper, 1 = solid black.
FILLED = 0.35  # at least this dark counts as filled in
FAINT = 0.12  # between FAINT and FILLED is a light mark worth a second look
# A tick, X or small dot only covers part of the middle, so it reads lighter
# than a filled bubble. It still counts, without a note, if it is at least
# this dark and every other bubble in the row is close to blank.
TICK = 0.18
# ...and if it is made of real pen or pencil strokes: at least this much of
# the middle must be properly dark. A faint smudge or a rubbed-out mark isn't.
TICK_STROKES = 0.15
BLANK = 0.04  # a bubble this light or lighter has nothing in it
# In a team member row with no name, a mark this dark (and much darker than
# the rest of the row) is too faint to read but worth pointing out.
SPECK = 0.05
# How much more ink than the blank rubric the space above a name line needs
# to count as a written name.
NAME_INK = 0.009
# Lighting is evened out by estimating the paper's brightness everywhere,
# ignoring any dark mark smaller than this (in points; bigger than a bubble).
PAPER_WINDOW = 40


@dataclass(frozen=True)
class Question:
    key: str  # used for the CSV column and the review screen's box
    label: str  # what the review screen calls it
    centres: tuple[tuple[float, float], ...]  # bubble centres, left to right
    scores: tuple[int, ...]  # the score each bubble stands for
    # Team member rows only: the space just above the name line, where the
    # name is written. A row with no name and no bubble is simply unused.
    name_box: tuple[float, float, float, float] | None = None

    @property
    def max_score(self) -> int:
        return max(self.scores)


QUESTIONS: tuple[Question, ...] = (
    Question("presentation", "Presentation", PRESENTATION_BUBBLES, TOPIC_SCORES),
    Question("quality", "Quality", QUALITY_BUBBLES, TOPIC_SCORES),
    Question("delivery", "Delivery", DELIVERY_BUBBLES, TOPIC_SCORES),
    Question("qa", "QA", QA_BUBBLES, TOPIC_SCORES),
) + tuple(
    Question(
        key=f"member_{n}",
        label=f"Member #{n}",
        centres=tuple((x, y) for x in PARTICIPATION_X),
        scores=PARTICIPATION_SCORES,
        name_box=(44, y - 18, 194, y + 5),  # the printed line is at y + 9
    )
    for n, y in enumerate(PARTICIPATION_Y, start=1)
)


@dataclass
class Reading:
    value: int | None  # the score that was filled in, or None if none was
    note: str = ""  # why the reader is unsure; empty when it is confident


def read_page(aligned: np.ndarray, template: Template, dpi: int = DEFAULT_DPI) -> dict[str, Reading]:
    # Read every question on a page that aligner.align() has straightened.
    # Returns {question key: Reading} in the order of QUESTIONS.
    gray = _to_gray(aligned)
    if gray.shape != template.gray.shape:
        raise ValueError("The page must be straightened to the template's size first (see aligner.align).")
    scale = dpi / 72
    ink = _ink(gray, scale)
    blank_ink = _ink(template.gray, scale)
    readings = {}
    for question in QUESTIONS:
        printed = np.array(question.centres, dtype=np.float64) * scale
        centres = _snap_row(gray, template.gray, printed, scale)
        middles = [_middle(ink, centre, scale) for centre in centres]
        darkness = [float(middle.mean()) for middle in middles]
        shift = centres[0] - printed[0]  # the name line moves with its row of bubbles
        named = question.name_box is not None and _has_writing(ink, blank_ink, question.name_box, scale, shift)
        readings[question.key] = _decide(question, darkness, named, middles)
    return readings


def _decide(question: Question, darkness: list[float], named: bool, middles: list[np.ndarray]) -> Reading:
    order = np.argsort(darkness)[::-1]
    best = int(order[0])
    value = question.scores[best]
    filled = [question.scores[i] for i in order if darkness[i] >= FILLED]
    light = [question.scores[i] for i in order if FAINT <= darkness[i] < FILLED]
    rest = [darkness[i] for i in order[1:]]

    if darkness[best] < FAINT:
        if question.name_box is None:
            return Reading(None, "No bubble is filled in.")
        if named:
            return Reading(None, "A name is written here, but no bubble is filled in.")
        if darkness[best] >= SPECK and darkness[best] >= 3 * max(float(np.median(rest)), 0.01):
            return Reading(None, f"There may be a very faint mark on {value}.")
        return Reading(None)  # unused team member row
    if not filled:
        strokes = float((middles[best] >= 0.5).mean())  # share of the middle that is properly dark
        if darkness[best] >= TICK and strokes >= TICK_STROKES and max(rest) <= BLANK:
            return Reading(value)  # a tick, X or dot in an otherwise empty row
        return Reading(value, f"The {value} bubble is only lightly marked.")
    if len(filled) > 1:
        listed = ", ".join(str(score) for score in sorted(filled))
        return Reading(value, f"More than one bubble is filled in ({listed}); {value} is the darkest.")
    if light:
        listed = ", ".join(str(score) for score in sorted(light))
        return Reading(value, f"There is also a light mark on {listed} - erased?")
    return Reading(value)


def _snap_row(gray: np.ndarray, blank: np.ndarray, centres: np.ndarray, scale: float) -> np.ndarray:
    # A page that was bent or curled when it was scanned can still be a few
    # pixels off after straightening, mostly far from the table. Slide the
    # blank rubric's copy of this row over the scan and keep the best fit.
    pad = 1.6 * BUBBLE_RADIUS * scale
    search = int(round(SNAP_DISTANCE * scale))
    x0 = int(centres[:, 0].min() - pad)
    x1 = int(centres[:, 0].max() + pad)
    y0 = int(centres[:, 1].min() - pad)
    y1 = int(centres[:, 1].max() + pad)
    height, width = gray.shape
    if x0 - search < 0 or y0 - search < 0 or x1 + search > width or y1 + search > height:
        return centres
    window = gray[y0 - search : y1 + search, x0 - search : x1 + search]
    fit = cv2.matchTemplate(window, blank[y0:y1, x0:x1], cv2.TM_CCOEFF_NORMED)
    _, _, _, (dx, dy) = cv2.minMaxLoc(fit)
    return centres + [dx - search, dy - search]


def _ink(gray: np.ndarray, scale: float) -> np.ndarray:
    # How dark each pixel is compared with the paper around it: 0 = paper,
    # 1 = black. The paper's brightness is estimated everywhere by wiping out
    # every mark smaller than PAPER_WINDOW (a morphological closing, done at
    # quarter size to be quick). Shadows and grey backgrounds are much bigger
    # than that, so they are evened out instead of counting as pencil.
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    size = int(PAPER_WINDOW * scale / 4) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    paper = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    paper = cv2.resize(paper, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(1 - gray.astype(np.float32) / np.maximum(paper, 1).astype(np.float32), 0, 1)


def _middle(ink: np.ndarray, centre: np.ndarray, scale: float) -> np.ndarray:
    # The ink values of the pixels in the middle of one bubble.
    x, y = (int(round(c)) for c in centre)
    r = int(round(SAMPLE_RADIUS * scale))
    inside = ink[y - r : y + r + 1, x - r : x + r + 1]
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    disk = xx * xx + yy * yy <= r * r
    if inside.shape != disk.shape:  # bubble hangs off the edge of the image
        return np.zeros(1, np.float32)
    return inside[disk]


def _has_writing(ink: np.ndarray, blank_ink: np.ndarray, box: tuple, scale: float, shift: np.ndarray) -> bool:
    # Is a name written on a team member's line? The box is the space just
    # above the printed line, which is empty on the blank rubric. shift is how
    # far the row was nudged to line up with the scan, in pixels.
    x0, y0, x1, y1 = (int(round(v * scale)) for v in box)
    dx, dy = (int(round(v)) for v in shift)
    scanned = ink[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    return float(scanned.mean() - blank_ink[y0:y1, x0:x1].mean()) > NAME_INK


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
