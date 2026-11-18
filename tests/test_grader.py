# Checks the bubble reader on fake filled-in rubrics and on a real phone photo.

import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import fakes  # noqa: E402
from aligner import TEMPLATE_PDF, align, load_template  # noqa: E402
from grader import QUESTIONS, Reading, read_page  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
TOPICS = ["presentation", "quality", "delivery", "qa"]

ANSWER_SETS = [
    {"presentation": 1, "quality": 4, "delivery": 5, "qa": 10, "member_1": 0, "member_2": 5, "member_3": 2},
    {"presentation": 10, "quality": 9, "delivery": 6, "qa": 3,
     "member_1": 1, "member_2": 3, "member_3": 4, "member_4": 5, "member_5": 0, "member_6": 2},
    {"presentation": 7, "quality": 8, "delivery": 2, "qa": 6, "member_6": 3},
]


@pytest.fixture(scope="module")
def template():
    return load_template()


def values(readings):
    return {key: reading.value for key, reading in readings.items()}


def notes(readings):
    return {key: reading.note for key, reading in readings.items() if reading.note}


def expected(answers):
    return {question.key: answers.get(question.key) for question in QUESTIONS}


def test_bubble_positions_match_the_template_pdf():
    # Every bubble listed in grader.py must be a ring on the blank rubric, and
    # every ring on the blank rubric must be listed.
    with pymupdf.open(TEMPLATE_PDF) as doc:
        rings = {
            (round((d["rect"].x0 + d["rect"].x1) / 2), round((d["rect"].y0 + d["rect"].y1) / 2))
            for d in doc[0].get_drawings()
            if 20 < d["rect"].width < 24 and 20 < d["rect"].height < 24
        }
    listed = [centre for question in QUESTIONS for centre in question.centres]
    assert len(listed) == len(set(listed)) == 76
    assert set(listed) == rings


def test_blank_rubric(template):
    readings = read_page(template.gray, template)
    assert all(reading.value is None for reading in readings.values())
    # Every topic needs a score; empty team member rows are normal.
    assert set(notes(readings)) == set(TOPICS)


@pytest.mark.parametrize("shade", [fakes.PEN, fakes.PENCIL])
@pytest.mark.parametrize("answers", ANSWER_SETS)
def test_filled_bubbles_are_read(template, answers, shade):
    readings = read_page(fakes.filled_page(template, answers, shade), template)
    assert values(readings) == expected(answers)
    assert notes(readings) == {}


@pytest.mark.parametrize("angle", [1.5, -3, 92, 181, 268])
def test_tilted_scans_are_read(template, angle):
    answers = ANSWER_SETS[1]
    scan = fakes.tilt(fakes.filled_page(template, answers, fakes.PENCIL), angle, seed=abs(int(angle)))
    readings = read_page(align(scan, template).image, template)
    assert values(readings) == expected(answers)
    assert notes(readings) == {}


def test_real_phone_photo(template):
    # A phone photo of a real filled-in rubric: sideways, on a table, with the paper slightly bent.
    # (cv2.imread can't open paths with non-English characters on Windows.)
    photo = cv2.imdecode(np.fromfile(FIXTURES / "real_photo.jpg", np.uint8), cv2.IMREAD_COLOR)
    readings = read_page(align(photo, template).image, template)
    assert values(readings) == {
        "presentation": 10, "quality": 5, "delivery": 10, "qa": 10,
        "member_1": 5, "member_2": 5, "member_3": 5, "member_4": 5, "member_5": 5, "member_6": None,
    }
    assert notes(readings) == {}


def test_small_marks_and_crosses_still_count(template):
    page = fakes.filled_page(template, {"presentation": 3, "quality": 7})
    fakes.mark(page, "delivery", 9, size=0.5)  # a dot that only half fills the bubble
    fakes.cross(page, "qa", 2)  # an X instead of filling in
    readings = read_page(page, template)
    assert values(readings) == expected({"presentation": 3, "quality": 7, "delivery": 9, "qa": 2})
    assert notes(readings) == {}


def test_thin_pen_ticks_need_no_second_look(template):
    # A ballpoint tick (about 0.5 mm, 3 px at 150 dpi) covers only part of the bubble.
    answers = {"presentation": 7, "quality": 4, "delivery": 9, "qa": 10, "member_1": 5}
    page = fakes.blank_page(template)
    for key, score in answers.items():
        fakes.tick(page, key, score, width=3)
    readings = read_page(page, template)
    assert values(readings) == expected(answers)
    assert notes(readings) == {}


def test_shadow_edge_is_not_a_mark(template):
    # A phone's shadow with a sharp edge across the unused team member rows.
    page = fakes.filled_page(template, {"presentation": 5, "quality": 5, "delivery": 5, "qa": 5, "member_1": 3})
    for edge_pt in (216, 222, 260, 300):
        shadowed = page.copy()
        x = round(edge_pt * fakes.SCALE)
        top = round(345 * fakes.SCALE)  # between the table and the team member rows
        shadowed[top:, :x] = (shadowed[top:, :x] * 0.6).astype(np.uint8)
        readings = read_page(shadowed, template)
        assert values(readings)["member_5"] is None and values(readings)["member_6"] is None
        assert notes(readings) == {}, f"shadow edge at {edge_pt} pt"


def test_faint_tick_next_to_a_short_pencil_name_is_flagged(template):
    page = fakes.filled_page(template, {"presentation": 5, "quality": 5, "delivery": 5, "qa": 5})
    fakes.write_name(page, 5, "Li", shade=140)
    fakes.tick(page, "member_5", 3, width=3, shade=190)
    reading = read_page(page, template)["member_5"]
    assert reading.note  # either read with a note, or blank with a note - never silently blank


def test_two_filled_bubbles_are_flagged(template):
    page = fakes.filled_page(template, ANSWER_SETS[0])
    fakes.mark(page, "quality", 7, shade=70)  # a second, slightly lighter answer
    reading = read_page(page, template)["quality"]
    assert reading.value == 4  # the darker one
    assert "More than one bubble" in reading.note and "4, 7" in reading.note


def test_light_mark_is_read_but_flagged(template):
    page = fakes.filled_page(template, {"presentation": 5, "quality": 5, "qa": 5})
    fakes.mark(page, "delivery", 9, fakes.FAINT_PENCIL)
    reading = read_page(page, template)["delivery"]
    assert reading.value == 9
    assert "lightly marked" in reading.note


def test_rubbed_out_mark_is_flagged(template):
    page = fakes.filled_page(template, {"presentation": 5, "quality": 5, "delivery": 6, "qa": 5})
    fakes.mark(page, "delivery", 8, fakes.FAINT_PENCIL)
    reading = read_page(page, template)["delivery"]
    assert reading.value == 6
    assert "light mark on 8" in reading.note


def test_named_member_without_a_score_is_flagged(template):
    page = fakes.filled_page(template, {"presentation": 5, "quality": 5, "delivery": 5, "qa": 5, "member_1": 4})
    fakes.write_name(page, 1, "Ann")
    fakes.write_name(page, 2, "Bo")
    readings = read_page(page, template)
    assert readings["member_1"] == Reading(4)
    assert readings["member_2"].value is None
    assert "name is written" in readings["member_2"].note
    assert readings["member_3"] == Reading(None)  # no name, no score: an unused row


def test_page_must_be_straightened_first(template):
    with pytest.raises(ValueError):
        read_page(np.full((100, 100), 255, np.uint8), template)
