# Checks reading whole PDFs, checking typed-in scores, and writing the Excel file.

import sys
from pathlib import Path

import numpy as np
import openpyxl
import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import fakes  # noqa: E402
from pipeline import (  # noqa: E402
    COLUMNS,
    PageResult,
    _render_dpi,
    check_answer,
    count_pages,
    get_template,
    scan_pdf,
    write_xlsx,
)

ANSWERS = {"presentation": 8, "quality": 10, "delivery": 9, "qa": 10, "member_1": 4, "member_2": 3, "member_3": 5}
EXPECTED_TEXT = {
    "presentation": "8", "quality": "10", "delivery": "9", "qa": "10",
    "member_1": "4", "member_2": "3", "member_3": "5", "member_4": "", "member_5": "", "member_6": "",
}


@pytest.fixture(scope="module")
def template():
    return get_template()


def test_scan_pdf_reads_every_page(tmp_path, template):
    good = fakes.filled_page(template, ANSWERS)
    unsure = fakes.filled_page(template, {**ANSWERS, "delivery": 6})
    fakes.mark(unsure, "delivery", 7)
    not_a_rubric = np.full_like(good, 255)
    pdf = fakes.save_pdf([fakes.tilt(good, 2), unsure, not_a_rubric], tmp_path / "batch.pdf")
    assert count_pages(pdf) == 3

    results = list(scan_pdf(pdf, template))
    assert [page.page_number for page, _ in results] == [1, 2, 3]
    assert all(page.source == pdf for page, _ in results)

    first, image = results[0]
    assert first.answers == EXPECTED_TEXT
    assert first.notes == {} and first.error == ""
    assert image.shape[:2] == template.gray.shape  # the straightened page

    second, _ = results[1]
    assert set(second.notes) == {"delivery"}

    third, _ = results[2]
    assert third.error
    assert third.answers == {key: "" for key in EXPECTED_TEXT}
    # The topic scores still need typing in, so they start out marked for checking.
    assert set(third.notes) == {"presentation", "quality", "delivery", "qa"}


def test_one_bad_page_doesnt_stop_the_rest(tmp_path, template, monkeypatch):
    import pipeline

    real_align = pipeline.align
    calls = []

    def flaky_align(image, template):
        calls.append(1)
        if len(calls) == 2:
            raise np.linalg.LinAlgError("Singular matrix")
        return real_align(image, template)

    monkeypatch.setattr(pipeline, "align", flaky_align)
    page = fakes.filled_page(template, ANSWERS)
    pdf = fakes.save_pdf([page, page, page], tmp_path / "three.pdf")
    results = [result for result, _ in scan_pdf(pdf, template)]
    assert [bool(result.error) for result in results] == [False, True, False]
    assert "LinAlgError" in results[1].error
    assert results[2].answers == EXPECTED_TEXT


def test_huge_pages_are_rendered_smaller(tmp_path, template):
    # A phone photo saved as a PDF at 20 dpi makes a page about 8 feet across.
    pdf = fakes.save_pdf([fakes.filled_page(template, ANSWERS)], tmp_path / "photo.pdf", dpi=20)
    with pymupdf.open(pdf) as doc:
        assert _render_dpi(doc[0]) == 32
    [(page, _)] = list(scan_pdf(pdf, template))
    assert page.answers == EXPECTED_TEXT


def test_password_protected_pdf_is_refused(tmp_path):
    pdf = tmp_path / "locked.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(pdf, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    with pytest.raises(ValueError, match="password"):
        count_pages(pdf)


def test_broken_pdf_is_refused(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a PDF")
    with pytest.raises(pymupdf.FileDataError):
        count_pages(broken)


@pytest.mark.parametrize(
    "key, text, ok",
    [
        ("quality", "", True),
        ("quality", "0", True),
        ("quality", "10", True),
        ("quality", " 7 ", True),
        ("quality", "11", False),
        ("quality", "-1", False),
        ("quality", "4.5", False),
        ("quality", "x", False),
        ("member_2", "5", True),
        ("member_2", "6", False),
    ],
)
def test_check_answer(key, text, ok):
    assert (check_answer(key, text) == "") == ok


def test_write_xlsx(tmp_path):
    pages = [
        PageResult(Path("scans/monday.pdf"), 1, dict(EXPECTED_TEXT), comments='Great demo, "clear" slides\nsecond line'),
        PageResult(Path("scans/tuesday.pdf"), 3, {"presentation": " 7 "}, error="Could not find the table."),
    ]
    out = tmp_path / "grades.xlsx"
    write_xlsx(pages, out)

    sheet = openpyxl.load_workbook(out).active
    rows = list(sheet.iter_rows(values_only=True))
    assert list(rows[0]) == COLUMNS == [
        "File", "Page", "Presentation", "Quality", "Delivery", "QA",
        "Member #1", "Member #2", "Member #3", "Member #4", "Member #5", "Member #6", "Comments",
    ]
    assert rows[1] == ("monday.pdf", 1, 8, 10, 9, 10, 4, 3, 5, None, None, None, 'Great demo, "clear" slides\nsecond line')
    assert rows[2] == ("tuesday.pdf", 3, 7) + (None,) * 10
    assert len(rows) == 3

    # The header row and the File and Page columns are bold and stay on screen.
    assert all(cell.font.bold for cell in sheet[1])
    assert sheet["A2"].font.bold and sheet["B2"].font.bold and not sheet["C2"].font.bold
    assert sheet.freeze_panes == "C2"


def test_xlsx_comments_cant_turn_into_formulas_and_scores_are_tidied(tmp_path):
    comments = ["- needs more eye contact", "+ great energy", "=1+1", "@Glenn good", "fine"]
    pages = [PageResult(Path("a.pdf"), n, {"quality": "05"}, comments=text) for n, text in enumerate(comments, 1)]
    out = tmp_path / "grades.xlsx"
    write_xlsx(pages, out)
    sheet = openpyxl.load_workbook(out).active
    assert [cell.value for cell in sheet["M"][1:]] == comments
    assert {cell.data_type for cell in sheet["M"][1:]} == {"s"}  # text, not formulas
    assert {cell.value for cell in sheet["D"][1:]} == {5}
