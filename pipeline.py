# Turn uploaded rubric PDFs into pages ready for review, and write the
# reviewed scores to an Excel workbook.
#
# This is everything the app does apart from drawing the screens, so it can
# be used and tested without a window:
#
#     template = get_template()
#     for page, image in scan_pdf("scans.pdf", template):
#         print(page.page_number, page.answers, page.notes)

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import pymupdf
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from aligner import AlignmentError, Template, align, load_template
from grader import QUESTIONS, read_page
from rasterizer import DEFAULT_DPI, rasterize_page

# Phone photos saved as PDF can have enormous pages (a 72 dpi photo of a
# sheet of paper is several feet across). Render those smaller: the aligner
# scales every page to the template's size anyway.
MAX_RENDER_SIDE = 3000  # pixels

COLUMNS = ["File", "Page"] + [question.label for question in QUESTIONS] + ["Comments"]
LABEL_COLUMNS = 2  # File and Page name each row

QUESTIONS_BY_KEY = {question.key: question for question in QUESTIONS}

UNREAD_NOTE = "This page couldn't be read. Type the score from the paper."

HEADER_FILL = PatternFill("solid", fgColor="FFD9B3")
LABEL_FILL = PatternFill("solid", fgColor="F2F2F2")
HEADER_BORDER = Border(bottom=Side("medium"))
LABEL_BORDER = Border(right=Side("thin"))


@dataclass
class PageResult:
    source: Path  # the PDF this page came from
    page_number: int  # counting from 1
    answers: dict[str, str]  # text for each score box by question key, "" if blank
    notes: dict[str, str] = field(default_factory=dict)  # boxes worth a second look, and why
    comments: str = ""
    error: str = ""  # set when the page couldn't be straightened, so nothing was read


@cache
def get_template() -> Template:
    # The blank rubric, rendered once and reused for every page.
    return load_template(dpi=DEFAULT_DPI)


def count_pages(pdf_path: str | Path) -> int:
    with pymupdf.open(pdf_path) as doc:
        _check_readable(doc)
        return doc.page_count


def scan_pdf(pdf_path: str | Path, template: Template) -> Iterator[tuple[PageResult, np.ndarray]]:
    # Straighten and read every page of a PDF, one page at a time.
    # Yields (result, image) where image is the straightened page, or the page
    # as it was if it couldn't be straightened.
    pdf_path = Path(pdf_path)
    with pymupdf.open(pdf_path) as doc:
        _check_readable(doc)
        for index, page in enumerate(doc):
            image = rasterize_page(page, _render_dpi(page))
            yield scan_page(image, pdf_path, index + 1, template)


def scan_page(image: np.ndarray, source: Path, page_number: int, template: Template) -> tuple[PageResult, np.ndarray]:
    try:
        alignment = align(image, template)
        readings = read_page(alignment.image, template, DEFAULT_DPI)
    except Exception as error:  # one odd page shouldn't stop the rest of the PDF
        message = str(error) if isinstance(error, AlignmentError) else f"{type(error).__name__}: {error}"
        answers = {question.key: "" for question in QUESTIONS}
        # Every topic needs a score, so mark those boxes for checking until they're typed in.
        notes = {question.key: UNREAD_NOTE for question in QUESTIONS if question.name_box is None}
        return PageResult(source, page_number, answers, notes, error=message), image

    answers = {key: "" if reading.value is None else str(reading.value) for key, reading in readings.items()}
    notes = {key: reading.note for key, reading in readings.items() if reading.note}
    return PageResult(source, page_number, answers, notes), alignment.image


def check_answer(key: str, text: str) -> str:
    # What's wrong with a typed-in score, or "" if it's fine. Blank is fine.
    text = text.strip()
    if not text:
        return ""
    highest = QUESTIONS_BY_KEY[key].max_score
    if not text.isdigit() or int(text) > highest:
        return f"Enter a whole number from 0 to {highest}."
    return ""


def write_xlsx(pages: Iterable[PageResult], xlsx_path: str | Path) -> None:
    # One row per page under a header row naming every column. The File and
    # Page columns label each row, and stay on screen with the header while
    # scrolling. Scores are saved as numbers so Excel can add them up.
    book = Workbook()
    sheet = book.active
    sheet.title = "Grades"
    sheet.append(COLUMNS)
    for page in pages:
        scores = [_score_value(page.answers.get(question.key, "")) for question in QUESTIONS]
        sheet.append([page.source.name, page.page_number, *scores, page.comments.strip() or None])
        # A comment like "= needs work" would otherwise be saved as a formula.
        sheet.cell(sheet.max_row, len(COLUMNS)).data_type = "s"

    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.fill = HEADER_FILL
        cell.border = HEADER_BORDER
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in sheet.iter_rows(min_row=2):
        for cell in row[:LABEL_COLUMNS]:
            cell.font = Font(bold=True)
            cell.fill = LABEL_FILL
            cell.alignment = Alignment(horizontal="left", vertical="top")
        row[LABEL_COLUMNS - 1].border = LABEL_BORDER
        for cell in row[LABEL_COLUMNS:-1]:
            cell.alignment = Alignment(horizontal="center", vertical="top")
        row[-1].alignment = Alignment(vertical="top", wrap_text=True)

    file_width = max([len(COLUMNS[0])] + [len(str(cell.value)) for cell in sheet["A"][1:]])
    widths = [min(file_width + 2, 50), 8] + [max(len(question.label) + 2, 8) for question in QUESTIONS] + [60]
    for index, width in enumerate(widths, 1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.freeze_panes = sheet.cell(2, LABEL_COLUMNS + 1)
    sheet.auto_filter.ref = sheet.dimensions  # sort and filter buttons on the header
    book.save(xlsx_path)


def _score_value(text: str) -> int | str | None:
    text = text.strip()
    if text.isdigit():
        return int(text)  # "05" -> 5
    return text or None  # blank stays an empty cell


def _check_readable(doc: pymupdf.Document) -> None:
    if doc.needs_pass:
        raise ValueError("The PDF is password-protected.")


def _render_dpi(page: pymupdf.Page) -> int:
    # PyMuPDF only accepts a whole number of dots per inch.
    longest_side_inches = max(page.rect.width, page.rect.height) / 72
    return max(1, min(DEFAULT_DPI, int(MAX_RENDER_SIDE / longest_side_inches)))
