# The rubric grader app.
#
# 1. Upload screen: drop scanned rubric PDFs on the orange panel, or click
#    "Upload PDF(s)". Every page is straightened and its bubbles are read.
# 2. Review screen: the pages are listed on the left. Click one to see the
#    scores read from it on the right, and type over any that are wrong.
#    Yellow boxes are ones the reader wasn't sure about (hover for why).
# 3. "Export to CSV" saves every page's scores to the export destination.
#
# Run it with:  python app.py

import sys
from itertools import count
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRect, QRectF, QRegularExpression, QSettings, QSize, QStandardPaths, Qt, QThread, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPixmapCache,
    QRegularExpressionValidator,
    QShortcut,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from grader import QUESTIONS
from pipeline import PageResult, check_answer, count_pages, get_template, scan_pdf, write_csv

UPLOAD_ORANGE = "#FFA042"
UPLOAD_ORANGE_HOVER = "#FF8E1F"  # while files are dragged over the panel
PANEL_ORANGE = "#FFAD5C"
SELECTED_BLUE = "#058CAE"
CHECK_YELLOW = "#FFF07A"
CHECK_BORDER = "#8A5A00"
ERROR_RED = "#C62828"

# Page images are kept as JPEGs and only decoded while on screen, so a big
# batch of scans doesn't fill up memory.
PREVIEW_MAX_SIDE = 2000
PIXMAP_CACHE_KB = 200 * 1024

STYLE = f"""
QMainWindow, #uploadScreen, #pageList, #pageListInner {{ background: white; }}

#dropZone {{ background: {UPLOAD_ORANGE}; border-radius: 100px; }}
#dropZone[dragging="true"] {{ background: {UPLOAD_ORANGE_HOVER}; }}
#prompt {{ font-size: 26px; font-weight: bold; color: black; }}
#progressText {{ font-size: 22px; font-weight: bold; color: black; }}
#message {{ font-size: 17px; color: black; }}
QProgressBar {{ background: white; border: none; border-radius: 9px; max-height: 18px; min-width: 420px; }}
QProgressBar::chunk {{ background: black; border-radius: 9px; }}

QPushButton#primary {{
    background: black; color: white; border: none; border-radius: 36px;
    font-size: 26px; font-weight: bold; min-height: 72px; padding: 0 44px;
}}
QPushButton#primary:hover {{ background: #2E2E2E; }}
QPushButton#primary:disabled {{ background: #6E6E6E; }}
QPushButton#secondary {{
    background: #D5D5D5; color: white; border: none; border-radius: 28px;
    font-size: 22px; font-weight: bold; min-height: 56px; padding: 0 26px;
}}
QPushButton#secondary:hover {{ background: #B9B9B9; }}
#destination {{ font-size: 15px; color: #6B6B6B; }}

#reviewPanel {{ background: {PANEL_ORANGE}; }}
#reviewScroll, #reviewContent {{ background: transparent; }}
#reviewTitle {{ font-size: 30px; font-weight: bold; color: black; padding-top: 6px; }}
#hint {{ font-size: 15px; color: black; background: rgba(255, 255, 255, 0.55); border-radius: 8px; padding: 8px 12px; }}
#fieldLabel {{ font-size: 25px; color: black; }}
#rule {{ background: black; min-height: 3px; max-height: 3px; margin: 4px 0; }}
#reviewPanel QPushButton#primary {{ min-height: 64px; border-radius: 32px; font-size: 25px; }}
QPushButton#link {{ background: transparent; border: none; font-size: 16px; color: black; text-decoration: underline; padding: 4px; }}
QPushButton#link:hover {{ color: #5A3200; }}
#panelDestination {{ font-size: 14px; color: #4A2E00; }}

QLineEdit {{
    background: white; color: black; border: 3px solid white; border-radius: 0;
    font-size: 24px; selection-background-color: {SELECTED_BLUE};
}}
QLineEdit:focus {{ border-color: black; }}
QLineEdit[state="check"] {{ background: {CHECK_YELLOW}; border-color: {CHECK_BORDER}; }}
QLineEdit[state="check"]:focus {{ border-color: black; }}
QLineEdit[state="invalid"] {{ background: #FFD3D3; border-color: {ERROR_RED}; }}
QPlainTextEdit {{ background: white; color: black; border: 3px solid white; font-size: 17px; }}
QPlainTextEdit:focus {{ border-color: black; }}

QScrollBar:vertical {{ background: white; width: 16px; margin: 0; }}
#reviewScroll QScrollBar:vertical {{ background: {PANEL_ORANGE}; }}
QScrollBar::handle:vertical {{ background: #D8D8D8; min-height: 48px; }}
QScrollBar::handle:vertical:hover {{ background: #BDBDBD; }}
#reviewScroll QScrollBar::handle:vertical {{ background: {CHECK_BORDER}; border-radius: 4px; margin: 2px 4px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
QToolTip {{ font-size: 14px; color: black; background: white; border: 1px solid #999; padding: 4px; }}
"""


# Dialogs go through these small functions so the tests can answer them.
def ask_yes_no(parent: QWidget, title: str, text: str) -> bool:
    return QMessageBox.question(parent, title, text) == QMessageBox.StandardButton.Yes


def ask_replace(parent: QWidget, path: Path) -> str:
    # Returns "replace", "elsewhere" or "cancel".
    box = QMessageBox(QMessageBox.Icon.Question, "Replace file?", f"{path.name} already exists in {path.parent}.\n\nReplace it?", parent=parent)
    replace = box.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
    elsewhere = box.addButton("Save somewhere else...", QMessageBox.ButtonRole.ActionRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(replace)
    box.exec()
    if box.clickedButton() is replace:
        return "replace"
    return "elsewhere" if box.clickedButton() is elsewhere else "cancel"


def choose_csv_path(parent: QWidget, start: Path) -> Path | None:
    path, _ = QFileDialog.getSaveFileName(parent, "Choose where to save the CSV", str(start), "CSV files (*.csv)")
    if not path:
        return None
    path = Path(path)
    return path if path.suffix else path.with_suffix(".csv")


def choose_pdfs(parent: QWidget) -> list[Path]:
    paths, _ = QFileDialog.getOpenFileNames(parent, "Choose rubric PDFs", "", "PDF files (*.pdf)")
    return [Path(path) for path in paths]


def show_warning(parent: QWidget, title: str, text: str) -> None:
    QMessageBox.warning(parent, title, text)


def show_info(parent: QWidget, title: str, text: str) -> None:
    QMessageBox.information(parent, title, text)


def plural(number: int, word: str, words: str | None = None) -> str:
    return f"{number} {word if number == 1 else words or word + 's'}"


class ScanWorker(QObject):
    # Reads PDFs on a background thread so the window stays responsive.
    progress = Signal(int, int)  # pages finished, total pages
    page_scanned = Signal(object, object, int, int)  # PageResult, JPEG bytes, image width, image height
    finished = Signal(list)  # [(file name, reason)] for files that couldn't be read

    def __init__(self, paths: list[Path]):
        super().__init__()
        self.paths = paths
        self.cancelled = False  # set from the window when it closes

    @Slot()
    def run(self) -> None:
        failures = []
        try:
            template = get_template()
        except Exception as error:  # e.g. the template PDF is missing
            self.finished.emit([("templates/FoD_rubric_blank.pdf", str(error))])
            return

        page_counts = {}
        for path in self.paths:
            try:
                page_counts[path] = count_pages(path)
            except Exception as error:
                failures.append((path.name, _reason(error)))
        total = sum(page_counts.values())
        done = 0
        self.progress.emit(done, total)

        for path in page_counts:
            try:
                for page, image in scan_pdf(path, template):
                    if self.cancelled:
                        break
                    jpeg, width, height = encode_preview(image)
                    self.page_scanned.emit(page, jpeg, width, height)
                    done += 1
                    self.progress.emit(done, total)
            except Exception as error:
                failures.append((path.name, _reason(error)))
            if self.cancelled:
                break
        self.finished.emit(failures)


def _reason(error: Exception) -> str:
    return str(error) or type(error).__name__


def encode_preview(image: np.ndarray) -> tuple[bytes, int, int]:
    height, width = image.shape[:2]
    if max(height, width) > PREVIEW_MAX_SIDE:
        scale = PREVIEW_MAX_SIDE / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("Couldn't encode the page image.")
    return data.tobytes(), image.shape[1], image.shape[0]


class FolderIcon(QWidget):
    # The white folder on the upload screen, drawn to match the mockup.
    WIDTH, HEIGHT = 660, 532  # drawing units; the icon scales to fit the widget

    def __init__(self):
        super().__init__()
        self.outline = QColor(UPLOAD_ORANGE)  # gap between the folder's front and back
        # Full size is 330x266; it shrinks on short windows so the text below stays clear.
        self.setMinimumSize(170, 137)
        self.setMaximumSize(330, 266)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(330, 266)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        scale = min(self.width() / self.WIDTH, self.height() / self.HEIGHT)
        painter.translate((self.width() - self.WIDTH * scale) / 2, (self.height() - self.HEIGHT * scale) / 2)
        painter.scale(scale, scale)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#FDFDFD"))
        painter.drawRoundedRect(QRectF(418, 0, 200, 90), 42, 42)  # tab
        painter.drawRoundedRect(QRectF(86, 57, 572, 420), 30, 30)  # back

        # The front leans back, so draw it as a sheared rounded rectangle.
        painter.setTransform(QTransform().translate(0, 112).shear(58 / 420, 0), True)
        painter.setPen(QPen(self.outline, 16))
        painter.drawRoundedRect(QRectF(0, 0, 583, 420), 30, 30)


class UploadScreen(QWidget):
    files_chosen = Signal(list)  # list[Path]

    def __init__(self):
        super().__init__(objectName="uploadScreen")
        self.setAcceptDrops(True)
        self.busy = False

        self.drop_zone = QFrame(objectName="dropZone")
        self.folder = FolderIcon()
        self.prompt = QLabel("Drag and drop files OR:", objectName="prompt")
        self.upload_button = QPushButton("Upload PDF(s)", objectName="primary")
        self.upload_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.upload_button.clicked.connect(self._browse)
        self.progress_text = QLabel(objectName="progressText")
        self.progress_bar = QProgressBar(textVisible=False)
        self.message = QLabel(objectName="message", wordWrap=True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.hide()

        zone = QVBoxLayout(self.drop_zone)
        zone.setContentsMargins(60, 40, 60, 40)
        zone.setSpacing(0)
        zone.addStretch(3)
        zone.addWidget(self.folder, 0, Qt.AlignmentFlag.AlignHCenter)
        zone.addSpacing(22)
        zone.addWidget(self.prompt, 0, Qt.AlignmentFlag.AlignHCenter)
        zone.addWidget(self.progress_text, 0, Qt.AlignmentFlag.AlignHCenter)
        zone.addSpacing(40)
        zone.addWidget(self.upload_button, 0, Qt.AlignmentFlag.AlignHCenter)
        zone.addWidget(self.progress_bar, 0, Qt.AlignmentFlag.AlignHCenter)
        zone.addSpacing(18)
        zone.addWidget(self.message, 0, Qt.AlignmentFlag.AlignHCenter)
        zone.addStretch(2)

        self.destination_button = QPushButton("Edit Export Destination", objectName="secondary")
        self.destination_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.destination = QLabel(objectName="destination")

        bottom = QHBoxLayout()
        bottom.setSpacing(18)
        bottom.addWidget(self.destination_button)
        bottom.addWidget(self.destination, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(56, 46, 56, 40)
        layout.setSpacing(36)
        layout.addWidget(self.drop_zone, 1)
        layout.addLayout(bottom)
        self.set_busy(False)

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.prompt.setVisible(not busy)
        self.upload_button.setVisible(not busy)
        self.progress_text.setVisible(busy)
        self.progress_bar.setVisible(busy)
        if busy:
            self.show_message("")
            self.progress_text.setText("Getting ready...")
            self.progress_bar.setRange(0, 0)  # busy animation until the page count is known

    @Slot(int, int)
    def show_progress(self, done: int, total: int) -> None:
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(done)
        if total:
            self.progress_text.setText(f"Reading page {min(done + 1, total)} of {total}...")

    def show_message(self, text: str) -> None:
        self.message.setText(text)
        self.message.setVisible(bool(text))

    def _browse(self) -> None:
        paths = choose_pdfs(self)
        if paths:
            self.files_chosen.emit(paths)

    # Drops are always accepted as a copy. Accepting a "move" (Shift+drag in
    # Explorer) would tell Windows to delete the original scans.
    def _accept_copy(self, event) -> bool:
        if self.busy or not event.mimeData().hasUrls() or not event.possibleActions() & Qt.DropAction.CopyAction:
            event.ignore()
            return False
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
        return True

    def dragEnterEvent(self, event) -> None:
        if self._accept_copy(event):
            self._set_dragging(True)

    def dragMoveEvent(self, event) -> None:
        self._accept_copy(event)

    def dragLeaveEvent(self, event) -> None:
        self._set_dragging(False)

    def dropEvent(self, event) -> None:
        self._set_dragging(False)
        if not self._accept_copy(event):
            return
        files = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        pdfs = [path for path in files if path.suffix.lower() == ".pdf" and path.is_file()]
        if not pdfs:
            self.show_message("Only PDF files can be uploaded.")
            return
        skipped = len(files) - len(pdfs)
        self.files_chosen.emit(pdfs)
        if skipped:
            self.show_message(f"Skipped {plural(skipped, 'file')} that {'isn’t a PDF' if skipped == 1 else 'aren’t PDFs'}.")

    def _set_dragging(self, dragging: bool) -> None:
        self.drop_zone.setProperty("dragging", dragging)
        self.drop_zone.style().unpolish(self.drop_zone)
        self.drop_zone.style().polish(self.drop_zone)
        self.folder.outline = QColor(UPLOAD_ORANGE_HOVER if dragging else UPLOAD_ORANGE)
        self.folder.update()


class PageCard(QWidget):
    # One scanned page in the list on the left of the review screen.
    clicked = Signal()

    CAPTION = 30  # height of the caption line above the page, in pixels
    FRAME = 6  # width of the blue frame around the selected page
    _ids = count()

    def __init__(self, page: PageResult, jpeg: bytes, width: int, height: int):
        super().__init__()
        self.page = page
        self.jpeg = jpeg
        self.aspect = height / width
        self.selected = False
        self.cache_key = f"page-{next(self._ids)}"
        self.setToolTip(str(page.source))
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self.CAPTION + 2 * self.FRAME + round((width - 2 * self.FRAME) * self.aspect)

    def sizeHint(self) -> QSize:
        return QSize(640, self.heightForWidth(640))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Caption: "file name · page N" on the left, the status on the right.
        # A long file name is shortened in the middle so neither gets covered.
        caption = QRect(self.FRAME, 0, self.width() - 2 * self.FRAME, self.CAPTION - 6)
        font = QFont(self.font())
        font.setPixelSize(15)
        bold = QFont(font)
        bold.setBold(True)
        status, colour = self.status()
        status_width = QFontMetrics(bold).horizontalAdvance(status) + 16 if status else 0
        suffix = f"  ·  page {self.page.page_number}"
        metrics = QFontMetrics(font)
        room = caption.width() - status_width - metrics.horizontalAdvance(suffix)
        name = metrics.elidedText(self.page.source.name, Qt.TextElideMode.ElideMiddle, max(room, 0))
        painter.setFont(font)
        painter.setPen(QColor("#555555"))
        painter.drawText(caption, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name + suffix)
        if status:
            painter.setFont(bold)
            painter.setPen(QColor(colour))
            painter.drawText(caption, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, status)

        image = QRect(self.FRAME, self.CAPTION + self.FRAME, self.width() - 2 * self.FRAME, self.height() - self.CAPTION - 2 * self.FRAME)
        painter.drawPixmap(image, self.pixmap(image.size()))
        if self.selected:
            painter.setPen(QPen(QColor(SELECTED_BLUE), self.FRAME, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap, Qt.PenJoinStyle.MiterJoin))
            painter.drawRect(QRectF(image).adjusted(-self.FRAME / 2, -self.FRAME / 2, self.FRAME / 2, self.FRAME / 2))
        else:
            painter.setPen(QPen(QColor("#CFCFCF"), 1))
            painter.drawRect(QRectF(image).adjusted(-0.5, -0.5, 0.5, 0.5))

    def status(self) -> tuple[str, str]:
        if self.page.error and self.page.notes:
            return "Couldn't read this page", ERROR_RED
        if self.page.notes:
            return f"{len(self.page.notes)} to check", CHECK_BORDER
        return "", ""

    def pixmap(self, size: QSize) -> QPixmap:
        # Decode and scale the page image, reusing the last result while the size stays the same.
        ratio = self.devicePixelRatioF()
        key = f"{self.cache_key}-{size.width()}x{size.height()}@{ratio}"
        pixmap = QPixmapCache.find(key)
        if pixmap is None:
            image = QImage.fromData(self.jpeg)
            pixmap = QPixmap.fromImage(
                image.scaled(size * ratio, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
            pixmap.setDevicePixelRatio(ratio)
            QPixmapCache.insert(key, pixmap)
        return pixmap


class PageList(QScrollArea):
    # The scrolling list of scanned pages on the left of the review screen.
    page_selected = Signal(int)

    def __init__(self):
        super().__init__(objectName="pageList")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        inner = QWidget(objectName="pageListInner")
        self.column = QVBoxLayout(inner)
        self.column.setContentsMargins(22, 18, 22, 30)
        self.column.setSpacing(34)
        self.column.addStretch(1)
        self.setWidget(inner)
        self.cards: list[PageCard] = []
        self.current = -1

    def add_page(self, page: PageResult, jpeg: bytes, width: int, height: int) -> None:
        card = PageCard(page, jpeg, width, height)
        index = len(self.cards)
        card.clicked.connect(lambda: self.select(index))
        self.cards.append(card)
        self.column.insertWidget(self.column.count() - 1, card)

    def clear(self) -> None:
        for card in self.cards:
            card.deleteLater()
        self.cards = []
        self.current = -1

    def select(self, index: int) -> None:
        if not 0 <= index < len(self.cards):
            return
        if 0 <= self.current < len(self.cards):
            self.cards[self.current].selected = False
            self.cards[self.current].update()
        self.current = index
        card = self.cards[index]
        card.selected = True
        card.update()
        self.ensureWidgetVisible(card, 0, 0)
        self.page_selected.emit(index)

    def refresh_current(self) -> None:
        if 0 <= self.current < len(self.cards):
            self.cards[self.current].update()


class ScoreBox(QLineEdit):
    # One white score box on the review panel.
    def __init__(self, key: str):
        super().__init__()
        self.key = key
        self.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d{0,2}"), self))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(124, 38)

    def set_state(self, state: str, tooltip: str) -> None:
        # state is "" (normal), "check" (yellow: worth a second look) or "invalid" (red).
        self.setToolTip(tooltip)
        if self.property("state") != state:
            self.setProperty("state", state)
            self.style().unpolish(self)
            self.style().polish(self)


class ReviewPanel(QFrame):
    # The orange panel on the right of the review screen.
    edited = Signal()  # the page on show was changed

    def __init__(self):
        super().__init__(objectName="reviewPanel")
        self.page: PageResult | None = None
        self.loading = False
        self.boxes: dict[str, ScoreBox] = {}

        # "Start over" sits in the top-left corner, level with the centred title.
        self.start_over_button = QPushButton("‹ Start over", objectName="link")
        self.start_over_button.setCursor(Qt.CursorShape.PointingHandCursor)
        title = QLabel("Review:", objectName="reviewTitle")
        header = QGridLayout()
        header.addWidget(title, 0, 0, Qt.AlignmentFlag.AlignHCenter)
        header.addWidget(self.start_over_button, 0, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.hint = QLabel(objectName="hint", wordWrap=True)
        self.hint.hide()

        # One grid for all the score boxes so they line up. The outer columns
        # stretch to keep the labels and boxes centred; the rules span all four.
        form = QGridLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(5)
        form.setColumnStretch(0, 1)
        form.setColumnStretch(3, 1)
        row = 0
        for question in QUESTIONS:
            if question.key == "member_1":
                form.addWidget(self._rule(), row, 0, 1, 4)
                row += 1
            label = QLabel(f"{question.label}:", objectName="fieldLabel")
            box = ScoreBox(question.key)
            box.textEdited.connect(lambda text, key=question.key: self._box_edited(key, text))
            box.returnPressed.connect(lambda key=question.key: self._box_accepted(key))
            label.setBuddy(box)
            self.boxes[question.key] = box
            form.addWidget(label, row, 1, Qt.AlignmentFlag.AlignRight)
            form.addWidget(box, row, 2)
            row += 1
        form.addWidget(self._rule(), row, 0, 1, 4)

        self.comments = QPlainTextEdit()
        self.comments.setTabChangesFocus(True)
        self.comments.setMinimumHeight(70)
        self.comments.setMaximumHeight(160)
        self.comments.textChanged.connect(self._comments_changed)
        comments_label = QLabel("Comments:", objectName="fieldLabel")
        comments_row = QHBoxLayout()
        comments_row.setSpacing(18)
        comments_row.addWidget(comments_label, 0, Qt.AlignmentFlag.AlignTop)
        comments_row.addWidget(self.comments, 1)

        # Everything above the export button scrolls if the window is short,
        # so the button itself is always on screen.
        content = QWidget(objectName="reviewContent")
        column = QVBoxLayout(content)
        column.setContentsMargins(30, 10, 30, 6)
        column.setSpacing(10)
        column.addLayout(header)
        column.addWidget(self.hint)
        column.addLayout(form)
        column.addSpacing(4)
        column.addLayout(comments_row)
        column.addStretch(1)
        self.scroll = scroll = QScrollArea(objectName="reviewScroll")
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.export_button = QPushButton("Export to CSV", objectName="primary")
        self.export_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.destination = QLabel(objectName="panelDestination")
        self.destination.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 14)
        layout.setSpacing(8)
        layout.addWidget(scroll, 1)
        layout.addWidget(self.export_button, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.destination)
        self.setEnabled(False)

    def _rule(self) -> QFrame:
        rule = QFrame(objectName="rule")
        rule.setContentsMargins(0, 0, 0, 0)
        return rule

    def show_page(self, page: PageResult | None) -> None:
        self.page = page
        self.setEnabled(page is not None)
        self.loading = True
        for key, box in self.boxes.items():
            box.setText(page.answers.get(key, "") if page else "")
            self._restyle(key)
        self.comments.setPlainText(page.comments if page else "")
        self.loading = False
        self._update_hint()

    def _box_edited(self, key: str, text: str) -> None:
        if self.page is None:
            return
        self.page.answers[key] = text
        self.page.notes.pop(key, None)  # the person has looked at it
        self._restyle(key)
        self._update_hint()
        self.edited.emit()

    def _box_accepted(self, key: str) -> None:
        # Enter keeps a yellow box's value and moves on to the next box.
        if self.page is not None and self.page.notes.pop(key, None) is not None:
            self._restyle(key)
            self._update_hint()
            self.edited.emit()
        # Moving on from the box itself (not the panel) goes through the
        # scroll area, which scrolls the next box into view, just like Tab.
        self.boxes[key].focusNextChild()

    def focus_box(self, key: str) -> None:
        self.boxes[key].setFocus()
        self.scroll.ensureWidgetVisible(self.boxes[key])

    def _comments_changed(self) -> None:
        if self.page is not None and not self.loading:
            self.page.comments = self.comments.toPlainText()
            self.edited.emit()

    def _restyle(self, key: str) -> None:
        box = self.boxes[key]
        problem = check_answer(key, box.text())
        note = self.page.notes.get(key) if self.page else None
        if problem:
            box.set_state("invalid", problem)
        elif note:
            box.set_state("check", f"{note}\nPress Enter to keep this value, or type the right one.")
        else:
            box.set_state("", "")

    def _update_hint(self) -> None:
        page = self.page
        if page is None:
            text = ""
        elif page.error and page.notes:
            text = "Couldn't line this page up with the rubric, so nothing was read. Type the scores in from the paper."
        elif page.notes:
            verb = "needs" if len(page.notes) == 1 else "need"
            text = f"{plural(len(page.notes), 'box', 'boxes')} {verb} a second look. Hover over a yellow box to see why."
        else:
            text = ""
        self.hint.setText(text)
        self.hint.setToolTip(page.error if page else "")
        self.hint.setVisible(bool(text))


class ReviewScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.page_list = PageList()
        self.panel = ReviewPanel()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.page_list, 56)
        layout.addWidget(self.panel, 44)


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None):
        super().__init__()
        self.setWindowTitle("FoD Rubric Grader")
        self.setMinimumSize(980, 600)  # fits a 1366x768 laptop, or a 1080p screen at 150% scaling
        self.settings = settings or QSettings("FoD Rubric Grader", "FoD Rubric Grader")
        self.pages: list[PageResult] = []
        self.unsaved = False  # edits or new pages since the last export
        self.thread: QThread | None = None
        self.worker: ScanWorker | None = None
        QPixmapCache.setCacheLimit(PIXMAP_CACHE_KB)

        self.upload = UploadScreen()
        self.review = ReviewScreen()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.upload)
        self.stack.addWidget(self.review)
        self.setCentralWidget(self.stack)

        self.upload.files_chosen.connect(self.start_scan)
        self.upload.destination_button.clicked.connect(self.choose_destination)
        self.review.page_list.page_selected.connect(self.show_page)
        self.review.panel.edited.connect(self.page_edited)
        self.review.panel.export_button.clicked.connect(self.export_csv)
        self.review.panel.start_over_button.clicked.connect(self.start_over)
        # Ctrl+PageDown / Ctrl+PageUp: next / previous page, without the mouse.
        for keys, step in (("Ctrl+PgDown", 1), ("Ctrl+PgUp", -1)):
            shortcut = QShortcut(QKeySequence(keys), self.review)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda step=step: self.step_page(step))
        self.update_destination()

    # --- export destination -------------------------------------------------

    def export_path(self) -> Path:
        saved = self.settings.value("export_path", "")
        if saved:
            return Path(saved)
        documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        return Path(documents) / "rubric_grades.csv"

    def choose_destination(self) -> bool:
        path = choose_csv_path(self, self.export_path())
        if path is None:
            return False
        self.settings.setValue("export_path", str(path))
        self.update_destination()
        return True

    def update_destination(self) -> None:
        # Show the folder and file name; the full path is in the tooltip.
        path = self.export_path()
        short = Path(path.parent.name) / path.name
        for label, verb in ((self.upload.destination, "Exports to"), (self.review.panel.destination, "Saves to")):
            label.setText(f"{verb} {short}")
            label.setToolTip(str(path))

    # --- reading PDFs -------------------------------------------------------

    def start_scan(self, paths: list[Path]) -> None:
        if self.thread is not None:
            return
        self.upload.set_busy(True)
        self.thread = QThread(self)
        self.worker = ScanWorker(paths)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.upload.show_progress)
        self.worker.page_scanned.connect(self.add_page)
        self.worker.finished.connect(self.scan_finished)
        self.thread.start()

    @Slot(object, object, int, int)
    def add_page(self, page: PageResult, jpeg: bytes, width: int, height: int) -> None:
        self.pages.append(page)
        self.review.page_list.add_page(page, jpeg, width, height)
        self.unsaved = True

    @Slot(list)
    def scan_finished(self, failures: list) -> None:
        if self.thread is None:  # the window is closing
            return
        self.stop_scan()
        self.upload.set_busy(False)
        if failures:
            lines = "\n".join(f"•  {name}: {reason}" for name, reason in failures)
            show_warning(self, "Some files couldn't be read", f"These files were skipped:\n\n{lines}")
        if not self.pages:
            self.upload.show_message("No pages could be read. Try another PDF.")
            return
        self.stack.setCurrentWidget(self.review)
        if self.review.page_list.current < 0:
            self.review.page_list.select(0)

    def stop_scan(self) -> None:
        if self.thread is None:
            return
        self.worker.cancelled = True
        self.thread.quit()
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.thread = self.worker = None

    # --- reviewing ----------------------------------------------------------

    @Slot(int)
    def show_page(self, index: int) -> None:
        self.review.panel.show_page(self.pages[index])

    @Slot()
    def page_edited(self) -> None:
        self.unsaved = True
        self.review.page_list.refresh_current()

    def step_page(self, step: int) -> None:
        page_list = self.review.page_list
        target = page_list.current + step
        if 0 <= target < len(page_list.cards):
            page_list.select(target)
            self.review.panel.focus_box(QUESTIONS[0].key)

    def start_over(self) -> None:
        if self.unsaved and not ask_yes_no(self, "Start over?", "The scores on this screen haven't been exported yet. Start over anyway?"):
            return
        self.pages = []
        self.unsaved = False
        self.review.page_list.clear()
        self.review.panel.show_page(None)
        self.upload.show_message("")
        self.stack.setCurrentWidget(self.upload)

    # --- export -------------------------------------------------------------

    def export_csv(self) -> None:
        for index, page in enumerate(self.pages):
            for question in QUESTIONS:
                problem = check_answer(question.key, page.answers.get(question.key, ""))
                if problem:
                    self.review.page_list.select(index)
                    self.review.panel.focus_box(question.key)
                    show_warning(self, "Can't export yet", f"{page.source.name}, page {page.page_number}, {question.label}: {problem}")
                    return

        # Yellow boxes include the topic scores of pages that couldn't be read.
        to_check = sum(len(page.notes) for page in self.pages)
        if to_check and not ask_yes_no(
            self,
            "Export anyway?",
            f"{plural(to_check, 'yellow box', 'yellow boxes')} {'hasn’t' if to_check == 1 else 'haven’t'} been checked yet.\n\nExport anyway?",
        ):
            return

        path = self.export_path()
        if path.exists():
            choice = ask_replace(self, path)
            if choice == "cancel" or (choice == "elsewhere" and not self.choose_destination()):
                return
            path = self.export_path()
        while True:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                write_csv(self.pages, path)
                break
            except OSError as error:
                if isinstance(error, PermissionError) and path.exists():
                    reason = "If the file is open in Excel, close it and try again."
                else:
                    reason = "The folder may be missing, or you may not be allowed to save there."
                question = f"Couldn't save {path}.\n\n{reason}\n({error.strerror or error})\n\nSave somewhere else?"
                if not ask_yes_no(self, "Export failed", question) or not self.choose_destination():
                    return
                path = self.export_path()
        self.unsaved = False
        show_info(self, "Exported", f"Saved {plural(len(self.pages), 'page')} to\n{path}")

    def closeEvent(self, event) -> None:
        if self.unsaved and self.pages and not ask_yes_no(self, "Quit without exporting?", "The scores haven't been exported to CSV yet. Quit anyway?"):
            event.ignore()
            return
        self.stop_scan()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    screen = app.primaryScreen().availableGeometry()
    title_bar = 40  # roughly; the window's frame isn't known until it's shown
    width = min(1300, int(screen.width() * 0.92))
    height = min(920, int(screen.height() * 0.92), screen.height() - title_bar)
    window.resize(max(width, window.minimumWidth()), max(height, window.minimumHeight()))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
