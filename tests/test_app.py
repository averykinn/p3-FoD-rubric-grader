# Checks the app's screens without opening a real window: uploading PDFs,
# fixing scores on the review screen, and exporting the CSV.

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no window on screen

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QEventLoop, QMimeData, QPointF, QSettings, Qt, QTimer, QUrl  # noqa: E402
from PySide6.QtGui import QDropEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

import app as rubric_app  # noqa: E402
import fakes  # noqa: E402
from pipeline import get_template  # noqa: E402

GOOD = {"presentation": 8, "quality": 10, "delivery": 9, "qa": 10, "member_1": 4, "member_2": 3}


class Dialogs:
    # Stands in for every pop-up, recording what was shown and giving set answers.
    def __init__(self):
        self.shown = []
        self.yes = True  # answer to yes/no questions
        self.replace = "replace"  # answer when the CSV already exists
        self.elsewhere = None  # the path picked in the "save somewhere else" file chooser (None = cancel)

    def choose_csv_path(self, parent, start):
        self.shown.append(("choose", "Choose where to save the CSV", ""))
        return self.elsewhere

    def choose_pdfs(self, parent):
        raise AssertionError("A test opened the real file chooser.")

    def ask_yes_no(self, parent, title, text):
        self.shown.append(("ask", title, text))
        return self.yes

    def ask_replace(self, parent, path):
        self.shown.append(("replace", str(path), ""))
        return self.replace

    def show_warning(self, parent, title, text):
        self.shown.append(("warning", title, text))

    def show_info(self, parent, title, text):
        self.shown.append(("info", title, text))

    def titles(self, kind):
        return [title for shown_kind, title, _ in self.shown if shown_kind == kind]


@pytest.fixture(scope="session")
def qapp():
    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(rubric_app.STYLE)
    return application


@pytest.fixture(scope="session")
def template():
    return get_template()


@pytest.fixture
def dialogs(monkeypatch):
    fake = Dialogs()
    for name in ("ask_yes_no", "ask_replace", "show_warning", "show_info", "choose_csv_path", "choose_pdfs"):
        monkeypatch.setattr(rubric_app, name, getattr(fake, name))
    return fake


@pytest.fixture
def window(qapp, dialogs, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue("export_path", str(tmp_path / "out" / "grades.csv"))
    main_window = rubric_app.MainWindow(settings)
    main_window.resize(1300, 900)
    main_window.show()
    yield main_window
    main_window.unsaved = False
    main_window.close()


@pytest.fixture
def demo_pdf(tmp_path, template):
    # Page 1 reads cleanly, page 2 has two doubtful boxes, page 3 isn't a rubric.
    good = fakes.filled_page(template, GOOD)
    unsure = fakes.filled_page(template, {"presentation": 6, "quality": 7, "qa": 4, "member_1": 2})
    fakes.mark(unsure, "quality", 8, shade=90)  # a second, lighter answer: the reader picks 7
    fakes.mark(unsure, "delivery", 3, fakes.FAINT_PENCIL)
    blank = np.full_like(good, 255)
    return fakes.save_pdf([fakes.tilt(good, 3), unsure, blank], tmp_path / "demo.pdf")


def scan(window, paths):
    # Read the PDFs and wait until the review screen is ready. Wait inside
    # Qt's event loop like the real app does: polling with QTest.qWait keeps
    # hold of Python's lock and starves the scanning thread.
    if paths:
        window.start_scan(paths)
    loop = QEventLoop()
    poll = QTimer()
    poll.timeout.connect(lambda: window.thread is None and loop.quit())
    poll.start(10)
    QTimer.singleShot(60_000, loop.quit)  # give up after a minute
    loop.exec()
    poll.stop()
    assert window.thread is None, "Scanning didn't finish."


def type_into(box, text):
    box.setFocus()
    box.selectAll()
    QTest.keyClicks(box, text)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_upload_screen_to_review_screen(window, demo_pdf):
    assert window.stack.currentWidget() is window.upload
    assert "grades.csv" in window.upload.destination.text()

    scan(window, [demo_pdf])

    assert window.stack.currentWidget() is window.review
    assert [page.page_number for page in window.pages] == [1, 2, 3]
    assert window.review.page_list.current == 0
    panel = window.review.panel
    assert {key: box.text() for key, box in panel.boxes.items()} == {
        "presentation": "8", "quality": "10", "delivery": "9", "qa": "10",
        "member_1": "4", "member_2": "3", "member_3": "", "member_4": "", "member_5": "", "member_6": "",
    }
    assert not panel.hint.isVisible()


def test_doubtful_boxes_are_yellow_until_checked(window, demo_pdf):
    scan(window, [demo_pdf])
    window.review.page_list.select(1)
    panel = window.review.panel
    page = window.pages[1]
    card = window.review.page_list.cards[1]
    assert panel.boxes["quality"].property("state") == "check"
    assert panel.boxes["delivery"].property("state") == "check"
    assert "More than one bubble" in panel.boxes["quality"].toolTip()
    assert panel.hint.isVisible()
    assert card.status()[0] == "2 to check"

    assert panel.boxes["quality"].text() == "7"
    type_into(panel.boxes["quality"], "8")  # the reader picked 7; say the right answer is 8
    assert page.answers["quality"] == "8"
    assert panel.boxes["quality"].property("state") == ""
    assert card.status()[0] == "1 to check"

    panel.boxes["delivery"].setFocus()
    QTest.keyClick(panel.boxes["delivery"], Qt.Key.Key_Return)  # keep the 3
    assert page.answers["delivery"] == "3"
    assert page.notes == {}
    assert panel.boxes["delivery"].property("state") == ""
    assert card.status()[0] == ""
    assert not panel.hint.isVisible()


def test_edits_stay_with_their_page(window, demo_pdf):
    scan(window, [demo_pdf])
    panel = window.review.panel
    type_into(panel.boxes["member_3"], "5")
    panel.comments.setFocus()
    QTest.keyClicks(panel.comments, "Strong demo")
    window.review.page_list.select(1)
    assert panel.boxes["member_3"].text() == ""
    assert panel.comments.toPlainText() == ""
    window.review.page_list.select(0)
    assert panel.boxes["member_3"].text() == "5"
    assert panel.comments.toPlainText() == "Strong demo"
    assert window.pages[0].comments == "Strong demo"


def test_out_of_range_score_blocks_export(window, demo_pdf, dialogs):
    scan(window, [demo_pdf])
    window.review.page_list.select(1)
    box = window.review.panel.boxes["member_1"]
    type_into(box, "9")  # participation only goes up to 5
    assert box.property("state") == "invalid"
    window.review.page_list.select(0)

    window.export_csv()
    assert dialogs.titles("warning") == ["Can't export yet"]
    assert window.review.page_list.current == 1  # it jumped to the problem
    assert not window.export_path().exists()


def test_export_writes_every_page(window, demo_pdf, dialogs):
    scan(window, [demo_pdf])
    panel = window.review.panel
    panel.comments.setFocus()
    QTest.keyClicks(panel.comments, "Good, clear talk")
    window.review.page_list.select(2)  # the page that couldn't be read
    assert panel.hint.isVisible()
    type_into(panel.boxes["presentation"], "4")

    window.export_csv()

    # Page 2 still has two yellow boxes and page 3 three blank topic scores, so it asked first.
    assert dialogs.titles("ask") == ["Export anyway?"]
    assert "5 yellow boxes" in dialogs.shown[0][2]
    rows = read_csv(window.export_path())
    assert [(row["File"], row["Page"]) for row in rows] == [("demo.pdf", "1"), ("demo.pdf", "2"), ("demo.pdf", "3")]
    assert rows[0]["Presentation"] == "8" and rows[0]["Member #2"] == "3" and rows[0]["Comments"] == "Good, clear talk"
    assert rows[1]["Quality"] == "7"
    assert rows[2]["Presentation"] == "4" and rows[2]["Quality"] == ""
    assert not window.unsaved
    assert dialogs.titles("info") == ["Exported"]


def test_export_can_be_cancelled(window, demo_pdf, dialogs):
    scan(window, [demo_pdf])
    dialogs.yes = False
    window.export_csv()
    assert not window.export_path().exists()
    assert window.unsaved


def test_existing_csv_is_only_replaced_when_asked(window, demo_pdf, dialogs):
    scan(window, [demo_pdf])
    path = window.export_path()
    path.parent.mkdir(parents=True)
    path.write_text("old")

    dialogs.replace = "cancel"
    window.export_csv()
    assert path.read_text() == "old"

    dialogs.replace = "replace"
    window.export_csv()
    assert len(read_csv(path)) == 3


def test_failed_export_offers_somewhere_else(window, demo_pdf, dialogs, tmp_path):
    scan(window, [demo_pdf])
    window.export_path().mkdir(parents=True)  # a folder where the file should go, so saving fails
    dialogs.replace = "replace"
    window.export_csv()  # ...and the file chooser is cancelled
    assert dialogs.titles("ask") == ["Export anyway?", "Export failed"]
    assert window.unsaved

    dialogs.shown.clear()
    dialogs.elsewhere = tmp_path / "usb" / "grades.csv"
    window.export_csv()
    assert dialogs.titles("ask") == ["Export anyway?", "Export failed"]
    assert len(read_csv(dialogs.elsewhere)) == 3
    assert window.export_path() == dialogs.elsewhere
    assert not window.unsaved


def test_unreadable_page_stays_marked_until_the_topics_are_typed(window, demo_pdf):
    scan(window, [demo_pdf])
    window.review.page_list.select(2)
    panel = window.review.panel
    card = window.review.page_list.cards[2]
    assert [panel.boxes[key].property("state") for key in ("presentation", "quality", "delivery", "qa")] == ["check"] * 4
    type_into(panel.boxes["member_1"], "4")
    assert card.status()[0] == "Couldn't read this page"
    assert panel.hint.isVisible()
    for key in ("presentation", "quality", "delivery", "qa"):
        type_into(panel.boxes[key], "6")
    assert card.status()[0] == ""
    assert not panel.hint.isVisible()


def test_ctrl_page_down_moves_between_pages(window, demo_pdf):
    scan(window, [demo_pdf])
    box = window.review.panel.boxes["member_2"]
    box.setFocus()
    QTest.keyClick(box, Qt.Key.Key_PageDown, Qt.KeyboardModifier.ControlModifier)
    assert window.review.page_list.current == 1
    assert QApplication.focusWidget() is window.review.panel.boxes["presentation"]
    QTest.keyClick(QApplication.focusWidget(), Qt.Key.Key_PageUp, Qt.KeyboardModifier.ControlModifier)
    assert window.review.page_list.current == 0


def test_choosing_the_export_destination(window, monkeypatch, tmp_path):
    chosen = tmp_path / "elsewhere" / "fod_scores.csv"
    monkeypatch.setattr(rubric_app, "choose_csv_path", lambda parent, start: chosen)
    window.upload.destination_button.click()
    assert window.export_path() == chosen
    assert window.settings.value("export_path") == str(chosen)
    assert "fod_scores.csv" in window.upload.destination.text()
    assert window.upload.destination.toolTip() == str(chosen)


def drop(window, paths, modifiers=Qt.KeyboardModifier.NoModifier):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    actions = Qt.DropAction.CopyAction | Qt.DropAction.MoveAction | Qt.DropAction.LinkAction
    event = QDropEvent(QPointF(20, 20), actions, mime, Qt.MouseButton.LeftButton, modifiers)
    window.upload.dropEvent(event)
    return event


def test_shift_drop_is_a_copy_not_a_move(window, demo_pdf):
    # Accepting a move would tell Windows Explorer to delete the original PDF.
    event = drop(window, [demo_pdf], Qt.KeyboardModifier.ShiftModifier)
    assert event.isAccepted()
    assert event.dropAction() == Qt.DropAction.CopyAction
    scan(window, [])


def test_dropping_files(window, demo_pdf, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("not a rubric")
    drop(window, [notes])
    assert window.thread is None
    assert "Only PDF files" in window.upload.message.text()

    drop(window, [demo_pdf, notes])
    assert window.thread is not None  # started reading the PDF
    assert "Skipped 1 file" in window.upload.message.text()
    scan(window, [])  # wait for it to finish
    assert len(window.pages) == 3


def test_unreadable_files_are_reported(window, tmp_path, dialogs):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a PDF")
    scan(window, [broken])
    assert dialogs.titles("warning") == ["Some files couldn't be read"]
    assert "broken.pdf" in dialogs.shown[0][2]
    assert window.stack.currentWidget() is window.upload
    assert "No pages could be read" in window.upload.message.text()


def test_start_over(window, demo_pdf, dialogs):
    scan(window, [demo_pdf])
    dialogs.yes = False
    window.review.panel.start_over_button.click()
    assert window.stack.currentWidget() is window.review  # said no

    dialogs.yes = True
    window.review.panel.start_over_button.click()
    assert window.stack.currentWidget() is window.upload
    assert window.pages == [] and window.review.page_list.cards == []
    assert not window.review.panel.isEnabled()


def test_closing_with_unexported_scores_asks_first(window, demo_pdf, dialogs):
    scan(window, [demo_pdf])
    dialogs.yes = False
    window.close()
    assert window.isVisible()
    assert dialogs.titles("ask") == ["Quit without exporting?"]
