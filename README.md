# p3-FoD-rubric-grader
Our project #3 for Software Development, creating a program to grade FoD student rubrics.

## Setup

```
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Running the app

```
.venv\Scripts\python app.py
```

1. **Upload screen:** drag scanned rubric PDFs onto the orange panel, or click
   **Upload PDF(s)**. Every page is straightened and its bubbles are read.
   **Edit Export Destination** picks where the CSV is saved (it's remembered).
2. **Review screen:** the pages are listed on the left. Click one to see its scores
   on the right, and type over any that are wrong. Yellow boxes are ones the reader
   wasn't sure about (hover for why); press Enter to keep a yellow box's value.
   Ctrl+PageDown / Ctrl+PageUp move to the next / previous page. If a page couldn't
   be lined up with the rubric, its topic boxes stay yellow until you type the
   scores in from the paper.
3. **Export to CSV** writes one row per page: file, page, the four topic scores,
   each member's participation score and the comments.

The "Underachieved (score = 0-4)" column only has four bubbles, which are read as
1-4. A score of 0 has to be typed in on the review screen.

## How it works

- `rasterizer.py` renders PDF pages to OpenCV images.
- `aligner.py` finds the border of the scoring table, uses its four corners to
  undo skew and perspective, and picks the right way up by comparing with the
  blank rubric in `templates/FoD_rubric_blank.pdf`.
- `grader.py` knows where every bubble is on the blank rubric, measures how dark
  each one is on the straightened page, and notes anything doubtful (nothing
  filled in, a faint mark, two bubbles filled in).
- `pipeline.py` ties those together for whole PDFs and writes the CSV.
- `app.py` is the window: the upload and review screens.

A point at `(x, y)` PDF points on the blank rubric is at `(x, y) * dpi / 72` pixels
in an aligned image (default `dpi` is 150).

### Straightening scans from the command line

```
.venv\Scripts\python aligner.py scans.pdf -o aligned --debug
```

Each page of `scans.pdf` is written to `aligned/` as a PNG that lines up with the
blank rubric. `--debug` also saves the original scan with the detected table corners
drawn on it.

## Tests

```
.venv\Scripts\python -m pytest tests
```

`tests/fixtures/real_photo.jpg` is a phone photo of a real filled-in rubric; the
other tests use fake scans made from the blank rubric (see `tests/fakes.py`).
