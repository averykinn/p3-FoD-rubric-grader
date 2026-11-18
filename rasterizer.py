#Turn PDF pages into OpenCV images.

from pathlib import Path

import cv2
import numpy as np
import pymupdf

DEFAULT_DPI = 150


def rasterize_page(page: pymupdf.Page, dpi: int = DEFAULT_DPI) -> np.ndarray:
    #Render one PDF page to a BGR image (the format OpenCV expects).
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def rasterize_pdf(pdf_path: str | Path, dpi: int = DEFAULT_DPI) -> list[np.ndarray]:
    #Render every page of a PDF to a BGR image.
    with pymupdf.open(pdf_path) as doc:
        return [rasterize_page(page, dpi) for page in doc]
