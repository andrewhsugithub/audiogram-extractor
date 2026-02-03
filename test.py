import cv2 as cv
import numpy as np
import pytesseract  # pip install pytesseract (and install Tesseract on your system)


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Return a binarized, inverted image where lines/text are white."""
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    # Adaptive thresholding is a good default for non-uniform lighting.
    bw = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 31, 10
    )  # OpenCV thresholding docs: :contentReference[oaicite:1]{index=1}

    bw = (
        255 - bw
    )  # invert so foreground is white (helps contour logic) :contentReference[oaicite:2]{index=2}
    return bw


def extract_lines(bw_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract horizontal/vertical line masks and a combined table mask."""
    h, w = bw_inv.shape[:2]

    # Directional kernels scaled to image size (tune divisor per your data)
    h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (max(10, w // 30), 1))
    v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, max(10, h // 30)))

    # Erode->Dilate isolates lines aligned to the kernel direction :contentReference[oaicite:3]{index=3}
    horiz = cv.dilate(cv.erode(bw_inv, h_kernel, iterations=1), h_kernel, iterations=1)
    vert = cv.dilate(cv.erode(bw_inv, v_kernel, iterations=1), v_kernel, iterations=1)

    table_mask = cv.add(horiz, vert)
    return horiz, vert, table_mask


def find_table_bbox(table_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the main table bounding box (largest external contour)."""
    contours, _ = cv.findContours(
        table_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )  # contour basics :contentReference[oaicite:4]{index=4}

    if not contours:
        return None

    cnt = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(cnt)
    return x, y, w, h


def find_cell_boxes(table_mask_roi: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find rectangular 'cell-like' boxes inside the table ROI."""
    # A small close can help connect broken grid segments :contentReference[oaicite:5]{index=5}
    k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    cleaned = cv.morphologyEx(table_mask_roi, cv.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv.findContours(cleaned, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv.boundingRect(c)

        # Basic filters -- tune for your docs
        if w < 15 or h < 15:
            continue
        if w * h < 300:
            continue

        boxes.append((x, y, w, h))

    # Remove likely duplicates by area/position heuristics (simple pass)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes


def group_boxes_into_rows(boxes: list[tuple[int, int, int, int]], y_tol: int = 10):
    """Group boxes into rows by approximate y alignment."""
    rows = []
    current = []

    for b in boxes:
        if not current:
            current = [b]
            continue

        # Compare y of current box to y of the row's first box
        if abs(b[1] - current[0][1]) <= y_tol:
            current.append(b)
        else:
            rows.append(sorted(current, key=lambda x: x[0]))
            current = [b]

    if current:
        rows.append(sorted(current, key=lambda x: x[0]))

    return rows


def ocr_cell(img_roi: np.ndarray) -> str:
    """OCR a single cell crop."""
    # psm 6 is a common choice for block-like text; pytesseract supports config strings :contentReference[oaicite:6]{index=6}
    return pytesseract.image_to_string(img_roi, config="--oem 3 --psm 6").strip()


def detect_table_and_read_cells(img_bgr: np.ndarray):
    bw = preprocess(img_bgr)
    _, _, table_mask = extract_lines(bw)

    bbox = find_table_bbox(table_mask)
    if bbox is None:
        raise RuntimeError(
            "No table-like region found. Try a DL approach or adjust preprocessing."
        )

    x, y, w, h = bbox
    table_roi = img_bgr[y : y + h, x : x + w]
    mask_roi = table_mask[y : y + h, x : x + w]

    cell_boxes = find_cell_boxes(mask_roi)
    rows = group_boxes_into_rows(cell_boxes, y_tol=12)

    data = []
    for row in rows:
        row_text = []
        for cx, cy, cw, ch in row:
            pad = 2
            crop = table_roi[
                max(0, cy - pad) : min(table_roi.shape[0], cy + ch + pad),
                max(0, cx - pad) : min(table_roi.shape[1], cx + cw + pad),
            ]
            row_text.append(ocr_cell(crop))
        data.append(row_text)

    return data, (x, y, w, h)
