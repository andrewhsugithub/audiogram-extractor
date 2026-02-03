from dataclasses import dataclass

import cv2
import gradio as gr
import numpy as np
from PIL import Image


@dataclass
class GridResult:
    bbox_debug: np.ndarray
    cropped: np.ndarray
    grid_debug: np.ndarray
    x_pixels: list[int]
    y_pixels: list[int]
    frequencies: list[int]
    db_levels: list[int]


# === Constants ===
DEFAULT_FREQUENCIES = [
    125,
    250,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    4000,
    6000,
    8000,
    12000,
]
DEFAULT_DB_LEVELS = [-10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]


# === Image Processing Functions ===
def detect_grid_bbox(img: np.ndarray) -> tuple[int, int, int, int]:
    """Detect the bounding box of the audiogram grid."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (300, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 300))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

    grid_mesh = cv2.add(h_lines, v_lines)
    contours, _ = cv2.findContours(
        grid_mesh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No grid detected in image")

    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest_contour)


def refine_bbox(
    img: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Refine the grid bounding box using projection analysis."""
    x, y, w, h = bbox
    crop = img[y : y + h, x : x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    row_sums = np.sum(binary, axis=1)
    col_sums = np.sum(binary, axis=0)

    def find_bounds(arr, min_val):
        indices = np.where(arr > min_val)[0]
        if len(indices) == 0:
            return 0, len(arr)
        return int(indices[0]), int(indices[-1])

    y_start, y_end = find_bounds(row_sums, w * 0.2 * 255)
    x_start, x_end = find_bounds(col_sums, h * 0.2 * 255)

    return x + x_start, y + y_start, x_end - x_start, y_end - y_start


def detect_grid_lines(img: np.ndarray) -> tuple[list[int], list[int]]:
    """Detect grid line positions in a cropped audiogram image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    h_img, w_img = img.shape[:2]

    # Connect and extract horizontal lines
    h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    h_open = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, h_close)
    h_lines = cv2.morphologyEx(h_connected, cv2.MORPH_OPEN, h_open)

    # Connect and extract vertical lines
    v_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    v_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, v_close)
    v_lines = cv2.morphologyEx(v_connected, cv2.MORPH_OPEN, v_open)

    # Find line positions via projection
    row_sums = np.sum(h_lines, axis=1)
    col_sums = np.sum(v_lines, axis=0)

    y_indices = np.where(row_sums > (w_img * 0.9 * 255))[0]
    x_indices = np.where(col_sums > (h_img * 0.9 * 255))[0]

    def group_indices(indices, gap=10):
        if len(indices) == 0:
            return []
        groups = []
        curr = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] <= indices[i - 1] + gap:
                curr.append(indices[i])
            else:
                groups.append(int(np.mean(curr)))
                curr = [indices[i]]
        groups.append(int(np.mean(curr)))
        return groups

    return group_indices(x_indices), group_indices(y_indices)


def create_bbox_debug_image(
    img: np.ndarray, initial: tuple, refined: tuple
) -> np.ndarray:
    """Create debug image showing bounding boxes."""
    vis = img.copy()
    x1, y1, w1, h1 = initial
    x2, y2, w2, h2 = refined
    cv2.rectangle(vis, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)  # Green: initial
    cv2.rectangle(vis, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)  # Blue: refined
    return vis


def create_grid_debug_image(
    img: np.ndarray, x_lines: list[int], y_lines: list[int]
) -> np.ndarray:
    """Create debug image showing detected grid lines."""
    vis = img.copy()
    h, w = img.shape[:2]
    for x in x_lines:
        cv2.line(vis, (x, 0), (x, h), (0, 0, 255), 2)  # Red vertical
    for y in y_lines:
        cv2.line(vis, (0, y), (w, y), (0, 255, 0), 2)  # Green horizontal
    return vis


def format_mapping(
    x_pixels: list[int],
    y_pixels: list[int],
    frequencies: list[int],
    db_levels: list[int],
) -> str:
    """Format the pixel-to-value mapping as horizontal tables."""
    lines = []

    # --- X-Axis (Frequency) ---
    # Zip ensures we only display valid pairs if lengths differ
    if x_pixels and frequencies:
        # Create pairs and unzip them to separate lists again
        # This guarantees alignment and handles length mismatches (truncates to shortest)
        freq_vals, px_vals = zip(*zip(frequencies, x_pixels))

        lines.append("## X-axis Mapping (Frequency → Pixel)")
        # Header: Frequency values
        lines.append("| Frequency (Hz) | " + " | ".join(map(str, freq_vals)) + " |")
        # Separator
        lines.append("| :--- | " + " | ".join(["---"] * len(freq_vals)) + " |")
        # Data: Pixel values
        lines.append("| **Pixel X** | " + " | ".join(map(str, px_vals)) + " |")
    else:
        lines.append("## X-axis Mapping\nNo data detected.")

    lines.append("\n")  # Spacer

    # --- Y-Axis (dB HL) ---
    if y_pixels and db_levels:
        db_vals, py_vals = zip(*zip(db_levels, y_pixels))

        lines.append("## Y-axis Mapping (dB HL → Pixel)")
        # Header: dB values
        lines.append("| dB HL | " + " | ".join(map(str, db_vals)) + " |")
        # Separator
        lines.append("| :--- | " + " | ".join(["---"] * len(db_vals)) + " |")
        # Data: Pixel values
        lines.append("| **Pixel Y** | " + " | ".join(map(str, py_vals)) + " |")
    else:
        lines.append("## Y-axis Mapping\nNo data detected.")

    return "\n".join(lines)


def process_audiogram(
    input_image: Image.Image,
) -> tuple[Image.Image, Image.Image, Image.Image, str]:
    """Main processing function for Gradio interface."""
    # Convert PIL to OpenCV format
    img = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    # Detect and refine bounding box
    initial_bbox = detect_grid_bbox(img)
    refined_bbox = refine_bbox(img, initial_bbox)

    # Create bbox debug image
    bbox_debug = create_bbox_debug_image(img, initial_bbox, refined_bbox)
    bbox_debug_rgb = cv2.cvtColor(bbox_debug, cv2.COLOR_BGR2RGB)

    # Crop to refined bbox
    x, y, w, h = refined_bbox
    cropped = img[y : y + h, x : x + w]
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    # Detect grid lines
    x_lines, y_lines = detect_grid_lines(cropped)

    # Remove border lines
    x_lines_inner = x_lines[1:-1] if len(x_lines) > 2 else x_lines
    y_lines_inner = y_lines[1:-1] if len(y_lines) > 2 else y_lines

    # Create grid debug image
    grid_debug = create_grid_debug_image(cropped, x_lines_inner, y_lines_inner)
    grid_debug_rgb = cv2.cvtColor(grid_debug, cv2.COLOR_BGR2RGB)

    # Create mapping
    frequencies = DEFAULT_FREQUENCIES[: len(x_lines_inner)]
    db_levels = DEFAULT_DB_LEVELS[: len(y_lines_inner)]
    mapping_text = format_mapping(x_lines_inner, y_lines_inner, frequencies, db_levels)

    return (
        Image.fromarray(bbox_debug_rgb),
        Image.fromarray(cropped_rgb),
        Image.fromarray(grid_debug_rgb),
        mapping_text,
    )


# === Gradio Interface ===
def create_interface() -> gr.Blocks:
    with gr.Blocks(title="Audiogram Extractor", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎧 Audiogram Grid Extractor
            
            Upload an audiogram image to detect the grid and extract pixel-to-value mappings.
            
            **Instructions:**
            1. Upload an audiogram image
            2. Click "Extract Grid"
            3. View the debug images and mappings
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Input Audiogram")
                extract_btn = gr.Button("🔍 Extract Grid", variant="primary")

        with gr.Row():
            with gr.Column(scale=1):
                bbox_output = gr.Image(
                    label="Bounding Box Detection (Green=Initial, Blue=Refined)"
                )
            with gr.Column(scale=1):
                cropped_output = gr.Image(label="Cropped Audiogram")
            with gr.Column(scale=1):
                grid_output = gr.Image(
                    label="Grid Lines (Red=Vertical, Green=Horizontal)"
                )

        with gr.Row():
            mapping_output = gr.Markdown(label="Pixel Mappings")

        extract_btn.click(
            fn=process_audiogram,
            inputs=[input_image],
            outputs=[bbox_output, cropped_output, grid_output, mapping_output],
        )

        gr.Markdown(
            """
            ---
            **Legend:**
            - 🟢 Green box: Initial detection
            - 🔵 Blue box: Refined detection  
            - 🔴 Red lines: Vertical grid lines (frequencies)
            - 🟢 Green lines: Horizontal grid lines (dB levels)
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
