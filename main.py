import cv2
import numpy as np


def detect_and_draw_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    # Connect dashed lines
    kernel = np.ones((3, 3), np.uint8)
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Isolate H/V lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(connected, cv2.MORPH_OPEN, h_kernel)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_lines = cv2.morphologyEx(connected, cv2.MORPH_OPEN, v_kernel)

    # Combine
    grid_mesh = cv2.add(h_lines, v_lines)

    # Find contours - FIX: Using findContours
    contours, _ = cv2.findContours(
        grid_mesh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Draw Rectangle (Green)
    # img_with_rect = img.copy()
    # cv2.rectangle(img_with_rect, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Crop
    # cropped_audiogram = img[y : y + h, x : x + w]

    # cv2.imwrite("image_with_rect_org.png", img_with_rect)
    # cv2.imwrite("cropped_audiogram_org.png", cropped_audiogram)

    return x, y, w, h


def refine_crop(img, initial_bbox):
    x, y, w, h = initial_bbox
    crop = img[y : y + h, x : x + w]

    # 1. Get horizontal and vertical projections
    # We sum up the pixel values. Rows with grid lines will have many black pixels (low sum).
    # Note: If your image is white-on-black, invert this logic.
    # Assuming standard black-ink-on-white-paper:
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Threshold to make ink purely black (0) and paper purely white (255)
    _, binary = cv2.threshold(gray_crop, 200, 255, cv2.THRESH_BINARY_INV)

    # 2. Find exact boundaries
    # Sum along rows (axis 1) -> locates horizontal lines
    row_sums = np.sum(binary, axis=1)
    # Sum along cols (axis 0) -> locates vertical lines
    col_sums = np.sum(binary, axis=0)

    # Find the first and last index where the sum is significant (contains a line)
    # We look for where the line mass exceeds a threshold (e.g., > 50% of width/height)

    def find_bounds(arr, min_val):
        indices = np.where(arr > min_val)[0]
        if len(indices) == 0:
            return 0, len(arr)
        return indices[0], indices[-1]

    # Threshold: Line must be at least 20% of the dimension length to count as a grid line
    y_start, y_end = find_bounds(row_sums, w * 0.2 * 255)
    x_start, x_end = find_bounds(col_sums, h * 0.2 * 255)

    # 3. Adjust the original coordinates
    new_x = x + x_start
    new_y = y + y_start
    new_w = x_end - x_start
    new_h = y_end - y_start

    # Draw Rectangle (Blue)
    img_with_rect = img.copy()
    cv2.rectangle(
        img_with_rect, (x, y), (x + w, y + h), (0, 255, 0), 2
    )  # Original in Green
    cv2.rectangle(
        img_with_rect, (new_x, new_y), (new_x + new_w, new_y + new_h), (255, 0, 0), 2
    )
    # Crop
    cropped_audiogram = img[new_y : new_y + new_h, new_x : new_x + new_w]

    cv2.imwrite("image_with_rect_v2.png", img_with_rect)
    cv2.imwrite("cropped_audiogram_v2.png", cropped_audiogram)

    return new_x, new_y, new_w, new_h


def mark_lines(image_path, output_path):
    # 1. Load and Preprocess
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to get binary image (lines = white, background = black)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    h_img, w_img = img.shape[:2]

    # --- STRATEGY: CONNECT DASHES FIRST, THEN REMOVE SYMBOLS ---

    # 2. Extract Horizontal Lines
    # Step A: Close gaps (Connect dashes)
    # Use a kernel slightly larger than the gap between dashes (e.g., 20px)
    h_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    h_connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, h_close_kernel)

    # Step B: Open (Remove symbols/noise)
    # Now that the grid line is solid/long, we can safely use a large kernel (40px)
    # to delete any text or symbols that didn't merge into a long line.
    h_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(h_connected, cv2.MORPH_OPEN, h_open_kernel)

    # 3. Extract Vertical Lines
    # Step A: Close gaps (Connect dashes)
    v_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    v_connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, v_close_kernel)

    # Step B: Open (Remove symbols/noise)
    v_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(v_connected, cv2.MORPH_OPEN, v_open_kernel)

    # 4. Detect Coordinates (Projection Method)

    # Y-Coordinates (Horizontal)
    row_sums = np.sum(horizontal_lines, axis=1)
    # Threshold: Line must be substantial (e.g., > 40% of image width)
    y_indices = np.where(row_sums > (w_img * 0.9 * 255))[0]

    # X-Coordinates (Vertical)
    col_sums = np.sum(vertical_lines, axis=0)
    # Threshold: Line must be substantial (e.g., > 40% of image height)
    x_indices = np.where(col_sums > (h_img * 0.9 * 255))[0]

    # 5. Group nearby lines (Deduplicate)
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

    final_x = group_indices(x_indices)
    final_y = group_indices(y_indices)

    # 6. Visualization
    vis_img = img.copy()

    # Draw straight lines based on detected coordinates
    for x in final_x:
        cv2.line(vis_img, (x, 0), (x, h_img), (0, 0, 255), 2)  # Red Vertical
    for y in final_y:
        cv2.line(vis_img, (0, y), (w_img, y), (0, 255, 0), 2)  # Green Horizontal

    # Save results
    cv2.imwrite(output_path, vis_img)

    # Optional: Save intermediate steps to debug
    cv2.imwrite("debug_h_connected.png", h_connected)  # See dashes connected
    cv2.imwrite("debug_h_clean.png", horizontal_lines)  # See symbols removed

    print(f"Detected {len(final_x)} Vertical Lines at: {final_x}")
    print(f"Detected {len(final_y)} Horizontal Lines at: {final_y}")
    return final_x, final_y


def main():
    IMAGE = "assets/2024/test.jpg"
    img = cv2.imread(IMAGE)
    x, y, w, h = detect_and_draw_grid(img)
    final_x, final_y, final_w, final_h = refine_crop(img, (x, y, w, h))
    print(f"Rectangle: x={final_x}, y={final_y}, w={final_w}, h={final_h}")
    # Run the function
    final_x, final_y = mark_lines("cropped_audiogram_v2.png", "grid_verification.png")
    final_x = final_x[1:-1]  # Remove first and last (borders)
    final_y = final_y[1:-1]  # Remove first and last (borders)

    x = [125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000]
    y = [-10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    print("\nX-axis mapping (pixels → frequency):")
    for px, freq in zip(final_x, x):
        print(f"  {px:4d} px → {freq:5d} Hz")

    print("\nY-axis mapping (pixels → dB HL):")
    for py, db in zip(final_y, y):
        print(f"  {py:4d} px → {db:4d} dB HL")


if __name__ == "__main__":
    main()
