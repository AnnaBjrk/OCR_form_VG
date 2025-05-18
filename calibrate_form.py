import cv2
import numpy as np
import os
from pdf2image import convert_from_path
from form_template import FormTemplate, FieldType


def load_original_form(pdf_path):
    """Load the original form and convert to images."""
    print(f"Loading original form from {pdf_path}...")
    images = convert_from_path(pdf_path, dpi=300)
    return images


def detect_form_corners(image):
    """Detect the corners of the form using edge detection."""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (should be the form)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) == 4:  # We found 4 corners
            return approx.reshape(4, 2)

    return None


def calibrate_form(image, corners):
    """Calibrate the form by finding rotation and scale."""
    if corners is None:
        return None

    # Sort corners: top-left, top-right, bottom-right, bottom-left
    corners = corners[corners[:, 0].argsort()]
    left = corners[:2]
    right = corners[2:]
    top_left = left[left[:, 1].argmin()]
    bottom_left = left[left[:, 1].argmax()]
    top_right = right[right[:, 1].argmin()]
    bottom_right = right[right[:, 1].argmax()]

    # Calculate rotation angle
    angle = np.arctan2(bottom_right[1] - bottom_left[1],
                       bottom_right[0] - bottom_left[0]) * 180 / np.pi

    # Calculate scale
    width = np.linalg.norm(top_right - top_left)
    height = np.linalg.norm(bottom_left - top_left)

    return {
        'angle': angle,
        'width': width,
        'height': height,
        'corners': [top_left, top_right, bottom_right, bottom_left]
    }


def visualize_calibration(image, calibration_data):
    """Create visualization of the calibration."""
    if calibration_data is None:
        return None

    # Convert PIL image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Draw corners
    for corner in calibration_data['corners']:
        cv2.circle(cv_image, tuple(map(int, corner)), 10, (0, 0, 255), -1)

    # Draw lines between corners
    corners = calibration_data['corners']
    for i in range(4):
        pt1 = tuple(map(int, corners[i]))
        pt2 = tuple(map(int, corners[(i + 1) % 4]))
        cv2.line(cv_image, pt1, pt2, (0, 255, 0), 2)

    # Add calibration info
    info = f"Angle: {calibration_data['angle']:.2f}°"
    cv2.putText(cv_image, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw grid
    width = int(calibration_data['width'])
    height = int(calibration_data['height'])
    grid_size = 100

    for x in range(0, width, grid_size):
        cv2.line(cv_image, (x, 0), (x, height), (255, 0, 0), 1)
        cv2.putText(cv_image, str(x), (x, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    for y in range(0, height, grid_size):
        cv2.line(cv_image, (0, y), (width, y), (255, 0, 0), 1)
        cv2.putText(cv_image, str(y), (5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return cv_image


def main():
    # Create debug directory
    debug_dir = "debug_rotated_original"
    os.makedirs(debug_dir, exist_ok=True)

    # Load original form
    original_pdf = "files_to_process/Af_aktivitetsrapport_orginal.pdf"
    images = load_original_form(original_pdf)

    # Process each page
    for i, image in enumerate(images, 1):
        print(f"\nProcessing page {i}...")

        # Detect corners
        corners = detect_form_corners(image)
        if corners is None:
            print(f"Could not detect form corners on page {i}")
            continue

        # Calibrate form
        calibration = calibrate_form(image, corners)
        if calibration is None:
            print(f"Could not calibrate form on page {i}")
            continue

        # Create visualization
        vis_image = visualize_calibration(image, calibration)
        if vis_image is not None:
            output_path = f"{debug_dir}/page_{i}_calibration.png"
            cv2.imwrite(output_path, vis_image)
            print(f"Saved calibration visualization to {output_path}")

            # Save the detected corners
            corners_path = f"{debug_dir}/page_{i}_corners.txt"
            with open(corners_path, 'w') as f:
                f.write(f"Angle: {calibration['angle']:.2f}°\n")
                f.write(f"Width: {calibration['width']:.2f}\n")
                f.write(f"Height: {calibration['height']:.2f}\n")
                f.write("\nCorners (x,y):\n")
                for corner in calibration['corners']:
                    f.write(f"({corner[0]:.2f}, {corner[1]:.2f})\n")
            print(f"Saved corner coordinates to {corners_path}")


if __name__ == "__main__":
    main()
