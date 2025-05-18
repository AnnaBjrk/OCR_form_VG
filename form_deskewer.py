import cv2
import numpy as np
import os
import pytesseract
from typing import Tuple, Optional


class FormDeskewer:
    def __init__(self):
        pass

    def deskew_form(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """Deskew a form document using form elements like tables and borders."""
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Method 1: Table-based deskewing (good for forms with tables)
        angle_table = self._detect_table_orientation(gray)

        # Method 2: Form border-based deskewing
        angle_border = self._detect_border_orientation(gray)

        # Method 3: Text block orientation
        angle_text = self._detect_text_block_orientation(gray)

        # Choose the most reliable angle (could be weighted average or most confident)
        angles = []
        if abs(angle_table) < 45:  # Only use if reasonable
            angles.append(angle_table)
        if abs(angle_border) < 45:
            angles.append(angle_border)
        if abs(angle_text) < 45:
            angles.append(angle_text)

        if not angles:
            print("Could not reliably detect orientation. Using original image.")
            final_angle = 0
        else:
            # Use median to avoid outliers
            final_angle = np.median(angles)

        print(
            f"Detected angles - Table: {angle_table:.2f}°, Border: {angle_border:.2f}°, Text: {angle_text:.2f}°")
        print(f"Final correction angle: {final_angle:.2f}°")

        # Apply rotation if needed
        if abs(final_angle) > 0.5:
            deskewed = self._rotate_image(
                image, -final_angle)  # Note the negative sign
        else:
            deskewed = image.copy()

        # Save if output path provided
        if output_path:
            directory = os.path.dirname(output_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            cv2.imwrite(output_path, deskewed)
            print(f"Saved deskewed image to {output_path}")

        return deskewed

    def _detect_table_orientation(self, gray: np.ndarray) -> float:
        """Detect orientation using table lines."""
        # Binary thresholding
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Dilate to connect table lines
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # Find horizontal lines (table rows)
        h_kernel = np.ones((1, 40), np.uint8)
        h_lines = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, h_kernel)

        # Find vertical lines (table columns)
        v_kernel = np.ones((40, 1), np.uint8)
        v_lines = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, v_kernel)

        # Combine horizontal and vertical lines
        table_lines = cv2.bitwise_or(h_lines, v_lines)

        # Find contours of the table structure
        contours, _ = cv2.findContours(
            table_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # Find the largest contour (likely the main form)
        max_contour = max(contours, key=cv2.contourArea)

        # Fit a rotated rectangle to the contour
        rect = cv2.minAreaRect(max_contour)
        angle = rect[2]

        # Normalize the angle
        if angle < -45:
            angle = 90 + angle

        return angle

    def _detect_border_orientation(self, gray: np.ndarray) -> float:
        """Detect orientation using form borders."""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Standard Hough Transform to find major lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)

        if lines is None or len(lines) == 0:
            return 0.0

        # Separate horizontal and vertical lines
        h_angles = []
        v_angles = []

        for line in lines:
            rho, theta = line[0]
            # Near horizontal (within 10 degrees of 0 or 180 degrees)
            if theta < 0.175 or theta > 2.967:  # ~10 degrees from 0 or 180
                angle_rad = theta if theta < 0.175 else theta - np.pi
                h_angles.append(np.degrees(angle_rad))
            # Near vertical (within 10 degrees of 90 or 270 degrees)
            elif 1.396 < theta < 1.745 or 4.538 < theta < 4.887:  # ~10 degrees from 90 or 270
                angle_rad = theta - np.pi/2 if theta < 1.745 else theta - 3*np.pi/2
                v_angles.append(np.degrees(angle_rad))

        # Use the group with more lines for better reliability
        if len(h_angles) > len(v_angles) and h_angles:
            return np.median(h_angles)
        elif v_angles:
            return np.median(v_angles)
        else:
            return 0.0

    def _detect_text_block_orientation(self, gray: np.ndarray) -> float:
        """Detect orientation using text blocks."""
        # Use adaptiveThreshold for better text isolation
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Close small gaps in text to form blocks
        # Horizontal-dominant kernel to connect text in lines
        kernel = np.ones((5, 20), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours of text blocks
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # Filter to get only reasonably sized text blocks
        min_area = 500
        text_blocks = [
            cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if not text_blocks:
            return 0.0

        # Get angles of each text block
        angles = []
        for block in text_blocks:
            rect = cv2.minAreaRect(block)
            # Get width and height to determine if we should use the angle directly or add 90 degrees
            width, height = rect[1]
            angle = rect[2]

            # Adjust angle based on rectangle orientation
            if width < height:
                angle = 90 + angle

            # Normalize to -45 to 45 degrees
            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90

            angles.append(angle)

        # Use median to reduce impact of outliers
        return np.median(angles) if angles else 0.0

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate the image by the given angle in degrees."""
        # Get image dimensions
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new image dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Perform the rotation
        rotated = cv2.warpAffine(image, M, (new_w, new_h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def process_directory(self, input_dir: str, output_dir: Optional[str] = None) -> None:
        """Process all images in a directory."""
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory {input_dir} does not exist")

        if output_dir is None:
            output_dir = input_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Supported image formats
        img_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf')

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(img_extensions):
                input_path = os.path.join(input_dir, filename)

                # Create output filename
                base, ext = os.path.splitext(filename)
                output_filename = f"{base}_deskewed{ext}"
                output_path = os.path.join(output_dir, output_filename)

                try:
                    self.deskew_form(input_path, output_path)
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")


# Example usage
if __name__ == "__main__":
    deskewer = FormDeskewer()
    deskewer.deskew_form("path/to/form.jpg", "path/to/deskewed_form.jpg")
