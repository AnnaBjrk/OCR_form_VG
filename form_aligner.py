import cv2
import numpy as np
import os
from typing import Tuple, Optional


class FormAligner:
    def __init__(self):
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()

        # Initialize FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Minimum number of good matches required
        self.min_matches = 10

    def _ensure_color_image(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is in color format (3 channels)."""
        if len(image.shape) == 2:  # Grayscale
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:  # Already color
            return image
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            raise ValueError(
                f"Unexpected image format with shape {image.shape}")

    def align_form(self,
                   scanned_image: np.ndarray,
                   template_image: np.ndarray,
                   output_path: Optional[str] = None) -> Tuple[np.ndarray, bool]:
        """
        Align a scanned form image to match the template using feature matching.

        Args:
            scanned_image: The scanned form image to align
            template_image: The template image to align against
            output_path: Optional path to save the aligned image

        Returns:
            Tuple of (aligned_image, success)
        """
        try:
            # Ensure both images are in color format
            scanned_color = self._ensure_color_image(scanned_image)
            template_color = self._ensure_color_image(template_image)

            # Convert to grayscale for feature detection
            scanned_gray = cv2.cvtColor(scanned_color, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)

            # Detect keypoints and compute descriptors
            kp1, des1 = self.sift.detectAndCompute(scanned_gray, None)
            kp2, des2 = self.sift.detectAndCompute(template_gray, None)

            if des1 is None or des2 is None or len(kp1) < self.min_matches or len(kp2) < self.min_matches:
                print(
                    f"Not enough keypoints found. Scanned: {len(kp1)}, Template: {len(kp2)}")
                return scanned_image, False

            # Match descriptors
            matches = self.matcher.knnMatch(des1, des2, k=2)

            # Apply ratio test to find good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < self.min_matches:
                print(f"Not enough good matches found: {len(good_matches)}")
                return scanned_image, False

            # Get coordinates of good matches
            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Apply homography to align the image
            h, w = template_gray.shape
            aligned = cv2.warpPerspective(scanned_color, H, (w, h))

            # Save aligned image if output path is provided
            if output_path:
                cv2.imwrite(output_path, aligned)

            return aligned, True

        except Exception as e:
            print(f"Error aligning image: {str(e)}")
            return scanned_image, False

    def process_directory(self,
                          input_dir: str,
                          template_image: np.ndarray,
                          output_dir: str) -> None:
        """
        Process all images in a directory, aligning them to the template.

        Args:
            input_dir: Directory containing images to process
            template_image: Template image to align against
            output_dir: Directory to save aligned images
        """
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"aligned_{filename}")

                # Read image
                image = cv2.imread(input_path)
                if image is None:
                    print(f"Could not read image: {input_path}")
                    continue

                # Align image
                aligned, success = self.align_form(
                    image, template_image, output_path)

                if success:
                    print(f"Successfully aligned {filename}")
                else:
                    print(f"Failed to align {filename}")


# Example usage
if __name__ == "__main__":
    # Example of aligning a scanned form with a template
    aligner = FormAligner()

    # Load images
    scanned = cv2.imread("scanned_form.jpg")
    template = cv2.imread("template.jpg")

    if scanned is not None and template is not None:
        # Align the form
        aligned, success = aligner.align_form(
            scanned, template, "aligned_form.jpg")

        if success:
            print("Form aligned successfully")
        else:
            print("Failed to align form")
