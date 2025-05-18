import cv2
import numpy as np
import os
from typing import Tuple, Optional, Dict, Any


class FormAligner:
    def __init__(self):
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        # FLANN matcher parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def align_form(self, scanned_image: np.ndarray, template_image: np.ndarray, output_path: Optional[str] = None) -> np.ndarray:
        """
        Align a scanned form image to match the template using feature matching.

        Args:
            scanned_image: The scanned form image to align
            template_image: The template form image to match against
            output_path: Optional path to save the aligned image

        Returns:
            The aligned form image
        """
        # Convert images to grayscale
        scanned_gray = cv2.cvtColor(scanned_image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        kp1, des1 = self.sift.detectAndCompute(scanned_gray, None)
        kp2, des2 = self.sift.detectAndCompute(template_gray, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            print("Not enough keypoints found for matching")
            return scanned_image

        # Match descriptors
        matches = self.matcher.knnMatch(des1, des2, k=2)

        # Apply ratio test to get good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            print("Not enough good matches found")
            return scanned_image

        # Get coordinates of good matches
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Get image dimensions
        h, w = template_gray.shape

        # Apply transformation
        aligned = cv2.warpPerspective(scanned_image, H, (w, h))

        # Save if output path provided
        if output_path:
            directory = os.path.dirname(output_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            cv2.imwrite(output_path, aligned)
            print(f"Saved aligned image to {output_path}")

        return aligned

    def process_directory(self, input_dir: str, template_path: str, output_dir: Optional[str] = None) -> None:
        """
        Process all images in a directory, aligning them to the template.

        Args:
            input_dir: Directory containing scanned form images
            template_path: Path to the template form image
            output_dir: Optional directory to save aligned images
        """
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory {input_dir} does not exist")

        if output_dir is None:
            output_dir = input_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load template image
        template = cv2.imread(template_path)
        if template is None:
            raise ValueError(
                f"Could not read template image at {template_path}")

        # Supported image formats
        img_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(img_extensions):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"aligned_{filename}")

                try:
                    # Read input image
                    scanned = cv2.imread(input_path)
                    if scanned is None:
                        print(f"Could not read image: {input_path}")
                        continue

                    # Align image
                    aligned = self.align_form(scanned, template, output_path)
                    print(f"Processed {filename}")

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")


# Example usage
if __name__ == "__main__":
    aligner = FormAligner()
    aligner.align_form("path/to/scanned_form.jpg",
                       "path/to/template.jpg", "path/to/aligned_form.jpg")
