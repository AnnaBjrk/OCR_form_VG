import cv2
import numpy as np
from PIL import Image
import easyocr
from skimage import exposure
from skimage.filters import threshold_local
import os


class DocumentProcessor:
    def __init__(self, template_dir=None):
        """Initialize the DocumentProcessor with EasyOCR reader and template information."""
        self.reader = easyocr.Reader(['sv'])
        self.template_dir = template_dir
        self.page1_markers = None
        self.page2_markers = None
        self.form_dimensions = None
        self.load_templates()

    def load_templates(self):
        """Load template information for page identification and form markers."""
        if self.template_dir and os.path.exists(self.template_dir):
            # Load template images and markers
            self.page1_template = cv2.imread(os.path.join(
                self.template_dir, 'page1_template.png'))
            self.page2_template = cv2.imread(os.path.join(
                self.template_dir, 'page2_template.png'))

            # Define known form markers (coordinates of key elements)
            # These should be adjusted based on your specific form
            self.page1_markers = {
                'header': [(100, 50), (400, 50)],  # Example coordinates
                'logo': [(50, 50), (150, 150)],    # Example coordinates
                'footer': [(100, 1000), (400, 1000)]  # Example coordinates
            }

            self.page2_markers = {
                'header': [(100, 50), (400, 50)],  # Example coordinates
                'footer': [(100, 1000), (400, 1000)]  # Example coordinates
            }

            # Define standard form dimensions
            self.form_dimensions = {
                'width': 2100,  # A4 width in pixels at 300 DPI
                'height': 2970  # A4 height in pixels at 300 DPI
            }

    def load_image(self, image_path):
        """Load an image from file path."""
        return cv2.imread(image_path)

    def reduce_noise(self, image):
        """Remove noise and spots from the image."""
        # Apply Gaussian blur to reduce noise
        denoised = cv2.GaussianBlur(image, (5, 5), 0)
        # Apply bilateral filter to preserve edges while removing noise
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        return denoised

    def correct_skew(self, image):
        """Correct skewed or rotated documents."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply threshold to get binary image
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Find all contours
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # Adjust angle
        if angle < -45:
            angle = 90 + angle

        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def enhance_contrast(self, image):
        """Enhance contrast between text and background."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge channels and convert back to BGR
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        return enhanced

    def normalize_size(self, image, target_width=1000):
        """Scale image to a consistent size."""
        # Calculate aspect ratio
        aspect_ratio = image.shape[1] / image.shape[0]
        target_height = int(target_width / aspect_ratio)

        # Resize image
        resized = cv2.resize(image, (target_width, target_height),
                             interpolation=cv2.INTER_AREA)
        return resized

    def correct_lighting(self, image):
        """Correct uneven lighting or shadows."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge channels and convert back to BGR
        merged = cv2.merge((cl, a, b))
        corrected = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        return corrected

    def sharpen_edges(self, image):
        """Improve text edges for better detection."""
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])

        # Apply sharpening
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def convert_to_grayscale(self, image):
        """Convert image to grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def identify_page(self, image):
        """Identify if the image is page 1 or 2 based on specific markers."""
        # Perform OCR on the image
        results = self.reader.readtext(image)

        # Extract all text from results
        detected_texts = [text for _, text, _ in results]

        # Check for page 1 markers
        if (self.page1_markers['title'] in detected_texts or
                self.page1_markers['page_number'] in detected_texts):
            return 1

        # Check for page 2 markers
        if (self.page2_markers['title'] in detected_texts or
                self.page2_markers['page_number'] in detected_texts):
            return 2

        # If no clear markers found, try to find the most similar match
        page1_score = sum(1 for text in detected_texts if any(
            marker in text for marker in self.page1_markers.values()))
        page2_score = sum(1 for text in detected_texts if any(
            marker in text for marker in self.page2_markers.values()))

        return 1 if page1_score > page2_score else 2

    def detect_form_corners(self, image):
        """Detect the corners of the form using known markers."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold to get binary image
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (should be the form)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            return box

        return None

    def correct_skew_with_markers(self, image):
        """Correct skew using form markers and known corners."""
        corners = self.detect_form_corners(image)
        if corners is None:
            # Fallback to generic skew correction
            return self.correct_skew(image)

        # Get the minimum area rectangle
        rect = cv2.minAreaRect(corners)
        angle = rect[-1]

        # Adjust angle
        if angle < -45:
            angle = 90 + angle

        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def normalize_form_size(self, image):
        """Scale image to match the standard form dimensions."""
        if self.form_dimensions is None:
            # Fallback to generic normalization
            return self.normalize_size(image)

        # Get current dimensions
        current_height, current_width = image.shape[:2]

        # Calculate scaling factors
        width_scale = self.form_dimensions['width'] / current_width
        height_scale = self.form_dimensions['height'] / current_height

        # Use the smaller scale to maintain aspect ratio
        scale = min(width_scale, height_scale)

        # Calculate new dimensions
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)

        # Resize image
        resized = cv2.resize(image, (new_width, new_height),
                             interpolation=cv2.INTER_AREA)

        return resized

    def process_document(self, image_path, apply_all=True):
        """Process document with all preprocessing steps."""
        # Load image
        image = self.load_image(image_path)
        if image is None:
            raise ValueError("Could not load image from path: " + image_path)

        if apply_all:
            # Identify page number
            page_number = self.identify_page(image)
            print(f"Detected page {page_number}")

            # Apply preprocessing steps
            image = self.reduce_noise(image)
            image = self.correct_skew_with_markers(image)
            image = self.enhance_contrast(image)
            image = self.normalize_form_size(image)
            image = self.correct_lighting(image)
            image = self.sharpen_edges(image)
            image = self.convert_to_grayscale(image)

        return image

    def perform_ocr(self, image):
        """Perform OCR on the processed image."""
        # Convert image to RGB if it's grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Perform OCR
        results = self.reader.readtext(image)
        return results
