import cv2
import numpy as np
import easyocr
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from form_template import FormTemplate, FieldType, CheckboxField
import re
from datetime import datetime
import os
from form_aligner import FormAligner


@dataclass
class ProcessingConfig:
    """Configuration for field processing."""
    # OCR settings
    ocr_languages: List[str] = None
    ocr_gpu: bool = True
    ocr_model_storage: str = "models"

    # Image processing
    contrast_threshold: float = 1.5
    brightness_threshold: int = 10
    checkbox_threshold: float = 0.15  # 15% black pixels = checked

    # Validation
    strict_validation: bool = True
    allow_partial_matches: bool = True

    def __post_init__(self):
        if self.ocr_languages is None:
            self.ocr_languages = ['sv']  # Only Swedish


class FormProcessor:
    """Class for optimized form processing."""

    def __init__(self, template: FormTemplate, config: Optional[ProcessingConfig] = None):
        """Initialize the form processor with a template and optional configuration."""
        self.template = template
        self.config = config or ProcessingConfig()
        self.reader = None
        self.aligner = FormAligner()
        self._initialize_ocr()

    def _initialize_ocr(self) -> easyocr.Reader:
        """Initialize OCR with appropriate settings."""
        print("\nInitializing OCR...")
        print(f"Languages: {self.config.ocr_languages}")
        print(f"GPU enabled: {self.config.ocr_gpu}")
        print(f"Model storage: {self.config.ocr_model_storage}")

        try:
            reader = easyocr.Reader(
                self.config.ocr_languages,
                gpu=self.config.ocr_gpu,
                model_storage_directory=self.config.ocr_model_storage
            )
            print("OCR initialization successful")
            self.reader = reader
            return reader
        except Exception as e:
            print(f"Error initializing OCR: {str(e)}")
            raise

    def _preprocess_field_image(self, image: np.ndarray, field_type: FieldType) -> np.ndarray:
        """Apply minimal preprocessing to field image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply minimal contrast enhancement
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)

        # Apply adaptive thresholding for better text detection
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        return binary

    def _expand_field_region(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
        """Expand field region to include overlapping text."""
        # Get field region
        field_region = image[y:y+h, x:x+w]

        # Convert to grayscale if needed
        if len(field_region.shape) == 3:
            field_region = cv2.cvtColor(field_region, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        _, binary = cv2.threshold(field_region, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return x, y, w, h

        # Find bounding box of all contours
        x_min, y_min = x + w, y + h
        x_max, y_max = x, y

        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            x_min = min(x_min, x + bx)
            y_min = min(y_min, y + by)
            x_max = max(x_max, x + bx + bw)
            y_max = max(y_max, y + by + bh)

        # Add padding
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)

        return x_min, y_min, x_max - x_min, y_max - y_min

    def process_text_field(self, image: np.ndarray, field_name: str, page_number: int) -> Dict[str, str]:
        """Process a text field with minimal preprocessing."""
        fields = self.template.get_page_fields(page_number)
        if field_name not in fields:
            return {"value": "", "handwritten_text": ""}

        field = fields[field_name]
        x, y = field.position
        w, h = field.size

        # Extract field region
        field_image = image[y:y+h, x:x+w]

        # Debug: Print field image dimensions
        print(
            f"Field {field_name} - Extracted image dimensions: {field_image.shape}")

        # Debug: Save original field image
        debug_dir = "debug_rotated_original"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/{field_name}_original.png", field_image)

        # Apply preprocessing
        processed_image = self._preprocess_field_image(
            field_image, field.field_type)

        # Debug: Save preprocessed image
        cv2.imwrite(f"{debug_dir}/{field_name}_preprocessed.png",
                    processed_image)

        try:
            # Perform OCR with both original and preprocessed images
            print(f"\nProcessing field {field_name}...")
            print(f"Field type: {field.field_type}")
            print(f"Field region: x={x}, y={y}, w={w}, h={h}")

            # Try OCR on both original and preprocessed images
            results = self.reader.readtext(field_image, detail=1)
            if not results:
                results = self.reader.readtext(processed_image, detail=1)

            # Debug print raw OCR results
            print(f"Field {field_name} - Raw OCR results:")
            for result in results:
                print(
                    f"  Text: {result[1]}, Confidence: {result[2]}, Box: {result[0]}")

            # Get the raw OCR results as handwritten text
            handwritten_text = " ".join([result[1] for result in results])

            # Debug print
            print(f"Field {field_name} - Raw OCR text: {handwritten_text}")
            print(
                f"Field {field_name} - OCR confidence: {[result[2] for result in results]}")

            # Apply field-specific post-processing
            if field.field_type == FieldType.DATE:
                processed_text = self._normalize_date(handwritten_text)
            elif field.field_type == FieldType.PERSONAL_NUMBER:
                processed_text = self._normalize_personal_number(
                    handwritten_text)
            elif field.field_type == FieldType.POSTAL_CODE:
                processed_text = self._normalize_postal_code(handwritten_text)
            else:
                processed_text = handwritten_text

            # Debug print
            print(f"Field {field_name} - Processed text: {processed_text}")

            # Return both the processed value and the original handwritten text
            return {
                "value": processed_text,
                "handwritten_text": handwritten_text
            }
        except Exception as e:
            print(f"Error processing field {field_name}: {str(e)}")
            return {"value": "", "handwritten_text": ""}

    def process_checkbox(self, image: np.ndarray, field_name: str, page_number: int) -> bool:
        """Process a checkbox field."""
        fields = self.template.get_page_fields(page_number)
        if field_name not in fields or not isinstance(fields[field_name], CheckboxField):
            return False

        field = fields[field_name]
        x, y = field.position
        w, h = field.size

        # Extract checkbox region
        checkbox_region = image[y:y+h, x:x+w]

        # Convert to grayscale if needed
        if len(checkbox_region.shape) == 3:
            checkbox_region = cv2.cvtColor(checkbox_region, cv2.COLOR_BGR2GRAY)

        # Calculate black pixel ratio
        black_pixels = np.sum(checkbox_region < 128)
        total_pixels = w * h
        black_ratio = black_pixels / total_pixels

        # Check if checkbox is marked
        return black_ratio > self.config.checkbox_threshold

    def _normalize_date(self, text: str) -> str:
        """Normalize date format to YYYY-MM-DD."""
        # Remove any non-digit characters
        digits = re.sub(r'\D', '', text)

        if len(digits) == 8:
            # Assume YYYYMMDD format
            return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
        elif len(digits) == 6:
            # Assume YYMMDD format
            year = int(digits[:2])
            year += 1900 if year > 50 else 2000
            return f"{year}-{digits[2:4]}-{digits[4:]}"

        return text

    def _normalize_personal_number(self, text: str) -> str:
        """Normalize Swedish personal number format."""
        # Remove any non-digit characters
        digits = re.sub(r'\D', '', text)

        if len(digits) == 12:
            # Full format (YYYYMMDDXXXX)
            return f"{digits[:8]}-{digits[8:]}"
        elif len(digits) == 10:
            # Short format (YYMMDDXXXX)
            return f"{digits[:6]}-{digits[6:]}"

        return text

    def _normalize_postal_code(self, text: str) -> str:
        """Normalize Swedish postal code format."""
        # Remove any non-digit characters
        digits = re.sub(r'\D', '', text)

        if len(digits) == 5:
            return f"{digits[:3]} {digits[3:]}"

        return text

    def validate_field(self, field_name: str, value: str, page_number: int) -> bool:
        """Validate a field value with enhanced rules."""
        return self.template.validate_field_value(field_name, value, page_number)

    def process_form(self, image: np.ndarray, page_number: int) -> Dict[str, Dict[str, str]]:
        """Process all fields on a form page."""
        print(f"\n[process_form] Aligning image for page {page_number}...")
        aligned_image = self._align_image(image, page_number)
        print(f"[process_form] Aligned image shape: {aligned_image.shape}")

        results = {}

        # Process text fields
        fields = self.template.get_page_fields(page_number)
        for field_name, field in fields.items():
            if field.field_type == FieldType.CHECKBOX:
                is_checked = self.process_checkbox(
                    aligned_image, field_name, page_number)
                results[field_name] = {
                    "value": str(is_checked),
                    "handwritten_text": "X" if is_checked else ""
                }
            else:
                field_result = self.process_text_field(
                    aligned_image, field_name, page_number)
                results[field_name] = field_result
                print(f"Processed field {field_name}: {field_result}")

        # Process table cells with enhanced debugging
        tables = self.template.get_page_tables(page_number)
        for table_name, table in tables.items():
            print(f"\nProcessing table: {table_name}")
            print(f"Table position: {table.position}")
            print(f"Number of rows: {table.num_rows}")
            print(f"Row height: {table.row_height}")

            for row in range(table.num_rows):
                for col, column in enumerate(table.columns):
                    field_name = f"{table_name}_row{row}_col{col}"
                    x, y = self.template.get_table_cell_coordinates(
                        table_name, row, col, page_number)

                    if x is not None and y is not None:
                        w, h = column.width, table.row_height
                        print(f"\nProcessing cell {field_name}:")
                        print(f"Position: x={x}, y={y}, width={w}, height={h}")

                        # Extract and save cell image for debugging
                        cell_image = aligned_image[y:y+h, x:x+w]
                        debug_dir = "debug_table_cells"
                        os.makedirs(debug_dir, exist_ok=True)
                        cv2.imwrite(
                            f"{debug_dir}/{field_name}.png", cell_image)

                        # Process the cell with OCR directly
                        try:
                            # Apply preprocessing to the cell image
                            processed_cell = self._preprocess_field_image(
                                cell_image, column.field_type)

                            # Try OCR on both original and preprocessed images
                            ocr_results = self.reader.readtext(
                                cell_image, detail=1)
                            if not ocr_results:
                                ocr_results = self.reader.readtext(
                                    processed_cell, detail=1)

                            # Get the raw OCR results as handwritten text
                            handwritten_text = " ".join(
                                [result[1] for result in ocr_results])

                            # Apply field-specific post-processing based on column type
                            if column.field_type == FieldType.DATE:
                                processed_text = self._normalize_date(
                                    handwritten_text)
                            elif column.field_type == FieldType.PERSONAL_NUMBER:
                                processed_text = self._normalize_personal_number(
                                    handwritten_text)
                            elif column.field_type == FieldType.POSTAL_CODE:
                                processed_text = self._normalize_postal_code(
                                    handwritten_text)
                            else:
                                processed_text = handwritten_text

                            # Store the results
                            results[field_name] = {
                                "value": processed_text,
                                "handwritten_text": handwritten_text
                            }
                            print(f"Cell OCR result: {results[field_name]}")
                        except Exception as e:
                            print(
                                f"Error processing cell {field_name}: {str(e)}")
                            results[field_name] = {
                                "value": "", "handwritten_text": ""}
                    else:
                        print(
                            f"Warning: Invalid coordinates for cell {field_name}")

        return results

    def _draw_grid(self, image: np.ndarray, grid_size: int = 100) -> np.ndarray:
        """Draw a grid overlay on the image."""
        height, width = image.shape[:2]
        grid_image = image.copy()

        # Draw vertical lines
        for x in range(0, width, grid_size):
            cv2.line(grid_image, (x, 0), (x, height), (128, 128, 128), 1)
            # Add x-coordinate labels
            cv2.putText(grid_image, str(x), (x + 5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        # Draw horizontal lines
        for y in range(0, height, grid_size):
            cv2.line(grid_image, (0, y), (width, y), (128, 128, 128), 1)
            # Add y-coordinate labels
            cv2.putText(grid_image, str(y), (5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        return grid_image

    def _align_image(self, image: np.ndarray, page_number: int) -> np.ndarray:
        """Align the image using template matching."""
        print(f"\n[process_form] Aligning image for page {page_number}...")

        # Create debug directory if it doesn't exist
        debug_dir = "debug_rotated_original"
        os.makedirs(debug_dir, exist_ok=True)

        # Save original image for debugging
        cv2.imwrite(f"{debug_dir}/page_{page_number}_original.png", image)

        # Get template image for this page
        template_path = f"templates/form1_page{page_number}.png"
        if not os.path.exists(template_path):
            print(f"Warning: Template image not found at {template_path}")
            return image

        template = cv2.imread(template_path)
        if template is None:
            print(
                f"Warning: Could not read template image from {template_path}")
            return image

        # Save template image for debugging
        cv2.imwrite(f"{debug_dir}/page_{page_number}_template.png", template)

        # Align the image
        aligned_image, success = self.aligner.align_form(image, template)

        if not success:
            print(f"Warning: Image alignment failed for page {page_number}")
            return image

        # Draw rectangles around all field regions
        debug_image = aligned_image.copy()
        fields = self.template.get_page_fields(page_number)
        for field_name, field in fields.items():
            x, y = field.position
            w, h = field.size
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_image, field_name, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw rectangles around all table cells
        tables = self.template.get_page_tables(page_number)
        for table_name, table in tables.items():
            for row_idx in range(table.num_rows):
                for col_idx, column in enumerate(table.columns):
                    x, y = self.template.get_table_cell_coordinates(
                        table_name, row_idx, col_idx, page_number)
                    if x is not None and y is not None:
                        w, h = column.width, table.row_height
                        cv2.rectangle(debug_image, (x, y),
                                      (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(debug_image, f"{table_name}_r{row_idx}_c{col_idx}", (
                            x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Add grid overlay
        grid_image = self._draw_grid(debug_image)

        # Save the debug image with rectangles and grid
        cv2.imwrite(
            f"{debug_dir}/page_{page_number}_aligned_with_regions.png", debug_image)
        cv2.imwrite(
            f"{debug_dir}/page_{page_number}_aligned_with_grid.png", grid_image)

        # Save aligned image for debugging
        cv2.imwrite(
            f"{debug_dir}/page_{page_number}_aligned.png", aligned_image)

        print(f"[process_form] Aligned image shape: {aligned_image.shape}")
        return aligned_image

    def process_page(self, image: np.ndarray, page_number: int) -> Dict[str, Any]:
        """Process a page and extract all field values."""
        print(f"\nProcessing page {page_number}...")
        print(f"Input image shape: {image.shape}")

        # Align the image using the reference point
        try:
            aligned_image = self._align_image(image, page_number)
            print(f"Aligned image shape: {aligned_image.shape}")
        except Exception as e:
            print(f"Error aligning image: {str(e)}")
            aligned_image = image

        # Debug: Save the aligned image
        debug_dir = "debug_rotated_original"
        os.makedirs(debug_dir, exist_ok=True)
        aligned_path = f"{debug_dir}/page_{page_number}_aligned.png"
        success = cv2.imwrite(aligned_path, aligned_image)
        print(f"Saved aligned image: {aligned_path} (success: {success})")

        results = {}
        fields = self.template.get_page_fields(page_number)
        print(f"Found {len(fields)} fields to process")

        # Debug: Draw field bounding boxes on the aligned image
        full_page_debug = aligned_image.copy()
        for field_name, field in fields.items():
            x, y = field.position
            w, h = field.size
            cv2.rectangle(full_page_debug, (x, y),
                          (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(full_page_debug, field_name, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        fields_path = f"{debug_dir}/page_{page_number}_fields.png"
        success_fields = cv2.imwrite(fields_path, full_page_debug)
        print(
            f"Saved field boxes image: {fields_path} (success: {success_fields})")

        for field_name, field in fields.items():
            if field.field_type == FieldType.CHECKBOX:
                results[field_name] = self.process_checkbox(
                    aligned_image, field_name, page_number)
            elif field.field_type == FieldType.TABLE:
                results[field_name] = self.process_table(
                    aligned_image, field_name, page_number)
            else:
                results[field_name] = self.process_text_field(
                    aligned_image, field_name, page_number)

        return results
