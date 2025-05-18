import fitz  # PyMuPDF
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os


class PDFProcessor:
    """Class for processing PDF files and extracting images."""

    def __init__(self):
        self.pdf_document = None
        self.images = {}

    def load_pdf(self, pdf_path: str) -> bool:
        """Load a PDF file."""
        try:
            print(f"Loading PDF: {pdf_path}")
            self.pdf_document = fitz.open(pdf_path)
            print(
                f"PDF loaded successfully. Total pages: {len(self.pdf_document)}")
            return True
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return False

    def extract_images(self) -> Dict[int, np.ndarray]:
        """Extract images from all pages in the PDF."""
        if not self.pdf_document:
            raise ValueError("No PDF document loaded")

        self.images = {}
        total_pages = len(self.pdf_document)

        print("\nExtracting and processing pages:")
        for page_num in range(total_pages):
            print(f"Processing page {page_num + 1}/{total_pages}...")
            page = self.pdf_document[page_num]

            # Get page as image with higher resolution
            # 2x zoom for better balance of quality and speed
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            # Convert to numpy array
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3)

            # Convert from RGB to BGR (OpenCV format)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Save debug image
            debug_dir = "debug_rotated_original"
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(f"{debug_dir}/page_{page_num + 1}_original.png", img)

            # Apply initial preprocessing
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Enhance contrast
            enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

            # Save preprocessed image
            cv2.imwrite(
                f"{debug_dir}/page_{page_num + 1}_preprocessed.png", enhanced)

            self.images[page_num + 1] = enhanced  # Store grayscale image
            print(f"Page {page_num + 1} processed successfully")

        print("\nAll pages processed successfully!")
        return self.images

    def get_page_count(self) -> int:
        """Get the number of pages in the PDF."""
        if not self.pdf_document:
            raise ValueError("No PDF document loaded")
        return len(self.pdf_document)

    def close(self):
        """Close the PDF document."""
        if self.pdf_document:
            self.pdf_document.close()
            self.pdf_document = None
            self.images = {}
