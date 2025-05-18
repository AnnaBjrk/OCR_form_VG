from pdf2image import convert_from_path
import cv2
import numpy as np
import easyocr
import os


def analyze_form_pages(pdf_path):
    """Analyze the form pages to identify distinguishing features."""
    # Convert PDF to images
    images = convert_from_path(pdf_path)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['sv'])

    # Process each page
    for i, image in enumerate(images):
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Perform OCR
        results = reader.readtext(opencv_image)

        print(f"\nAnalyzing Page {i+1}:")
        print("-" * 50)

        # Print all detected text with their positions
        for (bbox, text, prob) in results:
            # Get the top-left corner coordinates
            top_left = bbox[0]
            print(f"Text: {text}")
            print(f"Position: {top_left}")
            print(f"Confidence: {prob:.2f}")
            print("-" * 30)


if __name__ == "__main__":
    pdf_path = "Af_00331_Aktivitetsrapport_(Q).pdf"
    analyze_form_pages(pdf_path)
