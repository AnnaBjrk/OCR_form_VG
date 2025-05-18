import os
from pdf_processor import PDFProcessor
from form_template import FormTemplate
from form_processor import FormProcessor
from result_processor import ResultProcessor
import easyocr
import numpy as np
from typing import Dict, Union, List
from datetime import datetime


def get_available_pdfs():
    """Get list of PDF files in the files_to_process directory"""
    files_dir = "files_to_process"
    if not os.path.exists(files_dir):
        os.makedirs(files_dir)
        print(f"Created directory: {files_dir}")
        return []

    pdf_files = [f for f in os.listdir(
        files_dir) if f.lower().endswith('.pdf')]
    return pdf_files, files_dir


def get_user_pdf_selection():
    """Let user select a PDF file to process"""
    pdf_files, files_dir = get_available_pdfs()

    if not pdf_files:
        print(f"No PDF files found in {files_dir} directory.")
        print(f"Please place your PDF files in the {files_dir} directory.")
        return None

    print("\nAvailable PDF files:")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"{i}. {pdf}")

    while True:
        try:
            choice = int(
                input("\nEnter the number of the PDF file you want to process: "))
            if 1 <= choice <= len(pdf_files):
                return os.path.join(files_dir, pdf_files[choice-1])
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def process_forms(pdf_path: str, output_dir: str):
    """Process multiple forms in a PDF file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create debug directory
    debug_dir = "debug_rotated_original"
    os.makedirs(debug_dir, exist_ok=True)

    # Initialize components
    pdf_processor = PDFProcessor()
    template = FormTemplate(form_id="form1")
    form_processor = FormProcessor(template)
    result_processor = ResultProcessor(template)

    # Load and process PDF
    if not pdf_processor.load_pdf(pdf_path):
        print(f"Failed to load PDF: {pdf_path}")
        return

    try:
        # Extract images from PDF
        images = pdf_processor.extract_images()

        # Group pages into forms (2 pages per form)
        forms = []
        current_form = {}
        for page_num, image in images.items():
            current_form[page_num] = image
            if len(current_form) == 2:  # Complete form
                forms.append(current_form)
                current_form = {}

        # Add any remaining pages as a form
        if current_form:
            forms.append(current_form)

        # Process each form
        for form_num, form_pages in enumerate(forms, 1):
            print(f"\nProcessing form {form_num}...")

            # Process each page in the form
            page_results = {}
            for page_num, image in form_pages.items():
                print(f"  Processing page {page_num}...")

                # Process the form using FormProcessor
                field_values = form_processor.process_form(image, page_num)
                page_results[page_num] = field_values

            # Process results for this form
            form_result = result_processor.process_form_results(
                page_results, form_pages)

            # Generate output files for this form
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            md_path = os.path.join(
                output_dir, f"{base_name}_form{form_num}_{timestamp}.md")

            result_processor.generate_markdown_output(md_path)
            print(f"  Form {form_num} results saved to {md_path}")

        print(f"\nProcessing complete. Results saved to {output_dir}")

    finally:
        pdf_processor.close()


if __name__ == "__main__":
    # Get user input for PDF selection
    pdf_path = get_user_pdf_selection()
    if not pdf_path:
        print("No PDF file selected. Exiting...")
        exit(1)

    output_dir = "results"
    process_forms(pdf_path, output_dir)
