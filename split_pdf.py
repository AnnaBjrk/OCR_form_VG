import fitz  # PyMuPDF
import sys
import os
from datetime import datetime

# --- CONFIG ---
FILES_DIR = "files_to_process"
DEBUG_DIR = "debug_rotated_original"


def get_available_pdfs():
    """Get list of PDF files in the files_to_process directory"""
    if not os.path.exists(FILES_DIR):
        os.makedirs(FILES_DIR)
        print(f"Created directory: {FILES_DIR}")
        return []

    pdf_files = [f for f in os.listdir(
        FILES_DIR) if f.lower().endswith('.pdf')]
    return pdf_files


def get_user_pdf_selection():
    """Let user select a PDF file to process"""
    pdf_files = get_available_pdfs()

    if not pdf_files:
        print(f"No PDF files found in {FILES_DIR} directory.")
        print(f"Please place your PDF files in the {FILES_DIR} directory.")
        return None

    print("\nAvailable PDF files:")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"{i}. {pdf}")

    while True:
        try:
            choice = int(
                input("\nEnter the number of the PDF file you want to process: "))
            if 1 <= choice <= len(pdf_files):
                return os.path.join(FILES_DIR, pdf_files[choice-1])
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def generate_output_filenames(input_pdf):
    """Generate output filenames based on input PDF name"""
    base_name = os.path.splitext(os.path.basename(input_pdf))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        'output_pdf1': f"{base_name}_pages_1_2.pdf",
        'output_pdf2': f"{base_name}_pages_3_4.pdf",
        'debug_original': os.path.join(DEBUG_DIR, f"debug_original_{timestamp}.png"),
        'debug_rotated': os.path.join(DEBUG_DIR, f"debug_rotated_{timestamp}.png")
    }


def split_pdf(input_pdf, output_pdf1, output_pdf2):
    """Split PDF into two files with pages 1-2 and 3-4"""
    doc = fitz.open(input_pdf)
    doc1 = fitz.open()
    doc2 = fitz.open()

    # Add pages 1 and 2 to doc1, pages 3 and 4 to doc2 (1-based indexing)
    for i in range(len(doc)):
        if i < 2:  # 0,1 = pages 1,2
            doc1.insert_pdf(doc, from_page=i, to_page=i)
        else:  # 2,3 = pages 3,4
            doc2.insert_pdf(doc, from_page=i, to_page=i)

    doc1.save(output_pdf1)
    doc2.save(output_pdf2)
    doc1.close()
    doc2.close()
    doc.close()
    print(f"Saved {output_pdf1} (pages 1,2) and {output_pdf2} (pages 3,4)")


def main():
    # Create debug directory if it doesn't exist
    if not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR)
        print(f"Created directory: {DEBUG_DIR}")

    # Get user input for PDF selection
    input_pdf = get_user_pdf_selection()
    if not input_pdf:
        return

    # Generate output filenames
    filenames = generate_output_filenames(input_pdf)

    # Process the PDF
    split_pdf(input_pdf, filenames['output_pdf1'], filenames['output_pdf2'])


if __name__ == "__main__":
    main()
