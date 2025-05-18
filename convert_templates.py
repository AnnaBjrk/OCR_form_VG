from pdf2image import convert_from_path
import os


def convert_pdf_to_templates(pdf_path, output_dir):
    """Convert PDF pages to PNG images for template matching."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert PDF to images
    print(f"Converting {pdf_path} to images...")
    images = convert_from_path(pdf_path)

    # Save each page as a PNG
    for i, image in enumerate(images, start=1):
        output_path = os.path.join(output_dir, f"form1_page{i}.png")
        image.save(output_path, "PNG")
        print(f"Saved page {i} to {output_path}")


if __name__ == "__main__":
    pdf_path = "templates/Af_aktivitetsrapport_orginal.pdf"
    output_dir = "templates"
    convert_pdf_to_templates(pdf_path, output_dir)
