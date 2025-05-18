import cv2
import numpy as np
import os
from form_template import FormTemplate, FieldType


def visualize_field_regions(image_path, template, page_number):
    """Draw rectangles around field regions on the full page image."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Create a copy for drawing
    debug_image = image.copy()

    # Get fields for the page
    fields = template.get_page_fields(page_number)

    # Draw rectangles for each field
    for field_name, field in fields.items():
        x, y = field.position
        w, h = field.size

        # Draw rectangle
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add field name
        cv2.putText(debug_image, field_name, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Add coordinates
        coord_text = f"({x},{y}) {w}x{h}"
        cv2.putText(debug_image, coord_text, (x, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Create debug directory if it doesn't exist
    debug_dir = "debug_rotated_original"
    os.makedirs(debug_dir, exist_ok=True)

    # Save the visualization
    output_path = f"{debug_dir}/page_{page_number}_field_regions.png"
    cv2.imwrite(output_path, debug_image)
    print(f"Saved field region visualization to {output_path}")


def main():
    # Create template instance
    template = FormTemplate()

    # Process each page
    for page_num in [1, 2]:
        # Use the original page images
        image_path = f"debug_rotated_original/page_{page_num}_original.png"
        if os.path.exists(image_path):
            print(f"\nProcessing page {page_num}...")
            visualize_field_regions(image_path, template, page_num)
        else:
            print(f"Warning: Image not found: {image_path}")


if __name__ == "__main__":
    main()
