import cv2
import numpy as np
import os
from form_template import FormTemplate, FieldType


def visualize_coordinates(image_path, template, page_number):
    """Draw rectangles around field regions with template coordinates."""
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

        # Add field name and coordinates
        info_text = f"{field_name} ({x},{y}) {w}x{h}"
        cv2.putText(debug_image, info_text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Create debug directory if it doesn't exist
    debug_dir = "debug_rotated_original"
    os.makedirs(debug_dir, exist_ok=True)

    # Save the visualization
    output_path = f"{debug_dir}/page_{page_number}_coordinates.png"
    cv2.imwrite(output_path, debug_image)
    print(f"Saved coordinate visualization to {output_path}")

    # Also save a version with grid lines for reference
    grid_image = image.copy()
    # Draw horizontal grid lines every 100 pixels
    for y in range(0, image.shape[0], 100):
        cv2.line(grid_image, (0, y), (image.shape[1], y), (0, 0, 255), 1)
        cv2.putText(grid_image, str(y), (5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw vertical grid lines every 100 pixels
    for x in range(0, image.shape[1], 100):
        cv2.line(grid_image, (x, 0), (x, image.shape[0]), (0, 0, 255), 1)
        cv2.putText(grid_image, str(x), (x, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    grid_path = f"{debug_dir}/page_{page_number}_grid.png"
    cv2.imwrite(grid_path, grid_image)
    print(f"Saved grid visualization to {grid_path}")


def main():
    # Create template instance
    template = FormTemplate()

    # Process each page
    for page_num in [1, 2]:
        # Use the original page images
        image_path = f"debug_rotated_original/page_{page_num}_original.png"
        if os.path.exists(image_path):
            print(f"\nProcessing page {page_num}...")
            visualize_coordinates(image_path, template, page_num)
        else:
            print(f"Warning: Image not found: {image_path}")


if __name__ == "__main__":
    main()
