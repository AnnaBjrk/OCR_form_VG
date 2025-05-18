import cv2
import numpy as np
from form_template import FormTemplate, FieldType


class FormVisualizer:
    """Class for visualizing the form template."""

    def __init__(self, template: FormTemplate):
        self.template = template
        self.page_width = 2100  # A4 width at 300 DPI
        self.page_height = 2970  # A4 height at 300 DPI
        self.margin = 50

    def create_blank_page(self) -> np.ndarray:
        """Create a blank white page."""
        return np.ones((self.page_height, self.page_width, 3), dtype=np.uint8) * 255

    def draw_field(self, image: np.ndarray, field, color: tuple = (0, 0, 255)):
        """Draw a field on the image."""
        x, y = field.position
        w, h = field.size

        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Special handling for checkboxes
        if field.field_type == FieldType.CHECKBOX:
            # Draw checkbox border
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # If checkbox is checked, draw a check mark
            if isinstance(field, CheckboxField) and field.is_checked():
                # Draw a check mark (✓)
                center_x = x + w // 2
                center_y = y + h // 2
                # Draw the check mark in green
                check_color = (0, 255, 0)  # Green color for check mark
                # Draw the check mark using lines
                cv2.line(image, (x + 2, center_y),
                         (center_x - 2, y + h - 2), check_color, 2)
                cv2.line(image, (center_x - 2, y + h - 2),
                         (x + w - 2, y + 2), check_color, 2)

            # Draw field name to the right of the checkbox
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            cv2.putText(image, field.name, (x + w + 10, y + h),
                        font, font_scale, color, thickness)
            return

        # Draw field name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Split long text into multiple lines
        words = field.name.split()
        lines = []
        current_line = []

        for word in words:
            current_line.append(word)
            text = ' '.join(current_line)
            (text_width, _), _ = cv2.getTextSize(
                text, font, font_scale, thickness)
            if text_width > w - 10:  # Leave some margin
                if len(current_line) > 1:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(text)
                    current_line = []

        if current_line:
            lines.append(' '.join(current_line))

        # Draw each line of text
        for i, line in enumerate(lines):
            y_pos = y + 20 + (i * 20)  # 20 pixels between lines
            if y_pos < y + h:  # Only draw if within field bounds
                cv2.putText(image, line, (x + 5, y_pos), font,
                            font_scale, color, thickness)

    def draw_table(self, image: np.ndarray, table, color: tuple = (0, 0, 255)):
        """Draw a table on the image."""
        x, y = table.position

        # Draw table header
        current_x = x
        for column in table.columns:
            cv2.rectangle(image,
                          (current_x, y),
                          (current_x + column.width, y + table.row_height),
                          color, 2)

            # Draw column name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            # Split column name into multiple lines if needed
            words = column.name.split()
            lines = []
            current_line = []

            for word in words:
                current_line.append(word)
                text = ' '.join(current_line)
                (text_width, _), _ = cv2.getTextSize(
                    text, font, font_scale, thickness)
                if text_width > column.width - 10:
                    if len(current_line) > 1:
                        current_line.pop()
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(text)
                        current_line = []

            if current_line:
                lines.append(' '.join(current_line))

            # Draw each line of text
            for i, line in enumerate(lines):
                y_pos = y + 20 + (i * 20)
                if y_pos < y + table.row_height:
                    cv2.putText(image, line, (current_x + 5, y_pos),
                                font, font_scale, color, thickness)

            current_x += column.width

        # Draw table rows
        for row in range(table.num_rows):
            row_y = y + (row + 1) * table.row_height
            current_x = x
            for column in table.columns:
                cv2.rectangle(image,
                              (current_x, row_y),
                              (current_x + column.width, row_y + table.row_height),
                              color, 1)
                current_x += column.width

    def visualize_page(self, page_number: int) -> np.ndarray:
        """Create a visualization of a specific page."""
        # Create blank page
        image = self.create_blank_page()

        # Draw fields
        fields = self.template.get_page_fields(page_number)
        for field in fields.values():
            self.draw_field(image, field)

        # Draw tables
        tables = self.template.get_page_tables(page_number)
        for table in tables.values():
            self.draw_table(image, table)

        # Add page number
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"Page {page_number}",
                    (self.page_width - 200, 30), font, 1, (0, 0, 0), 2)

        return image

    def save_visualization(self, page_number: int, output_path: str):
        """Save the visualization to a file."""
        image = self.visualize_page(page_number)
        cv2.imwrite(output_path, image)

    def show_visualization(self, page_number: int):
        """Show the visualization in a window."""
        image = self.visualize_page(page_number)
        cv2.imshow(f"Form Template - Page {page_number}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create template and visualizer
    template = FormTemplate()
    visualizer = FormVisualizer(template)

    # Visualize both pages
    visualizer.show_visualization(1)
    visualizer.show_visualization(2)

    # Save visualizations
    visualizer.save_visualization(1, "page1_template.png")
    visualizer.save_visualization(2, "page2_template.png")
