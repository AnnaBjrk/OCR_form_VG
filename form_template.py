from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import re
import cv2
import os
import numpy as np


class FieldType(Enum):
    """Enum for different types of form fields with their validation patterns."""
    # Text fields
    TEXT = "text"                    # General text, no restrictions
    ALPHABETIC = "alphabetic"        # Only letters
    ALPHANUMERIC = "alphanumeric"    # Letters and numbers
    SWEDISH_TEXT = "swedish_text"    # Swedish characters allowed

    # Numeric fields
    NUMERIC = "numeric"              # Only numbers
    PERSONAL_NUMBER = "personal_number"  # Swedish personal number
    PHONE_NUMBER = "phone_number"    # Swedish phone number
    POSTAL_CODE = "postal_code"      # Swedish postal code

    # Date and time fields
    DATE = "date"                    # YYYY-MM-DD
    MONTH = "month"                  # MM
    YEAR = "year"                    # YYYY

    # Special fields
    CHECKBOX = "checkbox"            # Boolean selection
    SIGNATURE = "signature"          # Signature field
    HEADER = "header"                # Section header
    INFO_TEXT = "info_text"          # Informational text

    # Table specific
    TABLE_HEADER = "table_header"    # Table column header
    TABLE_CELL = "table_cell"        # Table cell content

    @property
    def validation_pattern(self) -> Optional[str]:
        """Get the validation pattern for this field type."""
        patterns = {
            FieldType.ALPHABETIC: r"^[a-zA-ZåäöÅÄÖ\s-]+$",
            FieldType.ALPHANUMERIC: r"^[a-zA-ZåäöÅÄÖ0-9\s-]+$",
            FieldType.SWEDISH_TEXT: r"^[a-zA-ZåäöÅÄÖ0-9\s.,!?-]+$",
            FieldType.NUMERIC: r"^\d+$",
            FieldType.PERSONAL_NUMBER: r"^\d{8}-\d{4}$|^\d{6}-\d{4}$",
            FieldType.PHONE_NUMBER: r"^(\+46|0)[1-9]\d{8}$",
            FieldType.POSTAL_CODE: r"^\d{3}\s?\d{2}$",
            FieldType.DATE: r"^\d{4}-\d{2}-\d{2}$",
            FieldType.MONTH: r"^(0[1-9]|1[0-2])$",
            FieldType.YEAR: r"^\d{4}$"
        }
        return patterns.get(self)

    @property
    def format_description(self) -> Optional[str]:
        """Get the format description for this field type."""
        formats = {
            FieldType.DATE: "YYYY-MM-DD",
            FieldType.MONTH: "MM",
            FieldType.YEAR: "YYYY",
            FieldType.PERSONAL_NUMBER: "ååååmmdd-xxxx eller ååmmdd-xxxx",
            FieldType.PHONE_NUMBER: "+46XXXXXXXXX eller 0XXXXXXXXX",
            FieldType.POSTAL_CODE: "XXX XX"
        }
        return formats.get(self)

    def validate(self, value: str) -> bool:
        """Validate a value against this field type's rules."""
        if not value and self != FieldType.CHECKBOX:
            return False

        if self == FieldType.CHECKBOX:
            # Accept empty string or any checked indicator
            return not value or CheckboxField.is_checked_indicator(value)

        if self == FieldType.SIGNATURE:
            return True  # Signature validation would be handled separately

        if self == FieldType.HEADER or self == FieldType.INFO_TEXT:
            return True  # These are display-only fields

        if self.validation_pattern:
            return bool(re.match(self.validation_pattern, value))

        return True


@dataclass
class Field:
    """Class representing a form field."""
    name: str
    field_type: FieldType
    position: Tuple[int, int]  # (x, y) coordinates
    size: Tuple[int, int]     # (width, height)
    format: Optional[str] = None
    required: bool = False
    max_length: Optional[int] = None
    related_fields: List[str] = None
    validation_pattern: Optional[str] = None


@dataclass
class TableColumn:
    """Class representing a table column."""
    name: str
    field_type: FieldType
    width: int
    format: Optional[str] = None
    validation_pattern: Optional[str] = None


@dataclass
class Table:
    """Class representing a table in the form."""
    name: str
    position: Tuple[int, int]
    columns: List[TableColumn]
    num_rows: int
    row_height: int
    required: bool = False


@dataclass
class CheckboxField(Field):
    """Class representing a checkbox field."""
    checked: bool = False
    # For grouped checkboxes (e.g., radio-like behavior)
    group: Optional[str] = None

    # Characters that indicate a checked state
    CHECKED_INDICATORS = {"X", "x", "✓", "✔",
                          "☑", "■", "●", "•", "1", "V", "v", "√"}

    def toggle(self) -> None:
        """Toggle the checkbox state."""
        self.checked = not self.checked

    def is_checked(self) -> bool:
        """Check if the checkbox is checked."""
        return self.checked

    def set_checked(self, checked: bool) -> None:
        """Set the checkbox state."""
        self.checked = checked

    def get_value(self) -> str:
        """Get the checkbox value as a string."""
        return "X" if self.checked else ""

    @classmethod
    def is_checked_indicator(cls, value: str) -> bool:
        """Check if a value indicates a checked state."""
        return value.strip() in cls.CHECKED_INDICATORS

    def set_from_value(self, value: str) -> None:
        """Set the checked state from a string value."""
        self.checked = self.is_checked_indicator(value)


class FormTemplate:
    """Class defining the structure of the activity report form."""

    def __init__(self, form_id: str):
        self.form_id = form_id
        self.pages: Dict[int, Dict[str, Any]] = {}
        self.template_images: Dict[int, np.ndarray] = {}
        self._initialize_template()

    def _initialize_template(self):
        """Initialize the form template with field definitions and template images."""
        # Load template images
        template_dir = "templates"
        if os.path.exists(template_dir):
            for filename in os.listdir(template_dir):
                if filename.startswith(f"{self.form_id}_page") and filename.endswith((".png", ".jpg", ".jpeg")):
                    try:
                        # Extract page number from filename (e.g., "form1_page1.png" -> 1)
                        page_num = int(filename.split("page")[1].split(".")[0])
                        template_path = os.path.join(template_dir, filename)
                        template_image = cv2.imread(template_path)
                        if template_image is not None:
                            self.template_images[page_num] = template_image
                            print(f"Loaded template image for page {page_num}")
                    except Exception as e:
                        print(f"Error loading template image {filename}: {e}")

        # Initialize page 1
        self.pages[1] = {
            "fields": {
                "form_title": Field(
                    name="Aktivitetsrapport",
                    field_type=FieldType.HEADER,
                    position=(100, 200),
                    size=(300, 50)
                ),
                "page_number": Field(
                    name="Sida 1(2)",
                    field_type=FieldType.TEXT,
                    position=(1030, 30),
                    size=(100, 30)
                ),

                # Personal information section
                "first_name": Field(
                    name="Förnamn",
                    field_type=FieldType.ALPHABETIC,
                    position=(100, 300),
                    size=(350, 80),
                    required=True,
                    max_length=50
                ),
                "last_name": Field(
                    name="Efternamn",
                    field_type=FieldType.ALPHABETIC,
                    position=(440, 300),
                    size=(350, 80),
                    required=True,
                    max_length=50
                ),
                "personal_number": Field(
                    name="Personnummer/Samordningsnummer",
                    field_type=FieldType.ALPHANUMERIC,
                    position=(790, 300),
                    size=(350, 80),
                    format="ååååmmdd-xxxx eller ååmmdd-xxxx",
                    required=True,
                    validation_pattern=r"^\d{8}-\d{4}$|^\d{6}-\d{4}$"
                ),

                # Reporting period section
                "month": Field(
                    name="Månad",
                    field_type=FieldType.NUMERIC,
                    position=(100, 440),
                    size=(25, 30),
                    format="MM",
                    required=True,
                    validation_pattern=r"^(0[1-9]|1[0-2])$"
                ),
                "year": Field(
                    name="År",
                    field_type=FieldType.NUMERIC,
                    position=(130, 440),
                    size=(25, 30),
                    format="YYYY",
                    required=True,
                    validation_pattern=r"^\d{4}$"
                )
            },
            "tables": {
                "advertised_jobs": Table(
                    name="Annonserade jobb",
                    position=(100, 585),
                    columns=[
                        TableColumn("Datum", FieldType.DATE, 100, "YYYY-MM-DD",
                                    r"^\d{4}-\d{2}-\d{2}$"),
                        TableColumn("Vilket jobb har du sökt?",
                                    FieldType.ALPHABETIC, 230),
                        TableColumn("Arbetsgivare", FieldType.ALPHABETIC, 620),
                        TableColumn("Arbetsort", FieldType.ALPHABETIC, 880)
                    ],
                    num_rows=22,
                    row_height=50,
                    required=True
                )
            }
        }

        # Initialize page 2
        self.pages[2] = {
            "fields": {
                "page_title": Field(
                    name="Har du anmält intresse för ett jobb som inte annonserats?",
                    field_type=FieldType.HEADER,
                    position=(100, 100),
                    size=(800, 50)
                ),
                "page_number": Field(
                    name="Sida 2(2)",
                    field_type=FieldType.TEXT,
                    position=(1030, 30),
                    size=(100, 40)
                ),

                # Signature section
                "signature_date": Field(
                    name="Datum",
                    field_type=FieldType.DATE,
                    position=(100, 1520),
                    size=(520, 50),
                    format="YYYY-MM-DD",
                    required=True,
                    validation_pattern=r"^\d{4}-\d{2}-\d{2}$"
                ),
                "name_clarification": Field(
                    name="Namnförtydligande",
                    field_type=FieldType.ALPHABETIC,
                    position=(100, 1590),
                    size=(520, 50),
                    required=True,
                    max_length=100
                ),
                "signature": Field(
                    name="Namnteckning",
                    field_type=FieldType.SIGNATURE,
                    position=(620, 1520),
                    size=(520, 100),
                    required=True
                ),

                # Nothing to report section
                "nothing_to_report": CheckboxField(
                    name="Jag har inga sökta jobb, intervjuer eller aktiviteter att rapportera",
                    field_type=FieldType.CHECKBOX,
                    position=(100, 1470),
                    size=(25, 25),
                    group="reporting_status"  # Group with other reporting status checkboxes
                )
            },
            "tables": {
                "unadvertised_jobs": Table(
                    name="Icke-annonserade jobb",
                    position=(100, 190),
                    columns=[
                        TableColumn("Datum", FieldType.DATE, 100, "YYYY-MM-DD",
                                    r"^\d{4}-\d{2}-\d{2}$"),
                        TableColumn(
                            "Vilket jobb har du anmält intresse för?", FieldType.ALPHABETIC, 220),
                        TableColumn("Arbetsgivare", FieldType.ALPHABETIC, 530),
                        TableColumn("Arbetsort", FieldType.ALPHABETIC, 830)
                    ],
                    num_rows=7,
                    row_height=50
                ),
                "interviews": Table(
                    name="Intervjuer",
                    position=(100, 620),
                    columns=[
                        TableColumn("Datum", FieldType.DATE, 100, "YYYY-MM-DD",
                                    r"^\d{4}-\d{2}-\d{2}$"),
                        TableColumn("Vilket jobb gällde intervjun?",
                                    FieldType.ALPHABETIC, 220),
                        TableColumn("Arbetsgivare", FieldType.ALPHABETIC, 530),
                        TableColumn("Arbetsort", FieldType.ALPHABETIC, 830)
                    ],
                    num_rows=6,
                    row_height=50
                ),
                "other_activities": Table(
                    name="Övriga aktiviteter",
                    position=(100, 1060),
                    columns=[
                        TableColumn("Datum", FieldType.DATE, 100, "YYYY-MM-DD",
                                    r"^\d{4}-\d{2}-\d{2}$"),
                        TableColumn("Beskrivning av aktiviteten",
                                    FieldType.ALPHABETIC, 260)
                    ],
                    num_rows=7,
                    row_height=50
                )
            }
        }

    def get_template_image(self, page_number: int) -> Optional[np.ndarray]:
        """Get the template image for a specific page."""
        return self.template_images.get(page_number)

    def get_page_fields(self, page_number: int) -> Dict[str, Field]:
        """Get all fields for a specific page."""
        return self.pages[page_number]["fields"]

    def get_page_tables(self, page_number: int) -> Dict[str, Table]:
        """Get all tables for a specific page."""
        return self.pages[page_number]["tables"]

    def validate_field_value(self, field_name: str, value: str, page_number: int) -> bool:
        """Validate a field value against its defined format and type."""
        fields = self.get_page_fields(page_number)
        if field_name not in fields:
            return False

        field = fields[field_name]

        # Check required fields
        if field.required and not value:
            return False

        # Check max length
        if field.max_length and len(value) > field.max_length:
            return False

        # Use the field type's validation
        if not field.field_type.validate(value):
            return False

        # Check field-specific validation pattern if exists
        if field.validation_pattern and not re.match(field.validation_pattern, value):
            return False

        return True

    def validate_table_cell(self, table_name: str, column_index: int,
                            value: str, page_number: int) -> bool:
        """Validate a table cell value."""
        tables = self.get_page_tables(page_number)
        if table_name not in tables:
            return False

        table = tables[table_name]
        if column_index >= len(table.columns):
            return False

        column = table.columns[column_index]

        # Use the column's field type validation
        if not column.field_type.validate(value):
            return False

        # Check column-specific validation pattern if exists
        if column.validation_pattern and not re.match(column.validation_pattern, value):
            return False

        return True

    def get_field_coordinates(self, field_name: str, page_number: int) -> Optional[Tuple[int, int]]:
        """Get the coordinates of a specific field."""
        fields = self.get_page_fields(page_number)
        if field_name in fields:
            return fields[field_name].position
        return None

    def get_table_cell_coordinates(self, table_name: str, row: int, column: int, page_number: int) -> Optional[Tuple[int, int]]:
        """Get the coordinates of a specific table cell."""
        tables = self.get_page_tables(page_number)
        if table_name in tables:
            table = tables[table_name]
            if 0 <= row < table.num_rows and 0 <= column < len(table.columns):
                x = table.position[0] + \
                    sum(col.width for col in table.columns[:column])
                y = table.position[1] + (row * table.row_height)
                return (x, y)
        return None

    def toggle_checkbox(self, field_name: str, page_number: int) -> bool:
        """Toggle a checkbox field's state."""
        fields = self.get_page_fields(page_number)
        if field_name not in fields or not isinstance(fields[field_name], CheckboxField):
            return False

        checkbox = fields[field_name]

        # If checkbox is part of a group, uncheck others in the same group
        if checkbox.group:
            for field in fields.values():
                if (isinstance(field, CheckboxField) and
                    field.group == checkbox.group and
                        field != checkbox):
                    field.set_checked(False)

        checkbox.toggle()
        return True

    def is_checkbox_checked(self, field_name: str, page_number: int) -> bool:
        """Check if a checkbox is checked."""
        fields = self.get_page_fields(page_number)
        if field_name not in fields or not isinstance(fields[field_name], CheckboxField):
            return False
        return fields[field_name].is_checked()

    def set_checkbox_state(self, field_name: str, checked: bool, page_number: int) -> bool:
        """Set a checkbox's state."""
        fields = self.get_page_fields(page_number)
        if field_name not in fields or not isinstance(fields[field_name], CheckboxField):
            return False

        checkbox = fields[field_name]

        # If checkbox is part of a group, uncheck others in the same group
        if checked and checkbox.group:
            for field in fields.values():
                if (isinstance(field, CheckboxField) and
                    field.group == checkbox.group and
                        field != checkbox):
                    field.set_checked(False)

        checkbox.set_checked(checked)
        return True
