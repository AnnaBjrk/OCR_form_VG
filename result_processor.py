import json
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import cv2
import numpy as np
from form_template import FormTemplate, FieldType, CheckboxField
import re
from pathlib import Path
import os


@dataclass
class FieldResult:
    """Class representing the result of processing a single field."""
    value: Union[str, bool]
    confidence: float
    validation_status: bool
    validation_messages: List[str]
    image_quality: float
    requires_review: bool
    page_number: int
    field_type: FieldType
    position: Tuple[int, int]
    size: Tuple[int, int]
    handwritten_text: str = ""  # Original handwritten text from the form


@dataclass
class SectionResult:
    """Class representing the result of processing a form section."""
    name: str
    fields: Dict[str, FieldResult]
    confidence: float
    validation_status: bool
    validation_messages: List[str]


@dataclass
class PageResult:
    """Class representing the result of processing a form page."""
    page_number: int
    sections: Dict[str, SectionResult]
    confidence: float
    validation_status: bool
    validation_messages: List[str]


@dataclass
class FormResult:
    """Class representing the complete form processing result."""
    form_id: str
    timestamp: str
    pages: Dict[int, PageResult]
    confidence: float
    validation_status: bool
    validation_messages: List[str]
    requires_review: bool
    cross_page_validations: List[Dict[str, Any]]


class ResultProcessor:
    """Class for intelligent result processing and output generation."""

    def __init__(self, template: FormTemplate):
        self.template = template
        self.form_result = None
        self.page_orientations = {}  # Store detected orientations

    def _calculate_image_quality(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """Calculate basic image quality score for a field region."""
        # Check if region is valid
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            return 0.0

        region = image[y:y+h, x:x+w]

        # Check if region is empty
        if region.size == 0:
            return 0.0

        # Convert to grayscale if needed
        if len(region.shape) == 3:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Calculate basic quality metrics
        blur_score = cv2.Laplacian(region, cv2.CV_64F).var()
        contrast_score = np.std(region)

        # Normalize scores
        blur_score = min(blur_score / 100, 1.0)
        contrast_score = min(contrast_score / 128, 1.0)

        # Combine scores (equal weights)
        quality_score = 0.5 * blur_score + 0.5 * contrast_score
        return quality_score

    def _calculate_field_confidence(self,
                                    ocr_confidence: float,
                                    validation_status: bool,
                                    image_quality: float) -> float:
        """Calculate overall confidence score for a field."""
        # Weights can be adjusted based on importance
        weights = {
            'ocr': 0.5,
            'validation': 0.3,
            'image_quality': 0.2
        }

        confidence = (
            weights['ocr'] * ocr_confidence +
            weights['validation'] * float(validation_status) +
            weights['image_quality'] * image_quality
        )

        return confidence

    def _validate_cross_page_relations(self) -> List[Dict[str, Any]]:
        """Validate relationships between fields on different pages."""
        validations = []

        # Example: Validate that personal information is consistent
        page1_fields = self.template.get_page_fields(1)
        page2_fields = self.template.get_page_fields(2)

        # Check name consistency
        if 'first_name' in page1_fields and 'first_name' in page2_fields:
            name1 = self.form_result.pages[1].sections['personal_info'].fields['first_name'].value
            name2 = self.form_result.pages[2].sections['personal_info'].fields['first_name'].value

            if name1 != name2:
                validations.append({
                    'type': 'cross_page_validation',
                    'fields': ['first_name'],
                    'pages': [1, 2],
                    'status': False,
                    'message': 'Name mismatch between pages'
                })

        return validations

    def _detect_page_orientation(self, image: np.ndarray, page_number: int) -> bool:
        """Detect if page is upside down based on form title position.
        Returns True if page is correctly oriented, False if upside down."""

        print(f"\nChecking orientation for page {page_number}...")

        # Get the form title field
        fields = self.template.get_page_fields(page_number)
        if "form_title" not in fields:
            print("No form title field found, using default orientation")
            return True

        title_field = fields["form_title"]
        x, y = title_field.position
        w, h = title_field.size

        # Extract the title region
        title_region = image[y:y+h, x:x+w]

        # Convert to grayscale if needed
        if len(title_region.shape) == 3:
            title_region = cv2.cvtColor(title_region, cv2.COLOR_BGR2GRAY)

        # Calculate text density in title region
        text_density = np.mean(title_region < 128)  # Black pixel density
        print(f"Title region text density: {text_density:.3f}")

        # The title should be at the top of the page (y < 20% of page height)
        # and have some text content (density > 0.01)
        is_correct_orientation = y < image.shape[0] * \
            0.2 and text_density > 0.01

        # Store the detected orientation
        self.page_orientations[page_number] = is_correct_orientation

        print(
            f"Page {page_number} is {'correctly' if is_correct_orientation else 'incorrectly'} oriented\n")

        return is_correct_orientation

    def _check_top_elements(self, image: np.ndarray, page_number: int) -> bool:
        """Check if form elements that should be at the top are correctly positioned."""
        # Get fields that should be at the top of the page
        top_fields = []
        fields = self.template.get_page_fields(page_number)

        print(f"Checking top elements for page {page_number}")
        print(f"Total fields on page: {len(fields)}")

        for field_name, field in fields.items():
            # Check if field is in top 20% of the page
            y_pos = field.position[1]
            if y_pos < image.shape[0] * 0.2:
                top_fields.append(field)
                print(f"Found top field: {field_name} at y={y_pos}")

        if not top_fields:
            print("No fields found in top 20% of page")
            return True  # No top fields to check

        # Check if these fields contain text
        for field in top_fields:
            x, y = field.position
            w, h = field.size
            field_region = image[y:y+h, x:x+w]

            # Convert to grayscale if needed
            if len(field_region.shape) == 3:
                field_region = cv2.cvtColor(field_region, cv2.COLOR_BGR2GRAY)

            # Check for text density
            text_density = np.mean(field_region < 128)
            print(f"Field at ({x}, {y}) has text density: {text_density:.3f}")
            if text_density > 0.1:  # If more than 10% black pixels
                print("Found text in top field")
                return True

        print("No text found in top fields")
        return False

    def _correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """Rotate image 180 degrees if it's upside down."""
        print("Rotating image 180 degrees")
        # Rotate the image
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        return rotated

    def process_results(self,
                        raw_results: Dict[str, Dict[str, str]],
                        image: np.ndarray,
                        page_number: int) -> PageResult:
        """Process raw OCR results into structured data."""
        # Check and correct page orientation
        is_correct_orientation = self._detect_page_orientation(
            image, page_number)
        if not is_correct_orientation:
            image = self._correct_orientation(image)

        sections = {}

        # Group fields by section
        for field_name, field_data in raw_results.items():
            # Only print if field has content
            if field_data.get('value') or field_data.get('handwritten_text'):
                print(f"Processing field {field_name}: {field_data}")

            section_name = self._get_section_name(field_name)

            if section_name not in sections:
                sections[section_name] = SectionResult(
                    name=section_name,
                    fields={},
                    confidence=0.0,
                    validation_status=True,
                    validation_messages=[]
                )

            # Get field from template if it exists
            fields = self.template.get_page_fields(page_number)
            field = fields.get(field_name)

            # For table cells, create a virtual field
            if field is None and '_row' in field_name and '_col' in field_name:
                # Extract position from table template
                table_name = field_name.split('_row')[0]
                row = int(field_name.split('_row')[1].split('_col')[0])
                col = int(field_name.split('_col')[1])

                x, y = self.template.get_table_cell_coordinates(
                    table_name, row, col, page_number)
                if x is not None and y is not None:
                    table = self.template.get_page_tables(page_number)[
                        table_name]
                    w, h = table.columns[col].width, table.row_height
                    field = type('Field', (), {
                        'position': (x, y),
                        'size': (w, h),
                        'field_type': FieldType.TEXT
                    })()

            if field is None:
                continue

            # Calculate image quality
            x, y = field.position
            w, h = field.size
            image_quality = self._calculate_image_quality(image, x, y, w, h)

            # Get value and handwritten text
            value = field_data.get('value', '')
            handwritten_text = field_data.get('handwritten_text', value)

            # Only print if field has content
            if value or handwritten_text:
                print(
                    f"Field {field_name} - Value: {value}, Handwritten: {handwritten_text}")

            # Validate field
            validation_status = self.template.validate_field_value(
                field_name, str(value), page_number)
            validation_messages = []

            # Calculate confidence
            ocr_confidence = 1.0  # This should come from the OCR engine
            confidence = self._calculate_field_confidence(
                ocr_confidence, validation_status, image_quality
            )

            # Determine if review is needed
            requires_review = confidence < 0.7 or not validation_status

            # Create field result
            field_result = FieldResult(
                value=value,
                confidence=confidence,
                validation_status=validation_status,
                validation_messages=validation_messages,
                image_quality=image_quality,
                requires_review=requires_review,
                page_number=page_number,
                field_type=getattr(field, 'field_type', FieldType.TEXT),
                position=field.position,
                size=field.size,
                handwritten_text=handwritten_text
            )

            sections[section_name].fields[field_name] = field_result

        # Calculate section-level metrics
        for section in sections.values():
            section.confidence = np.mean(
                [f.confidence for f in section.fields.values()])
            section.validation_status = all(
                f.validation_status for f in section.fields.values())
            section.validation_messages = [
                msg for f in section.fields.values()
                for msg in f.validation_messages
            ]

        # Create page result
        page_result = PageResult(
            page_number=page_number,
            sections=sections,
            confidence=np.mean([s.confidence for s in sections.values()]),
            validation_status=all(
                s.validation_status for s in sections.values()),
            validation_messages=[
                msg for s in sections.values()
                for msg in s.validation_messages
            ]
        )

        return page_result

    def _get_section_name(self, field_name: str) -> str:
        """Get the section name for a field."""
        # Handle table cell fields
        if '_row' in field_name and '_col' in field_name:
            # Extract the table name (everything before _row)
            table_name = field_name.split('_row')[0]
            return table_name

        # Handle regular fields
        if field_name.startswith('personal_'):
            return 'personal_info'
        elif field_name.startswith('activity_'):
            activity_type = field_name.split('_')[1]
            return activity_type
        elif field_name == 'signature_date':
            return 'signature'
        else:
            return 'other'

    def process_form_results(self,
                             page_results: Dict[int, Dict[str, Union[str, bool]]],
                             images: Dict[int, np.ndarray]) -> FormResult:
        """Process results from all pages and create a complete form result."""
        pages = {}

        # First pass: Check and correct orientation for all pages
        for page_num, image in images.items():
            print(f"\nChecking orientation for page {page_num}...")
            is_correct_orientation = self._detect_page_orientation(
                image, page_num)
            if not is_correct_orientation:
                print(f"Rotating page {page_num} 180 degrees")
                images[page_num] = self._correct_orientation(image)
                # Clear any existing results for this page since we rotated it
                if page_num in page_results:
                    page_results[page_num] = {}

        # Second pass: Process the corrected images
        for page_num, raw_results in page_results.items():
            print(f"\nProcessing page {page_num} after orientation check...")
            pages[page_num] = self.process_results(
                raw_results, images[page_num], page_num
            )

        # Create form result with orientation information
        self.form_result = FormResult(
            form_id=self.template.form_id,
            timestamp=datetime.now().isoformat(),
            pages=pages,
            confidence=np.mean([p.confidence for p in pages.values()]),
            validation_status=all(p.validation_status for p in pages.values()),
            validation_messages=[
                msg for p in pages.values()
                for msg in p.validation_messages
            ],
            requires_review=any(
                any(
                    any(f.requires_review for f in s.fields.values())
                    for s in p.sections.values()
                )
                for p in pages.values()
            ),
            cross_page_validations=self._validate_cross_page_relations()
        )

        return self.form_result

    def generate_json_output(self, output_path: str):
        """Generate JSON output file."""
        if not self.form_result:
            raise ValueError("No form result available")

        def convert_value(value):
            """Convert values to JSON serializable format."""
            if isinstance(value, bool):
                return str(value).lower()
            elif isinstance(value, (np.integer, np.floating)):
                return float(value)
            elif isinstance(value, (list, tuple)):
                return [convert_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            return value

        output = {
            'form_id': self.form_result.form_id,
            'timestamp': self.form_result.timestamp,
            'confidence': float(self.form_result.confidence),
            'validation_status': str(self.form_result.validation_status).lower(),
            'requires_review': str(self.form_result.requires_review).lower(),
            'page_orientations': {str(k): str(v).lower() for k, v in self.page_orientations.items()},
            'pages': {}
        }

        for page_num, page in self.form_result.pages.items():
            output['pages'][str(page_num)] = {
                'confidence': float(page.confidence),
                'validation_status': str(page.validation_status).lower(),
                'sections': {}
            }

            for section_name, section in page.sections.items():
                output['pages'][str(page_num)]['sections'][section_name] = {
                    'confidence': float(section.confidence),
                    'validation_status': str(section.validation_status).lower(),
                    'fields': {}
                }

                for field_name, field in section.fields.items():
                    output['pages'][str(page_num)]['sections'][section_name]['fields'][field_name] = {
                        'value': convert_value(field.value),
                        'confidence': float(field.confidence),
                        'validation_status': str(field.validation_status).lower(),
                        'requires_review': str(field.requires_review).lower(),
                        'image_quality': float(field.image_quality)
                    }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    def generate_markdown_output(self, output_path: str) -> None:
        """Generate markdown output for the form results."""
        if not self.form_result:
            raise ValueError("No form result available")

        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.splitext(output_path)[0]
        content_path = f"{base_path}_{timestamp}_content.md"
        processing_path = f"{base_path}_{timestamp}_processing.md"

        # Extract form data
        form_data = {
            'personal_info': {},
            'activities': {
                'advertised_jobs': [],
                'unadvertised_jobs': [],
                'interviews': [],
                'other_activities': []
            },
            'signature_date': None,
            'handwritten_text': {}
        }

        # Process each page
        for page_num, page_result in self.form_result.pages.items():
            for section_name, section in page_result.sections.items():
                for field_name, field in section.fields.items():
                    # Store both value and handwritten text
                    form_data['handwritten_text'][field_name] = {
                        'value': field.value,
                        'handwritten': field.handwritten_text
                    }

                    # Process field based on its type and name
                    if field_name.startswith('personal_'):
                        key = field_name.replace('personal_', '')
                        form_data['personal_info'][key] = field.value
                    elif field_name.startswith('activity_'):
                        activity_type = field_name.split('_')[1]
                        activity_num = int(field_name.split('_')[2])

                        # Initialize activity if needed
                        while len(form_data['activities'][activity_type]) < activity_num:
                            form_data['activities'][activity_type].append({})

                        # Add field to activity
                        field_key = '_'.join(field_name.split('_')[3:])
                        form_data['activities'][activity_type][activity_num -
                                                               1][field_key] = field.value
                    elif field_name == 'signature_date':
                        form_data['signature_date'] = field.value

        # Generate content markdown
        content_md = []

        # Add header
        content_md.append("# Aktivitetsrapport - Sammanställning")
        content_md.append("")
        content_md.append(
            f"**Rapport-ID:** {form_data['personal_info'].get('report_id', '')}")
        content_md.append(
            f"**Skannad:** {datetime.now().strftime('%Y-%m-%d')}")
        content_md.append("")

        # Add personal information with handwritten text
        content_md.append("## Personuppgifter")
        content_md.append("")
        for field_name, field_data in form_data['handwritten_text'].items():
            if field_name.startswith('personal_'):
                key = field_name.replace('personal_', '')
                content_md.append(f"**{key}:**")
                content_md.append(f"- Processed: {field_data['value']}")
                content_md.append(
                    f"- Handwritten: {field_data['handwritten']}")
                content_md.append("")
        content_md.append("")

        # Add activity summaries
        content_md.append("## Sammanfattning av aktiviteter")
        content_md.append("")

        # Count activities
        advertised_jobs = len(
            [a for a in form_data['activities']['advertised_jobs'] if any(a.values())])
        unadvertised_jobs = len(
            [a for a in form_data['activities']['unadvertised_jobs'] if any(a.values())])
        interviews = len([a for a in form_data['activities']
                         ['interviews'] if any(a.values())])
        other = len([a for a in form_data['activities']
                    ['other_activities'] if any(a.values())])
        total = advertised_jobs + unadvertised_jobs + interviews + other

        content_md.append(f"- **Sökta annonserade jobb:** {advertised_jobs}")
        content_md.append(f"- **Intresseanmälningar:** {unadvertised_jobs}")
        content_md.append(f"- **Genomförda intervjuer:** {interviews}")
        content_md.append(f"- **Övriga aktiviteter:** {other}")
        content_md.append(f"- **Totalt antal aktiviteter:** {total}")
        content_md.append("")

        # Add detailed activities with handwritten text
        if advertised_jobs > 0:
            content_md.append("## Sökta annonserade jobb")
            content_md.append("")
            for i, activity in enumerate(form_data['activities']['advertised_jobs'], 1):
                if any(activity.values()):
                    content_md.append(f"### Ansökan {i}")
                    content_md.append("")
                    for field_name, field_data in form_data['handwritten_text'].items():
                        if field_name.startswith(f'activity_advertised_jobs_{i}_'):
                            key = '_'.join(field_name.split('_')[3:])
                            content_md.append(f"**{key}:**")
                            content_md.append(
                                f"- Processed: {field_data['value']}")
                            content_md.append(
                                f"- Handwritten: {field_data['handwritten']}")
                            content_md.append("")
            content_md.append("")

        if unadvertised_jobs > 0:
            content_md.append("## Intresseanmälningar för ej annonserade jobb")
            content_md.append("")
            for i, activity in enumerate(form_data['activities']['unadvertised_jobs'], 1):
                if any(activity.values()):
                    content_md.append(f"### Intresseanmälan {i}")
                    content_md.append("")
                    for field_name, field_data in form_data['handwritten_text'].items():
                        if field_name.startswith(f'activity_unadvertised_jobs_{i}_'):
                            key = '_'.join(field_name.split('_')[3:])
                            content_md.append(f"**{key}:**")
                            content_md.append(
                                f"- Processed: {field_data['value']}")
                            content_md.append(
                                f"- Handwritten: {field_data['handwritten']}")
                            content_md.append("")
            content_md.append("")

        if interviews > 0:
            content_md.append("## Genomförda intervjuer")
            content_md.append("")
            for i, activity in enumerate(form_data['activities']['interviews'], 1):
                if any(activity.values()):
                    content_md.append(f"### Intervju {i}")
                    content_md.append("")
                    for field_name, field_data in form_data['handwritten_text'].items():
                        if field_name.startswith(f'activity_interviews_{i}_'):
                            key = '_'.join(field_name.split('_')[3:])
                            content_md.append(f"**{key}:**")
                            content_md.append(
                                f"- Processed: {field_data['value']}")
                            content_md.append(
                                f"- Handwritten: {field_data['handwritten']}")
                            content_md.append("")
            content_md.append("")

        if other > 0:
            content_md.append("## Övriga aktiviteter")
            content_md.append("")
            for i, activity in enumerate(form_data['activities']['other_activities'], 1):
                if any(activity.values()):
                    content_md.append(f"### Aktivitet {i}")
                    content_md.append("")
                    for field_name, field_data in form_data['handwritten_text'].items():
                        if field_name.startswith(f'activity_other_activities_{i}_'):
                            key = '_'.join(field_name.split('_')[3:])
                            content_md.append(f"**{key}:**")
                            content_md.append(
                                f"- Processed: {field_data['value']}")
                            content_md.append(
                                f"- Handwritten: {field_data['handwritten']}")
                            content_md.append("")
            content_md.append("")

        # Add table cell results to the markdown output
        for table_name, table in form_data['activities'].items():
            if table:  # Check if the table has any entries
                content_md.append(f"## {table_name}")
                content_md.append("")
                for i, row in enumerate(table, 1):
                    if any(row.values()):  # Check if the row has any values
                        content_md.append(f"### Row {i}")
                        content_md.append("")
                        for key, value in row.items():
                            if value:  # Only show non-empty values
                                content_md.append(f"**{key}:** {value}")
                        content_md.append("")
                content_md.append("")

        # Add status overview
        content_md.append("## Statusöversikt")
        content_md.append("")
        content_md.append(
            f"- **Ingen aktivitet rapporterad:** {'Ja' if total == 0 else 'Nej'}")
        content_md.append(
            f"- **Rapporten komplett:** {'Ja' if form_data['signature_date'] else 'Nej'}")
        if form_data['signature_date']:
            content_md.append(
                f"- **Underskrift:** Signerad {form_data['signature_date']}")
        content_md.append(
            f"- **Namnförtydligande:** {form_data['personal_info'].get('name', '')}")
        content_md.append("")

        # Add footer
        content_md.append("---")
        content_md.append("")
        content_md.append(
            "*Detta är en maskinellt genererad sammanställning av Arbetsförmedlingens aktivitetsrapport.*")
        content_md.append(
            "*Vid frågor kontakta Arbetsförmedlingen på 0771-416 416.*")

        # Write content to file
        with open(content_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_md))

        # Generate processing results markdown
        processing_md = []
        processing_md.append("# Form Processing Results")
        processing_md.append("")

        # Add form-level information
        processing_md.append("## Form Information")
        processing_md.append("")
        processing_md.append(f"**Total Pages:** {len(self.form_result.pages)}")
        processing_md.append(
            f"**Overall Confidence:** {self.form_result.confidence:.2f}")
        processing_md.append(
            f"**Validation Status:** {'Valid' if self.form_result.validation_status else 'Invalid'}")
        processing_md.append("")

        # Add handwritten text section
        processing_md.append("## Handwritten Text")
        processing_md.append("")
        for field_name, field_data in form_data['handwritten_text'].items():
            if field_data['handwritten']:  # Only show fields with handwritten text
                processing_md.append(f"**{field_name}:**")
                processing_md.append(f"- Processed: {field_data['value']}")
                processing_md.append(
                    f"- Handwritten: {field_data['handwritten']}")
                processing_md.append("")
        processing_md.append("")

        # Write processing results to file
        with open(processing_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processing_md))
