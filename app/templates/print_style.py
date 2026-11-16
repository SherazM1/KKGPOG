"""PDF section styling shared by the editor and renderer."""
import re


DEFAULT_HEADER_COLOR = '#1B4F9C'
HEADER_COLOR_KEY = 'pdf_header_color'
VERTICAL_SPACING_KEY = 'pdf_vertical_spacing'
PDF_IDENTIFIER_FIELDS = ('UPC', 'ID#', 'Item #')
PDF_TEXT_FIELDS = ('Description', 'CPP', *PDF_IDENTIFIER_FIELDS)
DEFAULT_TEXT_FIELDS = ('Description', 'CPP', 'UPC')


def normalize_text_fields(value):
    if value is None:
        return DEFAULT_TEXT_FIELDS
    try:
        fields = (value,) if isinstance(value, str) else tuple(value)
    except TypeError as exc:
        raise ValueError('Choose UPC, ID#, or Item # for the third PDF line.') from exc
    # Migrate the former optional ID line and empty/multiple selections.
    fields = tuple('ID#' if field == 'ID' else field for field in fields)
    if any(field not in PDF_TEXT_FIELDS for field in fields):
        raise ValueError('Choose UPC, ID#, or Item # for the third PDF line.')
    identifiers = [field for field in fields if field in PDF_IDENTIFIER_FIELDS]
    return ('Description', 'CPP', identifiers[-1] if identifiers else 'UPC')


def normalize_header_color(value):
    if not isinstance(value, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', value):
        raise ValueError('Choose a section header color in #RRGGBB format.')
    return value.upper()


def normalize_vertical_spacing(value):
    if not isinstance(value, bool):
        raise ValueError('Choose whether to keep vertical spacing in the PDF.')
    return 'preserve' if value else 'fill'


def section_preserves_spacing(region):
    return region.metadata.get(VERTICAL_SPACING_KEY, 'preserve') != 'fill'


def section_header_color(region):
    try:
        return normalize_header_color(region.metadata.get(HEADER_COLOR_KEY, DEFAULT_HEADER_COLOR))
    except ValueError:
        return DEFAULT_HEADER_COLOR
