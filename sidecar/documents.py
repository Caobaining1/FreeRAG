"""Non-PDF document readers: text, Markdown, DOCX, DOC/RTF (plan.md §5.1).

The parse pipeline was written for PDFs: every block carries a page number, a
page-local bbox, and a font size, because that is what a PDF gives you. None of
those exist in a .txt or a .docx — but the chunker downstream only needs blocks
with a type and text (its geometry gate treats a zero-width box as "nothing to
judge", see `chunking._stacked`). So the readers here produce exactly the same
`Block` type with `page_num = 1` and a zero bbox, and everything after this
module is unchanged.

Deliberately dependency-free, because the app ships a bundled interpreter:

* DOCX is a zip of XML, read with `zipfile` + `xml.etree`. Heading structure
  comes from the document's own styles, so a Word heading is a real `Title`
  rather than a font-size guess.
* DOC (Word 97) and RTF have no stdlib reader. On macOS `textutil` converts
  them; elsewhere the caller gets a clear error instead of a wrong parse.
* TXT/MD are read as UTF-8 with a fallback, and split on blank lines.
"""
import os
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from typing import List, Optional, Tuple

WORD_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

PDF_SUFFIXES = ('.pdf',)
DOCX_SUFFIXES = ('.docx',)
#: Formats `textutil` can turn into text on macOS.
CONVERTED_SUFFIXES = ('.doc', '.rtf')
TEXT_SUFFIXES = ('.txt', '.md', '.markdown')

SUPPORTED_SUFFIXES = PDF_SUFFIXES + DOCX_SUFFIXES + CONVERTED_SUFFIXES + TEXT_SUFFIXES

#: A paragraph shorter than this is kept as-is but never treated as a heading;
#: longer lines are prose that merely happens to sit under a heading's style.
HEADING_MAX_CHARS = 120

#: A blank-line paragraph longer than this is split on its line breaks too, so a
#: wrapped or line-per-line file does not arrive as one enormous block.
LONG_PARAGRAPH_CHARS = 1200

_HEADING_STYLE_RE = re.compile(r'^(heading|title|标题|标题\s*\d)', re.IGNORECASE)


def suffix_of(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def is_supported(path: str) -> bool:
    return suffix_of(path) in SUPPORTED_SUFFIXES


def _decode(raw: bytes) -> str:
    """UTF-8 first, then the encodings a Windows-authored .txt usually carries."""
    for encoding in ('utf-8', 'utf-16', 'gb18030', 'latin-1'):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode('utf-8', errors='replace')


def _clean(text: str) -> str:
    """Collapse the whitespace a text layer or a Word run leaves behind."""
    text = text.replace('\u00a0', ' ').replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def paragraphs_from_text(text: str, markdown: bool = False) -> List[Tuple[str, str]]:
    """``(block_type, text)`` per paragraph, in reading order.

    Blank lines separate paragraphs — but text extracted from a PDF, or written
    by an editor that wraps at column 80, often has none at all, and one file
    then reads as one 8,000-character paragraph that the chunker has to cut
    mid-sentence. So a "paragraph" longer than LONG_PARAGRAPH_CHARS is split on
    its line breaks as well. Line breaks are the safer place to cut: the chunker
    merges the pieces back up to the merge ceiling anyway, and a break that used
    to be a paragraph keeps the boundary where the author put it.
    """
    result: List[Tuple[str, str]] = []
    for block in re.split(r'\n\s*\n', text):
        body = _clean(block)
        if not body:
            continue
        if markdown and body.startswith('#'):
            heading = body.lstrip('#').strip()
            if heading:
                result.append(('Title', heading))
                continue
        if len(body) <= LONG_PARAGRAPH_CHARS:
            result.append(('Text', body))
            continue
        for line in body.split('\n'):
            piece = _clean(line)
            if piece:
                result.append(('Text', piece))
    return result


def paragraphs_from_docx(path: str) -> List[Tuple[str, str]]:
    """Paragraphs (and Markdown-rendered tables) from the DOCX package itself."""
    with zipfile.ZipFile(path) as archive:
        heading_styles = _docx_heading_styles(archive)
        document = ET.fromstring(archive.read('word/document.xml'))

    result: List[Tuple[str, str]] = []
    body = document.find(WORD_NS + 'body')
    if body is None:
        return result

    for node in body:
        tag = node.tag
        if tag == WORD_NS + 'p':
            text = _docx_paragraph(node)
            if not text:
                continue
            style = _docx_style(node)
            is_heading = heading_styles.get(style, False) or _HEADING_STYLE_RE.match(style or '')
            if is_heading and len(text) <= HEADING_MAX_CHARS:
                result.append(('Title', text))
            else:
                result.append(('Text', text))
        elif tag == WORD_NS + 'tbl':
            markdown = _docx_table(node)
            if markdown:
                result.append(('Table', markdown))
    return result


def _docx_heading_styles(archive: zipfile.ZipFile) -> dict:
    """styleId -> whether that style is a heading, from word/styles.xml."""
    try:
        styles = ET.fromstring(archive.read('word/styles.xml'))
    except KeyError:
        return {}

    headings = {}
    for style in styles.iter(WORD_NS + 'style'):
        style_id = style.get(WORD_NS + 'styleId') or ''
        name_node = style.find(WORD_NS + 'name')
        name = (name_node.get(WORD_NS + 'val') if name_node is not None else '') or ''
        # Word's own ids are "Heading1".."Heading9"; a localized template keeps
        # the id but renames the style, so both are checked.
        headings[style_id] = bool(
            re.match(r'^heading\s*\d*$', style_id, re.IGNORECASE)
            or _HEADING_STYLE_RE.match(name)
        )
    return headings


def _docx_style(paragraph: ET.Element) -> str:
    node = paragraph.find(WORD_NS + 'pPr/' + WORD_NS + 'pStyle')
    return (node.get(WORD_NS + 'val') or '') if node is not None else ''


def _docx_paragraph(paragraph: ET.Element) -> str:
    parts: List[str] = []
    for node in paragraph.iter():
        if node.tag == WORD_NS + 't':
            parts.append(node.text or '')
        elif node.tag == WORD_NS + 'tab':
            parts.append('\t')
        elif node.tag in (WORD_NS + 'br', WORD_NS + 'cr'):
            parts.append('\n')
    return _clean(''.join(parts))


def _docx_table(table: ET.Element) -> str:
    """A table as Markdown, matching what the PDF path's TSR produces."""
    rows: List[List[str]] = []
    for row in table.findall(WORD_NS + 'tr'):
        cells = [_clean(' '.join(
            _docx_paragraph(p) for p in cell.findall(WORD_NS + 'p')
        )) for cell in row.findall(WORD_NS + 'tc')]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ''

    width = max(len(row) for row in rows)
    lines = ['| ' + ' | '.join(row + [''] * (width - len(row))) + ' |' for row in rows]
    lines.insert(1, '|' + ' --- |' * width)
    return '\n'.join(lines)


def _convert_with_textutil(path: str) -> str:
    """DOC/RTF -> text. macOS only; the error says so rather than guessing."""
    try:
        finished = subprocess.run(
            ['textutil', '-convert', 'txt', '-stdout', path],
            check=True, capture_output=True,
        )
    except FileNotFoundError as error:
        raise ValueError(
            f'cannot read {os.path.basename(path)}: needs "textutil" (macOS)'
        ) from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or b'').decode('utf-8', errors='replace').strip()
        raise ValueError(f'textutil failed on {os.path.basename(path)}: {detail}') from error
    return _decode(finished.stdout)


def load_paragraphs(path: str) -> List[Tuple[str, str]]:
    """``(block_type, text)`` for a non-PDF document, in reading order."""
    suffix = suffix_of(path)
    if suffix in DOCX_SUFFIXES:
        return paragraphs_from_docx(path)
    if suffix in CONVERTED_SUFFIXES:
        return paragraphs_from_text(_convert_with_textutil(path))
    if suffix in TEXT_SUFFIXES:
        with open(path, 'rb') as handle:
            raw = handle.read()
        return paragraphs_from_text(_decode(raw), markdown=suffix in ('.md', '.markdown'))
    raise ValueError(f'unsupported document format: {suffix or path}')
