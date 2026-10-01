"""Readers for txt / md / docx / doc / rtf (sidecar/documents.py)."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from documents import (  # noqa: E402
    SUPPORTED_SUFFIXES,
    is_supported,
    load_paragraphs,
    paragraphs_from_docx,
    paragraphs_from_text,
)
from pipeline import parse_document  # noqa: E402

#: The namespace URI as an attribute value, and the same thing in Clark
#: notation for tag lookups. They are not interchangeable: the braces belong to
#: the tag syntax, and putting them in `xmlns` makes the document invalid XML
#: (which is exactly how this fixture was broken the first time).
NS_URI = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
W = '{' + NS_URI + '}'

DOCUMENT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="{uri}">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Methods</w:t></w:r></w:p>
    <w:p><w:r><w:t>We describe the </w:t></w:r><w:r><w:t>method</w:t></w:r>
         <w:r><w:t> in this paragraph.</w:t></w:r></w:p>
    <w:tbl>
      <w:tr><w:tc><w:p><w:r><w:t>Setting</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Score</w:t></w:r></w:p></w:tc></w:tr>
      <w:tr><w:tc><w:p><w:r><w:t>baseline</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>0.42</w:t></w:r></w:p></w:tc></w:tr>
    </w:tbl>
  </w:body>
</w:document>
""".format(uri=NS_URI)

STYLES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<w:styles xmlns:w="{uri}">
  <w:style w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>
  <w:style w:styleId="BodyText"><w:name w:val="Body Text"/></w:style>
</w:styles>
""".format(uri=NS_URI)


def write_docx(path):
    """A minimal but valid enough DOCX: the reader only wants these two parts."""
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('word/document.xml', DOCUMENT_XML)
        archive.writestr('word/styles.xml', STYLES_XML)


class TextReadingTest(unittest.TestCase):
    def test_paragraphs_split_on_blank_lines(self):
        text = 'First paragraph line one\nline two.\n\nSecond paragraph.\n'
        self.assertEqual(
            paragraphs_from_text(text),
            [('Text', 'First paragraph line one\nline two.'), ('Text', 'Second paragraph.')],
        )

    def test_markdown_headings_become_titles(self):
        self.assertEqual(
            paragraphs_from_text('# Title\n\nBody text.', markdown=True),
            [('Title', 'Title'), ('Text', 'Body text.')],
        )

    def test_a_hash_in_a_plain_text_file_is_not_a_heading(self):
        # Only a Markdown file treats '#' as structure; in .txt it is a comment.
        self.assertEqual(
            paragraphs_from_text('# not a heading'), [('Text', '# not a heading')]
        )

    def test_a_long_unbroken_paragraph_is_split_on_its_lines(self):
        # Text pulled out of a PDF has no blank lines; without this it arrives as
        # one 8,000-character block and the chunker cuts it mid-sentence.
        text = '\n'.join('Line %d of the document, long enough to count.' % i for i in range(200))
        paragraphs = paragraphs_from_text(text)

        self.assertGreater(len(paragraphs), 100)
        self.assertTrue(all(len(body) < 200 for _kind, body in paragraphs))

    def test_a_short_wrapped_paragraph_stays_whole(self):
        text = 'A sentence that was wrapped\nacross two lines.'
        self.assertEqual(paragraphs_from_text(text), [('Text', 'A sentence that was wrapped\nacross two lines.')])

    def test_non_utf8_files_are_decoded(self):
        text = '这是一个用 GB18030 编码的文档段落，需要正确读出。'
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as handle:
            handle.write(text.encode('gb18030'))
            path = handle.name
        try:
            self.assertEqual(load_paragraphs(path), [('Text', text)])
        finally:
            os.unlink(path)


class DocxReadingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'sample.docx')
        write_docx(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_heading_style_becomes_a_title_and_runs_are_joined(self):
        paragraphs = paragraphs_from_docx(self.path)
        self.assertEqual(paragraphs[0], ('Title', 'Methods'))
        # Three runs, one paragraph: splitting them would break a sentence.
        self.assertEqual(paragraphs[1], ('Text', 'We describe the method in this paragraph.'))

    def test_a_table_becomes_markdown(self):
        table = [p for p in paragraphs_from_docx(self.path) if p[0] == 'Table']
        self.assertEqual(len(table), 1)
        self.assertIn('| Setting | Score |', table[0][1])
        self.assertIn('| --- | --- |', table[0][1])
        self.assertIn('| baseline | 0.42 |', table[0][1])

    def test_a_docx_without_styles_still_finds_its_heading(self):
        # Word's own style id is "Heading1"; a template can drop styles.xml.
        with zipfile.ZipFile(self.path, 'w') as archive:
            archive.writestr('word/document.xml', DOCUMENT_XML)
        self.assertEqual(paragraphs_from_docx(self.path)[0], ('Title', 'Methods'))


@unittest.skipUnless(shutil.which('textutil'), 'needs macOS textutil')
class ConvertedFormatTest(unittest.TestCase):
    """DOC and RTF have no stdlib reader; they go through textutil."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.txt = os.path.join(self.tmp.name, 'source.txt')
        with open(self.txt, 'w') as handle:
            handle.write('First paragraph of the converted document.\n\nSecond paragraph.\n')

    def tearDown(self):
        self.tmp.cleanup()

    def convert(self, fmt):
        target = os.path.join(self.tmp.name, 'converted.' + fmt)
        subprocess.run(['textutil', '-convert', fmt, '-output', target, self.txt], check=True)
        return target

    def test_doc_round_trips_through_textutil(self):
        paragraphs = load_paragraphs(self.convert('doc'))
        self.assertIn(('Text', 'First paragraph of the converted document.'), paragraphs)

    def test_rtf_round_trips_through_textutil(self):
        paragraphs = load_paragraphs(self.convert('rtf'))
        self.assertTrue(any('Second paragraph' in text for _kind, text in paragraphs))


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        path = os.path.join(self.tmp.name, name)
        with open(path, 'w') as handle:
            handle.write(text)
        return path

    def test_supported_suffixes(self):
        for suffix in ('.pdf', '.docx', '.doc', '.rtf', '.txt', '.md'):
            self.assertIn(suffix, SUPPORTED_SUFFIXES, suffix)
            self.assertTrue(is_supported('document' + suffix))
        self.assertFalse(is_supported('archive.zip'))

    def test_a_text_document_parses_into_chunks(self):
        path = self.write('notes.txt', '第一段正文，需要足够长才会成为一个 chunk。\n\n第二段正文。\n')
        result = parse_document(path, use_cache=False)

        self.assertEqual(result['page_count'], 1)
        self.assertEqual(result['layout_provider'], 'txt')
        self.assertEqual(result['chunk_count'], 1, result['chunks'])
        self.assertIn('第一段正文', result['chunks'][0]['text'])
        self.assertEqual(result['chunks'][0]['metadata']['page_num'], 1)

    def test_a_docx_document_parses_into_chunks(self):
        path = os.path.join(self.tmp.name, 'sample.docx')
        write_docx(path)
        result = parse_document(path, use_cache=False)

        self.assertEqual(result['layout_provider'], 'docx')
        types = [c['metadata']['block_type'] for c in result['chunks']]
        self.assertTrue(all(t in ('Section', 'Title', 'Text', 'TableWithCaption', 'Table')
                            for t in types), types)
        self.assertTrue(any('| Setting |' in c['text'] for c in result['chunks']))

    def test_an_unsupported_format_says_so(self):
        path = self.write('archive.zip', 'not a document')
        with self.assertRaises(ValueError) as caught:
            parse_document(path, use_cache=False)
        self.assertIn('unsupported', str(caught.exception))

    def test_a_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            parse_document(os.path.join(self.tmp.name, 'nope.txt'), use_cache=False)


if __name__ == '__main__':
    unittest.main()
