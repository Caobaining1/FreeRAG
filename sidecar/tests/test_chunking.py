"""Unit tests for chunking + short-block consolidation (plan.md §5.3 / §5.3.1)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunking import (  # noqa: E402
    Block,
    consolidate_blocks,
    is_short,
    max_chars_for,
    split_at_sentences,
    to_chunks,
)

WIDE = 600.0  # page width used by the usefulness heuristic


def text_block(text, page=1, top=0.0, left=0.0, width=400.0, height=12.0, section="", kind="Text"):
    return Block(
        text=text,
        page_num=page,
        block_type=kind,
        bbox=(left, top, left + width, top + height),
        parent_section=section,
    )


class MarginNoiseTest(unittest.TestCase):
    """Header/footer filtering (the layout model has no Header/Footer class).

    Measured caveat: on the arXiv survey used for end-to-end testing this rule
    dropped nothing — the existing digit-only decoration rule and the
    repeated-text noise filter already covered that document. These cases prove
    the rule fires where those two cannot: a running head with words in it, and
    page numbers of the form "Page 3 of 20".
    """

    PAGE_HEIGHT = 800.0

    def test_short_block_in_the_header_band_is_dropped(self):
        body = ("body " * 40).strip()
        blocks = [text_block("Page 3 of 20", top=10.0), text_block(body, top=200.0)]
        kept = consolidate_blocks(blocks, page_width=WIDE, page_height=self.PAGE_HEIGHT)
        self.assertEqual([b.render().strip() for b in kept], [body])

    def test_short_block_in_the_footer_band_is_dropped(self):
        body = ("body " * 40).strip()
        blocks = [text_block(body, top=200.0), text_block("Manuscript submitted", top=780.0)]
        kept = consolidate_blocks(blocks, page_width=WIDE, page_height=self.PAGE_HEIGHT)
        self.assertEqual([b.render().strip() for b in kept], [body])

    def test_the_same_block_in_the_body_is_kept(self):
        blocks = [text_block("Page 3 of 20", top=400.0)]
        kept = consolidate_blocks(blocks, page_width=WIDE, page_height=self.PAGE_HEIGHT)
        self.assertEqual(len(kept), 1, "position, not content, decides this rule")

    def test_long_block_in_the_band_is_kept(self):
        # Real content that happens to sit high on the page.
        long_text = "This paragraph continues for a while and is clearly body text. " * 3
        blocks = [text_block(long_text, top=10.0)]
        kept = consolidate_blocks(blocks, page_width=WIDE, page_height=self.PAGE_HEIGHT)
        self.assertEqual(len(kept), 1)

    def test_a_title_in_the_band_survives(self):
        # A heading legitimately sits at the top of a page.
        blocks = [text_block("3. Methods", top=8.0, kind="Title")]
        kept = consolidate_blocks(blocks, page_width=WIDE, page_height=self.PAGE_HEIGHT)
        self.assertEqual(len(kept), 1)

    def test_without_a_page_height_nothing_is_dropped(self):
        blocks = [text_block("Page 3 of 20", top=10.0)]
        kept = consolidate_blocks(blocks, page_width=WIDE)
        self.assertEqual(len(kept), 1, "the rule must stay inert when the height is unknown")


class ShortnessTest(unittest.TestCase):
    def test_cjk_short_threshold(self):
        self.assertTrue(is_short("短标题"))
        self.assertFalse(is_short("这是一段足够长的正文内容用来超过四十个字符的阈值判断结果应当是假因为长度已经够了"))

    def test_latin_short_threshold(self):
        self.assertTrue(is_short("two words"))
        self.assertFalse(is_short(" ".join(["word"] * 25)))


class ProfileTest(unittest.TestCase):
    def test_profile_limits(self):
        self.assertEqual(max_chars_for("zh"), 8000)
        self.assertEqual(max_chars_for("en"), 30000)
        self.assertEqual(max_chars_for("mixed"), 12000)
        self.assertEqual(max_chars_for("unknown"), 12000)


class SentenceSplitTest(unittest.TestCase):
    def test_split_on_sentence_boundary(self):
        text = "第一句话。第二句话。第三句话。"
        parts = split_at_sentences(text, 10)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertLessEqual(len(part), 10, part)
        self.assertEqual("".join(parts).replace(" ", ""), text)

    def test_short_text_untouched(self):
        self.assertEqual(split_at_sentences("hello", 100), ["hello"])

    def test_overlong_single_sentence_is_hard_sliced(self):
        parts = split_at_sentences("x" * 25, 10)
        self.assertEqual([len(p) for p in parts], [10, 10, 5])


class ConsolidationTest(unittest.TestCase):
    def test_exempt_types_survive_even_when_tiny(self):
        blocks = [text_block("A", kind="Title"), text_block("B", kind="Figure")]
        out = consolidate_blocks(blocks, page_width=WIDE)
        self.assertEqual([b.block_type for b in out], ["Title", "Figure"])

    def test_page_number_and_decor_dropped(self):
        blocks = [text_block("12"), text_block("* * *"), text_block("这是正文内容足够长不应该被过滤掉的一段话啊啊啊啊啊啊")]
        out = consolidate_blocks(blocks, page_width=WIDE)
        self.assertEqual(len(out), 1)
        self.assertIn("正文内容", out[0].text)

    def test_repeated_header_is_dropped(self):
        header = "ACME Confidential"
        blocks = [text_block(header, page=p, top=0) for p in (1, 2, 3)] + [
            text_block("这是正文内容足够长不应该被过滤掉的一段话啊啊啊啊啊啊", page=1, top=100)
        ]
        out = consolidate_blocks(blocks, page_width=WIDE)
        self.assertEqual(len(out), 1)
        self.assertNotIn("Confidential", out[0].text)

    def test_short_block_merges_into_adjacent_text(self):
        blocks = [
            text_block("这是一个没有句号的段落结尾", top=0, height=12),
            text_block("而这是紧随其后的一小段补充文字", top=14, height=12),
        ]
        out = consolidate_blocks(blocks, page_width=WIDE, mean_height=12.0)
        self.assertEqual(len(out), 1)
        self.assertIn("补充文字", out[0].text)

    def test_no_merge_when_previous_sentence_closed(self):
        blocks = [
            text_block("这是一个已经结束的段落。", top=0, height=12),
            text_block("紧随其后的一小段文字", top=14, height=12),
        ]
        out = consolidate_blocks(blocks, page_width=WIDE, mean_height=12.0)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1].attached_to, out[0].chunk_id)

    def test_no_merge_across_sections(self):
        blocks = [
            text_block("第一部分没有句号的段落", top=0, height=12, section="S1"),
            text_block("第二部分的一小段文字", top=14, height=12, section="S2"),
        ]
        out = consolidate_blocks(blocks, page_width=WIDE, mean_height=12.0)
        self.assertEqual(len(out), 2)

    def test_no_merge_when_far_apart(self):
        blocks = [
            text_block("第一段没有句号", top=0, height=12),
            text_block("很远的一小段文字", top=400, height=12),
        ]
        out = consolidate_blocks(blocks, page_width=WIDE, mean_height=12.0)
        self.assertEqual(len(out), 2)

    def test_tiny_latin_fragment_dropped(self):
        blocks = [text_block("ab"), text_block("这是正文内容足够长不应该被过滤掉的一段话啊啊啊啊啊啊")]
        out = consolidate_blocks(blocks, page_width=WIDE)
        self.assertEqual(len(out), 1)

    def test_oversize_block_is_split(self):
        sentence = "这是一个足够长的句子用来测试拆分。"
        blocks = [text_block(sentence * 40, top=0, height=12)]
        out = consolidate_blocks(blocks, max_chars=100, page_width=WIDE)
        self.assertGreater(len(out), 1)
        for block in out:
            self.assertLessEqual(len(block.text), 100)

    def test_chunk_ids_unique_and_stable(self):
        blocks = [text_block("正文一" + "啊" * 60, top=0), text_block("正文二" + "啊" * 60, top=40)]
        chunks = to_chunks(consolidate_blocks(blocks, page_width=WIDE))
        ids = [c["chunk_id"] for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))
        for chunk in chunks:
            self.assertIn("chars", chunk["metadata"])
            self.assertIn("block_type", chunk["metadata"])


if __name__ == "__main__":
    unittest.main()
