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
    merge_max_for,
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

    def test_merge_ceiling_stays_below_the_split_threshold(self):
        for profile in ("zh", "en", "mixed"):
            self.assertLess(merge_max_for(profile), max_chars_for(profile))
        self.assertEqual(merge_max_for("unknown"), merge_max_for("mixed"))


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
    def test_bare_latin_fragments_are_dropped(self):
        # TUIrag's final filter: no chunk under 10 characters reaches the index,
        # so a lone table number or a bare heading is dropped rather than indexed.
        blocks = [
            text_block("Abstract", kind="Title"),
            text_block("(1)", kind="Caption"),
            text_block("Steam Deck OLED review", kind="Text"),
        ]
        out = consolidate_blocks(blocks, page_width=WIDE)
        self.assertEqual([b.render() for b in out], ["Steam Deck OLED review"])

    def test_cjk_and_substantial_exempts_survive(self):
        blocks = [text_block("方法", kind="Title"), text_block("A figure placeholder worth indexing", kind="Figure")]
        out = consolidate_blocks(blocks, page_width=WIDE)
        self.assertEqual([b.block_type for b in out], ["Title", "Figure"])

    def test_page_number_and_decor_dropped(self):
        blocks = [text_block("12"), text_block("* * *"), text_block("这是正文内容足够长不应该被过滤掉的一段话啊啊啊啊啊啊")]
        out = consolidate_blocks(blocks, page_width=WIDE)
        self.assertEqual(len(out), 1)
        self.assertIn("正文内容", out[0].text)

    def test_a_repeated_title_is_not_dropped_as_a_running_head(self):
        # The document's own title is also printed as the running head on every
        # later page. The repeated References must go; the Title must stay.
        title = "LLM-Oriented Information Retrieval: A Denoising-First Perspective"
        blocks = [text_block(title, page=1, top=0, kind="Title")]
        blocks += [text_block(title, page=p, top=0, kind="Reference") for p in (3, 5, 7)]
        blocks += [text_block("这是正文内容足够长不应该被过滤掉的一段话啊啊啊啊啊啊", page=1, top=200)]
        out = consolidate_blocks(blocks, page_width=WIDE)

        self.assertFalse([b for b in out if b.block_type == "Reference"])
        headings = [b for b in out if b.block_type in ("Title", "Section")]
        self.assertTrue(headings, "the document title must survive")
        self.assertIn("Denoising-First", headings[0].render())
        self.assertIn("正文内容", headings[0].render(), "the body should merge under it")

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

    def test_consecutive_paragraphs_merge_into_one_chunk(self):
        # TUIrag strategy B buffers consecutive Text regardless of a closing
        # marker: a paragraph and its sequel read as one retrieval unit.
        blocks = [
            text_block("这是一个已经结束的段落。", top=0, height=12),
            text_block("紧随其后的一小段文字", top=14, height=12),
        ]
        out = consolidate_blocks(blocks, page_width=WIDE, mean_height=12.0)
        self.assertEqual(len(out), 1)
        self.assertIn("已经结束", out[0].text)
        self.assertIn("紧随其后", out[0].text)

    def test_title_absorbs_the_body_under_it(self):
        blocks = [
            text_block("3. Methods", top=0, height=18, kind="Title"),
            text_block("We describe the method in this paragraph and explain it at length.", top=30, height=12),
        ]
        out = consolidate_blocks(blocks, page_width=WIDE)
        self.assertEqual(len(out), 1, "a heading must not be a standalone chunk")
        self.assertEqual(out[0].block_type, "Section")
        self.assertIn("3. Methods", out[0].text)
        self.assertIn("describe the method", out[0].text)
        # The section names itself so retrieval metadata still points at a heading.
        self.assertEqual(out[0].parent_section, "3. Methods")

    def test_table_takes_its_caption(self):
        blocks = [
            text_block("| a | b |\n| --- | --- |\n| 1 | 2 |", top=0, height=40, kind="Table"),
            text_block("Table 1: measured results", top=44, height=12, kind="Caption"),
        ]
        out = consolidate_blocks(blocks, page_width=WIDE)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].block_type, "TableWithCaption")
        self.assertTrue(out[0].text.startswith("Table 1"))

    def test_repeated_caption_is_skipped_without_breaking_the_run(self):
        header = "ACME Confidential"  # the running head, seen on every page
        blocks = [
            text_block("这是一个没有句号的段落开头", page=1, top=200, height=12),
            text_block(header, page=1, top=0, kind="Caption"),
            text_block("紧接着的正文会与上一段合并成一块", page=1, top=220, height=12),
        ] + [text_block(header, page=p, top=0, kind="Caption") for p in (2, 3)]
        out = consolidate_blocks(blocks, page_width=WIDE)

        self.assertEqual(len(out), 1)
        self.assertNotIn("Confidential", out[0].text)
        self.assertIn("紧接着的正文", out[0].text)

    def test_a_run_does_not_cross_a_page_break(self):
        # bbox is page-local, so a chunk folded across two pages would carry a box
        # made of coordinates from two different pages — drawn on the first page,
        # covering nothing, while the text also holds the second page's.
        blocks = [
            text_block("第一页最后一段没有句号", page=1, top=700, height=12),
            text_block("第二页第一段正文", page=2, top=90, height=12),
        ]
        out = consolidate_blocks(blocks, page_width=WIDE)

        self.assertEqual(len(out), 2)
        self.assertEqual([chunk.page_num for chunk in out], [1, 2])

    def test_merge_stops_at_the_ceiling(self):
        block = text_block("句子。" * 30, top=0, height=12)
        # A ceiling below the first block means each block stands alone.
        out = consolidate_blocks([block], merge_max=20, page_width=WIDE)
        for chunk in out:
            self.assertLessEqual(len(chunk.text), 20, chunk.text)

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


class MissedCaptionTest(unittest.TestCase):
    """Captions the detector handed back as body text ("Table 4: ...")."""

    def test_a_caption_typed_as_text_joins_the_table_above_it(self):
        table = text_block("| a | b |\n| --- | --- |\n| 1 | 2 |", top=0, height=80, kind="Table")
        caption = text_block("Table 4: Results of ablation studies.", top=84, height=40)
        out = consolidate_blocks([table, caption], page_width=WIDE)

        self.assertEqual([chunk.block_type for chunk in out], ["TableWithCaption"])
        self.assertTrue(out[0].render().startswith("Table 4"))
        self.assertIn("| --- |", out[0].render())

    def test_a_caption_typed_as_text_joins_the_figure_below_it(self):
        caption = text_block("Fig. 3. Comparison on task performance", top=0, height=12)
        figure = text_block("[Figure OCR]: drafts vs score", top=16, height=120, kind="Figure")
        out = consolidate_blocks([caption, figure], page_width=WIDE)

        self.assertEqual([chunk.block_type for chunk in out], ["FigureWithCaption"])

    def test_a_sentence_about_a_table_is_not_a_caption(self):
        # No separator after the number: this is prose that mentions Table 4.
        table = text_block("| a | b |\n| --- | --- |\n| 1 | 2 |", top=0, height=80, kind="Table")
        prose = text_block("Table 4 shows that the results differ across settings.", top=84, height=12)
        out = consolidate_blocks([table, prose], page_width=WIDE)

        self.assertEqual([chunk.block_type for chunk in out], ["Table", "Text"])

    def test_a_caption_away_from_any_table_stays_body_text(self):
        caption = text_block("Table 4: results", top=0, height=12)
        body = text_block("这一段正文跟在后面并应当与它合并成一块。", top=14, height=12)
        out = consolidate_blocks([caption, body], page_width=WIDE)

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].block_type, "Text")
        self.assertIn("Table 4", out[0].render())

    def test_a_caption_in_the_other_column_is_not_attached(self):
        caption = text_block("Table 9: something else", top=600, left=50, width=200, height=12)
        table = text_block("| a | b |\n| --- | --- |\n| 1 | 2 |", top=60, left=320, width=200, height=80, kind="Table")
        out = consolidate_blocks([caption, table], page_width=WIDE)

        self.assertEqual([chunk.block_type for chunk in out], ["Text", "Table"])


class StackingTest(unittest.TestCase):
    """A chunk is one rectangle, so a run only folds blocks stacked in a column."""

    def test_a_column_jump_ends_the_run(self):
        # Bottom of the left column, then the top of the right column, as XY-cut
        # orders a two-column page. Folding them draws one box over the whole
        # corner between the two — the whole reason for this rule.
        left = text_block("左栏的最后一段没有句号", top=400, left=50, width=200, height=100)
        right = text_block("右栏的第一段文字", top=60, left=320, width=200, height=100)
        out = consolidate_blocks([left, right], page_width=WIDE)

        # Two chunks, each box exactly its own block's box — not their union,
        # which would span from the top-left of one to the bottom-right of the
        # other and cover the page between them.
        self.assertEqual(len(out), 2)
        self.assertEqual(
            [[float(v) for v in chunk.bbox] for chunk in out],
            [[float(v) for v in left.bbox], [float(v) for v in right.bbox]],
        )

    def test_blocks_side_by_side_do_not_fold(self):
        # A three-column author grid is read across the row, not down a column.
        first = text_block("第一栏作者信息", top=100, left=50, width=200, height=40)
        second = text_block("第二栏作者信息", top=100, left=300, width=200, height=40)
        out = consolidate_blocks([first, second], page_width=WIDE)

        self.assertEqual(len(out), 2)
        self.assertIn("第一栏", out[0].text)
        self.assertIn("第二栏", out[1].text)

    def test_a_run_continues_while_it_keeps_going_down(self):
        lines = [text_block("正文第一段没有句号", top=0, height=12)]
        lines.append(text_block("正文第二段", top=14, height=12))
        lines.append(text_block("正文第三段", top=28, height=12))
        out = consolidate_blocks(lines, page_width=WIDE)
        self.assertEqual(len(out), 1)

    def test_a_caption_across_the_column_gap_stays_separate(self):
        table = text_block("| a | b |\n| --- | --- |\n| 1 | 2 |", top=0, left=50, width=200, height=80, kind="Table")
        caption = text_block("Table 1: measured results", top=300, left=320, width=200, height=12, kind="Caption")
        out = consolidate_blocks([table, caption], page_width=WIDE)

        self.assertEqual([chunk.block_type for chunk in out], ["Table", "Caption"])

    def test_a_caption_below_its_table_is_absorbed(self):
        table = text_block("| a | b |\n| --- | --- |\n| 1 | 2 |", top=0, left=50, width=200, height=80, kind="Table")
        caption = text_block("Table 1: measured results", top=84, left=50, width=200, height=12, kind="Caption")
        out = consolidate_blocks([table, caption], page_width=WIDE)

        self.assertEqual([chunk.block_type for chunk in out], ["TableWithCaption"])

    def test_a_caption_above_its_table_is_absorbed(self):
        # Journal style puts the table caption above the table. Reading forward
        # only, it becomes a chunk of its own — a fragment of one line.
        caption = text_block("Table 3: ablation results", top=0, left=50, width=200, height=12, kind="Caption")
        table = text_block("| a | b |\n| --- | --- |\n| 1 | 2 |", top=16, left=50, width=200, height=80, kind="Table")
        out = consolidate_blocks([caption, table], page_width=WIDE)

        self.assertEqual([chunk.block_type for chunk in out], ["TableWithCaption"])
        self.assertTrue(out[0].render().startswith("Table 3"))
        self.assertIn("| --- |", out[0].render())

    def test_a_caption_above_a_figure_is_absorbed(self):
        caption = text_block("Figure 4: analysis of the run", top=0, left=50, width=200, height=12, kind="Caption")
        figure = text_block("[Figure OCR]: drafts vs score", top=16, left=50, width=200, height=120, kind="Figure")
        out = consolidate_blocks([caption, figure], page_width=WIDE)

        self.assertEqual([chunk.block_type for chunk in out], ["FigureWithCaption"])

    def test_a_caption_is_not_absorbed_from_another_column(self):
        # Above it in reading order is not the same as on top of it: a caption at
        # the foot of the left column and a table at the head of the right one are
        # not one object, and joining them draws a box over both columns.
        caption = text_block("Table 9: something else", top=600, left=50, width=200, height=12, kind="Caption")
        table = text_block("| a | b |\n| --- | --- |\n| 1 | 2 |", top=60, left=320, width=200, height=80, kind="Table")
        out = consolidate_blocks([caption, table], page_width=WIDE)

        self.assertEqual([chunk.block_type for chunk in out], ["Caption", "Table"])

    def test_a_title_only_absorbs_the_body_directly_below_it(self):
        # A full-width heading over a two-column region: the left column starts
        # below it and is absorbed; the right column's first block starts at the
        # same height as the left one, so it is not "below" and does not fold in.
        title = text_block("2. Method", top=100, left=0, width=400, height=20, kind="Title")
        left = text_block("左栏的第一段正文", top=130, left=0, width=180, height=60)
        right = text_block("右栏的第一段正文", top=130, left=220, width=180, height=60)
        out = consolidate_blocks([title, left, right], page_width=WIDE)

        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].block_type, "Section")
        self.assertIn("左栏的第一段正文", out[0].text)
        self.assertNotIn("右栏的第一段正文", out[0].text)


if __name__ == "__main__":
    unittest.main()
