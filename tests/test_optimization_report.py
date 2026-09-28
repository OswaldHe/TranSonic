# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The report's two rendering helpers, both of which mislead when they get a detail wrong.

Neither is arithmetic, so neither can be wrong in a way a number would reveal. A heading spliced in
at the wrong level restructures the document's outline; a table cell cut in the wrong place either
reads as a rendering fault or bleeds emphasis across every row after it. Both shipped once in a
report a human read before being noticed, which is why they are pinned here.
"""

from __future__ import annotations

import pytest

from optimization.report import _GIST_WIDTH, _embed, _slot_gist

pytestmark = pytest.mark.optimization


def test_embed_demotes_a_documents_own_title_below_the_section_holding_it():
    embedded = _embed("# Planned placement\n\ntext\n\n## What it gave up\n")
    assert "### Planned placement" in embedded
    assert "#### What it gave up" in embedded


def test_embed_leaves_an_indented_comment_alone():
    """`describe()` emits indented code blocks, and a `#` inside one is not a heading."""
    assert _embed("    # not a heading\n") == "    # not a heading"


def test_embed_does_not_run_past_the_deepest_heading_level():
    assert _embed("##### five\n", demote=3) == "###### five"


def test_embed_leaves_a_bare_hash_run_alone():
    """ATX headings need the space; `###` on its own is not one."""
    assert _embed("###\n") == "###"


def test_slot_gist_keeps_short_prose_whole():
    assert _slot_gist("**NKI only.** Nothing else.") == "**NKI only.** Nothing else."


def test_slot_gist_marks_the_cut_it_made():
    gist = _slot_gist("word " * 80)
    assert gist.endswith("…")
    assert len(gist) <= _GIST_WIDTH + 3


def test_slot_gist_closes_a_bold_run_the_cut_split():
    """An unpaired `**` bleeds emphasis across the rest of the table, not just its own cell."""
    gist = _slot_gist("**" + "long lead " * 30 + "**and the rest")
    assert gist.count("**") % 2 == 0


def test_slot_gist_escapes_a_pipe_that_would_split_the_row():
    assert _slot_gist("use a | b") == "use a \\| b"


def test_slot_gist_collapses_the_newlines_a_yaml_block_carries():
    assert _slot_gist("first line\n\nsecond line\n") == "first line second line"


def test_slot_gist_reports_empty_prose_as_empty():
    assert _slot_gist("   \n  ") == "—"


def test_slot_gist_distinguishes_two_slots_sharing_a_lead():
    """4-6 and 9-10 both open the same way and differ only in the sentence after it."""
    lead = "**NKI and torch-xla are both allowed.** "
    assert _slot_gist(lead + "Mix them however serves the kernel, " * 4) != \
           _slot_gist(lead + "Consolidate what iterations 7-8 learned, " * 4)
