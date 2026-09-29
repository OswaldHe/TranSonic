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

from optimization.report import _GIST_WIDTH, _embed, _slot_enforcement, _slot_gist

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


# -- the enforcement column ----------------------------------------------------------


def _recorded(**over):
    """A slot as `Slot.to_dict` writes it into the checker manifest."""
    slot = {
        "iterations": [4],
        "label": "4",
        "text": "Use tensor_scalar.",
        "enforcement": "soft",
        "soften_last": True,
        "per_iteration": {"4": "soft"},
    }
    slot.update(over)
    return slot


def test_the_enforcement_column_reads_what_the_manifest_writes():
    """It asked for `enforce`, a key `to_dict` has never written, so every slot read as not
    enforced — including the ones that had just rejected a candidate and spent its iteration."""
    assert _slot_enforcement(_recorded(), "Use tensor_scalar.") == "soft"
    assert _slot_enforcement(
        _recorded(enforcement="hard", per_iteration={"4": "hard"}), "x",
    ) == "hard"


def test_the_enforcement_column_names_a_softened_last_iteration():
    slot = _recorded(
        iterations=[1, 2, 3], label="1-3", enforcement="hard",
        per_iteration={"1": "hard", "2": "hard", "3": "soft"},
    )
    assert _slot_enforcement(slot, "x") == "hard, soft on 3"


def test_the_enforcement_column_sorts_iterations_numerically():
    """String keys put "10" before "9", which would report the wrong iteration as the softened one."""
    slot = _recorded(
        iterations=[9, 10], label="9-10", enforcement="hard",
        per_iteration={"9": "hard", "10": "soft"},
    )
    assert _slot_enforcement(slot, "x") == "hard, soft on 10"


def test_an_unconstrained_slot_has_no_enforcement_to_show():
    assert _slot_enforcement(_recorded(), "") == "—"


def test_the_enforcement_column_falls_back_to_a_legacy_record():
    """A manifest written before `per_iteration` existed still renders."""
    assert _slot_enforcement(
        {"label": "1", "text": "x", "enforce": True, "per_iteration": {}}, "x",
    ) == "hard"
    assert _slot_enforcement(
        {"label": "1", "text": "x", "enforce": False, "per_iteration": {}}, "x",
    ) == "off"
