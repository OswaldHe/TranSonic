# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stripping an earlier agent's prose out of the code a later agent reads.

Two properties matter and both are easy to get wrong. The result has to still *parse* — a stripped
file is code the next agent may run, and a docstring-only function body becomes a syntax error if
the docstring is simply deleted. And the formatting has to survive: `tokenize.untokenize` renders
`import nki.language` as `import nki .language`, and code that looks mechanically mangled invites an
agent to fix the formatting instead of the kernel.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from optimization.strip import BANNER, STRIPPED_FILES, strip_file, strip_source, strip_tree

pytestmark = pytest.mark.optimization


def test_comments_and_docstrings_both_go():
    source = textwrap.dedent('''
        """Module docstring with a wrong claim about the hardware."""
        import nki

        # A comment block asserting something untested.
        # Second line of it.
        X = 5  # trailing claim


        def f(a):
            """What this does, allegedly."""
            return a + X
    ''')
    result, comments, docstrings = strip_source(source)
    assert comments == 3
    assert docstrings == 2
    assert "wrong claim" not in result
    assert "untested" not in result
    assert "allegedly" not in result
    assert "X = 5" in result
    assert "return a + X" in result
    ast.parse(result)


def test_formatting_is_preserved_exactly():
    """The reason this is position-based rather than a token round-trip."""
    source = "import nki.language as nl\nDIM = 5120  # the model dim\nY = DIM * 2\n"
    result, _, _ = strip_source(source)
    assert "import nki.language as nl" in result
    assert "DIM = 5120" in result
    assert "nki .language" not in result
    assert "DIM =5120" not in result


def test_a_docstring_only_body_keeps_a_body():
    """Deleting the docstring outright would leave `def f():` with nothing under it."""
    source = textwrap.dedent('''
        def f():
            """Only a docstring."""

        class C:
            """Also only a docstring."""
    ''')
    result, _, docstrings = strip_source(source)
    assert docstrings == 2
    assert result.count("pass") == 2
    ast.parse(result)


def test_indentation_of_the_substituted_pass_matches():
    source = 'class C:\n    def f(self):\n        """doc"""\n'
    result, _, _ = strip_source(source)
    assert "\n        pass" in result
    ast.parse(result)


def test_a_hash_inside_a_string_is_not_a_comment():
    """The reason `tokenize` finds the comments rather than a regex."""
    source = 'MARKER = "##autohelix[latency_ms=1]"  # a real comment\nP = "# not a comment"\n'
    result, comments, _ = strip_source(source)
    assert comments == 1
    assert "##autohelix[latency_ms=1]" in result
    assert "# not a comment" in result
    assert "a real comment" not in result


def test_a_multiline_docstring_goes_entirely():
    source = textwrap.dedent('''
        def f():
            """First line.

            A long explanation spanning
            several lines, possibly wrong.
            """
            return 1
    ''')
    result, _, docstrings = strip_source(source)
    assert docstrings == 1
    assert "possibly wrong" not in result
    assert "several lines" not in result
    assert "return 1" in result
    ast.parse(result)


def test_runs_of_blank_lines_are_collapsed():
    """A removed comment block otherwise leaves a hole where it used to be."""
    source = "A = 1\n# one\n# two\n# three\n# four\nB = 2\n"
    result, _, _ = strip_source(source)
    assert "\n\n\n" not in result


def test_a_file_that_does_not_parse_is_left_alone(tmp_path):
    """Loud, but not fatal: the point is to remove a hazard, not to block the run."""
    path = tmp_path / "source.py"
    path.write_text("def broken(:\n")
    result = strip_file(path)
    assert not result.ok
    assert "source.py" in result.describe() and "left as-is" in result.describe()
    assert path.read_text() == "def broken(:\n"


def test_a_missing_file_is_reported_not_raised(tmp_path):
    result = strip_file(tmp_path / "source.py")
    assert not result.ok and "not present" in result.skipped


def test_the_banner_says_what_happened(tmp_path):
    """Without it a kernel with no commentary looks accidental, and an agent may restore it."""
    path = tmp_path / "source.py"
    path.write_text("# a claim\nX = 1\n")
    strip_file(path)
    text = path.read_text()
    assert text.startswith(BANNER)
    assert "a claim" not in text
    # And the banner itself makes no claim about the hardware, so it cannot become the next
    # wrong fact.
    for word in ("NKI", "SBUF", "HBM", "fp8", "TFLOPS", "faster", "slower"):
        assert word not in BANNER


def test_only_source_and_inference_are_touched(tmp_path):
    """`reference_*.py` and `vendor/` are the specification and keep their comments."""
    (tmp_path / "reference").mkdir()
    (tmp_path / "module").mkdir()
    (tmp_path / "module" / "source.py").write_text("# agent claim\nX = 1\n")
    (tmp_path / "module" / "inference.py").write_text("# agent claim\nY = 2\n")
    (tmp_path / "reference" / "reference_torch.py").write_text("# authoritative\nZ = 3\n")
    (tmp_path / "reference" / "reference_numerics.py").write_text("# authoritative\nW = 4\n")

    results = strip_tree(tmp_path)
    assert {r.path.name for r in results} == set(STRIPPED_FILES)
    assert "agent claim" not in (tmp_path / "module" / "source.py").read_text()
    assert "authoritative" in (tmp_path / "reference" / "reference_torch.py").read_text()
    assert "authoritative" in (tmp_path / "reference" / "reference_numerics.py").read_text()


def test_the_real_bootstrapped_moe_kernel_strips_and_still_parses():
    """The file this was written for. A stripped kernel an agent cannot run is worse than nothing."""
    import shutil

    repo = Path("/home/ubuntu/workspace/bootstrap-runs/01-MoE")
    if not (repo / "source.py").is_file():
        pytest.skip("the local 01-MoE bootstrap repo is not present")

    import tempfile

    tmp = Path(tempfile.mkdtemp())
    for name in STRIPPED_FILES:
        shutil.copy2(repo / name, tmp / name)
        result = strip_file(tmp / name)
        assert result.ok, result.describe()
        assert result.comments > 0 and result.docstrings > 0
        ast.parse((tmp / name).read_text())
    # The real kernel's own claims are gone, and the code that matters is not.
    stripped = (tmp / "source.py").read_text()
    assert "SBUF" not in stripped
    assert "def kernel" in stripped
