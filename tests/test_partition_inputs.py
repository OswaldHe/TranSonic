# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for sample-input loading, tokenization, and the bundled input sets."""

import json
from pathlib import Path

import pytest

from model_partition.inputs import (
    InputError,
    SampleInput,
    load_input_set,
    read_jsonl,
    summarize,
    tokenize_samples,
)
from model_partition.tokenization import (
    ByteTokenizer,
    TokenizerError,
    _as_ids,
    load_tokenizer,
)

INPUTS_DIR = Path(__file__).resolve().parents[1] / "partition" / "inputs"


class WordTokenizer:
    """One id per word, so token counts are predictable in tests."""

    eos_token_id = 0

    def encode(self, text, role="raw"):
        return [len(word) for word in text.split()]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


# -- jsonl reading -----------------------------------------------------------


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('\n{"id": "a", "prompt": "x"}\n\n{"id": "b", "prompt": "y"}\n')
    assert [r["id"] for r in read_jsonl(path)] == ["a", "b"]


def test_read_jsonl_rejects_a_comment_line(tmp_path):
    """JSONL is JSON: a comment is a malformed line, not something to skip."""
    path = tmp_path / "in.jsonl"
    path.write_text('# a comment\n{"id": "a", "prompt": "x"}\n')
    with pytest.raises(InputError, match=r"in\.jsonl:1 is not valid JSON"):
        read_jsonl(path)


def test_committed_input_sets_are_pure_jsonl():
    """The bundled sets must parse with any JSON reader, comments included nowhere."""
    import json
    from pathlib import Path

    inputs = Path(__file__).resolve().parent.parent / "partition" / "inputs"
    for name in ("short.jsonl", "long.jsonl"):
        for line in (inputs / name).read_text().splitlines():
            if line.strip():
                assert isinstance(json.loads(line), dict)


def test_read_jsonl_reports_the_bad_line_number(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('{"id": "a", "prompt": "x"}\n{broken\n')
    with pytest.raises(InputError, match=r"in\.jsonl:2 is not valid JSON"):
        read_jsonl(path)


def test_read_jsonl_rejects_non_objects(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text("[1, 2, 3]\n")
    with pytest.raises(InputError, match="must be a JSON object"):
        read_jsonl(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(InputError, match="Input file not found"):
        read_jsonl(tmp_path / "nope.jsonl")


# -- tokenization ------------------------------------------------------------


def test_prompts_are_tokenized_and_ids_recorded():
    records = [{"id": "a", "prompt": "one two three"}]
    samples = tokenize_samples(records, WordTokenizer())
    assert samples[0].id == "a"
    assert samples[0].n_tokens == 3
    assert not samples[0].is_long


def test_target_tokens_truncates_a_long_prompt():
    records = [{"id": "a", "prompt": " ".join(["word"] * 100), "target_tokens": 10}]
    assert tokenize_samples(records, WordTokenizer())[0].n_tokens == 10


def test_target_tokens_extends_a_short_prompt():
    records = [{"id": "a", "prompt": "aa bb", "target_tokens": 20}]
    sample = tokenize_samples(records, WordTokenizer())[0]
    assert sample.n_tokens == 20
    assert sample.target_tokens == 20


def test_padding_can_be_disabled():
    records = [{"id": "a", "prompt": "aa bb", "target_tokens": 20}]
    assert tokenize_samples(records, WordTokenizer(), pad_to_target=False)[0].n_tokens == 2


def test_empty_prompts_are_skipped():
    records = [{"id": "a", "prompt": "   "}, {"id": "b", "prompt": "real text"}]
    assert [s.id for s in tokenize_samples(records, WordTokenizer())] == ["b"]


def test_limit_caps_the_sample_count():
    records = [{"id": str(i), "prompt": "a b"} for i in range(10)]
    assert len(tokenize_samples(records, WordTokenizer(), limit=3)) == 3


def test_text_field_is_accepted_as_an_alias():
    assert tokenize_samples([{"id": "a", "text": "x y"}], WordTokenizer())[0].n_tokens == 2


def test_ids_default_to_positional_names():
    assert tokenize_samples([{"prompt": "a b"}], WordTokenizer())[0].id == "sample-000"


def test_long_threshold():
    assert SampleInput(id="s", prompt="", token_ids=[0] * 1023).is_long is False
    assert SampleInput(id="s", prompt="", token_ids=[0] * 1024).is_long is True


def test_sample_tensor_shape():
    pytest.importorskip("torch")
    tensor = SampleInput(id="s", prompt="", token_ids=[1, 2, 3]).tensor()
    assert tuple(tensor.shape) == (1, 3)


def test_sample_dict_truncates_the_prompt_preview():
    payload = SampleInput(id="s", prompt="x" * 1000, token_ids=[1]).to_dict()
    assert len(payload["prompt_preview"]) == 200
    assert payload["n_tokens"] == 1


# -- input sets --------------------------------------------------------------


def test_load_input_set_combines_short_and_long(tmp_path):
    short = tmp_path / "short.jsonl"
    short.write_text(json.dumps({"id": "s", "prompt": "a b c"}))
    long = tmp_path / "long.jsonl"
    long.write_text(json.dumps({"id": "l", "prompt": "x " * 2000, "target_tokens": 2048}))
    samples = load_input_set(short, long, WordTokenizer())
    assert [s.id for s in samples] == ["s", "l"]
    assert samples[1].is_long


def test_load_input_set_tolerates_an_absent_long_file(tmp_path):
    short = tmp_path / "short.jsonl"
    short.write_text(json.dumps({"id": "s", "prompt": "a b"}))
    assert len(load_input_set(short, tmp_path / "missing.jsonl", WordTokenizer())) == 1


def test_empty_input_set_is_an_error(tmp_path):
    short = tmp_path / "short.jsonl"
    short.write_text("")
    with pytest.raises(InputError, match="Input set is empty"):
        load_input_set(short, None, WordTokenizer())


def test_summarize_describes_both_bands():
    samples = [SampleInput(id="a", prompt="", token_ids=[0] * 128),
               SampleInput(id="b", prompt="", token_ids=[0] * 8192)]
    text = summarize(samples)
    assert "2 sample(s)" in text and "short: 1" in text and "long: 1" in text


# -- bundled sets ------------------------------------------------------------


def test_bundled_short_set_holds_complete_instructions():
    """No target_tokens: pinning a length truncates these mid-sentence, which
    would have the judge assess a broken instruction."""
    records = read_jsonl(INPUTS_DIR / "short.jsonl")
    assert len(records) >= 5
    assert not any("target_tokens" in r for r in records)
    assert len({r["id"] for r in records}) == len(records)
    # A mixture of chat-templated and raw continuation prompts.
    assert {r["role"] for r in records} >= {"user", "raw"}
    for record in records:
        # Each prompt is long enough to be a real instruction and ends cleanly.
        assert 90 <= len(record["prompt"].split()) <= 200, record["id"]
        assert record["prompt"].rstrip()[-1] in ".?\"" or record["role"] == "raw"


def test_bundled_long_set_pins_lengths_where_the_tail_is_filler():
    path = INPUTS_DIR / "long.jsonl"
    if not path.is_file():
        pytest.skip("long.jsonl not generated")
    records = read_jsonl(path)
    assert all(r["target_tokens"] >= 2048 for r in records)


def test_long_generator_produces_the_requested_budgets(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "long.jsonl"
    completed = subprocess.run(
        [sys.executable, str(INPUTS_DIR / "fetch_long_inputs.py"),
         "--out", str(out), "--budgets", "2048", "4096", "--per-budget", "2"],
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    records = read_jsonl(out)
    assert [r["target_tokens"] for r in records] == [2048, 2048, 4096, 4096]
    assert all(r["source"].startswith("synthetic") for r in records)


def test_long_generator_is_reproducible(tmp_path):
    import subprocess
    import sys

    def run(path):
        subprocess.run(
            [sys.executable, str(INPUTS_DIR / "fetch_long_inputs.py"),
             "--out", str(path), "--budgets", "2048", "--seed", "42"],
            capture_output=True, text=True, timeout=300, check=True,
        )
        return path.read_text()

    assert run(tmp_path / "a.jsonl") == run(tmp_path / "b.jsonl")


def test_long_prompts_embed_a_verifiable_needle(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "long.jsonl"
    subprocess.run(
        [sys.executable, str(INPUTS_DIR / "fetch_long_inputs.py"),
         "--out", str(out), "--budgets", "2048"],
        capture_output=True, text=True, timeout=300, check=True,
    )
    record = read_jsonl(out)[0]
    assert record["expected_substring"] in record["prompt"]
    assert "Question:" in record["prompt"]


# -- tokenizer loading -------------------------------------------------------


def test_as_ids_handles_a_flat_list():
    assert _as_ids([1, 2, 3]) == [1, 2, 3]


def test_as_ids_unwraps_a_batch_encoding():
    """Regression: iterating a BatchEncoding yields keys, giving a 2-token prompt."""

    class BatchEncoding(dict):
        @property
        def input_ids(self):
            return [[7, 8, 9]]

    assert _as_ids(BatchEncoding({"input_ids": [[7, 8, 9]], "attention_mask": [[1, 1, 1]]})) == [7, 8, 9]


def test_as_ids_unwraps_a_plain_mapping():
    assert _as_ids({"input_ids": [4, 5], "attention_mask": [1, 1]}) == [4, 5]


def test_as_ids_flattens_a_nested_batch():
    assert _as_ids([[1, 2, 3]]) == [1, 2, 3]


def test_as_ids_rejects_a_mapping_without_input_ids():
    with pytest.raises(TokenizerError, match="without input_ids"):
        _as_ids({"attention_mask": [1]})


def test_as_ids_rejects_an_unusable_type():
    with pytest.raises(TokenizerError, match="unusable type"):
        _as_ids(42)


def test_byte_tokenizer_round_trips():
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("hello")
    assert tokenizer.decode(ids) == "hello"


def test_load_tokenizer_falls_back_to_bytes(tmp_path):
    """A directory with no tokenizer must not block tracing."""
    tokenizer = load_tokenizer(tmp_path, vocab_size=64)
    assert isinstance(tokenizer, ByteTokenizer)
    assert tokenizer.encode("ab")
