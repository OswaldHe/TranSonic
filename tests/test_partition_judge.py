# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LLM judge: verdict parsing, CLI invocation, offline stub."""

import json
import subprocess

import pytest

from model_partition.verify.judge import (
    ClaudeJudge,
    JudgeError,
    StubJudge,
    Verdict,
    _extract_text,
    build_judge,
    parse_verdict,
)


# -- verdict parsing ---------------------------------------------------------


def test_parse_clean_json():
    verdict = parse_verdict('{"fluent": true, "score": 5, "reason": "reads well"}', "s0")
    assert verdict.fluent and verdict.score == 5
    assert verdict.reason == "reads well"
    assert verdict.sample_id == "s0"
    assert verdict.passed()


def test_parse_json_surrounded_by_prose():
    """Models often add a sentence before the JSON."""
    text = 'Here is my assessment:\n{"fluent": false, "score": 2, "reason": "repetitive"}\nDone.'
    verdict = parse_verdict(text)
    assert not verdict.fluent and verdict.score == 2
    assert not verdict.passed()


def test_parse_missing_json_is_an_error_not_a_pass():
    verdict = parse_verdict("I think it looks fine, honestly.")
    assert verdict.error and "no JSON object" in verdict.error
    assert not verdict.passed()
    assert verdict.score == 0


def test_parse_malformed_json_is_an_error():
    verdict = parse_verdict('{"fluent": true, "score": }')
    assert "invalid JSON" in verdict.error
    assert not verdict.passed()


def test_parse_json_array_is_an_error():
    verdict = parse_verdict("[1, 2, 3]")
    assert verdict.error and not verdict.passed()


def test_score_is_clamped_to_the_scale():
    assert parse_verdict('{"fluent": true, "score": 99}').score == 5
    assert parse_verdict('{"fluent": true, "score": -4}').score == 0


def test_non_numeric_score_becomes_zero():
    verdict = parse_verdict('{"fluent": true, "score": "great"}')
    assert verdict.score == 0 and not verdict.passed()


def test_missing_fields_default_to_failing():
    verdict = parse_verdict("{}")
    assert not verdict.fluent and verdict.score == 0


def test_long_reason_is_truncated():
    verdict = parse_verdict(json.dumps({"fluent": True, "score": 4, "reason": "x" * 2000}))
    assert len(verdict.reason) <= 500


def test_min_score_gate():
    verdict = Verdict(fluent=True, score=3)
    assert not verdict.passed(min_score=4)
    assert verdict.passed(min_score=3)


def test_verdict_with_error_never_passes():
    assert not Verdict(fluent=True, score=5, error="timeout").passed()


def test_verdict_serializes():
    payload = Verdict(fluent=True, score=4, reason="ok", sample_id="s1").to_dict()
    assert payload == {"sample_id": "s1", "fluent": True, "score": 4, "reason": "ok", "error": ""}


# -- CLI output extraction ---------------------------------------------------


def test_extract_text_from_result_field():
    stdout = json.dumps({"type": "result", "result": '{"fluent": true, "score": 5}'})
    assert '"fluent": true' in _extract_text(stdout)


def test_extract_text_from_content_blocks():
    stdout = json.dumps({"content": [{"text": "part one "}, {"text": "part two"}]})
    assert _extract_text(stdout) == "part one part two"


def test_extract_text_passes_through_plain_output():
    assert _extract_text("not json at all") == "not json at all"


# -- ClaudeJudge -------------------------------------------------------------


def test_missing_cli_is_reported_not_raised(monkeypatch):
    judge = ClaudeJudge(command="/nonexistent/claude")
    monkeypatch.setattr("shutil.which", lambda _: None)
    verdict = judge.judge("prompt", "continuation", "s0")
    assert "not found" in verdict.error
    assert not verdict.passed()


def test_judge_model_prefers_explicit_then_env(monkeypatch):
    monkeypatch.delenv("MODEL_PARTITION_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "env-sonnet")
    assert ClaudeJudge()._model() == "env-sonnet"
    assert ClaudeJudge(model="explicit")._model() == "explicit"
    monkeypatch.setenv("MODEL_PARTITION_JUDGE_MODEL", "override")
    assert ClaudeJudge()._model() == "override"


def test_judge_falls_back_to_sonnet_alias(monkeypatch):
    monkeypatch.delenv("MODEL_PARTITION_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_DEFAULT_SONNET_MODEL", raising=False)
    assert ClaudeJudge()._model() == "sonnet"


def test_successful_invocation_builds_the_expected_command(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(
            command, 0,
            stdout=json.dumps({"result": '{"fluent": true, "score": 5, "reason": "good"}'}),
            stderr="",
        )

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(subprocess, "run", fake_run)
    verdict = ClaudeJudge(model="m").judge("the prompt", "the continuation", "s0")
    assert verdict.passed()
    command = captured["command"]
    # Same transport as AutoHelix's Claude backend.
    assert "--print" in command
    assert command[command.index("--output-format") + 1] == "json"
    assert command[command.index("--model") + 1] == "m"
    assert "the continuation" in command[-1]


def test_nonzero_exit_is_reported(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(subprocess, "run", lambda c, **k: subprocess.CompletedProcess(
        c, 2, stdout="", stderr="credentials missing"))
    verdict = ClaudeJudge().judge("p", "c")
    assert "exited 2" in verdict.error and "credentials missing" in verdict.error


def test_timeout_is_reported(monkeypatch):
    def raise_timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 5)

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(subprocess, "run", raise_timeout)
    verdict = ClaudeJudge(timeout=5).judge("p", "c")
    assert "timed out after 5s" in verdict.error


def test_os_error_is_reported(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(subprocess, "run", lambda c, **k: (_ for _ in ()).throw(OSError("boom")))
    assert "judge failed" in ClaudeJudge().judge("p", "c").error


def test_prompt_and_continuation_are_truncated(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["body"] = command[-1]
        return subprocess.CompletedProcess(command, 0, stdout='{"fluent": true, "score": 4}', stderr="")

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(subprocess, "run", fake_run)
    ClaudeJudge().judge("p" * 20000, "c" * 20000)
    assert len(captured["body"]) < 12000


# -- StubJudge ---------------------------------------------------------------


def test_stub_accepts_varied_text():
    verdict = StubJudge().judge("p", "a reasonably varied continuation of some text")
    assert verdict.passed()
    assert "unique ratio" in verdict.reason


def test_stub_rejects_degenerate_repetition():
    verdict = StubJudge().judge("p", "the the the the the the the the the the")
    assert not verdict.passed()
    assert "degenerate repetition" in verdict.reason


def test_stub_rejects_empty_or_single_token():
    assert not StubJudge().judge("p", "").passed()
    assert not StubJudge().judge("p", "word").passed()


def test_stub_needs_no_network(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("stub must not shell out"))
    assert StubJudge().judge("p", "varied words here indeed").passed()


# -- factory -----------------------------------------------------------------


def test_build_judge_selects_backend():
    assert isinstance(build_judge("claude"), ClaudeJudge)
    assert isinstance(build_judge("stub"), StubJudge)
    with pytest.raises(JudgeError, match="Unknown judge kind"):
        build_judge("oracle")


# -- the passing bar ---------------------------------------------------------
#
# The bar is deliberately low: readable, correctly formed, and connected to the
# input. Rough style must not fail a run, because rough style is not evidence of a
# broken partition. These check the prompt states that bar and the parser applies
# it; the prompt's behaviour against a real model is exercised by the slow test.


def test_prompt_states_the_three_tests():
    from model_partition.verify.judge import JUDGE_PROMPT

    lowered = JUDGE_PROMPT.lower()
    assert "human-readable" in lowered
    assert "correctly formed" in lowered
    assert "connected to the input" in lowered


def test_prompt_excuses_terse_reasoning_style():
    from model_partition.verify.judge import JUDGE_PROMPT

    assert "telegraphic" in JUDGE_PROMPT
    assert "truncation" in JUDGE_PROMPT
    assert "<think>" in JUDGE_PROMPT


def test_prompt_names_the_failure_modes():
    from model_partition.verify.judge import JUDGE_PROMPT

    for failure in ("degenerate repetition", "gibberish", "unrelated"):
        assert failure in JUDGE_PROMPT


def test_rough_but_on_topic_output_passes():
    """A 4 means all three tests held with rough style; it must pass."""
    verdict = parse_verdict(json.dumps({
        "fluent": True, "score": 4,
        "reason": "Telegraphic style but clearly addresses the question",
    }))
    assert verdict.passed()


def test_one_failing_test_does_not_pass():
    verdict = parse_verdict(json.dumps({"fluent": False, "score": 2, "reason": "unrelated"}))
    assert not verdict.passed()


def test_missing_fields_do_not_pass():
    assert not parse_verdict("{}").passed()


@pytest.mark.slow
def test_real_judge_applies_the_stated_bar():
    """Terse on-topic output passes; damage does not."""
    judge = ClaudeJudge()
    if not judge.available():
        pytest.skip("claude CLI not available")
    prompt = "Explain why reading a 12 GB CSV with read().splitlines() exhausts memory."
    rough = ("We need answer user's question. Need explain problem: read().splitlines() "
             "loads entire file into memory plus list of lines, huge memory")
    assert judge.judge(prompt, rough).passed()
    assert not judge.judge(prompt, "the the the the the the the the").passed()
    assert not judge.judge(prompt, "Bananas ripen best at room temperature.").passed()


def test_score_four_or_five_passes_and_ends_the_loop():
    """4 and 5 are the passing scores; the loop exits on them."""
    for score in (4, 5):
        assert parse_verdict(json.dumps({"fluent": True, "score": score})).passed()
        # The score is authoritative even if the flag disagrees with it.
        assert parse_verdict(json.dumps({"fluent": False, "score": score})).passed()


def test_score_below_four_does_not_pass():
    for score in (0, 1, 2, 3):
        assert not parse_verdict(json.dumps({"fluent": True, "score": score})).passed()


def test_an_errored_verdict_never_passes_whatever_the_score():
    assert not Verdict(fluent=True, score=5, error="timed out").passed()
