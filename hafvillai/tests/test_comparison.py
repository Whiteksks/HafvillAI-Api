"""CPU regressions for comparison decoding, stopping, budgets, and metrics."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hafvillai.compare import generate_tokens, heldout_items
from hafvillai.comparison_metrics import (output_budget, repetition_metrics, text_metrics,
                                         token_kinds, validate_cases, visible_text)


class ScriptedModel:
    def __init__(self):
        self.offsets = []

    def __call__(self, ids, *, position_ids, inference_params, num_last_tokens):
        self.offsets.append((inference_params.seqlen_offset, ids.shape[1]))
        index = inference_params.key_value_memory_dict.get("index", 0)
        logits = torch.full((1, 1, 8), -100.0)
        logits[0, 0, (2, 3, 4, 5)[min(index, 3)]] = 100
        inference_params.key_value_memory_dict["index"] = index + 1
        return SimpleNamespace(logits=logits)


class ComparisonTests(unittest.TestCase):
    def test_prepared_row_numbers_are_one_based(self):
        rows = [{"name": "first"}, {"name": "second"}, {"name": "third"}]
        items = [{"row": 3, "input_ids": torch.zeros(10)}, {"row": 1, "input_ids": torch.zeros(10)}]
        manifest = {"signature": {"dataset_sha256": "test"}}
        config = SimpleNamespace(path=lambda name: Path("unused"))
        with patch("hafvillai.compare.load_prepared", return_value=(manifest, [], items)), \
                patch("hafvillai.compare.read_rows", return_value=rows):
            info, selected = heldout_items(config, 2, 2048)
        self.assertEqual(info["selected_rows_one_based"], [1, 3])
        self.assertEqual([row["name"] for item, row in selected], ["first", "third"])

    def test_generation_uses_fresh_state_and_accepts_multiple_stop_tokens(self):
        model = ScriptedModel()
        ids = torch.tensor([[6, 7, 6]])
        for _ in range(2):
            self.assertEqual(generate_tokens(model, ids, 8, {4, 7}, "greedy", 42), [2, 3, 4])
        self.assertEqual(model.offsets, [(0, 3), (3, 1), (4, 1)] * 2)

    def test_generation_budget_excludes_prompt(self):
        model = ScriptedModel()
        self.assertEqual(generate_tokens(model, torch.tensor([[6, 7]]), 2, {7}, "greedy", 42), [2, 3])
        self.assertEqual(len(model.offsets), 2)

    def test_penalty_affects_output_history_and_preserves_prompt_tokens(self):
        template = torch.tensor([[[0., 0., 4., -4., 0., 0., 6., 0.]]])
        model = lambda *args, **kwargs: SimpleNamespace(logits=template)
        calls = []
        def sample(logits, **kwargs):
            calls.append(logits.clone())
            return torch.tensor([len(calls)+1])
        with patch("mamba_ssm.utils.generation.sample", side_effect=sample):
            tokens = generate_tokens(model, torch.tensor([[6]]), 8, {4}, "greedy", 42, 1.2)
        self.assertEqual(tokens, [2, 3, 4])
        self.assertEqual(calls[0][0, 2].item(), 4.)
        self.assertAlmostEqual(calls[1][0, 2].item(), 4./1.2, places=6)
        self.assertAlmostEqual(calls[2][0, 3].item(), -4.*1.2, places=6)
        self.assertTrue(all(call[0, 6].item() == 6. for call in calls))
        self.assertEqual(template[0, 0, 2].item(), 4.)

    def test_total_budget_and_no_input_truncation(self):
        self.assertEqual(output_budget(100, 2048), 1948)
        self.assertEqual(output_budget(100, 2048, 256), 256)
        self.assertEqual(output_budget(2040, 2048, 256), 8)
        for length in (0, 2048, 3000):
            with self.assertRaises(ValueError):
                output_budget(length, 2048)

    def test_periodic_repetition_is_distinct_from_short_answers(self):
        metrics = repetition_metrics("a b c a b c a b c".split())
        self.assertEqual(metrics["longest_consecutive_repeat_words"], 9)
        self.assertEqual(metrics["repeat_period_words"], 3)
        self.assertEqual(repetition_metrics(["hello", "Mara"])["duplicate_4gram_fraction"], 0)
        self.assertEqual(repetition_metrics(["x"] * 30)["longest_consecutive_repeat_words"], 30)

    def test_closed_and_unfinished_planning_blocks(self):
        visible, unfinished = visible_text("<system>Plan.</system>Mara: Hello.<|eot_id|>")
        self.assertEqual(visible, "Mara: Hello.")
        self.assertFalse(unfinished)
        self.assertEqual(visible_text("<system_thinking>Still planning"), ("", True))
        metrics = text_metrics("<system_thinking>Mara</system_thinking>Hi.", ["Mara"])
        self.assertFalse(metrics["expected_mentions_in_visible_text"]["Mara"])
        self.assertTrue(text_metrics("<screen>Character Analysis: I am a teacher.")["planning_marker_present"])
        self.assertTrue(text_metrics("<system info>Context: Alex asks my name.")["planning_marker_present"])

    def test_nested_role_blocks_are_not_double_counted(self):
        text = "<system_thinking><role_thinking>x</role_thinking></system_thinking>Hi"
        kinds = token_kinds(text, [(i, i + 1) for i in range(len(text))])
        self.assertEqual(kinds.count("role_thinking"), 0)
        self.assertEqual(kinds[-2:], ["other", "other"])

    def test_fixture_and_path_validation(self):
        path = Path(__file__).resolve().parents[1] / "examples/coser_checks.json"
        for cases in json.loads(path.read_text()).values():
            validate_cases(cases)
        with self.assertRaises(ValueError):
            validate_cases([{"id": "../escape", "format": "raw", "prompt": "Hello"}])
        with self.assertRaises(ValueError):
            validate_cases([{"id": "x", "format": "chat", "messages": [{"role": "assistant", "content": "Hello"}]}])


if __name__ == "__main__":
    unittest.main()
