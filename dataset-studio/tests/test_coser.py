import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from coser_data import bounded_examples, character_exchanges, clean_utterances, remove_blocks
from training.config import TrainConfig
from hafvillai_runtime.serialization import ChatSerializer
from training.data import group_id, load_prepared, prepare, validation_groups
from training.tests.test_training import Tokenizer


class CoserTests(unittest.TestCase):
    def test_balanced_nested_multiline_thoughts_and_malformed_blocks(self):
        self.assertEqual(remove_blocks("[A [nested]\nthought] Hello. (nods)"), "Hello. (nods)")
        for text in ["[open", "close]", "[balanced] still [open"]:
            with self.assertRaises(ValueError):
                remove_blocks(text)

    def test_environment_and_private_thoughts_do_not_enter_dialogue(self):
        scene = {"dialogues": [{"character": "A", "message": "[Private] Hello."},
                               {"character": "Environment", "message": "[bad formatting"},
                               {"character": "B", "message": "(nods) Hi."}]}
        result = clean_utterances(scene)
        self.assertEqual([r["character"] for r in result], ["A", "B"])
        self.assertEqual(result[1]["speech"], "Hi.")
        self.assertEqual(result[0]["utterance_index"], 0)
        self.assertEqual(result[1]["utterance_index"], 2)
        scene["dialogues"][0]["message"] = "[unclosed"
        with self.assertRaises(ValueError):
            clean_utterances(scene)


    def test_character_views_keep_roles_and_end_on_complete_assistant(self):
        utterances = [{"character": name, "content": text, "utterance_index": i}
                      for i, (name, text) in enumerate([("A", "Hello."), ("A", "More."),
                                                       ("B", "Hi."), ("C", "Question?"),
                                                       ("A", "Answer."), ("B", "Tail.")])]
        pairs = character_exchanges(utterances, "A")
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][0]["content"], "Begin the conversation.")
        self.assertEqual(pairs[0][1]["content"], "A: Hello.\nMore.")
        self.assertEqual(pairs[1][0]["content"], "B: Hi.\nC: Question?")
        for pair in pairs:
            self.assertEqual([m["role"] for m in pair], ["user", "assistant"])


    def test_windows_preserve_whole_exchanges_without_truncation(self):
        serializer = ChatSerializer(Tokenizer())
        utterances = [{"character": "A" if i % 2 else "B", "content": f"Complete turn {i}.",
                       "utterance_index": i} for i in range(12)]
        pairs = character_exchanges(utterances, "A")
        rows, rejected = bounded_examples("Role A.", pairs, serializer, max_length=180, max_reply=80)
        self.assertGreater(len(rows), 1)
        self.assertFalse(rejected)
        expected = [m["content"] for pair in pairs for m in pair]
        actual = [m["content"] for row in rows for m in row["messages"][1:]]
        self.assertEqual(actual, expected)
        for row in rows:
            self.assertLessEqual(len(serializer.encode(row["messages"])[0]), 180)
            self.assertEqual(row["messages"][-1]["role"], "assistant")

    def test_oversized_reply_is_excluded_whole(self):
        serializer = ChatSerializer(Tokenizer())
        utterances = [{"character": "B", "content": "Question.", "utterance_index": 0},
                      {"character": "A", "content": "x" * 300, "utterance_index": 1},
                      {"character": "B", "content": "Next?", "utterance_index": 2},
                      {"character": "A", "content": "Yes.", "utterance_index": 3}]
        rows, rejected = bounded_examples("Role A.", character_exchanges(utterances, "A"),
                                         serializer, max_length=180, max_reply=80)
        self.assertEqual(rejected["oversized_exchange"], 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["utterance_start"], 2)
        self.assertNotIn("x" * 10, str(rows))

    def test_book_groups_prevent_cross_view_and_cross_scene_leakage(self):
        rows = [{"split_group": f"book:{i}", "messages": [
            {"role": "user", "content": f"scene {j}"}, {"role": "assistant", "content": f"view {j}"}]}
                for i in range(10) for j in range(3)]
        validation, groups = validation_groups(rows, 42, .2, "split_group")
        self.assertEqual(len(groups), 10)
        self.assertEqual(len(validation), 2)
        for index in range(10):
            self.assertEqual(len({group_id(r, "consciousness", "split_group") for r in rows[index*3:index*3+3]}), 1)
        with self.assertRaises(ValueError):
            validation_groups([{"split_group": ""}], 42, .2, "split_group")

    def test_preparation_roundtrip_with_explicit_source_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [{"split_group": f"book:{i}", "messages": [
                {"role": "system", "content": "Role."}, {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": "Answer."}]} for i in range(10)]
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps(r) + "\n" for r in rows))
            config = TrainConfig("consciousness", str(source), str(root / "output"),
                                 source_dataset=str(source), prepared_dir=str(root / "prepared"),
                                 split_group_field="split_group", max_seq_length=2048)
            with patch("training.data.load_tokenizer", return_value=Tokenizer()):
                prepare(config, offline=True)
            manifest, train, valid = load_prepared(config)
            self.assertEqual(manifest["signature"]["split_group_field"], "split_group")
            self.assertTrue({r["group"] for r in train}.isdisjoint({r["group"] for r in valid}))


if __name__ == "__main__":
    unittest.main()
