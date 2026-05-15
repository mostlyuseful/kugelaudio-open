import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kugelaudio_open.utils.chunking import ChunkPlan, split_text_into_chunks


class SplitTextIntoChunksTests(unittest.TestCase):
    def test_rejects_empty_text(self):
        with self.assertRaisesRegex(ValueError, "Text input is required"):
            split_text_into_chunks("   ", 10)

    def test_rejects_non_positive_max_words(self):
        with self.assertRaisesRegex(ValueError, "max_words_per_chunk"):
            split_text_into_chunks("hello world", 0)

    def test_short_text_stays_single_chunk(self):
        plan = split_text_into_chunks("Hello world.", 10)
        self.assertEqual(
            plan,
            ChunkPlan(
                chunks=["Hello world."],
                boundary_types=["sentence"],
                overlap_prefixes=[""],
                new_texts=["Hello world."],
            ),
        )

    def test_splits_on_sentence_boundaries(self):
        plan = split_text_into_chunks("One two. Three four. Five six.", 4)
        self.assertEqual(plan.chunks, ["One two. Three four.", "Five six."])
        self.assertEqual(plan.boundary_types, ["sentence", "sentence"])
        self.assertEqual(plan.overlap_prefixes, ["", ""])
        self.assertEqual(plan.new_texts, ["One two. Three four.", "Five six."])

    def test_falls_back_to_clause_boundaries(self):
        text = "Alpha beta gamma, delta epsilon zeta, eta theta iota."
        plan = split_text_into_chunks(text, 4)
        self.assertEqual(plan.chunks, ["Alpha beta gamma,", "delta epsilon zeta,", "eta theta iota."])
        self.assertEqual(plan.boundary_types, ["clause", "clause", "clause"])

    def test_hard_wraps_unpunctuated_text(self):
        text = "one two three four five six seven"
        plan = split_text_into_chunks(text, 3)
        self.assertEqual(plan.chunks, ["one two three", "four five six", "seven"])
        self.assertEqual(plan.boundary_types, ["hard_wrap", "hard_wrap", "hard_wrap"])

    def test_hard_wraps_oversized_clause(self):
        text = "one two three four five six, seven eight"
        plan = split_text_into_chunks(text, 3)
        self.assertEqual(plan.chunks, ["one two three", "four five six,", "seven eight"])
        self.assertEqual(plan.boundary_types, ["hard_wrap", "hard_wrap", "hard_wrap"])

    def test_rejects_negative_overlap_sentences(self):
        with self.assertRaisesRegex(ValueError, "overlap_sentences"):
            split_text_into_chunks("hello world", 10, overlap_sentences=-1)

    def test_overlap_disabled_preserves_existing_chunks(self):
        baseline = split_text_into_chunks("One two. Three four. Five six.", 4)
        overlapped = split_text_into_chunks("One two. Three four. Five six.", 4, overlap_sentences=0)
        self.assertEqual(overlapped, baseline)

    def test_overlap_adds_previous_sentence_prefix(self):
        plan = split_text_into_chunks("One two. Three four. Five six.", 4, overlap_sentences=1)
        self.assertEqual(plan.chunks, ["One two. Three four.", "Three four. Five six."])
        self.assertEqual(plan.overlap_prefixes, ["", "Three four."])
        self.assertEqual(plan.new_texts, ["One two. Three four.", "Five six."])

    def test_overlap_is_bounded_by_available_sentences(self):
        plan = split_text_into_chunks(
            "One two. Three four. Five six.",
            2,
            overlap_sentences=2,
        )
        self.assertEqual(
            plan.overlap_prefixes,
            ["", "One two.", "Three four."],
        )
        self.assertEqual(
            plan.chunks,
            [
                "One two.",
                "One two. Three four.",
                "Three four. Five six.",
            ],
        )


if __name__ == "__main__":
    unittest.main()
