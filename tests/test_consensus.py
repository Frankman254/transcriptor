import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

from consensus import (  # noqa: E402
    ConsensusWord,
    collapse_repetition_loops,
    consensus_words,
    is_filler_line,
    summarize,
    text_stability,
    to_segments,
)


def hypothesis(words, *, start=0.0, step=0.5, probability=0.9):
    """One pass' worth of words on a regular grid."""
    out = []
    time = start
    for word in words:
        out.append({"text": word, "start": time, "end": time + step * 0.8,
                    "probability": probability})
        time += step
    return out


def drop(words, target):
    return [w for w in words if w["text"] != target]


LINE = ["Esta", "noche", "seguimos", "bailando,", "porque", "la",
        "música", "nos", "llama."]


class VotingTests(unittest.TestCase):
    def test_word_only_one_pass_heard_is_dropped(self):
        clean = hypothesis(LINE)
        hallucinating = hypothesis(LINE) + [
            {"text": "gracias", "start": 4.6, "end": 5.0, "probability": 0.4}
        ]
        words, stats = consensus_words([clean, hallucinating, hypothesis(LINE)])
        self.assertNotIn("gracias", [w.text for w in words])
        self.assertEqual(stats.dropped, 1)
        self.assertEqual(stats.looped, 0)

    def test_word_the_majority_heard_survives_one_pass_omitting_it(self):
        full = hypothesis(LINE)
        words, _ = consensus_words([full, full, drop(hypothesis(LINE), "porque")])
        self.assertIn("porque", [w.text for w in words])
        recovered = next(w for w in words if w.text == "porque")
        self.assertEqual(recovered.votes, 2)
        self.assertAlmostEqual(recovered.agreement, 2 / 3)

    def test_spelling_variants_resolve_to_the_majority_surface_form(self):
        # "musica" and "música" are different keys — accents can be phonemic,
        # so the normalizer does not fold them. They still land in the same
        # slot because they sound alike and occur at the same instant, and the
        # slot then votes on which spelling to keep.
        accented = hypothesis(LINE)
        plain = hypothesis([w.replace("música", "musica") for w in LINE])
        words, _ = consensus_words([accented, plain, accented])
        chosen = next(w for w in words if w.text.endswith("sica"))
        self.assertEqual(chosen.text, "música")
        self.assertEqual(chosen.votes, 2)
        self.assertEqual(chosen.variants, ["musica", "música"])
        self.assertTrue(chosen.contested)
        self.assertEqual(len([w for w in words if "sic" in w.text]), 1)

    def test_unanimous_words_report_full_agreement(self):
        full = hypothesis(LINE)
        words, _ = consensus_words([full, full, full])
        self.assertEqual([w.text for w in words], LINE)
        self.assertTrue(all(w.agreement == 1.0 for w in words))
        self.assertFalse(any(w.contested for w in words))

    def test_a_single_pass_keeps_everything_it_heard(self):
        # With one voter there is no majority to appeal to; deleting anything
        # would be the engine inventing a disagreement.
        words, stats = consensus_words([hypothesis(LINE)])
        self.assertEqual([w.text for w in words], LINE)
        self.assertEqual(stats.removed, 0)

    def test_no_passes_yields_nothing(self):
        words, stats = consensus_words([])
        self.assertEqual(words, [])
        self.assertEqual(stats.removed, 0)

    def test_clips_never_overlap_even_when_pass_timings_disagree(self):
        a = hypothesis(LINE, step=0.5)
        b = hypothesis(LINE, start=0.18, step=0.5)
        c = hypothesis(LINE, start=-0.0, step=0.52)
        words, _ = consensus_words([a, b, c])
        for previous, word in zip(words, words[1:]):
            self.assertLessEqual(previous.end, word.start)
            self.assertLess(word.start, word.end)

    def test_repeated_chorus_far_apart_is_not_merged_into_one_slot(self):
        # Identical text three minutes apart is two different lines, and the
        # aligner must not fold them together.
        line = hypothesis(["why", "do", "I"], start=10.0)
        again = hypothesis(["why", "do", "I"], start=190.0)
        passes = [line + again, line + again, line + again]
        words, _ = consensus_words(passes)
        self.assertEqual(len(words), 6)
        self.assertLess(words[2].end, words[3].start)


class RepetitionLoopTests(unittest.TestCase):
    def words(self, tokens):
        return [ConsensusWord(token, i * 0.3, i * 0.3 + 0.25, 3, 3, 0.9, 1.0)
                for i, token in enumerate(tokens)]

    def test_runaway_phrase_is_trimmed_to_a_few_repetitions(self):
        kept, removed = collapse_repetition_loops(
            self.words(["おばあちゃん", "の"] * 20))
        self.assertEqual(len(kept), 8)
        self.assertEqual(removed, 32)

    def test_a_loop_every_pass_agrees_on_is_counted_apart_from_vote_drops(self):
        # Unanimous agreement on garbage is not the same failure as a word
        # only one pass heard, and the report must not blend the two.
        clean = hypothesis(["uno", "dos", "tres"])
        looping = hypothesis(["uno", "dos", "tres"] + ["la"] * 30)
        words, stats = consensus_words([clean, looping, looping])
        self.assertEqual(" ".join(w.text for w in words), "uno dos tres la la la la")
        self.assertEqual(stats.looped, 26)
        self.assertEqual(stats.dropped, 0)
        self.assertEqual(stats.removed, 26)

    def test_a_real_hook_of_five_repetitions_is_left_alone(self):
        tokens = ["why", "do", "I"] * 5
        kept, removed = collapse_repetition_loops(self.words(tokens))
        self.assertEqual(len(kept), len(tokens))
        self.assertEqual(removed, 0)

    def test_ordinary_text_is_untouched(self):
        tokens = LINE
        kept, removed = collapse_repetition_loops(self.words(tokens))
        self.assertEqual([w.text for w in kept], tokens)
        self.assertEqual(removed, 0)


class SegmentTests(unittest.TestCase):
    def test_segments_carry_agreement_and_reconstruct_their_own_text(self):
        full = hypothesis(LINE)
        words, _ = consensus_words([full, full, drop(hypothesis(LINE), "la")])
        segments = to_segments(words)
        self.assertTrue(segments)
        for segment in segments:
            rebuilt = "".join(word["word"] for word in segment["words"]).strip()
            self.assertEqual(rebuilt, segment["text"])
            self.assertIn("agreement", segment)
            self.assertLessEqual(segment["start"], segment["end"])

    def test_japanese_and_latin_mix_in_one_line_keeps_only_the_latin_spaces(self):
        tokens = ["誰", "の", "メモ", "気", "に", "しない", "で",
                  "You'll", "be", "getting", "high"]
        pass_words = hypothesis(tokens, step=0.3)
        words, _ = consensus_words([pass_words] * 3)
        segments = to_segments(words, language="ja")
        self.assertEqual(segments[0]["text"],
                         "誰のメモ気にしないで You'll be getting high")

    def test_a_long_pause_starts_a_new_segment(self):
        early = hypothesis(["uno", "dos"], start=0.0)
        late = hypothesis(["tres", "cuatro"], start=20.0)
        words, _ = consensus_words([early + late] * 3)
        segments = to_segments(words)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["text"], "uno dos")
        self.assertEqual(segments[1]["text"], "tres cuatro")


class ConvergenceTests(unittest.TestCase):
    def test_identical_results_are_perfectly_stable(self):
        words, _ = consensus_words([hypothesis(LINE)] * 3)
        self.assertEqual(text_stability(words, words), 1.0)

    def test_one_changed_word_in_nine_is_measured_as_such(self):
        before, _ = consensus_words([hypothesis(LINE)] * 3)
        changed = [w.replace("llama.", "llega.") for w in LINE]
        after, _ = consensus_words([hypothesis(changed)] * 3)
        self.assertAlmostEqual(text_stability(before, after), 1 - 1 / 9, places=6)

    def test_empty_versus_populated_is_zero_stability(self):
        words, _ = consensus_words([hypothesis(LINE)] * 3)
        self.assertEqual(text_stability([], words), 0.0)
        self.assertEqual(text_stability([], []), 1.0)

    def test_first_round_reports_no_stability_and_later_rounds_do(self):
        words, stats = consensus_words([hypothesis(LINE)] * 3)
        first = summarize(words, stats, 3, None)
        self.assertIsNone(first.stability)
        self.assertEqual(first.passes, 3)
        second = summarize(words, stats, 4, words)
        self.assertEqual(second.stability, 1.0)
        self.assertEqual(second.as_dict()["stability"], 1.0)


class FillerPhraseTests(unittest.TestCase):
    """Whisper's subtitle reflexes survive voting: every pass emits them."""

    def test_the_common_caption_signoffs_are_recognised(self):
        for phrase in ["ご視聴ありがとうございました", "Thank you.",
                       "字幕をご視聴頂きましてありがとうございました。",
                       "thanks for watching", "Bye"]:
            self.assertTrue(is_filler_line(phrase), phrase)

    def test_a_real_lyric_containing_those_words_is_left_alone(self):
        for phrase in ["Thank you for the love you gave me",
                       "おばあちゃんの", "Say goodbye, my love",
                       "Esta noche seguimos bailando"]:
            self.assertFalse(is_filler_line(phrase), phrase)

    def test_a_filler_line_never_reaches_the_segments(self):
        real = hypothesis(["Esta", "noche", "bailamos"], start=0.0)
        filler = hypothesis(["Thank", "you."], start=20.0)
        words, _ = consensus_words([real + filler] * 3)
        texts = [segment["text"] for segment in to_segments(words)]
        self.assertEqual(texts, ["Esta noche bailamos"])

    def test_segment_ids_stay_contiguous_after_a_line_is_dropped(self):
        real = hypothesis(["uno", "dos"], start=0.0)
        filler = hypothesis(["Thank", "you."], start=20.0)
        more = hypothesis(["tres", "cuatro"], start=40.0)
        words, _ = consensus_words([real + filler + more] * 3)
        segments = to_segments(words)
        self.assertEqual([s["id"] for s in segments], list(range(len(segments))))
        self.assertEqual([s["text"] for s in segments], ["uno dos", "tres cuatro"])


if __name__ == "__main__":
    unittest.main()
