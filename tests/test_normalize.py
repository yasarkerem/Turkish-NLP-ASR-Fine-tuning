import pytest

from turkish_asr.metrics import compute_pair_metrics, duration_bucket
from turkish_asr.normalize import (
    TurkishNormalizer,
    contains_turkish_specific,
    turkish_lower,
)


class TestTurkishLower:
    def test_dotted_capital_i_lowers_to_plain_i(self):
        assert turkish_lower("İSTANBUL") == "istanbul"

    def test_ascii_capital_i_lowers_to_dotless_i(self):
        assert turkish_lower("ISPARTA") == "ısparta"

    def test_mixed_word(self):
        assert turkish_lower("DİYARBAKIR") == "diyarbakır"

    def test_igdir(self):
        assert turkish_lower("IĞDIR") == "ığdır"

    def test_all_four_i_forms(self):
        assert turkish_lower("İIıi") == "iııi"

    def test_python_default_would_be_wrong(self):
        # The exact failure mode this function exists for.
        assert "İSTANBUL".lower() != "istanbul"
        assert "ISPARTA".lower() == "isparta"  # wrong for Turkish

    def test_combining_dot_artifact_is_repaired(self):
        # "İstanbul".lower() via default Unicode rules leaves i + U+0307.
        broken = "İstanbul".lower()
        assert "̇" in broken
        assert turkish_lower(broken) == "istanbul"

    def test_other_turkish_letters(self):
        assert turkish_lower("ÇĞÖŞÜ") == "çğöşü"

    def test_idempotent(self):
        for text in ["İSTANBUL", "Iğdır'ın", "ÇÖREK ve şeker", "IIİİııii"]:
            once = turkish_lower(text)
            assert turkish_lower(once) == once


class TestNormalizer:
    norm = TurkishNormalizer()

    def test_punctuation_stripped(self):
        assert self.norm("Merhaba, dünya...") == "merhaba dünya"

    def test_apostrophe_joins_suffix(self):
        assert self.norm("Ankara'da yaşıyor!") == "ankarada yaşıyor"

    def test_curly_apostrophe(self):
        assert self.norm("İstanbul’un tarihi") == "istanbulun tarihi"

    def test_whitespace_collapsed(self):
        assert self.norm("  çok\t\tgüzel \n bir gün ") == "çok güzel bir gün"

    def test_case_and_punct_together(self):
        assert self.norm('"IŞIK; ılık süt iç!"') == "ışık ılık süt iç"

    def test_symbols_become_spaces(self):
        assert self.norm("%50 indirim + kargo") == "50 indirim kargo"

    def test_idempotent(self):
        for text in ["Ankara'da!", "İZMİR — 35", "Şu ışık: IŞIK mı?"]:
            once = self.norm(text)
            assert self.norm(once) == once

    def test_options_can_be_disabled(self):
        keep_case = TurkishNormalizer(lowercase=False)
        assert keep_case("İyi Günler!") == "İyi Günler"


class TestContainsTurkishSpecific:
    def test_positive(self):
        for text in ["çay", "ığdır", "ŞEKER", "göl", "ütü", "İz"]:
            assert contains_turkish_specific(text)

    def test_negative(self):
        assert not contains_turkish_specific("merhaba dostum")
        assert not contains_turkish_specific("The quick brown fox")


class TestMetrics:
    def test_raw_and_norm_differ_on_case_and_punct(self):
        m = compute_pair_metrics(["Merhaba Dünya!"], ["merhaba dünya"])
        assert m.wer_raw == 1.0
        assert m.wer_norm == 0.0

    def test_perfect_hypothesis(self):
        m = compute_pair_metrics(["bir iki üç"], ["bir iki üç"])
        assert m.wer_raw == 0.0
        assert m.cer_norm == 0.0

    def test_dotless_i_confusion_is_an_error_after_normalization(self):
        # A model writing dotted i where Turkish needs dotless must be
        # penalized even after normalization.
        m = compute_pair_metrics(["ılık"], ["ilik"])
        assert m.wer_norm == 1.0
        assert 0 < m.cer_norm < 1

    def test_empty_norm_reference_skipped_not_crashed(self):
        m = compute_pair_metrics(["...", "iyi"], ["a", "iyi"])
        assert m.n_skipped_empty_norm == 1
        assert m.wer_norm == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_pair_metrics(["a"], ["a", "b"])


class TestDurationBucket:
    @pytest.mark.parametrize(
        "seconds,bucket",
        [(0.5, "<3s"), (2.99, "<3s"), (3.0, "3-6s"), (5.99, "3-6s"), (6.0, "6-10s"), (9.99, "6-10s"), (10.0, ">10s"), (42.0, ">10s")],
    )
    def test_boundaries(self, seconds, bucket):
        assert duration_bucket(seconds) == bucket
