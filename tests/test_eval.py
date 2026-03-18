"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest

from nirasa.eval.thaiqa import char_f1_score, exact_match_score


class TestCharF1:
    def test_exact_match(self):
        f1 = char_f1_score("กรุงเทพมหานคร", "กรุงเทพมหานคร")
        assert abs(f1 - 1.0) < 1e-6

    def test_partial_overlap(self):
        f1 = char_f1_score("กรุงเทพ", "กรุงเทพมหานคร")
        assert 0.0 < f1 < 1.0

    def test_no_overlap(self):
        f1 = char_f1_score("abc", "xyz")
        assert f1 == 0.0

    def test_both_empty(self):
        f1 = char_f1_score("", "")
        assert f1 == 1.0

    def test_one_empty_prediction(self):
        f1 = char_f1_score("", "กรุงเทพ")
        assert f1 == 0.0

    def test_one_empty_ground_truth(self):
        f1 = char_f1_score("กรุงเทพ", "")
        assert f1 == 0.0

    def test_thai_characters(self):
        f1 = char_f1_score("ประเทศไทย", "ประเทศไทยสวย")
        assert 0.5 < f1 < 1.0

    def test_symmetric_partial(self):
        """F1 should be symmetric when texts have partial overlap."""
        f1_a = char_f1_score("กรุงเทพ", "กรุงเทพมหานคร")
        f1_b = char_f1_score("กรุงเทพมหานคร", "กรุงเทพ")
        assert abs(f1_a - f1_b) < 1e-6


class TestExactMatch:
    def test_same_thai_string(self):
        em = exact_match_score("กรุงเทพมหานคร", "กรุงเทพมหานคร")
        assert em == 1.0

    def test_different_thai_string(self):
        em = exact_match_score("กรุงเทพ", "เชียงใหม่")
        assert em == 0.0

    def test_whitespace_normalization(self):
        em = exact_match_score("  กรุงเทพ  ", "กรุงเทพ")
        assert em == 1.0

    def test_empty_strings(self):
        em = exact_match_score("", "")
        assert em == 1.0

    def test_subset_not_match(self):
        em = exact_match_score("กรุงเทพ", "กรุงเทพมหานคร")
        assert em == 0.0
