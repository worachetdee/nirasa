"""Tests for the data pipeline components."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from nirasa.data.clean import clean_text, thai_ratio, clean_file
from nirasa.data.dedup import dedup_file
from nirasa.data.filter import filter_document, filter_file


# --- clean_text ---


class TestCleanText:
    def test_html_removal(self):
        text = "<p>สวัสดีครับ</p> <b>ทดสอบ</b>"
        result = clean_text(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "สวัสดีครับ" in result
        assert "ทดสอบ" in result

    def test_url_removal(self):
        text = "ดูข้อมูลที่ https://example.com/path?q=test และ http://test.org"
        result = clean_text(text)
        assert "https://" not in result
        assert "http://" not in result
        assert "ดูข้อมูลที่" in result

    def test_nfkc_normalization(self):
        # NFKC should normalize compatibility characters
        text = "ทดสอบ\ufeffข้อความ"
        result = clean_text(text)
        assert "\ufeff" not in result

    def test_control_char_removal(self):
        text = "สวัสดี\x00\x01ครับ"
        result = clean_text(text)
        assert "\x00" not in result
        assert "\x01" not in result

    def test_preserves_thai(self):
        text = "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย"
        result = clean_text(text)
        assert result == text

    def test_collapse_whitespace(self):
        text = "สวัสดี     ครับ"
        result = clean_text(text)
        assert "     " not in result


# --- thai_ratio ---


class TestThaiRatio:
    def test_pure_thai(self):
        ratio = thai_ratio("ภาษาไทยล้วน")
        assert ratio > 0.8

    def test_pure_english(self):
        ratio = thai_ratio("English only text")
        assert ratio == 0.0

    def test_mixed(self):
        ratio = thai_ratio("สวัสดี hello")
        assert 0.0 < ratio < 1.0

    def test_empty(self):
        ratio = thai_ratio("")
        assert ratio == 0.0

    def test_numbers_and_thai(self):
        ratio = thai_ratio("ประชากร 70 ล้านคน")
        assert 0.3 < ratio < 1.0


# --- dedup ---


class TestDedup:
    def test_detect_near_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "input.jsonl")
            output_file = os.path.join(tmpdir, "output.jsonl")

            base_text = "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย " * 10
            near_dup = base_text + " เพิ่มเติมเล็กน้อย"
            different = "เชียงใหม่เป็นจังหวัดทางภาคเหนือของประเทศไทย " * 10

            with open(input_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({"text": base_text}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"text": near_dup}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"text": different}, ensure_ascii=False) + "\n")

            stats = dedup_file(input_file, output_file, threshold=0.8, num_perm=64)

            assert stats["output_docs"] < stats["input_docs"]
            assert stats["duplicates_removed"] >= 1

    def test_no_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "input.jsonl")
            output_file = os.path.join(tmpdir, "output.jsonl")

            texts = [
                "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย มีประชากรมากที่สุด",
                "เชียงใหม่เป็นจังหวัดทางภาคเหนือ มีภูเขาสูง อากาศเย็นสบาย",
                "ภูเก็ตเป็นเกาะที่มีชื่อเสียงด้านการท่องเที่ยว มีหาดทรายสวยงาม",
            ]

            with open(input_file, "w", encoding="utf-8") as f:
                for text in texts:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

            stats = dedup_file(input_file, output_file, threshold=0.8, num_perm=64)
            assert stats["duplicates_removed"] == 0


# --- filter ---


class TestFilter:
    def test_pass_good_document(self):
        text = "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย " * 10
        assert filter_document(text, min_len=50) is True

    def test_reject_short(self):
        assert filter_document("สั้น", min_len=50) is False

    def test_reject_too_long(self):
        text = "ก" * 600000
        assert filter_document(text, max_len=500000) is False

    def test_reject_low_thai_ratio(self):
        text = "English text only, no Thai at all here. " * 10
        assert filter_document(text, min_len=10, min_thai_ratio=0.1) is False

    def test_filter_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "input.jsonl")
            output_file = os.path.join(tmpdir, "output.jsonl")

            with open(input_file, "w", encoding="utf-8") as f:
                # Good document
                good = "กรุงเทพมหานครเป็นเมืองหลวง " * 20
                f.write(json.dumps({"text": good}, ensure_ascii=False) + "\n")
                # Too short
                f.write(json.dumps({"text": "สั้น"}, ensure_ascii=False) + "\n")

            stats = filter_file(input_file, output_file, min_len=50)
            assert stats["output_docs"] == 1
            assert stats["filtered"] == 1
