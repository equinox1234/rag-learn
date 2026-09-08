"""测试文本清洗"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.libs.loader.text_cleaner import PdfTextCleaner


class TestPdfTextCleaner:
    def setup_method(self):
        self.cleaner = PdfTextCleaner()

    def test_merge_hyphenated_word(self):
        """测试连字符断词合并"""
        text = "detec-\ntion"
        result = self.cleaner.clean(text)
        assert "detection" in result

    def test_merge_abbreviation(self):
        """测试缩写点合并（V. + Villamayor）"""
        text = "V.\nVillamayor"
        result = self.cleaner.clean(text)
        assert "V. Villamayor" in result

    def test_merge_number(self):
        """测试数字连接（80x80 + FPA）"""
        text = "80x80\nFPA"
        result = self.cleaner.clean(text)
        assert "80x80 FPA" in result

    def test_preserve_paragraph(self):
        """测试段落分隔保留"""
        text = "sentence.\nAbstract\nIn this work"
        result = self.cleaner.clean(text)
        assert "sentence.\nAbstract" in result

    def test_merge_comma(self):
        """测试逗号结尾合并"""
        text = "first,\nsecond"
        result = self.cleaner.clean(text)
        assert "first, second" in result

    def test_empty_text(self):
        """测试空文本"""
        assert self.cleaner.clean("") == ""

    def test_no_trailing_newline(self):
        """测试清理后没有多余空格"""
        text = "hello world  "
        result = self.cleaner.clean(text)
        assert result == "hello world"

    def test_multiple_paragraphs(self):
        """测试多段落"""
        text = "First paragraph.\n\nSecond paragraph."
        result = self.cleaner.clean(text)
        assert "First paragraph." in result
        assert "Second paragraph." in result