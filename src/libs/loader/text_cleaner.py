"""PDF 文本清洗：修复 PDF 提取导致的换行混乱"""
import re


class PdfTextCleaner:
    """修复 PDF 文字提取中的假换行问题"""

    # 一行以这些结尾 → 说明没说完，应与下一行合并
    CONTINUATION_PATTERNS = [
        r'[a-z]$',           # 小写字母结尾（英文单词被截断）
        r'[a-z]\s*$',        # 小写字母 + 空格
        r'[;:,]\s*$',        # 分号/冒号/逗号结尾
        r'[A-Z]\.\s*$',      # 缩写点结尾（如 "V." 后面应该是名字）
        r'^\s*[-–]\s*',      # 以连字符开头（被分割的复合词）
        r'[a-z]\s*[-–]\s*$', # 连字符结尾（如 "detec-\ntion"）
    ]

    # 章节标题前缀（无论上下文如何，都不合并）
    SECTION_HEADERS = [
        r'^Abstract\b',
        r'^Introduction\b',
        r'^Conclusion\b',
        r'^References\b',
        r'^Acknowledgements\b',
        r'^Acknowledgments\b',
        r'^Fig(ure)?\.?\s+\d+',
        r'^Table\s+\d+',
    ]

    def clean(self, text: str) -> str:
        """清洗文本：合并假换行，保留真段落"""
        if not text:
            return text

        # 1. 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 2. 修复被连字符断开的词（如 "detec-\ntion" → "detection"）
        text = re.sub(r'([a-zA-Z])-\s*\n\s*([a-zA-Z])', r'\1\2', text)

        # 3. 逐行处理，判断哪些行应该合并
        lines = text.split('\n')
        merged = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # 如果是空行，保留（段落分隔）
            if not line:
                merged.append('')
                i += 1
                continue

            # 看下一行是否应该合并到当前行
            while i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if not next_line:
                    break  # 空行 = 段落结束，不合并

                if self._should_merge(line, next_line):
                    if line.endswith('-') or line.endswith('–'):
                        # 连字符连接，直接拼接
                        line = line.rstrip('-–') + next_line
                    else:
                        # 普通合并，加空格
                        line = line + ' ' + next_line
                    i += 1
                else:
                    break

            merged.append(line)
            i += 1

        # 4. 修复多余空格
        text = '\n'.join(merged)
        text = re.sub(r' {2,}', ' ', text)          # 多个空格 → 一个
        text = re.sub(r'\n{3,}', '\n\n', text)      # 多余空行

        return text.strip()

    def _should_merge(self, current_line: str, next_line: str) -> bool:
        """判断当前行和下一行是否应该合并"""

        # 当前行是章节标题 → 不合并（标题单独一行）
        if any(re.match(p, current_line) for p in self.SECTION_HEADERS):
            return False

        # 下一行是章节标题 → 绝对不合并
        if any(re.match(p, next_line) for p in self.SECTION_HEADERS):
            return False

        # 下一行是空行 → 不合并
        if not next_line.strip():
            return False

        # 下一行是数字编号 / 列表项 → 不合并
        if re.match(r'^\d+\.\s+', next_line) or re.match(r'^[-•]\s+', next_line):
            return False

        # 连字符结尾（如 "detec-" → 下一行是 "tion"）
        if current_line.rstrip().endswith('-') or current_line.rstrip().endswith('–'):
            return True

        # 缩写点结尾（如 "V." → 后面应该是 "Villamayor"）
        if re.search(r'[A-Z]\.\s*$', current_line):
            return True

        # 逗号/分号/冒号结尾 → 肯定没说完
        if re.search(r'[;:,]\s*$', current_line):
            return True

        # 数字结尾 → 可能没说完
        if re.search(r'\d\s*$', current_line):
            return True

        # 小写字母结尾 → 大概率没说完
        if re.search(r'[a-z]\s*$', current_line):
            return True

        # 当前行以句号/问号/感叹号结尾（句子结束了）
        if re.search(r'[.?!]\s*$', current_line):
            if re.match(r'^[a-z]', next_line):
                return True  # 下一行小写 → 说明句子没完
            return False       # 下一行大写 → 正常新句子

        return True