"""PDF 加载器：用 MarkItDown 把 PDF 转成 Markdown"""
from pathlib import Path
from src.core.types import Document
from src.libs.loader.text_cleaner import PdfTextCleaner


class PdfLoader:
    def __init__(self, extract_images: bool = True, enable_clean: bool = True):
        self.extract_images = extract_images
        self.cleaner = PdfTextCleaner() if enable_clean else None

    def load(self, file_path: str) -> Document:
        path = Path(file_path)
        try:
            import markitdown
            md = markitdown.MarkItDown()
            result = md.convert(str(path))
            text = result.text_content or ""
        except ImportError:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)

        # 清洗文本：修复 PDF 换行问题
        if self.cleaner:
            text = self.cleaner.clean(text)

        doc_id = path.stem.replace(" ", "_")
        return Document(
            id=doc_id,
            text=text,
            metadata={"source_path": str(path.resolve()), "doc_type": "pdf"},
        )