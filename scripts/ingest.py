"""摄取脚本：python scripts/ingest.py --file 你的文件.pdf"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.settings import load_settings
from src.ingestion.pipeline import IngestionPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="摄取文档到 RAG 知识库")
    parser.add_argument("--file", required=True, help="PDF 文件路径")
    args = parser.parse_args()

    settings = load_settings()
    pipeline = IngestionPipeline(settings)
    result = pipeline.run(args.file)
    print(f"结果: doc_id={result['doc_id']}, chunks={result['chunk_count']}")