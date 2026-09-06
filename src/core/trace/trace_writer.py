"""Trace 持久化：把追踪记录写入 JSON Lines 文件"""
import json
from pathlib import Path
from typing import Dict, Any


def write_trace(trace_dict: Dict[str, Any], path: str = "logs/traces.jsonl"):
    """把一条 trace 追加到 JSONL 文件"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(trace_dict, ensure_ascii=False)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_traces(path: str = "logs/traces.jsonl") -> list:
    """读取所有 trace 记录"""
    p = Path(path)
    if not p.exists():
        return []
    traces = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return traces