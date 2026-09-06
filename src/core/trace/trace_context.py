"""追踪上下文：记录每个环节的耗时和数据"""
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class TraceContext:
    """一次请求的追踪记录

    用法：
        trace = TraceContext(trace_type="query")
        trace.record_stage("检索", {"method": "dense"}, elapsed_ms=150.0)
        trace.finish()
        print(trace.to_dict())
    """

    def __init__(self, trace_type: str = "query"):
        self.trace_id = str(uuid.uuid4())[:8]       # 短 ID，方便看
        self.trace_type = trace_type                  # "query" 或 "ingestion"
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.finished_at: Optional[str] = None
        self.stages: List[Dict[str, Any]] = []        # 每个环节的记录
        self._start = time.monotonic()                # 用于算耗时

    def record_stage(self, name: str, data: Dict[str, Any],
                     elapsed_ms: Optional[float] = None):
        """记录一个环节"""
        entry = {
            "stage": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        if elapsed_ms is not None:
            entry["elapsed_ms"] = round(elapsed_ms, 2)
        self.stages.append(entry)

    def finish(self):
        """标记追踪结束"""
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典（方便转 JSON）"""
        total_ms = round((time.monotonic() - self._start) * 1000, 2)
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_ms": total_ms,
            "stages": self.stages,
        }