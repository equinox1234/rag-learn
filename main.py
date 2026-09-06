"""入口：启动 MCP Server（供 Claude Desktop 调用）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    from src.mcp_server.server import main
    import asyncio
    asyncio.run(main())