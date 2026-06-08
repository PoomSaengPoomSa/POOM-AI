import os
import sys
import json
import asyncio
import threading
import atexit
import re
import datetime
from langsmith import traceable
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ISO 8601 pattern detection for datetime and date serialization fallback
ISO_DATETIME_RE = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?$')
ISO_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')

def deserialize_datetime(obj):
    """
    Recursively parse ISO strings back to Python datetime/date objects
    to ensure full compatibility with existing agent codebases.
    """
    if isinstance(obj, str):
        if ISO_DATETIME_RE.match(obj):
            try:
                return datetime.datetime.fromisoformat(obj)
            except ValueError:
                pass
        elif ISO_DATE_RE.match(obj):
            try:
                return datetime.date.fromisoformat(obj)
            except ValueError:
                pass
    elif isinstance(obj, dict):
        return {k: deserialize_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deserialize_datetime(item) for item in obj]
    return obj

class MCPClientManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(MCPClientManager, cls).__new__(cls, *args, **kwargs)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        self.session = None
        self.client_context = None
        
        # Wait synchronously for client startup (up to 30 seconds)
        future = asyncio.run_coroutine_threadsafe(self._start_client(), self.loop)
        future.result(timeout=30)
        
        atexit.register(self.close)

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _start_client(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mcp_server_path = os.path.abspath(os.path.join(current_dir, "..", "mcp_server.py"))
        
        # Disable LangSmith tracing inside the MCP server subprocess to prevent deadlock/hang on Windows stdio
        env = os.environ.copy()
        env["LANGSMITH_TRACING"] = "false"
        env["LANGCHAIN_TRACING_V2"] = "false"
        
        # Run using current virtualenv interpreter
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[mcp_server_path],
            env=env
        )
        
        self.client_context = stdio_client(server_params)
        read_stream, write_stream = await self.client_context.__aenter__()
        
        self.session = ClientSession(read_stream, write_stream)
        await self.session.__aenter__()
        await self.session.initialize()

    def call_tool(self, name: str, arguments: dict):
        future = asyncio.run_coroutine_threadsafe(self._call_tool_async(name, arguments), self.loop)
        return future.result()

    async def _call_tool_async(self, name: str, arguments: dict):
        result = await self.session.call_tool(name, arguments)
        
        # Handle explicit tool execution errors from the MCP server
        if getattr(result, "isError", False):
            error_msg = result.content[0].text if result.content else "Unknown MCP error"
            raise RuntimeError(f"MCP Server Error in tool '{name}': {error_msg}")
            
        if not result.content:
            return None
        
        content = result.content[0]
        if hasattr(content, "text"):
            text_val = content.text
            try:
                raw_data = json.loads(text_val)
                return deserialize_datetime(raw_data)
            except json.JSONDecodeError:
                return text_val
        return None

    def close(self):
        try:
            if self.session:
                future = asyncio.run_coroutine_threadsafe(self.session.__aexit__(None, None, None), self.loop)
                future.result(timeout=5)
            if self.client_context:
                future = asyncio.run_coroutine_threadsafe(self.client_context.__aexit__(None, None, None), self.loop)
                future.result(timeout=5)
        except Exception:
            pass
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)

# Initialize the global singleton client
_mcp_manager = MCPClientManager()

def get_customer(customer_id: int):
    return _mcp_manager.call_tool("get_customer", {"customer_id": customer_id})

def get_trend_report():
    return _mcp_manager.call_tool("get_trend_report", {})

def get_customer_features(customer_id: int, months: int = 3):
    return _mcp_manager.call_tool("get_customer_features", {"customer_id": customer_id, "months": months})

def get_large_external_transactions(customer_id: int, threshold_amount: float = 10000000.0):
    return _mcp_manager.call_tool("get_large_external_transactions", {"customer_id": customer_id, "threshold_amount": threshold_amount})

def save_asset_insight(customer_id: int, insight: str):
    return _mcp_manager.call_tool("save_asset_insight", {"customer_id": customer_id, "insight": insight})

def save_churn_level(customer_id: int, grade: str, reason: str, explain_reason: str = ""):
    return _mcp_manager.call_tool("save_churn_level", {"customer_id": customer_id, "grade": grade, "reason": reason, "explain_reason": explain_reason})

def get_recent_consultation_report(customer_id: int):
    return _mcp_manager.call_tool("get_recent_consultation_report", {"customer_id": customer_id})

def get_main_products():
    return _mcp_manager.call_tool("get_main_products", {})

def save_product_matching(product_id: int, customer_id: int, is_suitable: int, reason: str):
    return _mcp_manager.call_tool("save_product_matching", {"product_id": product_id, "customer_id": customer_id, "is_suitable": is_suitable, "reason": reason})

def get_customer_relationship(customer_id: int):
    return _mcp_manager.call_tool("get_customer_relationship", {"customer_id": customer_id})

def get_customer_active_products(customer_id: int):
    return _mcp_manager.call_tool("get_customer_active_products", {"customer_id": customer_id})

def get_customer_accounts(customer_id: int):
    return _mcp_manager.call_tool("get_customer_accounts", {"customer_id": customer_id})

def get_customer_transactions(customer_id: int, months: int = 3):
    return _mcp_manager.call_tool("get_customer_transactions", {"customer_id": customer_id, "months": months})

@traceable(name="fetch_batch_target_c_ids", run_type="tool")
def fetch_batch_target_c_ids() -> list:
    return _mcp_manager.call_tool("fetch_batch_target_c_ids", {})

def get_customer_ids_by_pb(u_id: str) -> list:
    return _mcp_manager.call_tool("get_customer_ids_by_pb", {"u_id": u_id})
