import sys
import os
import datetime
import decimal
import json

# Ensure the customer-main-agent directory and its parent are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Fix Windows standard stream encoding issue for MCP process communication
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from tool import tools_direct
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Poom-Agent-Tools")

def serialize_datetime(obj):
    """
    Recursively convert datetime, date, and decimal.Decimal objects 
    to native JSON-compatible types.
    """
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        if obj % 1 == 0:
            return int(obj)
        return float(obj)
    if isinstance(obj, dict):
        return {k: serialize_datetime(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_datetime(item) for item in obj]
    return obj

def to_mcp_response(data):
    """
    Helper to serialize python structures to JSON string for safe MCP transmission.
    """
    return json.dumps(serialize_datetime(data), ensure_ascii=False)

@mcp.tool()
def get_customer(customer_id: int) -> str:
    """
    Get customer asset portfolio details.
    """
    return to_mcp_response(tools_direct.get_customer(customer_id))

@mcp.tool()
def get_trend_report() -> str:
    """
    Retrieve completed trend reports.
    """
    return to_mcp_response(tools_direct.get_trend_report())

@mcp.tool()
def get_customer_features(customer_id: int, months: int = 3) -> str:
    """
    Get customer features extracted from the database for the given period (months).
    """
    return to_mcp_response(tools_direct.get_customer_features(customer_id, months))

@mcp.tool()
def get_large_external_transactions(customer_id: int, threshold_amount: float = 10000000.0) -> str:
    """
    Retrieve external transactions where the customer transferred out a large amount of money.
    """
    return to_mcp_response(tools_direct.get_large_external_transactions(customer_id, threshold_amount))

@mcp.tool()
def save_asset_insight(customer_id: int, insight: str) -> str:
    """
    Save the LLM generated asset profile analysis result to customer's llm_insight column and update analysis_time.
    """
    return to_mcp_response(tools_direct.save_asset_insight(customer_id, insight))

@mcp.tool()
def save_churn_level(customer_id: int, grade: str, reason: str, explain_reason: str = "") -> str:
    """
    Insert a new churn risk level assessment into churn_level table.
    """
    return to_mcp_response(tools_direct.save_churn_level(customer_id, grade, reason, explain_reason))

@mcp.tool()
def get_recent_consultation_report(customer_id: int) -> str:
    """
    Get the latest consultation_report content for the customer.
    """
    return to_mcp_response(tools_direct.get_recent_consultation_report(customer_id))

@mcp.tool()
def get_main_products() -> str:
    """
    Retrieve active bank main products from the product table.
    """
    return to_mcp_response(tools_direct.get_main_products())

@mcp.tool()
def save_product_matching(product_id: int, customer_id: int, is_suitable: int, reason: str) -> str:
    """
    Upsert product matching suitability evaluation result.
    """
    return to_mcp_response(tools_direct.save_product_matching(product_id, customer_id, is_suitable, reason))

@mcp.tool()
def get_customer_relationship(customer_id: int) -> str:
    """
    Retrieve customer family relationships.
    """
    return to_mcp_response(tools_direct.get_customer_relationship(customer_id))

@mcp.tool()
def get_customer_active_products(customer_id: int) -> str:
    """
    Retrieve products currently held by the customer.
    """
    return to_mcp_response(tools_direct.get_customer_active_products(customer_id))

@mcp.tool()
def get_customer_accounts(customer_id: int) -> str:
    """
    Retrieve customer's account types and balances.
    """
    return to_mcp_response(tools_direct.get_customer_accounts(customer_id))

@mcp.tool()
def get_customer_transactions(customer_id: int, months: int = 3) -> str:
    """
    Retrieve all transactions for a specific customer in the last N months.
    """
    return to_mcp_response(tools_direct.get_customer_transactions(customer_id, months))

@mcp.tool()
def fetch_batch_target_c_ids() -> str:
    """
    DB 단일 스캔 쿼리를 통해 분석 후보 VVIP 고객 정보 및 스캔 조건 사유 추출
    """
    return to_mcp_response(tools_direct.fetch_batch_target_c_ids())

if __name__ == "__main__":
    mcp.run()
