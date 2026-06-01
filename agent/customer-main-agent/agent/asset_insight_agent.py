import os
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# db 및 tool 임포트
from db import root_env_path
from tool import tools

# API 키 및 모델 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        f"Configuration error: OPENAI_API_KEY is missing. "
        f"Please verify your root .env file at {root_env_path}"
    )

DEFAULT_MODEL = "gpt-4o-mini"

# 프롬프트 동적 로드 헬퍼 함수
def load_prompt(filename: str) -> str:
    """
    Utility to load prompt templates from the local prompt directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(os.path.dirname(current_dir), "prompt", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# 1. Pydantic 스키마 정의
class ToolSelection1(BaseModel):
    call_search_today_news: bool = Field(description="오늘자 금융/경제 뉴스를 검색할지 여부. 고객이 적극투자형이거나 자산 비중에서 투자가 높은 경우 시장 정보 수집을 위해 참(True)으로 설정합니다.")
    news_keyword: Optional[str] = Field(description="뉴스 검색 키워드 (예: 금리, 금, 부동산, 주식, 코스피 등), 필요 없으면 None")
    call_get_trend_report: bool = Field(description="경제 지표 트렌드 보고서(금값, 기준금리, 부동산)를 가져올지 여부. 고객의 연금, 투자, 대출 자산이 존재하고 거시 경제 지표와의 매칭 진단이 유용할 때 참(True)으로 설정합니다.")
    reason: str = Field(description="해당 데이터 수집 도구들을 선택한 이유에 대한 에이전트의 구체적인 분석 및 판단 근거 (한 문장)")

# 2. State 정의
class Agent1State(TypedDict):
    customer_id: int
    portfolio: Optional[Dict[str, Any]]
    tool_selection: Optional[Dict[str, Any]]
    today_news: Optional[List[Dict[str, Any]]]
    trend_reports: Optional[List[Dict[str, Any]]]
    asset_insight: Optional[str]
    errors: List[str]

# 3. 노드 구현체 정의
def load_basic_profile_node(state: Agent1State) -> Dict[str, Any]:
    customer_id = state["customer_id"]
    errors = list(state.get("errors", []))
    try:
        portfolio = tools.get_portfolio_weight(customer_id)
        if not portfolio:
            raise ValueError(f"Customer with ID {customer_id} not found in database.")
        return {"portfolio": portfolio, "errors": errors}
    except Exception as e:
        errors.append(f"load_basic_profile failed: {str(e)}")
        return {"errors": errors}

def determine_tools_node(state: Agent1State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    portfolio = state["portfolio"]
    try:
        system_prompt = load_prompt("asset_insight_determine_tools_system.md")
        user_prompt_template = load_prompt("asset_insight_determine_tools_user.md")
        
        net_worth = max(1, portfolio['net_worth'])
        user_prompt = user_prompt_template.format(
            name=portfolio['name'],
            grade=portfolio['grade'],
            tendency=portfolio['tendency'],
            total_assets=portfolio['total_assets'],
            deposit=portfolio['deposit'],
            deposit_ratio=(portfolio['deposit'] / net_worth * 100),
            investment=portfolio['investment'],
            investment_ratio=(portfolio['investment'] / net_worth * 100),
            pension=portfolio['pension'],
            pension_ratio=(portfolio['pension'] / net_worth * 100),
            loan=portfolio['loan'],
            loan_ratio=(portfolio['loan'] / net_worth * 100),
            net_worth=portfolio['net_worth']
        )

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.1, api_key=OPENAI_API_KEY)
        structured_llm = llm.with_structured_output(ToolSelection1)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{user_content}")
        ])
        chain = prompt | structured_llm
        selection: ToolSelection1 = chain.invoke({"user_content": user_prompt})
        return {"tool_selection": selection.dict()}
    except Exception as e:
        errors.append(f"determine_tools failed: {str(e)}")
        return {"errors": errors}

def execute_selected_tools_node(state: Agent1State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    tool_selection = state["tool_selection"]

    today_news = []
    trend_reports = []

    try:
        if tool_selection.get("call_search_today_news"):
            keyword = tool_selection.get("news_keyword")
            print(f"   [Tool Run - AssetInsight] Running search_today_news(keyword='{keyword}')")
            today_news = tools.search_today_news(keyword=keyword)
        
        if tool_selection.get("call_get_trend_report"):
            print("   [Tool Run - AssetInsight] Running get_trend_report()")
            trend_reports = tools.get_trend_report()

        return {
            "today_news": today_news,
            "trend_reports": trend_reports
        }
    except Exception as e:
        errors.append(f"execute_selected_tools failed: {str(e)}")
        return {"errors": errors}

def analyze_assets_node(state: Agent1State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    portfolio = state["portfolio"]
    today_news = state["today_news"]
    trend_reports = state["trend_reports"]

    portfolio_str = (
        f"- 고객명: {portfolio['name']}\n"
        f"- 투자성향: {portfolio['tendency']}\n"
        f"- 등급: {portfolio['grade']}\n"
        f"- 총자산: {portfolio['total_assets']:,}원\n"
        f"  - 예금: {portfolio['deposit']:,}원 (순자산 대비 비중: {portfolio['deposit']/max(1, portfolio['net_worth'])*100:.1f}%)\n"
        f"  - 투자: {portfolio['investment']:,}원 (순자산 대비 비중: {portfolio['investment']/max(1, portfolio['net_worth'])*100:.1f}%)\n"
        f"  - 연금: {portfolio['pension']:,}원 (순자산 대비 비중: {portfolio['pension']/max(1, portfolio['net_worth'])*100:.1f}%)\n"
        f"  - 대출: {portfolio['loan']:,}원 (순자산 대비 부채비율: {portfolio['loan']/max(1, portfolio['net_worth'])*100:.1f}%)\n"
        f"  - 순자산: {portfolio['net_worth']:,}원\n"
    )

    if state["tool_selection"].get("call_search_today_news"):
        news_list = []
        for n in today_news[:5]:
            news_list.append(f"[{n['source']}] {n['title']}\n{n['body'][:200]}...")
        news_str = "\n\n".join(news_list) if news_list else "당일 수집된 주요 뉴스 없음."
    else:
        news_str = "[참고] 에이전트의 수집 판단 제외: 해당 고객의 포트폴리오 성격상 당일 뉴스의 직접적인 필요성이 낮아 분석 데이터에서 제외되었습니다."

    if state["tool_selection"].get("call_get_trend_report"):
        reports_list = []
        for r in trend_reports:
            reports_list.append(f"[{r['type'].upper()} 트렌드 분석 보고서]\n{r['content']}")
        reports_str = "\n\n".join(reports_list) if reports_list else "활성화된 지표 트렌드 분석 보고서 없음."
    else:
        reports_str = "[참고] 에이전트의 수집 판단 제외: 금값/금리/부동산 거시 지표 변화 영향도가 낮아 분석 데이터에서 제외되었습니다."

    try:
        system_prompt = load_prompt("asset_analysis_system.md")
        user_prompt_template = load_prompt("asset_analysis_user.md")

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.7, api_key=OPENAI_API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt_template)
        ])
        chain = prompt | llm
        response = chain.invoke({
            "portfolio_str": portfolio_str,
            "news_str": news_str,
            "reports_str": reports_str,
            "tendency": portfolio['tendency']
        })
        return {"asset_insight": response.content.strip()}
    except Exception as e:
        errors.append(f"analyze_assets failed: {str(e)}")
        return {"errors": errors}

def save_results_node(state: Agent1State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    customer_id = state["customer_id"]
    asset_insight = state["asset_insight"]

    try:
        # 자산 분석 결과 DB 업데이트
        insight_saved = tools.save_asset_insight(customer_id, asset_insight)
        if not insight_saved:
            raise ValueError(f"Failed to update asset insight in customer table for ID {customer_id}")

        print(f"  [+] Sub Agent 1: 자산 분석 인사이트 DB 적재 완료! (고객 ID: {customer_id})")
        return {}
    except Exception as e:
        errors.append(f"save_results failed: {str(e)}")
        return {"errors": errors}

# LangGraph 그래프 조립
workflow1 = StateGraph(Agent1State)
workflow1.add_node("load_basic_profile", load_basic_profile_node)
workflow1.add_node("determine_tools", determine_tools_node)
workflow1.add_node("execute_selected_tools", execute_selected_tools_node)
workflow1.add_node("analyze_assets", analyze_assets_node)
workflow1.add_node("save_results", save_results_node)

workflow1.set_entry_point("load_basic_profile")
workflow1.add_edge("load_basic_profile", "determine_tools")
workflow1.add_edge("determine_tools", "execute_selected_tools")
workflow1.add_edge("execute_selected_tools", "analyze_assets")
workflow1.add_edge("analyze_assets", "save_results")
workflow1.add_edge("save_results", END)

compiled_app1 = workflow1.compile()


class AssetInsightAgent:
    """
    Sub Agent 1: 자산 보유 현황 인사이트 도출 Agent
    """
    def __init__(self, model_name: str = None):
        global DEFAULT_MODEL
        if model_name:
            DEFAULT_MODEL = model_name
        self.app = compiled_app1

    def run(self, customer_id: int) -> Dict[str, Any]:
        initial_state: Agent1State = {
            "customer_id": customer_id,
            "portfolio": None,
            "tool_selection": None,
            "today_news": None,
            "trend_reports": None,
            "asset_insight": None,
            "errors": []
        }
        final_state = self.app.invoke(
            initial_state,
            config={"run_name": "AssetInsightAgent", "tags": ["asset_insight_agent"]}
        )
        if final_state.get("errors"):
            raise RuntimeError(f"LangGraph execution error in AssetInsightAgent: {final_state['errors']}")
        return final_state
