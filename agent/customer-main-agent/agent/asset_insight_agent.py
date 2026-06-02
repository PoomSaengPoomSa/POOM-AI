import os
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langsmith import traceable

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
    call_customer: bool = Field(description="고객 기본 프로필 및 자산 비중 정보(customer)를 조회할지 여부.")
    call_customer_account: bool = Field(description="고객의 상세 계좌 유형 및 잔액 정보(customer_account)를 조회할지 여부. 유동성 자산 파악 및 정밀 리밸런싱 진단 시 유용.")
    call_customer_product: bool = Field(description="고객이 보유한 예적금/금융상품 가입 목록(customer_product)을 조회할지 여부. 만기 예정 자금 재투자 조언 시 유용.")
    call_customer_information: bool = Field(description="최근 1개월간 기록된 고객의 정성적 행동 특징 메모(customer_information)를 조회할지 여부.")
    call_trend_llm_report: bool = Field(description="경제 지표 트렌드 보고서(trend_llm_report)를 조회할지 여부. 거시 지표 변화 영향 진단 시 유용.")
    reason: str = Field(description="해당 데이터 수집 도구들을 선택한 이유에 대한 에이전트의 구체적인 분석 및 판단 근거 (한 문장)")

# 2. State 정의
class Agent1State(TypedDict):
    customer_id: int
    portfolio: Optional[Dict[str, Any]]
    tool_selection: Optional[Dict[str, Any]]
    customer_accounts: Optional[List[Dict[str, Any]]]
    customer_products: Optional[List[Dict[str, Any]]]
    customer_features: Optional[List[Dict[str, Any]]]
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
    customer_id = state["customer_id"]
    tool_selection = state["tool_selection"]

    portfolio = state.get("portfolio")
    customer_accounts = []
    customer_products = []
    customer_features = []
    trend_reports = []

    try:
        if tool_selection.get("call_customer"):
            print("   [Tool Run - AssetInsight] Fetching customer profile details (customer)")
            portfolio = tools.get_portfolio_weight(customer_id)
        
        if tool_selection.get("call_customer_account"):
            print("   [Tool Run - AssetInsight] Fetching customer accounts (customer_account)")
            customer_accounts = tools.get_customer_accounts(customer_id)
        
        if tool_selection.get("call_customer_product"):
            print("   [Tool Run - AssetInsight] Fetching customer active products (customer_product)")
            customer_products = tools.get_customer_active_products(customer_id)
        
        if tool_selection.get("call_customer_information"):
            print("   [Tool Run - AssetInsight] Fetching customer features (customer_information) - Last 1 Month")
            customer_features = tools.get_customer_features(customer_id, months=1)
        
        if tool_selection.get("call_trend_llm_report"):
            print("   [Tool Run - AssetInsight] Fetching trend reports (trend_llm_report)")
            trend_reports = tools.get_trend_report()

        return {
            "portfolio": portfolio,
            "customer_accounts": customer_accounts,
            "customer_products": customer_products,
            "customer_features": customer_features,
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
    customer_accounts = state.get("customer_accounts", [])
    customer_products = state.get("customer_products", [])
    customer_features = state.get("customer_features", [])
    trend_reports = state.get("trend_reports", [])
    tool_selection = state["tool_selection"]

    if tool_selection.get("call_customer"):
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
    else:
        portfolio_str = "[참고] 에이전트의 수집 판단 제외: 고객 자산 프로필 조회가 스킵되었습니다."

    if tool_selection.get("call_customer_account"):
        acc_list = []
        for acc in customer_accounts:
            acc_list.append(f"- 계좌번호: {acc['account_num']}, 유형: {acc['account_type']}, 잔액: {acc['balance']:,}원")
        accounts_str = "\n".join(acc_list) if acc_list else "보유 중인 상세 계좌 정보 없음."
    else:
        accounts_str = "[참고] 에이전트의 수집 판단 제외: 상세 계좌 유형 및 잔액 정보가 제외되었습니다."

    if tool_selection.get("call_customer_product"):
        prod_list = []
        for prod in customer_products:
            prod_list.append(f"- 상품 ID: {prod['pd_id']}, 상품명: {prod['product_name']} (가입일: {prod['opening_date']}, 만기일: {prod['expiration_date']})")
        products_str = "\n".join(prod_list) if prod_list else "보유 중인 금융 상품 목록 없음."
    else:
        products_str = "[참고] 에이전트의 수집 판단 제외: 가입 상품 및 만기 정보가 제외되었습니다."

    if tool_selection.get("call_customer_information"):
        feat_list = []
        for feat in customer_features:
            feat_list.append(f"[{feat['category']}] {feat['contents']} ({feat['created_date'].strftime('%Y-%m-%d')})")
        features_str = "\n".join(feat_list) if feat_list else "최근 1개월간 기록된 고객 특징 없음."
    else:
        features_str = "[참고] 에이전트의 수집 판단 제외: 최근 1개월간의 고객 정성 특징 정보가 제외되었습니다."

    if tool_selection.get("call_trend_llm_report"):
        reports_list = []
        for r in trend_reports:
            reports_list.append(f"[{r['type'].upper()} 트렌드 분석 보고서]\n{r['content']}")
        reports_str = "\n\n".join(reports_list) if reports_list else "활성화된 지표 트렌드 분석 보고서 없음."
    else:
        reports_str = "[참고] 에이전트의 수집 판단 제외: 금값/금리/부동산 거시 지표 변화 영향 분석이 제외되었습니다."

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
            "accounts_str": accounts_str,
            "products_str": products_str,
            "features_str": features_str,
            "reports_str": reports_str,
            "tendency": portfolio['tendency']
        })
        return {"asset_insight": response.content.strip()}
    except Exception as e:
        errors.append(f"analyze_assets failed: {str(e)}")
        return {"errors": errors}

def verify_insight_node(state: Agent1State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    insight = state.get("asset_insight")
    if not insight:
        errors.append("verify_insight failed: No asset_insight found in state.")
        return {"errors": errors}
    
    try:
        # 1. 2인칭("고객님") -> 3인칭("고객") 교정
        cleaned = insight.replace("고객님", "고객")
        
        # 2. 마크다운 기호 제거
        for char in ['*', '#', '_', '|', '`', '-']:
            cleaned = cleaned.replace(char, '')
            
        # 3. 3문장 이내 제약조건 검증 및 교정
        sentences = [s.strip() for s in cleaned.split('.') if s.strip()]
        if len(sentences) > 3:
            print(f"   [Verification Node] Warning: Insight has {len(sentences)} sentences. Truncating to 3 sentences.")
            cleaned = ". ".join(sentences[:3]) + "."
        elif sentences:
            if not cleaned.endswith('.'):
                cleaned += "."
                
        print(f"   [Verification Node] Verification completed. Original length: {len(insight)}, Cleaned length: {len(cleaned)}")
        return {"asset_insight": cleaned.strip()}
    except Exception as e:
        errors.append(f"verify_insight failed: {str(e)}")
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
workflow1.add_node("verify_insight", verify_insight_node)
workflow1.add_node("save_results", save_results_node)

workflow1.set_entry_point("load_basic_profile")
workflow1.add_edge("load_basic_profile", "determine_tools")
workflow1.add_edge("determine_tools", "execute_selected_tools")
workflow1.add_edge("execute_selected_tools", "analyze_assets")
workflow1.add_edge("analyze_assets", "verify_insight")
workflow1.add_edge("verify_insight", "save_results")
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

    @traceable(name="AssetInsightAgent", run_type="chain", tags=["AssetInsightAgent"])
    def run(self, customer_id: int) -> Dict[str, Any]:
        initial_state: Agent1State = {
            "customer_id": customer_id,
            "portfolio": None,
            "tool_selection": None,
            "customer_accounts": None,
            "customer_products": None,
            "customer_features": None,
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
