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
class ChurnAssessment2(BaseModel):
    grade: str = Field(
        description="이탈 위험 등급. 반드시 '양호', '주의', '위험' 중 하나여야 합니다."
    )
    reason: str = Field(
        description="판정 사유. 반드시 공백 포함 80자 이내의 한 문장(한국어 경어체)으로 간결하게 작성해 주세요. (VARCHAR(100) 길이 제약)"
    )

class ToolSelection2(BaseModel):
    call_customer: bool = Field(description="고객 기본 프로필 및 자산 비중 정보(customer)를 조회할지 여부.")
    call_customer_account: bool = Field(description="고객의 상세 계좌 유형 및 잔액 정보(customer_account)를 조회할지 여부.")
    call_customer_product: bool = Field(description="고객이 보유한 예적금/금융상품 가입 목록(customer_product)을 조회할지 여부.")
    call_customer_information: bool = Field(description="최근 1개월간 기록된 고객의 정성적 행동 특징 메모(customer_information)를 조회할지 여부.")
    call_customer_transaction: bool = Field(description="최근 3개월간 기록된 고객의 상세 거래 내역(customer_transaction)을 조회할지 여부.")
    reason: str = Field(description="도구 수집 판단 근거 (한 문장)")

# 2. State 정의
class Agent2State(TypedDict):
    customer_id: int
    portfolio: Optional[Dict[str, Any]]
    tool_selection: Optional[Dict[str, Any]]
    customer_accounts: Optional[List[Dict[str, Any]]]
    customer_products: Optional[List[Dict[str, Any]]]
    customer_features: Optional[List[Dict[str, Any]]]
    customer_transactions: Optional[List[Dict[str, Any]]]
    churn_grade: Optional[str]
    churn_reason: Optional[str]
    errors: List[str]

# 3. 노드 구현체 정의
def load_basic_profile_node(state: Agent2State) -> Dict[str, Any]:
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

def determine_tools_node(state: Agent2State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    portfolio = state["portfolio"]
    try:
        system_prompt = load_prompt("churn_risk_determine_tools_system.md")
        user_prompt_template = load_prompt("churn_risk_determine_tools_user.md")
        
        user_prompt = user_prompt_template.format(
            name=portfolio['name'],
            grade=portfolio['grade'],
            total_assets=portfolio['total_assets'],
            deposit=portfolio['deposit'],
            loan=portfolio['loan']
        )

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.1, api_key=OPENAI_API_KEY)
        structured_llm = llm.with_structured_output(ToolSelection2)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{user_content}")
        ])
        chain = prompt | structured_llm
        selection: ToolSelection2 = chain.invoke({"user_content": user_prompt})
        return {"tool_selection": selection.dict()}
    except Exception as e:
        errors.append(f"determine_tools failed: {str(e)}")
        return {"errors": errors}

def execute_selected_tools_node(state: Agent2State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    customer_id = state["customer_id"]
    tool_selection = state["tool_selection"]

    portfolio = state.get("portfolio")
    customer_accounts = []
    customer_products = []
    customer_features = []
    customer_transactions = []

    try:
        if tool_selection.get("call_customer"):
            print("   [Tool Run - ChurnRisk] Fetching customer profile details (customer)")
            portfolio = tools.get_portfolio_weight(customer_id)
        
        if tool_selection.get("call_customer_account"):
            print("   [Tool Run - ChurnRisk] Fetching customer accounts (customer_account)")
            customer_accounts = tools.get_customer_accounts(customer_id)
        
        if tool_selection.get("call_customer_product"):
            print("   [Tool Run - ChurnRisk] Fetching customer active products (customer_product)")
            customer_products = tools.get_customer_active_products(customer_id)
        
        if tool_selection.get("call_customer_information"):
            print("   [Tool Run - ChurnRisk] Fetching customer features (customer_information) - Last 1 Month")
            customer_features = tools.get_customer_features(customer_id, months=1)
        
        if tool_selection.get("call_customer_transaction"):
            print("   [Tool Run - ChurnRisk] Fetching customer transactions (customer_transaction) - Last 3 Months")
            customer_transactions = tools.get_customer_transactions(customer_id, months=3)

        return {
            "portfolio": portfolio,
            "customer_accounts": customer_accounts,
            "customer_products": customer_products,
            "customer_features": customer_features,
            "customer_transactions": customer_transactions
        }
    except Exception as e:
        errors.append(f"execute_selected_tools failed: {str(e)}")
        return {"errors": errors}

def analyze_churn_node(state: Agent2State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    portfolio = state["portfolio"]
    customer_accounts = state.get("customer_accounts", [])
    customer_products = state.get("customer_products", [])
    customer_features = state.get("customer_features", [])
    customer_transactions = state.get("customer_transactions", [])
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
        features_list = []
        for f in customer_features:
            features_list.append(f"[{f['category']} - {f['created_date'].strftime('%Y-%m-%d')}] {f['contents']}")
        features_str = "\n".join(features_list) if features_list else "최근 1개월 내 기록된 고객 특징 없음."
    else:
        features_str = "[참고] 에이전트의 수집 판단 제외: 고객 행동 특징 분석이 불필요하다고 AI가 진단하여 이력 제외함."

    if tool_selection.get("call_customer_transaction"):
        tx_list = []
        for t in customer_transactions[:10]:
            tx_list.append(
                f"- 일시: {t['ct_datetime'].strftime('%Y-%m-%d %H:%M:%S')}, "
                f"금액: {t['amount']:,}원, "
                f"구분: {'출금' if t['ct_type']=='W' else '입금'}, "
                f"상대행: {t['opp_bank_name']}, "
                f"적요: {t['briefs']}, "
                f"거래후잔액: {t['balance_after']:,}원"
            )
        tx_str = "\n".join(tx_list) if tx_list else "최근 3개월 내 거래 내역 없음."
    else:
        tx_str = "[참고] 에이전트의 수집 판단 제외: 거액 송금 유출 패턴 분석 불필요로 판단하여 거래 이력 제외함."

    try:
        system_prompt = load_prompt("churn_risk_system.md")
        user_prompt_template = load_prompt("churn_risk_user.md")

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.3, api_key=OPENAI_API_KEY)
        structured_llm = llm.with_structured_output(ChurnAssessment2)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt_template)
        ])
        chain = prompt | structured_llm
        assessment: ChurnAssessment2 = chain.invoke({
            "name": portfolio["name"],
            "grade": portfolio["grade"],
            "total_assets": portfolio["total_assets"],
            "deposit": portfolio["deposit"],
            "loan": portfolio["loan"],
            "features_str": features_str,
            "tx_str": tx_str
        })

        # 검증 레이어 (Verification Layer)
        grade = assessment.grade.strip()
        grade_map = {"Low": "양호", "Medium": "주의", "High": "위험", "low": "양호", "medium": "주의", "high": "위험"}
        grade = grade_map.get(grade, grade)
        if grade not in ["양호", "주의", "위험"]:
            grade = "양호"

        reason = assessment.reason.strip()
        if len(reason) > 100:
            reason = reason[:97] + "..."

        return {"churn_grade": grade, "churn_reason": reason}
    except Exception as e:
        errors.append(f"analyze_churn failed: {str(e)}")
        return {"errors": errors}

def save_results_node(state: Agent2State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    customer_id = state["customer_id"]
    churn_grade = state["churn_grade"]
    churn_reason = state["churn_reason"]

    try:
        # 이탈 등급 결과 DB 인서트
        churn_saved = tools.save_churn_level(customer_id, churn_grade, churn_reason)
        if not churn_saved:
            raise ValueError(f"Failed to insert churn risk level into churn_level table for ID {customer_id}")

        print(f"  [+] Sub Agent 2: 이탈 등급 분석 DB 적재 완료! (고객 ID: {customer_id}, 등급: {churn_grade})")
        return {}
    except Exception as e:
        errors.append(f"save_results failed: {str(e)}")
        return {"errors": errors}

# LangGraph 그래프 조립
workflow2 = StateGraph(Agent2State)
workflow2.add_node("load_basic_profile", load_basic_profile_node)
workflow2.add_node("determine_tools", determine_tools_node)
workflow2.add_node("execute_selected_tools", execute_selected_tools_node)
workflow2.add_node("analyze_churn", analyze_churn_node)
workflow2.add_node("save_results", save_results_node)

workflow2.set_entry_point("load_basic_profile")
workflow2.add_edge("load_basic_profile", "determine_tools")
workflow2.add_edge("determine_tools", "execute_selected_tools")
workflow2.add_edge("execute_selected_tools", "analyze_churn")
workflow2.add_edge("analyze_churn", "save_results")
workflow2.add_edge("save_results", END)

compiled_app2 = workflow2.compile()


class ChurnRiskAgent:
    """
    Sub Agent 2: 이탈 위험 수준 도출 Agent
    """
    def __init__(self, model_name: str = None):
        global DEFAULT_MODEL
        if model_name:
            DEFAULT_MODEL = model_name
        self.app = compiled_app2

    @traceable(name="ChurnRiskAgent", run_type="chain", tags=["ChurnRiskAgent"])
    def run(self, customer_id: int) -> Dict[str, Any]:
        initial_state: Agent2State = {
            "customer_id": customer_id,
            "portfolio": None,
            "tool_selection": None,
            "customer_accounts": None,
            "customer_products": None,
            "customer_features": None,
            "customer_transactions": None,
            "churn_grade": None,
            "churn_reason": None,
            "errors": []
        }
        final_state = self.app.invoke(
            initial_state,
            config={"run_name": "ChurnRiskAgent", "tags": ["churn_risk_agent"]}
        )
        if final_state.get("errors"):
            raise RuntimeError(f"LangGraph execution error in ChurnRiskAgent: {final_state['errors']}")
        return final_state
