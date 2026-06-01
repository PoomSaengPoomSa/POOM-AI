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
class ChurnAssessment2(BaseModel):
    grade: str = Field(
        description="이탈 위험 등급. 반드시 '양호', '주의', '위험' 중 하나여야 합니다."
    )
    reason: str = Field(
        description="판정 사유. 반드시 공백 포함 80자 이내의 한 문장(한국어 경어체)으로 간결하게 작성해 주세요. (VARCHAR(100) 길이 제약)"
    )

class ToolSelection2(BaseModel):
    call_get_customer_features: bool = Field(description="고객의 최근 3개월 특징 기록을 조회할지 여부. 고객 유지 및 행동 성향 파악을 위해 참(True)으로 설정합니다.")
    call_get_large_external_transactions: bool = Field(description="고객의 타행 거액 송금 내역을 조회할지 여부. 자산이 고액이거나 대출이 있어 이탈 징후 분석이 필요할 때 참(True)으로 설정합니다.")
    transaction_threshold: Optional[float] = Field(description="거액 송금 조회 기준 금액 (원 단위, 기본값 10,000,000원), 필요 없으면 None", default=10000000.0)
    reason: str = Field(description="도구 수집 판단 근거 (한 문장)")

# 2. State 정의
class Agent2State(TypedDict):
    customer_id: int
    portfolio: Optional[Dict[str, Any]]
    tool_selection: Optional[Dict[str, Any]]
    recent_features: Optional[List[Dict[str, Any]]]
    large_transactions: Optional[List[Dict[str, Any]]]
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

    recent_features = []
    large_transactions = []

    try:
        if tool_selection.get("call_get_customer_features"):
            print("   [Tool Run - ChurnRisk] Running get_customer_features(months=3)")
            recent_features = tools.get_customer_features(customer_id, months=3)
        
        if tool_selection.get("call_get_large_external_transactions"):
            threshold = tool_selection.get("transaction_threshold") or 10000000.0
            print(f"   [Tool Run - ChurnRisk] Running get_large_external_transactions(threshold={threshold:,}원)")
            large_transactions = tools.get_large_external_transactions(customer_id, threshold_amount=threshold)

        return {
            "recent_features": recent_features,
            "large_transactions": large_transactions
        }
    except Exception as e:
        errors.append(f"execute_selected_tools failed: {str(e)}")
        return {"errors": errors}

def analyze_churn_node(state: Agent2State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    portfolio = state["portfolio"]
    recent_features = state["recent_features"]
    large_transactions = state["large_transactions"]

    if state["tool_selection"].get("call_get_customer_features"):
        features_list = []
        for f in recent_features:
            features_list.append(f"[{f['category']} - {f['created_date'].strftime('%Y-%m-%d')}] {f['contents']}")
        features_str = "\n".join(features_list) if features_list else "최근 3개월 내 기록된 고객 특징 없음."
    else:
        features_str = "[참고] 에이전트의 수집 판단 제외: 고객 행동 특징 분석이 불필요하다고 AI가 진단하여 이력 제외함."

    if state["tool_selection"].get("call_get_large_external_transactions"):
        tx_list = []
        for t in large_transactions[:5]:
            tx_list.append(
                f"- 일시: {t['ct_datetime'].strftime('%Y-%m-%d %H:%M:%S')}, "
                f"금액: {t['amount']:,}원, "
                f"상대행: {t['opp_bank_name']}, "
                f"적요: {t['briefs']}, "
                f"거래후잔액: {t['balance_after']:,}원"
            )
        tx_str = "\n".join(tx_list) if tx_list else "최근 타행 거액 송금 이력 없음."
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

    def run(self, customer_id: int) -> Dict[str, Any]:
        initial_state: Agent2State = {
            "customer_id": customer_id,
            "portfolio": None,
            "tool_selection": None,
            "recent_features": None,
            "large_transactions": None,
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
