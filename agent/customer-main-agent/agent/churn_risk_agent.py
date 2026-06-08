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
    explain_reason: str = Field(
        description="판정 상세 설명. 공백 포함 200자 내외의 구체적인 근거와 배경을 포함한 한국어 경어체 설명을 작성해 주세요. (TEXT 컬럼)"
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
    churn_explain_reason: Optional[str]
    errors: List[str]

# 3. 노드 구현체 정의
def load_basic_profile_node(state: Agent2State) -> Dict[str, Any]:
    customer_id = state["customer_id"]
    errors = list(state.get("errors", []))
    try:
        portfolio = tools.get_customer(customer_id)
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
            portfolio = tools.get_customer(customer_id)
        
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

        return {"churn_grade": assessment.grade.strip(), "churn_reason": assessment.reason.strip(), "churn_explain_reason": assessment.explain_reason.strip()}
    except Exception as e:
        errors.append(f"analyze_churn failed: {str(e)}")
        return {"errors": errors}

def verify_churn_node(state: Agent2State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    
    portfolio = state["portfolio"]
    customer_transactions = state.get("customer_transactions", [])
    churn_grade = state.get("churn_grade")
    churn_reason = state.get("churn_reason")
    churn_explain_reason = state.get("churn_explain_reason")
    
    if not churn_grade or not churn_reason:
        errors.append("verify_churn failed: Churn grade or reason missing in state.")
        return {"errors": errors}
        
    try:
        # LLM 기반 이탈 위험 등급 및 사유 정합성 검증 레이어 가동
        verifier_system_prompt = (
            "당신은 은행 본점의 수석 고객 행동 분석 검증관입니다.\n"
            "제시된 [고객 자산 및 거래 팩트 데이터]와 1차 판정된 [이탈 등급 및 판단 사유]를 분석하여 비즈니스적 적절성을 평가하고, 모순이 있거나 미흡한 경우 등급과 사유를 교정해 주십시오.\n\n"
            "### [비즈니스 판단 검증 규칙]\n"
            "1. **임계치 초과 유출 시 '위험'**: 최근 7일 내 누적 타행 송금액이 순자산의 30% 이상이거나, 최근 7일 내 단일 1억 원 이상의 대형 출금이 있다면 이탈 위험 등급은 '위험'이어야 합니다. ('양호' 또는 '주의'로 판정된 경우 '위험'으로 격상)\n"
            "2. **사유와 등급의 일치성**: 판정 사유(reason) 내용이 결정된 등급(grade)과 비즈니스 논리적으로 일치해야 하며 모순이 없어야 합니다.\n\n"
            "### [출력 및 규격 제한 (엄격 적용)]\n"
            "- 판정 사유(reason)는 반드시 **공백 포함 80자 이내의 한 문장(한국어 경어체)**이어야 합니다. (마크다운 기호 금지)\n"
            "- 판정 상세 설명(explain_reason)은 **공백 포함 200자 내외의 한국어 경어체 문장**으로, 판정 근거가 되는 구체적 데이터 포인트(거래 금액, 타행명, 메모 내용 등)와 판단 배경을 서술해 주십시오. (마크다운 기호 금지)\n"
            "- 이탈 등급(grade)은 오직 '양호', '주의', '위험' 중 하나여야 합니다."
        )
        
        # 최근 3개월 거래 내역 팩트 요약
        tx_list = []
        if customer_transactions:
            for t in customer_transactions[:10]:
                tx_list.append(
                    f"- 금액: {t['amount']:,}원, 구분: {'출금' if t['ct_type']=='W' else '입금'}, 상대행: {t['opp_bank_name']}, 적요: {t['briefs']}"
                )
        tx_facts_str = "\n".join(tx_list) if tx_list else "최근 거래 내역 없음."
        
        net_worth = max(1, portfolio.get('net_worth', 1))
        customer_facts = (
            f"- 고객명: {portfolio.get('name')}\n"
            f"- 순자산: {portfolio.get('net_worth', 0):,}원\n"
            f"- 총자산: {portfolio.get('total_assets', 0):,}원\n"
            f"- 최근 3개월 거래 팩트:\n{tx_facts_str}"
        )
        
        user_content = (
            f"### [고객 자산 및 거래 팩트 데이터]\n{customer_facts}\n"
            f"### [1차 판정 결과]\n- 등급: {churn_grade}\n- 사유: {churn_reason}\n- 상세 설명: {churn_explain_reason}\n"
        )
        
        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.1, api_key=OPENAI_API_KEY)
        structured_llm = llm.with_structured_output(ChurnAssessment2)
        prompt = ChatPromptTemplate.from_messages([
            ("system", verifier_system_prompt),
            ("user", "{user_content}")
        ])
        chain = prompt | structured_llm
        verified: ChurnAssessment2 = chain.invoke({"user_content": user_content})
        
        # 룰 기반 안전망 및 포맷 교정 레이어
        grade = verified.grade.strip()
        grade_map = {"Low": "양호", "Medium": "주의", "High": "위험", "low": "양호", "medium": "주의", "high": "위험"}
        grade = grade_map.get(grade, grade)
        if grade not in ["양호", "주의", "위험"]:
            grade = "양호"
            
        reason = verified.reason.strip()
        explain_reason = verified.explain_reason.strip()
        # 마크다운 서식 소거
        for char in ['*', '#', '_', '|', '`', '-']:
            reason = reason.replace(char, '')
            explain_reason = explain_reason.replace(char, '')
            
        if len(reason) > 80:
            print(f"   [Verification Node - ChurnRisk] Warning: Reason length ({len(reason)}) exceeds 80 chars. Truncating.")
            reason = reason[:77] + "..."

        if len(explain_reason) > 300:
            print(f"   [Verification Node - ChurnRisk] Warning: Explain length ({len(explain_reason)}) exceeds 300 chars. Truncating.")
            explain_reason = explain_reason[:297] + "..."
            
        print(f"   [Verification Node - ChurnRisk] Verification completed. Grade: {churn_grade} -> {grade}, Reason length: {len(reason)}, Explain length: {len(explain_reason)}")
        return {"churn_grade": grade, "churn_reason": reason, "churn_explain_reason": explain_reason}
    except Exception as e:
        errors.append(f"verify_churn failed: {str(e)}")
        return {"errors": errors}

def save_results_node(state: Agent2State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    customer_id = state["customer_id"]
    churn_grade = state["churn_grade"]
    churn_reason = state["churn_reason"]
    churn_explain_reason = state.get("churn_explain_reason", "")

    try:
        # 이탈 등급 결과 DB 인서트
        churn_saved = tools.save_churn_level(customer_id, churn_grade, churn_reason, churn_explain_reason)
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
workflow2.add_node("verify_churn", verify_churn_node)
workflow2.add_node("save_results", save_results_node)

workflow2.set_entry_point("load_basic_profile")
workflow2.add_edge("load_basic_profile", "determine_tools")
workflow2.add_edge("determine_tools", "execute_selected_tools")
workflow2.add_edge("execute_selected_tools", "analyze_churn")
workflow2.add_edge("analyze_churn", "verify_churn")
workflow2.add_edge("verify_churn", "save_results")
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
            "churn_explain_reason": None,
            "errors": []
        }
        final_state = self.app.invoke(
            initial_state,
            config={"run_name": "ChurnRiskAgent", "tags": ["churn_risk_agent"]}
        )
        if final_state.get("errors"):
            raise RuntimeError(f"LangGraph execution error in ChurnRiskAgent: {final_state['errors']}")
        return final_state
