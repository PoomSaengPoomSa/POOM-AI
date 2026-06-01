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
class ToolSelection3(BaseModel):
    call_customer: bool = Field(description="고객 기본 프로필 및 자산 비중 정보(customer)를 조회할지 여부.")
    call_customer_account: bool = Field(description="고객의 상세 계좌 유형 및 잔액 정보(customer_account)를 조회할지 여부.")
    call_customer_product: bool = Field(description="고객이 이미 가입 중인 상품 목록(customer_product)을 조회할지 여부. 중복 추천 방지를 위해 권장.")
    call_customer_information: bool = Field(description="최근 1개월간 기록된 고객의 정성적 행동 특징 메모(customer_information)를 조회할지 여부.")
    call_customer_relationship: bool = Field(description="고객의 가족 관계 정보(customer_relationship)를 조회할지 여부.")
    reason: str = Field(description="도구 수집 판단 근거 (한 문장)")

class ProductMatchingDetail3(BaseModel):
    product_id: int = Field(description="주력 상품 ID (pd_id)")
    product_name: str = Field(description="주력 상품 명칭")
    is_suitable: int = Field(description="적합성 여부 (적합=1, 부적합=0, 보유 중=2)")
    reason: str = Field(description="개인화된 맞춤형 추천/제외 이유 (PB 상담 멘트용)")

class ProductMatchingList3(BaseModel):
    matchings: List[ProductMatchingDetail3]

# 2. State 정의
class Agent3State(TypedDict):
    customer_id: int
    report: Optional[Dict[str, Any]]
    portfolio: Optional[Dict[str, Any]]
    tool_selection: Optional[Dict[str, Any]]
    customer_relationship: Optional[List[Dict[str, Any]]]
    customer_products: Optional[List[Dict[str, Any]]]
    customer_accounts: Optional[List[Dict[str, Any]]]
    customer_features: Optional[List[Dict[str, Any]]]
    main_products: Optional[List[Dict[str, Any]]]
    product_matchings: List[Dict[str, Any]]
    errors: List[str]

# 3. 노드 구현체 정의
def load_report_node(state: Agent3State) -> Dict[str, Any]:
    customer_id = state["customer_id"]
    errors = list(state.get("errors", []))
    try:
        report = tools.get_recent_consultation_report(customer_id)
        if not report:
            raise ValueError(f"No consultation report found for customer ID {customer_id}.")
        return {"report": report, "errors": errors}
    except Exception as e:
        errors.append(f"load_report failed: {str(e)}")
        return {"errors": errors}

def determine_tools_node(state: Agent3State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    report = state["report"]
    try:
        system_prompt_ctx = load_prompt("product_matching_determine_tools_system.md")
        user_prompt_template = load_prompt("product_matching_determine_tools_user.md")
        
        user_prompt_ctx = user_prompt_template.format(report_content=report['content'])

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.1, api_key=OPENAI_API_KEY)
        structured_llm_ctx = llm.with_structured_output(ToolSelection3)
        prompt_ctx = ChatPromptTemplate.from_messages([
            ("system", system_prompt_ctx),
            ("user", "{user_content}")
        ])
        chain_ctx = prompt_ctx | structured_llm_ctx
        selection: ToolSelection3 = chain_ctx.invoke({"user_content": user_prompt_ctx})
        return {"tool_selection": selection.dict()}
    except Exception as e:
        errors.append(f"determine_tools failed: {str(e)}")
        return {"errors": errors}

def execute_selected_tools_node(state: Agent3State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    customer_id = state["customer_id"]
    tool_selection = state["tool_selection"]

    portfolio = None
    customer_relationship = []
    customer_products = []
    customer_accounts = []
    customer_features = []

    try:
        if tool_selection.get("call_customer"):
            print("   [Tool Run - ProductMatcher] Running get_portfolio_weight() (customer)")
            portfolio = tools.get_portfolio_weight(customer_id)
        if tool_selection.get("call_customer_account"):
            print("   [Tool Run - ProductMatcher] Running get_customer_accounts() (customer_account)")
            customer_accounts = tools.get_customer_accounts(customer_id)
        if tool_selection.get("call_customer_product"):
            print("   [Tool Run - ProductMatcher] Running get_customer_active_products() (customer_product)")
            customer_products = tools.get_customer_active_products(customer_id)
        if tool_selection.get("call_customer_information"):
            print("   [Tool Run - ProductMatcher] Running get_customer_features() (customer_information) - Last 1 Month")
            customer_features = tools.get_customer_features(customer_id, months=1)
        if tool_selection.get("call_customer_relationship"):
            print("   [Tool Run - ProductMatcher] Running get_customer_relationship() (customer_relationship)")
            customer_relationship = tools.get_customer_relationship(customer_id)

        return {
            "portfolio": portfolio,
            "customer_relationship": customer_relationship,
            "customer_products": customer_products,
            "customer_accounts": customer_accounts,
            "customer_features": customer_features
        }
    except Exception as e:
        errors.append(f"execute_selected_tools failed: {str(e)}")
        return {"errors": errors}

def load_matching_data_node(state: Agent3State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    customer_id = state["customer_id"]
    portfolio = state.get("portfolio")
    try:
        if not portfolio:
            print("   [Tool Run - ProductMatcher] Portfolio not loaded. Fetching portfolio weights as fallback.")
            portfolio = tools.get_portfolio_weight(customer_id)
        
        main_products = tools.get_main_products()
        return {
            "portfolio": portfolio,
            "main_products": main_products
        }
    except Exception as e:
        errors.append(f"load_matching_data failed: {str(e)}")
        return {"errors": errors}

def match_products_node(state: Agent3State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    portfolio = state["portfolio"]
    customer_features = state.get("customer_features")
    main_products = state["main_products"]
    tool_selection = state["tool_selection"]

    customer_products = state.get("customer_products", []) if tool_selection.get("call_customer_product") else None
    customer_relationship = state.get("customer_relationship", []) if tool_selection.get("call_customer_relationship") else None
    customer_accounts = state.get("customer_accounts", []) if tool_selection.get("call_customer_account") else None

    # 중복 추천 배제 검증 역할
    held_product_ids = set()
    if customer_products:
        held_product_ids = {ap["pd_id"] for ap in customer_products}

    held_products = []
    to_evaluate_products = []
    for p in main_products:
        if p["pd_id"] in held_product_ids:
            held_products.append(p)
        else:
            to_evaluate_products.append(p)

    matchings = []
    for p in held_products:
        matchings.append({
            "product_id": p["pd_id"],
            "product_name": p["name"],
            "is_suitable": 2,
            "reason": "고객님이 이미 보유 중인 상품이므로 추천 리스트에서 보류합니다."
        })

    if not to_evaluate_products:
        print("   [LLM Skip - ProductMatcher] All main products are already held. Skipping LLM matching.")
        return {"product_matchings": matchings}

    # 프롬프트 데이터 결합
    if customer_features is not None:
        features_list = [f"[{f['category']}] {f['contents']}" for f in customer_features]
        features_str = "\n".join(features_list) if features_list else "최근 1개월간 기록된 특징 없음."
    else:
        features_str = "[참고] 수집 제외"

    if customer_products is not None:
        ap_list = [f"- 상품 ID: {ap['pd_id']}, 상품명: {ap['product_name']} (만기: {ap['expiration_date']})" for ap in customer_products]
        active_products_str = "\n".join(ap_list) if ap_list else "보유 중인 상품 없음."
    else:
        active_products_str = "[참고] 수집 제외"

    if customer_relationship is not None:
        rel_list = []
        for r in customer_relationship:
            info = r.get("information") or "특이사항 없음"
            rel_list.append(f"- 관계: {r['relationship']}, 세부정보: {info}")
        relationship_str = "\n".join(rel_list) if rel_list else "등록된 가족 내역 없음."
    else:
        relationship_str = "[참고] 수집 제외"

    if customer_accounts is not None:
        acc_list = [f"- 유형: {ac['account_type']}, 계좌: {ac['account_num']}, 잔액: {ac['balance']:,}원" for ac in customer_accounts]
        accounts_str = "\n".join(acc_list) if acc_list else "보유 계좌 없음."
    else:
        accounts_str = "[참고] 수집 제외"

    prod_list = []
    for idx, p in enumerate(to_evaluate_products, 1):
        prod_list.append(
            f"--- [주력 상품 {idx}] ---\n"
            f"- 상품 ID (pd_id): {p['pd_id']}\n"
            f"- 상품명: {p['name']}\n"
            f"- 설명: {p['explanation']}\n"
            f"- 주요 특징: {p['features']}\n"
            f"- 대상 고객군: {p['target_customer']}\n"
            f"- 기대수익률: {p['expected_return']}% ({p['return_type']})\n"
        )
    products_str = "\n".join(prod_list)

    try:
        system_prompt = load_prompt("product_matching_system.md")
        user_prompt_template = load_prompt("product_matching_user.md")

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.3, api_key=OPENAI_API_KEY)
        structured_llm = llm.with_structured_output(ProductMatchingList3)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{user_content}")
        ])

        user_prompt = user_prompt_template.format(
            name=portfolio["name"],
            grade=portfolio["grade"],
            tendency=portfolio["tendency"],
            total_assets=portfolio["total_assets"],
            deposit=portfolio["deposit"],
            loan=portfolio["loan"],
            features_str=features_str,
            products_str=products_str,
            product_id_type="int",
            active_products_str=active_products_str,
            relationship_str=relationship_str,
            accounts_str=accounts_str
        )

        chain = prompt | structured_llm
        result: ProductMatchingList3 = chain.invoke({"user_content": user_prompt})

        for m in result.matchings:
            matchings.append({
                "product_id": m.product_id,
                "product_name": m.product_name,
                "is_suitable": m.is_suitable,
                "reason": m.reason.strip()
            })
        return {"product_matchings": matchings}
    except Exception as e:
        errors.append(f"match_products failed: {str(e)}")
        return {"errors": errors}

def verify_matchings_node(state: Agent3State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    product_matchings = state.get("product_matchings", [])
    
    cleaned_matchings = []
    try:
        for m in product_matchings:
            reason = m.get("reason", "")
            if reason:
                # 1. 2인칭 -> 3인칭 교정 ("고객님" -> "고객")
                cleaned = reason.replace("고객님", "고객")
                
                # 2. 마크다운 기호 제거
                for char in ['*', '#', '_', '|', '`', '-']:
                    cleaned = cleaned.replace(char, '')
                    
                # 3. 3문장 이내 제약조건 검증 및 교정
                sentences = [s.strip() for s in cleaned.split('.') if s.strip()]
                if len(sentences) > 3:
                    print(f"   [Verification Node - ProductMatcher] Warning: Product {m['product_id']} reason has {len(sentences)} sentences. Truncating to 3 sentences.")
                    cleaned = ". ".join(sentences[:3]) + "."
                elif sentences:
                    if not cleaned.endswith('.'):
                        cleaned += "."
                
                m["reason"] = cleaned.strip()
            cleaned_matchings.append(m)
        
        print(f"   [Verification Node - ProductMatcher] Cleaned and verified {len(cleaned_matchings)} product matchings.")
        return {"product_matchings": cleaned_matchings}
    except Exception as e:
        errors.append(f"verify_matchings failed: {str(e)}")
        return {"errors": errors}

def save_matching_node(state: Agent3State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    customer_id = state["customer_id"]
    product_matchings = state["product_matchings"]
    try:
        for m in product_matchings:
            tools.save_product_matching(
                product_id=m["product_id"],
                customer_id=customer_id,
                is_suitable=m["is_suitable"],
                reason=m["reason"]
            )
        print(f"  [+] Sub Agent 3: 주력 상품 매칭 분석 DB 적재 완료! (고객 ID: {customer_id}, 총 {len(product_matchings)}개 상품)")
        return {}
    except Exception as e:
        errors.append(f"save_matching failed: {str(e)}")
        return {"errors": errors}

# LangGraph 그래프 조립
workflow3 = StateGraph(Agent3State)
workflow3.add_node("load_report", load_report_node)
workflow3.add_node("determine_tools", determine_tools_node)
workflow3.add_node("execute_selected_tools", execute_selected_tools_node)
workflow3.add_node("load_matching_data", load_matching_data_node)
workflow3.add_node("match_products", match_products_node)
workflow3.add_node("verify_matchings", verify_matchings_node)
workflow3.add_node("save_matching", save_matching_node)

workflow3.set_entry_point("load_report")
workflow3.add_edge("load_report", "determine_tools")
workflow3.add_edge("determine_tools", "execute_selected_tools")
workflow3.add_edge("execute_selected_tools", "load_matching_data")
workflow3.add_edge("load_matching_data", "match_products")
workflow3.add_edge("match_products", "verify_matchings")
workflow3.add_edge("verify_matchings", "save_matching")
workflow3.add_edge("save_matching", END)

compiled_app3 = workflow3.compile()


class ProductMatchingAgent:
    """
    Sub Agent 3: 주력 금융 상품 적합성 평가 Agent
    """
    def __init__(self, model_name: str = None):
        global DEFAULT_MODEL
        if model_name:
            DEFAULT_MODEL = model_name
        self.app = compiled_app3

    @traceable(name="ProductMatchingAgent", run_type="chain", tags=["ProductMatchingAgent"])
    def run(self, customer_id: int) -> Dict[str, Any]:
        initial_state: Agent3State = {
            "customer_id": customer_id,
            "report": None,
            "portfolio": None,
            "tool_selection": None,
            "customer_relationship": [],
            "customer_products": [],
            "customer_accounts": [],
            "customer_features": [],
            "main_products": None,
            "product_matchings": [],
            "errors": []
        }
        final_state = self.app.invoke(
            initial_state,
            config={"run_name": "ProductMatchingAgent", "tags": ["product_matching_agent"]}
        )
        if final_state.get("errors"):
            raise RuntimeError(f"LangGraph execution error in ProductMatchingAgent: {final_state['errors']}")
        return final_state
