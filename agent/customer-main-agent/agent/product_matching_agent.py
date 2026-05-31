import os
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# db 및 tool 임포트
from ..db import root_env_path
from ..tool import tools

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
class ContextSelection32(BaseModel):
    call_get_customer_relationship: bool = Field(description="가족 관계 정보를 가져올지 여부.")
    call_get_customer_active_products: bool = Field(description="고객이 이미 가입 중인 상품 목록을 가져올지 여부.")
    call_get_customer_accounts: bool = Field(description="계좌 잔액 정보를 가져올지 여부.")
    reason: str = Field(description="데이터 수집 의사결정 판단 근거 (한 문장)")

class ProductMatchingDetail32(BaseModel):
    product_id: int = Field(description="주력 상품 ID (pd_id)")
    product_name: str = Field(description="주력 상품 명칭")
    is_suitable: int = Field(description="적합성 여부 (적합=1, 부적합=0, 보유 중=2)")
    reason: str = Field(description="개인화된 맞춤형 추천/제외 이유 (PB 상담 멘트용)")

class ProductMatchingList32(BaseModel):
    matchings: List[ProductMatchingDetail32]

# 2. State 정의
class Agent32State(TypedDict):
    customer_id: int
    report: Optional[Dict[str, Any]]
    context_selection: Optional[Dict[str, Any]]
    customer_relationship: Optional[List[Dict[str, Any]]]
    active_products: Optional[List[Dict[str, Any]]]
    customer_accounts: Optional[List[Dict[str, Any]]]
    customer_profile: Optional[Dict[str, Any]]
    recent_features_1m: Optional[List[Dict[str, Any]]]
    main_products: Optional[List[Dict[str, Any]]]
    product_matchings: List[Dict[str, Any]]
    errors: List[str]

# 3. 노드 구현체 정의
def load_report_node(state: Agent32State) -> Dict[str, Any]:
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

def determine_context_node(state: Agent32State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    report = state["report"]
    try:
        system_prompt_ctx = load_prompt("product_matching_determine_context_system.md")
        user_prompt_template = load_prompt("product_matching_determine_context_user.md")
        
        user_prompt_ctx = user_prompt_template.format(report_content=report['content'])

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.3, api_key=OPENAI_API_KEY)
        structured_llm_ctx = llm.with_structured_output(ContextSelection32)
        prompt_ctx = ChatPromptTemplate.from_messages([
            ("system", system_prompt_ctx),
            ("user", "{user_content}")
        ])
        chain_ctx = prompt_ctx | structured_llm_ctx
        selection: ContextSelection32 = chain_ctx.invoke({"user_content": user_prompt_ctx})
        return {"context_selection": selection.dict()}
    except Exception as e:
        errors.append(f"determine_context failed: {str(e)}")
        return {"errors": errors}

def fetch_context_data_node(state: Agent32State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    customer_id = state["customer_id"]
    context_selection = state["context_selection"]

    customer_relationship = []
    active_products = []
    customer_accounts = []

    try:
        if context_selection.get("call_get_customer_relationship"):
            print("   [Tool Run - ProductMatcher] Running get_customer_relationship()")
            customer_relationship = tools.get_customer_relationship(customer_id)
        if context_selection.get("call_get_customer_active_products"):
            print("   [Tool Run - ProductMatcher] Running get_customer_active_products()")
            active_products = tools.get_customer_active_products(customer_id)
        if context_selection.get("call_get_customer_accounts"):
            print("   [Tool Run - ProductMatcher] Running get_customer_accounts()")
            customer_accounts = tools.get_customer_accounts(customer_id)

        return {
            "customer_relationship": customer_relationship,
            "active_products": active_products,
            "customer_accounts": customer_accounts
        }
    except Exception as e:
        errors.append(f"fetch_context_data failed: {str(e)}")
        return {"errors": errors}

def load_matching_data_node(state: Agent32State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    customer_id = state["customer_id"]
    try:
        profile = tools.get_portfolio_weight(customer_id)
        recent_features_1m = tools.get_customer_features(customer_id, months=1)
        main_products = tools.get_main_products()
        return {
            "customer_profile": profile,
            "recent_features_1m": recent_features_1m,
            "main_products": main_products
        }
    except Exception as e:
        errors.append(f"load_matching_data failed: {str(e)}")
        return {"errors": errors}

def match_products_node(state: Agent32State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    profile = state["customer_profile"]
    features_1m = state["recent_features_1m"]
    main_products = state["main_products"]
    context_selection = state["context_selection"]

    active_products = state.get("active_products", []) if context_selection.get("call_get_customer_active_products") else None
    relationship = state.get("customer_relationship", []) if context_selection.get("call_get_customer_relationship") else None
    accounts = state.get("customer_accounts", []) if context_selection.get("call_get_customer_accounts") else None

    # 중복 추천 배제 검증 (Verification Node) 역할
    held_product_ids = set()
    if active_products:
        held_product_ids = {ap["pd_id"] for ap in active_products}

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
    features_list = [f"[{f['category']}] {f['contents']}" for f in features_1m]
    features_str = "\n".join(features_list) if features_list else "최근 1개월간 기록된 특징 없음."

    if active_products is not None:
        ap_list = [f"- 상품 ID: {ap['pd_id']}, 상품명: {ap['product_name']} (만기: {ap['expiration_date']})" for ap in active_products]
        active_products_str = "\n".join(ap_list) if ap_list else "보유 중인 상품 없음."
    else:
        active_products_str = "[참고] 수집 제외"

    if relationship is not None:
        rel_list = []
        for r in relationship:
            info = r.get("information") or "특이사항 없음"
            rel_list.append(f"- 관계: {r['relationship']}, 세부정보: {info}")
        relationship_str = "\n".join(rel_list) if rel_list else "등록된 가족 내역 없음."
    else:
        relationship_str = "[참고] 수집 제외"

    if accounts is not None:
        acc_list = [f"- 유형: {ac['account_type']}, 계좌: {ac['account_num']}, 잔액: {ac['balance']:,}원" for ac in accounts]
        accounts_str = "\n".join(acc_list) if acc_list else "보유 계좌 없음."
    else:
        accounts_str = "[참고] 보유 계좌 없음."

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
        structured_llm = llm.with_structured_output(ProductMatchingList32)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{user_content}")
        ])

        user_prompt = user_prompt_template.format(
            name=profile["name"],
            grade=profile["grade"],
            tendency=profile["tendency"],
            total_assets=profile["total_assets"],
            deposit=profile["deposit"],
            loan=profile["loan"],
            features_str=features_str,
            products_str=products_str,
            product_id_type="int",
            active_products_str=active_products_str,
            relationship_str=relationship_str,
            accounts_str=accounts_str
        )

        chain = prompt | structured_llm
        result: ProductMatchingList32 = chain.invoke({"user_content": user_prompt})

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

def save_matching_node(state: Agent32State) -> Dict[str, Any]:
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
        print(f"  [+] Sub Agent 4: 주력 상품 매칭 분석 DB 적재 완료! (고객 ID: {customer_id}, 총 {len(product_matchings)}개 상품)")
        return {}
    except Exception as e:
        errors.append(f"save_matching failed: {str(e)}")
        return {"errors": errors}

# LangGraph 그래프 조립
workflow32 = StateGraph(Agent32State)
workflow32.add_node("load_report", load_report_node)
workflow32.add_node("determine_context", determine_context_node)
workflow32.add_node("fetch_context_data", fetch_context_data_node)
workflow32.add_node("load_matching_data", load_matching_data_node)
workflow32.add_node("match_products", match_products_node)
workflow32.add_node("save_matching", save_matching_node)

workflow32.set_entry_point("load_report")
workflow32.add_edge("load_report", "determine_context")
workflow32.add_edge("determine_context", "fetch_context_data")
workflow32.add_edge("fetch_context_data", "load_matching_data")
workflow32.add_edge("load_matching_data", "match_products")
workflow32.add_edge("match_products", "save_matching")
workflow32.add_edge("save_matching", END)

compiled_app32 = workflow32.compile()


class ProductMatchingAgent:
    """
    Sub Agent 4: 주력 상품 매칭 Agent
    """
    def __init__(self, model_name: str = None):
        global DEFAULT_MODEL
        if model_name:
            DEFAULT_MODEL = model_name
        self.app = compiled_app32

    def run(self, customer_id: int) -> Dict[str, Any]:
        initial_state: Agent32State = {
            "customer_id": customer_id,
            "report": None,
            "context_selection": None,
            "customer_relationship": [],
            "active_products": [],
            "customer_accounts": [],
            "customer_profile": None,
            "recent_features_1m": None,
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
