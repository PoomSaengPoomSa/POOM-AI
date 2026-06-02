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
            print("   [Tool Run - ProductMatcher] Running get_customer() (customer)")
            portfolio = tools.get_customer(customer_id)
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
            portfolio = tools.get_customer(customer_id)
        
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
        
    portfolio = state["portfolio"]
    customer_products = state.get("customer_products", [])
    product_matchings = state.get("product_matchings", [])
    
    if not product_matchings:
        return {}
        
    try:
        # LLM 기반 상품 매칭 적합성 및 사유 정합성 검증 레이어 가동
        verifier_system_prompt = (
            "당신은 은행 본점의 금융 상품 추천 품질 심사역입니다.\n"
            "제시된 [고객 정보] 및 [상품 가입 목록], 그리고 1차 도출된 [상품별 매칭 및 추천 사유 리스트]를 대조하여, 추천이 금융 비즈니스 가이드라인을 준수하는지 정밀 평가하고 필요시 등급 및 추천 멘트를 교정해 주십시오.\n\n"
            "### [금융 비즈니스 검증 항목]\n"
            "1. **투자 성향 초과 제한**: 고객 투자 성향(예: 안정형, 안정추구형)에 맞지 않는 초고위험 주식형 펀드/글로벌 펀드가 '적합(1)'으로 판정되었다면, 반드시 '부적합(0)'으로 변경하고 사유를 성향 불일치로 교정하십시오.\n"
            "2. **중복 추천 배제**: 제공된 고객의 기가입 상품 목록에 존재하는 상품임에도 '보유 중(2)'이 아닌 '적합(1)' 등으로 판정되었다면, 무조건 '보유 중(2)'으로 판정하고 사유를 '고객이 이미 보유하고 있는 상품이므로 추천에서 배제합니다.'로 교정하십시오.\n"
            "3. **판단과 사유의 일치성**: 판정 결과(적합=1, 부적합=0, 보유 중=2)와 추천/제외 사유 내용이 비즈니스 논리적으로 일관되어야 하며 서로 모순이 없어야 합니다.\n\n"
            "### [출력 및 규격 제한 (엄격 적용)]\n"
            "- 반드시 제공된 JSON 스키마 구조를 준수하여 출력하십시오. 부가적인 텍스트는 완전히 배제하십시오.\n"
            "- 각 상품의 추천 사유는 마크다운 서식을 제거하고, **공백 포함 3문장 이내**의 한글 경어체(3인칭 호칭)여야 합니다."
        )
        
        # 기가입 상품 목록 팩트 요약
        held_list = []
        if customer_products:
            for ap in customer_products:
                held_list.append(f"- 상품 ID: {ap['pd_id']}, 상품명: {ap['product_name']}")
        held_products_str = "\n".join(held_list) if held_list else "보유 중인 상품 없음."
        
        customer_info_str = (
            f"- 고객명: {portfolio.get('name')}\n"
            f"- 투자성향: {portfolio.get('tendency')}\n"
            f"- 등급: {portfolio.get('grade')}\n"
            f"- 총자산: {portfolio.get('total_assets', 0):,}원\n"
        )
        
        # 1차 판정 리스트 요약
        eval_list = []
        for pm in product_matchings:
            eval_list.append(
                f"- 상품 ID (pd_id): {pm['product_id']}\n"
                f"  * 상품명: {pm['product_name']}\n"
                f"  * 1차 판정 적합성 (is_suitable): {pm['is_suitable']}\n"
                f"  * 1차 사유 (reason): {pm['reason']}"
            )
        eval_products_str = "\n".join(eval_list)
        
        user_content = (
            f"### [고객 정보]\n{customer_info_str}\n"
            f"### [상품 가입 목록]\n{held_products_str}\n"
            f"### [상품별 매칭 및 추천 사유 리스트]\n{eval_products_str}\n"
        )
        
        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.1, api_key=OPENAI_API_KEY)
        structured_llm = llm.with_structured_output(ProductMatchingList3)
        prompt = ChatPromptTemplate.from_messages([
            ("system", verifier_system_prompt),
            ("user", "{user_content}")
        ])
        chain = prompt | structured_llm
        verified: ProductMatchingList3 = chain.invoke({"user_content": user_content})
        
        # 룰 기반 안전망 및 포맷 교정 레이어
        cleaned_matchings = []
        held_product_ids = {ap["pd_id"] for ap in customer_products} if customer_products else set()
        
        for m in verified.matchings:
            is_suitable = m.is_suitable
            reason = m.reason.strip()
            
            # 중복 체크 룰베이스 강제 검증 (Safety Net)
            if m.product_id in held_product_ids:
                is_suitable = 2
                reason = "고객이 이미 보유하고 있는 상품이므로 추천에서 배제합니다."
            
            # 인칭 교정 ("고객님" -> "고객")
            cleaned = reason.replace("고객님", "고객")
            
            # 마크다운 기호 제거
            for char in ['*', '#', '_', '|', '`', '-']:
                cleaned = cleaned.replace(char, '')
                
            # 문장 수 제약조건 교정
            sentences = [s.strip() for s in cleaned.split('.') if s.strip()]
            if len(sentences) > 3:
                print(f"   [Verification Node - ProductMatcher] Warning: Product {m.product_id} reason has {len(sentences)} sentences. Truncating.")
                cleaned = ". ".join(sentences[:3]) + "."
            elif sentences:
                if not cleaned.endswith('.'):
                    cleaned += "."
            
            cleaned_matchings.append({
                "product_id": m.product_id,
                "product_name": m.product_name,
                "is_suitable": is_suitable,
                "reason": cleaned.strip()
            })
            
        print(f"   [Verification Node - ProductMatcher] Verified and cleaned {len(cleaned_matchings)} product matchings.")
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
