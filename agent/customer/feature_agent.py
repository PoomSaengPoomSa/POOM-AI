import os
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from . import tools

# Absolute path resolution to strictly load .env from agent/customer/.env
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Map LANGSMITH_ environment variables to LANGCHAIN_ standard tracing variables
if os.getenv("LANGSMITH_TRACING") == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
if os.getenv("LANGSMITH_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
if os.getenv("LANGSMITH_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT").strip('"\'')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        f"Configuration error: OPENAI_API_KEY is missing from the .env file. "
        f"Please verify your local .env file at {env_path}"
    )

# Global default model configuration
DEFAULT_MODEL = "gpt-4o-mini"

# Helper function to dynamically load prompts from markdown files
def load_prompt(filename: str) -> str:
    """
    Utility to load prompt templates from the prompt directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "prompt", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# 1. Pydantic Models for Customer Feature Matcher
class ExtractedFeature(BaseModel):
    category: str = Field(
        description="특징 카테고리. 반드시 '관계', '성향', '상품', '기호', '건강', '기타' 중 하나여야 합니다."
    )
    contents: str = Field(
        description="80자 이내의 한국어 경어체 특징 설명 한 문장. (VARCHAR(500) 길이 제한이 있으므로 반드시 간결하게 작성해야 합니다.)"
    )

class ExtractedFeatureList(BaseModel):
    features: List[ExtractedFeature]

class ContextSelection(BaseModel):
    call_get_customer_relationship: bool = Field(description="가족 관계(배우자, 자녀 등) 정보를 가져올지 여부. 상담 보고서 내용에 가족, 자녀 교육, 결혼, 상속, 배우자 관련 언급이 있을 때 참(True)으로 설정합니다.")
    call_get_customer_active_products: bool = Field(description="고객이 이미 보유/가입 중인 금융 상품 목록을 가져올지 여부. 추천 시 이미 보유 중인 상품의 중복 추천을 방지하기 위해 참(True)으로 설정합니다.")
    call_get_customer_accounts: bool = Field(description="고객의 예적금 계좌 잔액 및 세부 계좌 타입 정보를 가져올지 여부. 상담 보고서에 특정 계좌 예치 여력이나 구체적인 통장 리밸런싱이 언급된 경우 참(True)으로 설정합니다.")
    reason: str = Field(description="상담 보고서의 내용을 토대로 해당 추가 정보들이 필요하다고 에이전트가 진단한 판단 근거 (한 문장)")

class ProductMatchingDetail(BaseModel):
    product_id: int = Field(description="주력 상품 ID (pd_id)")
    product_name: str = Field(description="주력 상품 명칭")
    is_suitable: int = Field(description="적합성 여부 (적합=1, 부적합=0, 보유 중=2)")
    reason: str = Field(description="개인화된 맞춤형 추천/제외 이유 (PB 상담 멘트용)")

class ProductMatchingList(BaseModel):
    matchings: List[ProductMatchingDetail]

# 2. State Definition for Customer Feature Matcher
class Agent2State(TypedDict):
    customer_id: int
    report: Optional[Dict[str, Any]]
    context_selection: Optional[Dict[str, Any]]
    extracted_features: List[Dict[str, Any]]
    customer_relationship: Optional[List[Dict[str, Any]]]
    active_products: Optional[List[Dict[str, Any]]]
    customer_accounts: Optional[List[Dict[str, Any]]]
    customer_profile: Optional[Dict[str, Any]]
    recent_features_1m: Optional[List[Dict[str, Any]]]
    main_products: Optional[List[Dict[str, Any]]]
    product_matchings: List[Dict[str, Any]]
    errors: List[str]

# 3. Graph Node Implementations for Customer Feature Matcher
def load_report_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 1: Load the latest consultation report for the customer.
    """
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

def extract_features_and_determine_context_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 2: First extract features from consultation report,
    and then analyze report to decide what additional context to fetch.
    """
    errors = list(state.get("errors", []))
    if errors:
        return {}

    report = state["report"]
    try:
        # LLM 1: Extract features
        system_prompt_feat = load_prompt("feature_extraction_system.md")
        user_prompt_feat = load_prompt("feature_extraction_user.md")

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.3, api_key=OPENAI_API_KEY)
        structured_llm_feat = llm.with_structured_output(ExtractedFeatureList)

        prompt_feat = ChatPromptTemplate.from_messages([
            ("system", system_prompt_feat),
            ("user", user_prompt_feat)
        ])

        chain_feat = prompt_feat | structured_llm_feat
        result_feat: ExtractedFeatureList = chain_feat.invoke({
            "report_content": report["content"]
        })

        # Standardize and truncate features to DB constraints
        valid_features = []
        allowed_categories = ["관계", "성향", "상품", "기호", "건강", "기타"]
        for f in result_feat.features:
            cat = f.category.strip()
            if cat not in allowed_categories:
                continue
            cont = f.contents.strip()
            if len(cont) > 500:
                cont = cont[:497] + "..."
            valid_features.append({"category": cat, "contents": cont})

        # LLM 2: Determine context tools to run
        system_prompt_ctx = (
            "당신은 고객의 상담 보고서 원문을 정밀하게 분석하여, "
            "이 상담 내용을 근거로 해당 고객에게 금융 상품을 정밀 매칭/추천하기 위해 "
            "추가로 조회해야 할 내부 데이터베이스 정보 도구를 판단하는 의사결정 에이전트입니다.\n\n"
            "추가 조회 가능한 맥락 정보 목록:\n"
            "1. `get_customer_relationship` (가족 관계): 상담에 배우자, 자녀 교육, 상속, 결혼 등 가족/가구원 관련 이슈가 직접 언급되었을 때만 가져옵니다.\n"
            "2. `get_customer_active_products` (이미 보유 중인 상품): 상품을 추천할 때 이미 보유 중인 상품과의 중복 추천을 원천 차단하기 위해 반드시 참(True)으로 설정하여 조회합니다.\n"
            "3. `get_customer_accounts` (예적금 계좌 잔액): 상담에 통장 정리, 특정 계좌 잔액 리밸런싱, 예치금 재투자가 언급되었을 때 가져옵니다."
        )

        user_prompt_ctx = f"상담 보고서 본문:\n\"\"\"\n{report['content']}\n\"\"\"\n\n위 상담 내용에 기반하여 정밀 상품 추천을 위해 가져와야 할 정보들을 결정해 주십시오."
        
        structured_llm_ctx = llm.with_structured_output(ContextSelection)
        prompt_ctx = ChatPromptTemplate.from_messages([
            ("system", system_prompt_ctx),
            ("user", user_prompt_ctx)
        ])
        chain_ctx = prompt_ctx | structured_llm_ctx
        selection: ContextSelection = chain_ctx.invoke({})

        return {
            "extracted_features": valid_features,
            "context_selection": selection.dict()
        }
    except Exception as e:
        errors.append(f"extract_features_and_determine_context failed: {str(e)}")
        return {"errors": errors}

def save_features_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 3: Save extracted features to database customer_information table.
    """
    errors = list(state.get("errors", []))
    if errors:
        return {}

    customer_id = state["customer_id"]
    extracted_features = state["extracted_features"]

    try:
        for f in extracted_features:
            tools.save_customer_feature(
                customer_id=customer_id,
                category=f["category"],
                contents=f["contents"]
            )
        print(f"  [+] Saved {len(extracted_features)} extracted features to DB for customer {customer_id}")
        return {}
    except Exception as e:
        errors.append(f"save_features failed: {str(e)}")
        return {"errors": errors}

def fetch_context_data_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 4: Fetch context database records dynamically.
    """
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
            print("   [Tool Run] Running get_customer_relationship()")
            customer_relationship = tools.get_customer_relationship(customer_id)
        else:
            print("   [Tool Skip] Skipping get_customer_relationship based on AI judgment.")

        if context_selection.get("call_get_customer_active_products"):
            print("   [Tool Run] Running get_customer_active_products()")
            active_products = tools.get_customer_active_products(customer_id)
        else:
            print("   [Tool Skip] Skipping get_customer_active_products based on AI judgment.")

        if context_selection.get("call_get_customer_accounts"):
            print("   [Tool Run] Running get_customer_accounts()")
            customer_accounts = tools.get_customer_accounts(customer_id)
        else:
            print("   [Tool Skip] Skipping get_customer_accounts based on AI judgment.")

        return {
            "customer_relationship": customer_relationship,
            "active_products": active_products,
            "customer_accounts": customer_accounts
        }
    except Exception as e:
        errors.append(f"fetch_context_data failed: {str(e)}")
        return {"errors": errors}

def load_matching_data_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 5: Fetch customer profile and active main products.
    """
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
            "main_products": main_products,
            "errors": errors
        }
    except Exception as e:
        errors.append(f"load_matching_data failed: {str(e)}")
        return {"errors": errors}

def match_products_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 6: LangChain ChatModel call with Structured Output to match main products against customer profile.
    Uses dynamically gathered active products, family relationship, and account balances to make intelligent matches.
    """
    errors = list(state.get("errors", []))
    if errors:
        return {}

    profile = state["customer_profile"]
    features_1m = state["recent_features_1m"]
    main_products = state["main_products"]
    context_selection = state["context_selection"]

    # Retrieve dynamically loaded context
    active_products = state.get("active_products", []) if context_selection.get("call_get_customer_active_products") else None
    relationship = state.get("customer_relationship", []) if context_selection.get("call_get_customer_relationship") else None
    accounts = state.get("customer_accounts", []) if context_selection.get("call_get_customer_accounts") else None

    # Divide products into held and to-evaluate lists
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
    # Auto-populate matching results for held products
    for p in held_products:
        matchings.append({
            "product_id": p["pd_id"],
            "product_name": p["name"],
            "is_suitable": 2,
            "reason": "고객님이 보유 중인 상품입니다."
        })

    # If there are no products left to evaluate, skip LLM entirely
    if not to_evaluate_products:
        print(f"   [LLM Skip] All main products are already held by customer {customer_id}. Skipping LLM matching.")
        return {"product_matchings": matchings}

    # Format inputs for remaining products
    features_list = []
    for f in features_1m:
        features_list.append(f"[{f['category']}] {f['contents']}")
    features_str = "\n".join(features_list) if features_list else "최근 1개월간 기록된 고객 특징 정보 없음."

    # Format dynamic contexts or explain if skipped
    if active_products is not None:
        ap_list = []
        for ap in active_products:
            ap_list.append(f"- 상품 ID: {ap['pd_id']}, 상품명: {ap['product_name']} (가입일: {ap['opening_date']}, 만기일: {ap['expiration_date']})")
        active_products_str = "\n".join(ap_list) if ap_list else "현재 고객이 보유/가입 중인 상품 없음."
    else:
        active_products_str = "[참고] 에이전트 수집 제외: 보유 상품 정보가 적합성 매칭 데이터에서 제외되었습니다."

    if relationship is not None:
        rel_list = []
        for r in relationship:
            rel_list.append(f"- 관계: {r['relationship']}, 생년월일: {r['birthday']}, 직업: {r['job']}, 배우자여부: {r['is_spouse']}, 결혼기념일: {r['wedding_date']}")
        relationship_str = "\n".join(rel_list) if rel_list else "등록된 가족 관계 정보 없음."
    else:
        relationship_str = "[참고] 에이전트 수집 제외: 가족 관계 정보가 적합성 매칭 데이터에서 제외되었습니다."

    if accounts is not None:
        acc_list = []
        for ac in accounts:
            acc_list.append(f"- 계좌유형: {ac['account_type']}, 계좌번호: {ac['account_num']}, 잔액: {ac['balance']:,}원")
        accounts_str = "\n".join(acc_list) if acc_list else "보유 중인 계좌 내역 없음."
    else:
        accounts_str = "[참고] 에이전트 수집 제외: 계좌별 잔액 정보가 적합성 매칭 데이터에서 제외되었습니다."

    prod_list = []
    for idx, p in enumerate(to_evaluate_products, 1):
        prod_list.append(
            f"--- [주력 상품 {idx}] ---\n"
            f"- 상품 ID (pd_id): {p['pd_id']}\n"
            f"- 상품명: {p['name']}\n"
            f"- 설명: {p['explanation']}\n"
            f"- 종류: {p['type']}\n"
            f"- 주요 특징: {p['features']}\n"
            f"- 추천 대상 고객군: {p['target_customer']}\n"
            f"- 기대수익률: {p['expected_return']}% ({p['return_type']})\n"
        )
    products_str = "\n".join(prod_list) if prod_list else "활성화된 본점 주력 상품 정보 없음."

    try:
        system_prompt = load_prompt("product_matching_system.md")
        # Enhance system prompt rules to handle dynamic context logic (removed held products rule as they are handled in python)
        dynamic_matching_rules = (
            "\n\n## [중요 추가 매칭 규칙]\n"
            "1. 가족 관계 정보가 제공된 경우, 가구원의 생일, 자녀의 나이, 배우자 여부 등을 상품의 target_customer 및 특징과 대조하여 "
            "가족 결혼자금 준비, 자녀 학자금 마련, 은퇴 부부 생활 자금 등 개인화된 시나리오 기반의 적합도(1)를 추천 사유와 함께 작성해 주십시오.\n"
            "2. 계좌별 잔액 정보가 제공된 경우, 특정 통장의 가용 자금 여유가 상품 가입 조건에 맞는지 확인하여 구체적인 예치 권고 사유를 제안하십시오."
        )
        system_prompt += dynamic_matching_rules

        user_prompt_template = load_prompt("product_matching_user.md")

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.3, api_key=OPENAI_API_KEY)
        structured_llm = llm.with_structured_output(ProductMatchingList)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{user_content}")
        ])

        # Customize user prompt parameters to include dynamic context
        raw_user_prompt = user_prompt_template.format(
            name=profile["name"],
            grade=profile["grade"],
            tendency=profile["tendency"],
            total_assets=profile["total_assets"],
            deposit=profile["deposit"],
            loan=profile["loan"],
            features_str=features_str,
            products_str=products_str,
            product_id_type="int"
        )
        
        # Append dynamic contexts to raw user prompt
        dynamic_user_prompt = (
            f"{raw_user_prompt}\n\n"
            f"## [추가 에이전트 수집 맥락 정보]\n"
            f"1. 고객의 현재 보유/가입 중인 금융 상품 목록:\n{active_products_str}\n\n"
            f"2. 고객의 가구원 및 가족 관계 정보:\n{relationship_str}\n\n"
            f"3. 고객의 계좌유형별 잔액 정보:\n{accounts_str}\n"
        )

        chain = prompt | structured_llm
        result: ProductMatchingList = chain.invoke({"user_content": dynamic_user_prompt})

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

def save_matching_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 7: Save product matching suitability assessments into the product_matching table.
    """
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
        print(f"  [+] Saved {len(product_matchings)} product suitability matching results to DB for customer {customer_id}")
        return {}
    except Exception as e:
        errors.append(f"save_matching failed: {str(e)}")
        return {"errors": errors}


# 4. Compiled State Graph for Customer Feature Matcher
workflow2 = StateGraph(Agent2State)

workflow2.add_node("load_report", load_report_node)
workflow2.add_node("extract_features_and_determine_context", extract_features_and_determine_context_node)
workflow2.add_node("save_features", save_features_node)
workflow2.add_node("fetch_context_data", fetch_context_data_node)
workflow2.add_node("load_matching_data", load_matching_data_node)
workflow2.add_node("match_products", match_products_node)
workflow2.add_node("save_matching", save_matching_node)

workflow2.set_entry_point("load_report")

workflow2.add_edge("load_report", "extract_features_and_determine_context")
workflow2.add_edge("extract_features_and_determine_context", "save_features")
workflow2.add_edge("save_features", "fetch_context_data")
workflow2.add_edge("fetch_context_data", "load_matching_data")
workflow2.add_edge("load_matching_data", "match_products")
workflow2.add_edge("match_products", "save_matching")
workflow2.add_edge("save_matching", END)

compiled_app2 = workflow2.compile()


class CustomerFeatureAgent:
    """
    Customer Feature Agent (고객 특징 분석 에이전트)
    Extracts features from consultation reports and performs product suitability matching.
    """
    def __init__(self, model_name: str = None):
        global DEFAULT_MODEL
        if model_name:
            DEFAULT_MODEL = model_name
        self.app = compiled_app2

    def run(self, customer_id: int) -> Dict[str, Any]:
        """
        Run the complete compiled LangGraph workflow for Customer Feature Agent.
        """
        initial_state: Agent2State = {
            "customer_id": customer_id,
            "report": None,
            "context_selection": None,
            "extracted_features": [],
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
            config={"run_name": "CustomerFeatureAgent", "tags": ["feature_agent"]}
        )
        
        if final_state.get("errors"):
            raise RuntimeError(f"LangGraph execution encountered errors in CustomerFeatureAgent: {final_state['errors']}")
            
        return final_state
