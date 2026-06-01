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

# 1. Pydantic Models for Customer Feature Extractor, Refiner, and Keyword Extractor
class ExtractedFeature(BaseModel):
    category: str = Field(
        description="특징 카테고리. 반드시 '관계', '성향', '상품', '기호', '건강', '기타' 중 하나여야 합니다."
    )
    contents: str = Field(
        description="80자 이내의 한국어 경어체 특징 설명 한 문장. (VARCHAR(500) 길이 제한이 있으므로 반드시 간결하게 작성해야 합니다.)"
    )

class ExtractedFeatureList(BaseModel):
    features: List[ExtractedFeature]

class RefinedFeatureDecision(BaseModel):
    action: str = Field(
        description="결정된 작업. 반드시 'ADD', 'SKIP', 'UPDATE' 중 하나여야 합니다."
    )
    category: str = Field(
        description="특징 카테고리. 반드시 '관계', '성향', '상품', '기호', '건강', '기타' 중 하나여야 합니다."
    )
    contents: str = Field(
        description="80자 이내의 한국어 경어체 특징 설명 한 문장. (ADD 또는 UPDATE 시 필수, SKIP 시 공백 가능)"
    )
    target_ci_id: Optional[int] = Field(
        description="UPDATE 결정 시, 수정 대상이 되는 기존 특징의 ci_id. ADD 또는 SKIP 시에는 Null(또는 생략)."
    )
    reason: str = Field(
        description="이 결정을 내린 판단 근거 (한 문장)"
    )

class RefinedFeatureList(BaseModel):
    decisions: List[RefinedFeatureDecision]

class KeywordList(BaseModel):
    keywords: List[str] = Field(
        description="고객의 최근 한달 내 특징들로부터 추출한 핵심 키워드 리스트 (5~8개 내외)"
    )


# 2. State Definition for Customer Feature Matcher (Upgraded Version)
class Agent2State(TypedDict):
    customer_id: int
    report: Optional[Dict[str, Any]]
    existing_features: List[Dict[str, Any]]
    extracted_features: List[Dict[str, Any]]
    refined_decisions: List[Dict[str, Any]]
    features_last_1m: Optional[List[Dict[str, Any]]]
    keyword_features_str: Optional[str]
    errors: List[str]


# 3. Graph Node Implementations
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

def load_existing_features_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 2: Load existing customer features from DB for the last 12 months.
    """
    customer_id = state["customer_id"]
    errors = list(state.get("errors", []))
    try:
        existing = tools.get_customer_features(customer_id, months=12)
        return {"existing_features": existing, "errors": errors}
    except Exception as e:
        errors.append(f"load_existing_features failed: {str(e)}")
        return {"errors": errors}

def extract_features_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 3: Extract raw candidate features from the current consultation report.
    """
    errors = list(state.get("errors", []))
    if errors:
        return {}

    report = state["report"]
    try:
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

        return {"extracted_features": valid_features}
    except Exception as e:
        errors.append(f"extract_features failed: {str(e)}")
        return {"errors": errors}

def refine_and_deduplicate_features_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 4: Compare extracted features against existing features to decide ADD, SKIP, or UPDATE.
    """
    errors = list(state.get("errors", []))
    if errors:
        return {}

    extracted_features = state["extracted_features"]
    existing_features = state["existing_features"]

    if not extracted_features:
        return {"refined_decisions": []}

    try:
        system_prompt = load_prompt("feature_refinement_system.md")
        
        # Format existing features for LLM input
        existing_str = ""
        if existing_features:
            for f in existing_features:
                existing_str += f"- ID: {f.get('ci_id')}, 카테고리: {f.get('category')}, 내용: {f.get('contents')}\n"
        else:
            existing_str = "등록된 기존 특징 없음."

        # Format candidate features for LLM input
        candidates_str = ""
        for idx, f in enumerate(extracted_features, 1):
            candidates_str += f"{idx}. 카테고리: {f['category']}, 후보내용: {f['contents']}\n"

        user_content = (
            f"## 1. 기존 데이터베이스에 등록된 특징 목록 (Existing Features)\n"
            f"{existing_str}\n\n"
            f"## 2. 새롭게 추출된 후보 특징 목록 (Candidate Features)\n"
            f"{candidates_str}\n\n"
            f"위의 정보를 비교 분석하여 각 후보 특징별로 가장 적합한 업데이트 의사결정(ADD, SKIP, UPDATE)을 리스트로 도출해 주십시오."
        )

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
        structured_llm = llm.with_structured_output(RefinedFeatureList)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{user_content}")
        ])

        chain = prompt | structured_llm
        result: RefinedFeatureList = chain.invoke({"user_content": user_content})

        refined_decisions = []
        for d in result.decisions:
            refined_decisions.append({
                "action": d.action.upper(),
                "category": d.category,
                "contents": d.contents.strip(),
                "target_ci_id": d.target_ci_id,
                "reason": d.reason.strip()
            })

        return {"refined_decisions": refined_decisions}
    except Exception as e:
        errors.append(f"refine_and_deduplicate_features failed: {str(e)}")
        return {"errors": errors}

def save_features_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 5: Apply ADD, UPDATE, or SKIP decisions to the customer_information table.
    """
    errors = list(state.get("errors", []))
    if errors:
        return {}

    customer_id = state["customer_id"]
    refined_decisions = state.get("refined_decisions", [])

    try:
        add_count = 0
        update_count = 0
        skip_count = 0

        for d in refined_decisions:
            action = d["action"]
            category = d["category"]
            contents = d["contents"]
            target_ci_id = d["target_ci_id"]

            if action == "ADD":
                tools.save_customer_feature(customer_id, category, contents)
                print(f"  [+] [ADD] 카테고리: {category} | 내용: {contents}")
                add_count += 1
            elif action == "UPDATE":
                if target_ci_id is not None:
                    tools.update_customer_feature(target_ci_id, contents)
                    print(f"  [+] [UPDATE] 기존 ID: {target_ci_id} ➔ 새 내용: {contents}")
                    update_count += 1
                else:
                    # Fallback to ADD if target_ci_id is missing but action is UPDATE
                    tools.save_customer_feature(customer_id, category, contents)
                    print(f"  [!] [UPDATE->ADD] target_ci_id 누락으로 인한 강제 신규 추가 | 카테고리: {category} | 내용: {contents}")
                    add_count += 1
            elif action == "SKIP":
                print(f"  [-] [SKIP] 중복 제외 | 내용: {contents} (사유: {d['reason']})")
                skip_count += 1

        print(f"  [+] Feature Refinement Summary for customer {customer_id}: "
              f"Added: {add_count}, Updated: {update_count}, Skipped: {skip_count}")
        return {}
    except Exception as e:
        errors.append(f"save_features failed: {str(e)}")
        return {"errors": errors}

def load_features_last_1m_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 6: Load customer features registered within the last 1 month for keyword extraction.
    """
    customer_id = state["customer_id"]
    errors = list(state.get("errors", []))
    try:
        # Load existing features from the last 1 month
        features = tools.get_customer_features(customer_id, months=1)
        return {"features_last_1m": features, "errors": errors}
    except Exception as e:
        errors.append(f"load_features_last_1m failed: {str(e)}")
        return {"errors": errors}

def extract_keywords_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 7: Analyze recent features to extract representative keywords for Word Cloud.
    """
    errors = list(state.get("errors", []))
    if errors:
        return {}

    features_last_1m = state.get("features_last_1m", [])
    
    # Format monthly features for LLM input
    features_list = []
    if features_last_1m:
        for f in features_last_1m:
            features_list.append(f"[{f.get('category')}] {f.get('contents')}")
    features_str = "\n".join(features_list) if features_list else "최근 한 달 이내에 등록된 고객 특징 정보 없음."

    try:
        system_prompt = load_prompt("keyword_extraction_system.md")
        user_content = (
            f"## 대상 고객의 최근 1개월간 등록된 특징 정보:\n"
            f"\"\"\"\n{features_str}\n\"\"\"\n\n"
            f"위의 정보를 기반으로 워드 클라우드용 대표 핵심 키워드 리스트를 도출해 주십시오."
        )

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
        structured_llm = llm.with_structured_output(KeywordList)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{user_content}")
        ])

        chain = prompt | structured_llm
        result: KeywordList = chain.invoke({"user_content": user_content})

        # Strip, sanitize, and format as comma-separated string
        cleaned_keywords = [kw.strip() for kw in result.keywords if kw.strip()]
        joined_str = ",".join(cleaned_keywords)
        
        return {"keyword_features_str": joined_str}
    except Exception as e:
        errors.append(f"extract_keywords failed: {str(e)}")
        return {"errors": errors}

def save_keyword_features_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 8: Save the comma-separated keyword string into the features column of customer table.
    """
    errors = list(state.get("errors", []))
    if errors:
        return {}

    customer_id = state["customer_id"]
    keyword_features_str = state.get("keyword_features_str", "")

    try:
        if keyword_features_str:
            tools.save_customer_keyword_features(customer_id, keyword_features_str)
            print(f"  [+] [KEYWORDS SAVE] customer_id {customer_id}의 features 열에 적재 성공: '{keyword_features_str}'")
        else:
            print(f"  [!] [KEYWORDS SKIP] 추출된 키워드 문자열이 비어 있어 저장을 생략합니다.")
        return {}
    except Exception as e:
        errors.append(f"save_keyword_features failed: {str(e)}")
        return {"errors": errors}


# 4. Compiled State Graph for Customer Feature Matcher
workflow2 = StateGraph(Agent2State)

workflow2.add_node("load_report", load_report_node)
workflow2.add_node("load_existing_features", load_existing_features_node)
workflow2.add_node("extract_features", extract_features_node)
workflow2.add_node("refine_and_deduplicate_features", refine_and_deduplicate_features_node)
workflow2.add_node("save_features", save_features_node)
workflow2.add_node("load_features_last_1m", load_features_last_1m_node)
workflow2.add_node("extract_keywords", extract_keywords_node)
workflow2.add_node("save_keyword_features", save_keyword_features_node)

workflow2.set_entry_point("load_report")

workflow2.add_edge("load_report", "load_existing_features")
workflow2.add_edge("load_existing_features", "extract_features")
workflow2.add_edge("extract_features", "refine_and_deduplicate_features")
workflow2.add_edge("refine_and_deduplicate_features", "save_features")
workflow2.add_edge("save_features", "load_features_last_1m")
workflow2.add_edge("load_features_last_1m", "extract_keywords")
workflow2.add_edge("extract_keywords", "save_keyword_features")
workflow2.add_edge("save_keyword_features", END)

compiled_app2 = workflow2.compile()


class CustomerFeatureAgent:
    """
    Customer Feature Agent (고객 특징 분석 에이전트 - 고도화 버전)
    Extracts features from consultation reports, performs intelligent deduplication and refinement,
    and extracts representative keywords for Word Cloud database update.
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
            "existing_features": [],
            "extracted_features": [],
            "refined_decisions": [],
            "features_last_1m": [],
            "keyword_features_str": "",
            "errors": []
        }
        
        final_state = self.app.invoke(
            initial_state,
            config={"run_name": "CustomerFeatureAgent", "tags": ["feature_agent"]}
        )
        
        if final_state.get("errors"):
            raise RuntimeError(f"LangGraph execution encountered errors in CustomerFeatureAgent: {final_state['errors']}")
            
        return final_state
