import os
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator
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

def sanitize_val(val: Any) -> Any:
    """
    Sanitizes string inputs, converting representation of null/none values to actual Python None.
    """
    if val is None:
        return None
    if isinstance(val, str):
        v = val.strip()
        if v.lower() in ("null", "none", "", "nan", "undefined"):
            return None
        return v
    return val

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

class ExtractedRelationship(BaseModel):
    relationship: str = Field(
        description="지인 관계 유형 (예: 배우자, 자녀, 아들, 딸, 어머니, 아버지, 친구, 직장동료 등)"
    )
    information: str = Field(
        description="해당 지인에 대한 정보나 대화 내용 요약 (한국어 경어체 한 문장)"
    )
    birthday: Optional[str] = Field(
        default=None,
        description="지인의 생년월일 (YYYY-MM-DD 형식, 상담 내용에 생년월일이나 나이 등으로 유추 가능할 때 정확하게 계산 가능하면 YYYY-MM-DD 형식으로 작성, 그렇지 않으면 null)"
    )
    job: Optional[str] = Field(
        default=None,
        description="지인의 직업 또는 소속 (없으면 null)"
    )
    is_spouse: bool = Field(
        description="배우자(남편/아내/부인/신랑 등) 여부. 배우자인 경우 true, 그 외의 지인은 false"
    )
    wedding_date: Optional[str] = Field(
        default=None,
        description="결혼기념일 (YYYY-MM-DD 형식, 배우자(is_spouse=true)이고 결혼기념일 정보가 명확히 있을 때만 YYYY-MM-DD 형식으로 작성, 없으면 null)"
    )

class ExtractedRelationshipList(BaseModel):
    relationships: List[ExtractedRelationship]


# 2. State Definition for Customer Feature Matcher (Upgraded Version)
class Agent2State(TypedDict):
    customer_id: int
    report: Optional[Dict[str, Any]]
    existing_features: List[Dict[str, Any]]
    extracted_features: List[Dict[str, Any]]
    refined_decisions: List[Dict[str, Any]]
    features_last_1m: Optional[List[Dict[str, Any]]]
    keyword_features_str: Optional[str]
    existing_relationships: List[Dict[str, Any]]
    extracted_relationships: List[Dict[str, Any]]
    validated_relationships: List[Dict[str, Any]]
    errors: Annotated[List[str], operator.add]



# 3. Graph Node Implementations
def load_report_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 1: Load the latest consultation report for the customer.
    """
    customer_id = state["customer_id"]
    try:
        report = tools.get_recent_consultation_report(customer_id)
        if not report:
            raise ValueError(f"No consultation report found for customer ID {customer_id}.")
        return {"report": report}
    except Exception as e:
        return {"errors": [f"load_report failed: {str(e)}"]}

def load_existing_features_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 2: Load existing customer features from DB for the last 12 months.
    """
    customer_id = state["customer_id"]
    try:
        existing = tools.get_customer_features(customer_id, months=12)
        return {"existing_features": existing}
    except Exception as e:
        return {"errors": [f"load_existing_features failed: {str(e)}"]}


def extract_features_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 3: Extract raw candidate features from the current consultation report.
    """
    if state.get("errors"):
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
        return {"errors": [f"extract_features failed: {str(e)}"]}

def refine_and_deduplicate_features_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 4: Compare extracted features against existing features to decide ADD, SKIP, or UPDATE.
    """
    if state.get("errors"):
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

        user_prompt = load_prompt("feature_refinement_user.md")
        user_content = user_prompt.format(
            existing_str=existing_str,
            candidates_str=candidates_str
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
        return {"errors": [f"refine_and_deduplicate_features failed: {str(e)}"]}

def save_features_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 5: Apply ADD, UPDATE, or SKIP decisions to the customer_information table.
    """
    if state.get("errors"):
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
        return {"errors": [f"save_features failed: {str(e)}"]}

def load_features_last_1m_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 6: Load customer features registered within the last 1 month for keyword extraction.
    """
    if state.get("errors"):
        return {}
    customer_id = state["customer_id"]
    try:
        # Load existing features from the last 1 month
        features = tools.get_customer_features(customer_id, months=1)
        return {"features_last_1m": features}
    except Exception as e:
        return {"errors": [f"load_features_last_1m failed: {str(e)}"]}

def extract_keywords_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 7: Analyze recent features to extract representative keywords for Word Cloud.
    """
    if state.get("errors"):
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
        user_prompt = load_prompt("keyword_extraction_user.md")
        user_content = user_prompt.format(
            features_str=features_str
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
        return {"errors": [f"extract_keywords failed: {str(e)}"]}

def save_keyword_features_node(state: Agent2State) -> Dict[str, Any]:
    """
    Node 8: Save the comma-separated keyword string into the features column of customer table.
    """
    if state.get("errors"):
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
        return {"errors": [f"save_keyword_features failed: {str(e)}"]}


def load_existing_relationships_node(state: Agent2State) -> Dict[str, Any]:
    """
    Load existing relationships for the customer from customer_relationship table.
    """
    customer_id = state["customer_id"]
    try:
        existing = tools.get_customer_relationships_all(customer_id)
        return {"existing_relationships": existing}
    except Exception as e:
        return {"errors": [f"load_existing_relationships failed: {str(e)}"]}

def extract_relationships_node(state: Agent2State) -> Dict[str, Any]:
    """
    Extract raw relationship records from the consultation report.
    """
    if state.get("errors"):
        return {}

    report = state["report"]
    try:
        system_prompt = load_prompt("relationship_extraction_system.md")
        user_prompt = load_prompt("relationship_extraction_user.md")

        consult_date = report.get("consult_date")
        if consult_date:
            if hasattr(consult_date, "strftime"):
                consult_date_str = consult_date.strftime("%Y-%m-%d")
            else:
                consult_date_str = str(consult_date)
        else:
            consult_date_str = "제공되지 않음"

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.3, api_key=OPENAI_API_KEY)
        structured_llm = llm.with_structured_output(ExtractedRelationshipList)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])

        chain = prompt | structured_llm
        result: ExtractedRelationshipList = chain.invoke({
            "report_content": report["content"],
            "consult_date": consult_date_str
        })

        extracted = []
        for r in result.relationships:
            extracted.append({
                "relationship": sanitize_val(r.relationship),
                "information": sanitize_val(r.information),
                "birthday": sanitize_val(r.birthday),
                "job": sanitize_val(r.job),
                "is_spouse": 1 if r.is_spouse else 0,
                "wedding_date": sanitize_val(r.wedding_date)
            })

        return {"extracted_relationships": extracted}
    except Exception as e:
        return {"errors": [f"extract_relationships failed: {str(e)}"]}

def validate_relationships_node(state: Agent2State) -> Dict[str, Any]:
    """
    Validate and sanitize the extracted relationships to prevent hallucination
    and verify compliance with DB constraints.
    """
    if state.get("errors"):
        return {}

    report = state["report"]
    extracted = state.get("extracted_relationships", [])
    if not extracted:
        return {"validated_relationships": []}

    try:
        system_prompt = load_prompt("relationship_validation_system.md")
        user_prompt = load_prompt("relationship_validation_user.md")

        consult_date = report.get("consult_date")
        if consult_date:
            if hasattr(consult_date, "strftime"):
                consult_date_str = consult_date.strftime("%Y-%m-%d")
            else:
                consult_date_str = str(consult_date)
        else:
            consult_date_str = "제공되지 않음"

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
        structured_llm = llm.with_structured_output(ExtractedRelationshipList)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])

        chain = prompt | structured_llm
        result: ExtractedRelationshipList = chain.invoke({
            "report_content": report["content"],
            "consult_date": consult_date_str,
            "extracted_relationships": str(extracted)
        })

        validated = []
        for r in result.relationships:
            # Enforce database length constraints
            rel_name = sanitize_val(r.relationship) or ""
            if len(rel_name) > 50:
                rel_name = rel_name[:50]
                
            info = sanitize_val(r.information) or ""
            
            job = sanitize_val(r.job)
            if job and len(job) > 50:
                job = job[:50]

            validated.append({
                "relationship": rel_name,
                "information": info,
                "birthday": sanitize_val(r.birthday),
                "job": job,
                "is_spouse": 1 if r.is_spouse else 0,
                "wedding_date": sanitize_val(r.wedding_date)
            })

        print(f"  [+] Validation Node completed. Validated {len(validated)} relationships.")
        return {"validated_relationships": validated}
    except Exception as e:
        return {"errors": [f"validate_relationships failed: {str(e)}"]}

def refine_merged_relationship_info(existing_info: str, new_info: str) -> str:
    """
    Call LLM to merge and deduplicate existing and new relationship information
    into a single clean description.
    """
    if not existing_info.strip():
        return new_info.strip()
    if not new_info.strip():
        return existing_info.strip()

    try:
        system_prompt = load_prompt("relationship_merge_system.md")
        user_prompt = load_prompt("relationship_merge_user.md")
        user_content = user_prompt.format(
            existing_info=existing_info,
            new_info=new_info
        )

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
        messages = [
            ("system", system_prompt),
            ("user", user_content)
        ]
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"  [!] Failed to refine merged info via LLM: {e}. Falling back to simple concatenation.")
        if new_info not in existing_info:
            return f"{existing_info.strip()} {new_info.strip()}".strip()
        return existing_info

def save_relationships_node(state: Agent2State) -> Dict[str, Any]:
    """
    Merge validated relationships with existing ones and persist to database.
    """
    if state.get("errors"):
        return {}

    customer_id = state["customer_id"]
    existing = state.get("existing_relationships", [])
    validated = state.get("validated_relationships", [])

    try:
        # Create a lookup map of existing relationships (by type)
        existing_map = {}
        for r in existing:
            rel_type = r["relationship"].strip()
            existing_map[rel_type] = r

        for r in validated:
            rel_type = sanitize_val(r["relationship"])
            info = sanitize_val(r["information"])
            birthday = sanitize_val(r["birthday"])
            job = sanitize_val(r["job"])
            is_spouse = r["is_spouse"]
            wedding_date = sanitize_val(r["wedding_date"])

            if rel_type in existing_map:
                # Merge existing details
                existing_rec = existing_map[rel_type]
                cr_id = existing_rec["cr_id"]
                existing_info = existing_rec["information"] or ""
                
                # Combine and refine info using LLM to prevent redundancies
                combined_info = refine_merged_relationship_info(existing_info, info)

                
                # Merge fields: keep existing unless existing is None/empty and new is provided
                m_birthday = sanitize_val(existing_rec["birthday"]) or birthday
                m_job = sanitize_val(existing_rec["job"]) or job
                m_is_spouse = existing_rec["is_spouse"] if existing_rec["is_spouse"] is not None else is_spouse
                m_wedding_date = sanitize_val(existing_rec["wedding_date"]) or wedding_date

                # Convert date object to YYYY-MM-DD string if it is a datetime.date
                if hasattr(m_birthday, "strftime"):
                    m_birthday = m_birthday.strftime("%Y-%m-%d")
                if hasattr(m_wedding_date, "strftime"):
                    m_wedding_date = m_wedding_date.strftime("%Y-%m-%d")

                tools.update_customer_relationship(
                    cr_id=cr_id,
                    information=combined_info,
                    birthday=m_birthday,
                    job=m_job,
                    is_spouse=m_is_spouse,
                    wedding_date=m_wedding_date
                )
                print(f"  [+] [UPDATE RELATIONSHIP] 기존 관계 '{rel_type}' 업데이트 성공 (cr_id: {cr_id})")
            else:
                # Insert new relationship
                tools.save_customer_relationship(
                    customer_id=customer_id,
                    relationship=rel_type,
                    information=info,
                    birthday=birthday,
                    job=job,
                    is_spouse=is_spouse,
                    wedding_date=wedding_date
                )
                print(f"  [+] [ADD RELATIONSHIP] 신규 관계 '{rel_type}' 등록 성공")

        return {}
    except Exception as e:
        return {"errors": [f"save_relationships failed: {str(e)}"]}


# 4. Compiled State Graph for Customer Feature Matcher
workflow2 = StateGraph(Agent2State)

workflow2.add_node("load_report", load_report_node)
workflow2.add_node("load_existing_features", load_existing_features_node)
workflow2.add_node("load_existing_relationships", load_existing_relationships_node)
workflow2.add_node("extract_features", extract_features_node)
workflow2.add_node("refine_and_deduplicate_features", refine_and_deduplicate_features_node)
workflow2.add_node("save_features", save_features_node)
workflow2.add_node("extract_relationships", extract_relationships_node)
workflow2.add_node("validate_relationships", validate_relationships_node)
workflow2.add_node("save_relationships", save_relationships_node)
workflow2.add_node("load_features_last_1m", load_features_last_1m_node)
workflow2.add_node("extract_keywords", extract_keywords_node)
workflow2.add_node("save_keyword_features", save_keyword_features_node)

workflow2.set_entry_point("load_report")

# Branch 1: Feature Pipeline
workflow2.add_edge("load_report", "load_existing_features")
workflow2.add_edge("load_existing_features", "extract_features")
workflow2.add_edge("extract_features", "refine_and_deduplicate_features")
workflow2.add_edge("refine_and_deduplicate_features", "save_features")

# Branch 2: Relationship Pipeline
workflow2.add_edge("load_report", "load_existing_relationships")
workflow2.add_edge("load_existing_relationships", "extract_relationships")
workflow2.add_edge("extract_relationships", "validate_relationships")
workflow2.add_edge("validate_relationships", "save_relationships")

# Synchronization Join (Merge both branches before keyword analysis)
workflow2.add_edge("save_features", "load_features_last_1m")
workflow2.add_edge("save_relationships", "load_features_last_1m")

# Final Sequence
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
            "existing_relationships": [],
            "extracted_relationships": [],
            "validated_relationships": [],
            "errors": []
        }
        
        final_state = self.app.invoke(
            initial_state,
            config={"run_name": "CustomerFeatureAgent", "tags": ["feature_agent"]}
        )
        
        if final_state.get("errors"):
            raise RuntimeError(f"LangGraph execution encountered errors in CustomerFeatureAgent: {final_state['errors']}")
            
        return final_state
