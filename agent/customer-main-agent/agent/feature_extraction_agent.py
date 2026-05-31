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
class ExtractedFeature31(BaseModel):
    category: str = Field(description="특징 카테고리. 반드시 '관계', '성향', '상품', '기호', '건강', '기타' 중 하나여야 합니다.")
    contents: str = Field(description="80자 이내의 한국어 경어체 특징 설명 한 문장. (VARCHAR(500) 길이 제한)")

class ExtractedFeatureList31(BaseModel):
    features: List[ExtractedFeature31]

# 2. State 정의
class Agent31State(TypedDict):
    customer_id: int
    report: Optional[Dict[str, Any]]
    extracted_features: List[Dict[str, Any]]
    errors: List[str]

# 3. 노드 구현체 정의
def load_report_node(state: Agent31State) -> Dict[str, Any]:
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

def extract_features_node(state: Agent31State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    report = state["report"]
    try:
        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.3, api_key=OPENAI_API_KEY)
        system_prompt_feat = load_prompt("feature_extraction_system.md")
        user_prompt_feat = load_prompt("feature_extraction_user.md")
        structured_llm_feat = llm.with_structured_output(ExtractedFeatureList31)
        prompt_feat = ChatPromptTemplate.from_messages([
            ("system", system_prompt_feat),
            ("user", user_prompt_feat)
        ])
        chain_feat = prompt_feat | structured_llm_feat
        result_feat: ExtractedFeatureList31 = chain_feat.invoke({"report_content": report["content"]})

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

def save_features_node(state: Agent31State) -> Dict[str, Any]:
    errors = list(state.get("errors", []))
    if errors:
        return {}
    customer_id = state["customer_id"]
    extracted_features = state["extracted_features"]
    try:
        for f in extracted_features:
            tools.save_customer_feature(customer_id, f["category"], f["contents"])
        print(f"  [+] Sub Agent 3: 상담 기록 {len(extracted_features)}개 특징 추출 및 DB 적재 완료! (고객 ID: {customer_id})")
        return {}
    except Exception as e:
        errors.append(f"save_features failed: {str(e)}")
        return {"errors": errors}

# LangGraph 그래프 조립
workflow31 = StateGraph(Agent31State)
workflow31.add_node("load_report", load_report_node)
workflow31.add_node("extract_features", extract_features_node)
workflow31.add_node("save_features", save_features_node)

workflow31.set_entry_point("load_report")
workflow31.add_edge("load_report", "extract_features")
workflow31.add_edge("extract_features", "save_features")
workflow31.add_edge("save_features", END)

compiled_app31 = workflow31.compile()


class FeatureExtractionAgent:
    """
    Sub Agent 3: 상담 보고서 기반 특징 추출 Agent
    """
    def __init__(self, model_name: str = None):
        global DEFAULT_MODEL
        if model_name:
            DEFAULT_MODEL = model_name
        self.app = compiled_app31

    def run(self, customer_id: int) -> Dict[str, Any]:
        initial_state: Agent31State = {
            "customer_id": customer_id,
            "report": None,
            "extracted_features": [],
            "errors": []
        }
        final_state = self.app.invoke(
            initial_state,
            config={"run_name": "FeatureExtractionAgent", "tags": ["feature_extraction_agent"]}
        )
        if final_state.get("errors"):
            raise RuntimeError(f"LangGraph execution error in FeatureExtractionAgent: {final_state['errors']}")
        return final_state
