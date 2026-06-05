import os
import sys
import json
import operator
from typing import TypedDict, List, Dict, Any, Optional, Annotated

from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Absolute path resolution to strictly support imports from workspace
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Load environment variables
load_dotenv(find_dotenv())

DEFAULT_MODEL = "gpt-4o-mini"

from agent.simulator.tools import (
    fetch_from_tavily,
    query_knowledge_base,
    get_customer_held_products,
    get_all_products,
    get_customer_product_matching,
    format_products_context,
)

# 1. State Definition for Simulator Agent
class SimulatorState(TypedDict):
    customer_id: int
    question: str
    context_content: Optional[str]
    history: List[Dict[str, str]]
    intent: Optional[str]
    retrieved_knowledge: Optional[str]
    recent_features_1m: Optional[str]
    answer: Optional[str]
    errors: Annotated[List[str], operator.add]

# 2. Pydantic Model for Intent Routing
class IntentRoute(BaseModel):
    intent: str = Field(
        description="PB 질문의 핵심 대화 주제 및 의도. 반드시 'knowledge' 또는 'general' 중 하나여야 합니다."
    )
    reason: str = Field(
        description="이 대화 의도로 분류한 사유 (한 문장)"
    )

# 3. Prompt Utility Functions
def get_simulator_system_prompt() -> str:
    """Read the simulator system prompt from prompt/simulator_system_prompt.md."""
    prompt_path = os.path.join(current_dir, "prompt", "simulator_system_prompt.md")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def get_simulator_user_prompt(context_content: str, recent_features_1m: str = "", retrieved_knowledge: str = "") -> str:
    """Read the simulator user prompt template and fill in variables."""
    prompt_path = os.path.join(current_dir, "prompt", "simulator_user_prompt.md")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"User prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    # Remove recent features section if empty to prevent LLM confusion
    if not recent_features_1m.strip():
        template = template.replace("[고객의 최근 1개월 이내 특징 및 메모 (DB)]\n{recent_features_1m}", "")
        template = template.replace("[고객의 최근 1개월 이내 특징 및 메모 (DB)]\r\n{recent_features_1m}", "")
    else:
        template = template.replace("{recent_features_1m}", recent_features_1m)

    # Remove retrieved knowledge section if empty
    if not retrieved_knowledge.strip():
        template = template.replace("[검색된 행내 지식 및 세무/시장 정보 (RAG)]\n{retrieved_knowledge}", "")
        template = template.replace("[검색된 행내 지식 및 세무/시장 정보 (RAG)]\r\n{retrieved_knowledge}", "")
    else:
        template = template.replace("{retrieved_knowledge}", retrieved_knowledge)

    template = template.replace("{context_content}", context_content)
    return template

def get_intent_router_system_prompt() -> str:
    """Read the intent router system prompt from prompt/intent_router_system_prompt.md."""
    prompt_path = os.path.join(current_dir, "prompt", "intent_router_system_prompt.md")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Intent router prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def get_assistant_acknowledgment() -> str:
    """Read the assistant acknowledgment message from prompt/assistant_acknowledgment.md."""
    prompt_path = os.path.join(current_dir, "prompt", "assistant_acknowledgment.md")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Assistant acknowledgment file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# 4. Graph Node Implementations

def load_context_node(state: SimulatorState) -> Dict[str, Any]:
    """
    Node 1: Load customer profile markdown/txt and historical conversations.
    """
    customer_id = state["customer_id"]
    data_dir = os.path.join(current_dir, "data/history")
    
    # Load profile content
    md_path = os.path.join(data_dir, f"customer_{customer_id}.md")
    txt_path = os.path.join(data_dir, f"customer_{customer_id}.txt")
    
    context_content = ""
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            context_content = f.read()
    elif os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            context_content = f.read()
    else:
        context_content = "고객 정보가 존재하지 않습니다. 기본적인 금융 상담으로 대응해 주세요."
        
    # Load history
    history_path = os.path.join(data_dir, f"customer_{customer_id}_history.json")
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
            
    # Limit history to prevent context bloat
    history = history[-10:]
    
    return {"context_content": context_content, "history": history}


def route_intent_node(state: SimulatorState) -> Dict[str, Any]:
    """
    Node 2: Classify user query intent using LLM (gpt-4o-mini structured output).
    """
    if state.get("errors"):
        return {}
        
    question = state["question"]
    api_key = os.getenv("OPENAI_API_KEY")
    
    try:
        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.1, api_key=api_key)
        structured_llm = llm.with_structured_output(IntentRoute)
        
        system_prompt = get_intent_router_system_prompt()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "PB 질문: {question}")
        ])
        
        chain = prompt | structured_llm
        result: IntentRoute = chain.invoke({"question": question})
        
        sys.stderr.write(f"[Router] 대화 의도 판별 완료: '{result.intent}' (사유: {result.reason})\n")
        return {"intent": result.intent.lower()}
    except Exception as e:
        sys.stderr.write(f"[Router 오류] 의도 분류 실패: {str(e)}. 'general'로 대체합니다.\n")
        return {"intent": "general"}


def knowledge_node(state: SimulatorState) -> Dict[str, Any]:
    """
    Node 3: Unified Knowledge Node
    Fetches both VectorDB RAG (세법/시장전망) and MySQL DB (전체 상품, 보유 현황, AI 추천 매칭 및 1개월 특징)
    concurrently to build a complete unified context for non-general queries.
    """
    if state.get("errors"):
        return {}
        
    question = state["question"]
    customer_id = state["customer_id"]
    chroma_db_dir = os.path.join(current_dir, "data", "chroma_db")
    THRESHOLD = 1.2
    
    retrieved_knowledge_parts = []
    
    # 1. Retrieve VectorDB RAG (threshold 0.6 filter, fallback to Tavily)
    try:
        rag_knowledge = query_knowledge_base(question, chroma_db_dir, THRESHOLD)
        if rag_knowledge:
            retrieved_knowledge_parts.append(rag_knowledge)
        else:
            sys.stderr.write(f"[RAG->Tavily] 유사도 기준(0.6) 만족 지식 부재로 Tavily 실시간 웹 검색 수행\n")
            web_search = fetch_from_tavily(question)
            retrieved_knowledge_parts.append(web_search)
    except Exception as e:
        sys.stderr.write(f"[RAG 오류] 지식 검색 중 에러 발생: {str(e)}\n")
        sys.stderr.write(f"[RAG->Tavily] 에러 발생으로 인해 Tavily 실시간 웹 검색 수행\n")
        web_search = fetch_from_tavily(question)
        retrieved_knowledge_parts.append(web_search)
        
    # 2. Retrieve MySQL Financial Products & matching information (using split tools)
    try:
        held_products = get_customer_held_products(customer_id)
        all_products = get_all_products()
        matching_records = get_customer_product_matching(customer_id)
        product_knowledge = format_products_context(held_products, all_products, matching_records)
        if product_knowledge:
            retrieved_knowledge_parts.append(product_knowledge)
    except Exception as e:
        sys.stderr.write(f"[DB 오류] 금융 상품 매칭 조회 실패: {str(e)}\n")
        retrieved_knowledge_parts.append("실시간 금융 상품 정보를 조회하지 못했습니다.")
        
    # Combine knowledge parts
    combined_knowledge = "\n\n".join(retrieved_knowledge_parts)
    
    # 3. Retrieve customer's features in the last 1 month from MySQL customer_information table
    recent_features_str = ""
    try:
        from agent.customer.tools import get_customer_features
        features = get_customer_features(customer_id, months=1)
        
        if not features:
            recent_features_str = "최근 1개월 이내에 기록된 특징이나 특이사항 메모가 없습니다."
        else:
            feature_lines = []
            for f in features:
                cat = f.get("category") or "기타"
                cont = f.get("contents") or ""
                feature_lines.append(f"- [{cat}] {cont}")
            recent_features_str = "\n".join(feature_lines)
            
        sys.stderr.write(f"[DB] 최근 1개월 특징 {len(features)}개 로드 완료\n")
    except Exception as e:
        sys.stderr.write(f"[DB 오류] 최근 특징 조회 실패: {str(e)}\n")
        recent_features_str = "특징 정보를 가져오는 중 오류가 발생했습니다."
        
    return {
        "retrieved_knowledge": combined_knowledge,
        "recent_features_1m": recent_features_str
    }


def route_conditional_edge(state: SimulatorState) -> str:
    """
    Determine next node based on intent.
    If 'general', skip knowledge retrieval and go straight to generate_answer.
    Otherwise, go to knowledge node.
    """
    intent = state.get("intent")
    if intent == "general":
        return "generate_answer"
    else:
        return "knowledge"


def extract_sources_from_knowledge(retrieved_knowledge: str) -> list[str]:
    """
    Extract clean source descriptions from retrieved knowledge text.
    """
    if not retrieved_knowledge:
        return []
    
    sources = []
    for line in retrieved_knowledge.splitlines():
        line = line.strip()
        if line.startswith("[") and "출처" in line:
            parts = line.split("출처:")
            if len(parts) > 1:
                source_val = parts[1].strip()
                if " (유사 거리:" in source_val:
                    source_val = source_val.split(" (유사 거리:")[0].strip()
                sources.append(source_val)
            else:
                parts_web = line.split("출처 (웹):")
                if len(parts_web) > 1:
                    sources.append(parts_web[1].strip())
    
    unique_sources = []
    for s in sources:
        if s not in unique_sources:
            unique_sources.append(s)
    return unique_sources


def generate_answer_node(state: SimulatorState) -> Dict[str, Any]:
    """
    Node 5: Construct complete prompt list, invoke LLM, and update/persist conversation history.
    """
    if state.get("errors"):
        return {}
        
    customer_id = state["customer_id"]
    question = state["question"]
    context_content = state["context_content"]
    history = state["history"]
    retrieved_knowledge = state.get("retrieved_knowledge", "")
    recent_features_1m = state.get("recent_features_1m", "")
    api_key = os.getenv("OPENAI_API_KEY")
    
    try:
        # 메시지 히스토리 조립
        messages = []
        # System
        messages.append({"role": "system", "content": get_simulator_system_prompt()})
        # User Context (Profile + RAG + 1M Features)
        user_context_prompt = get_simulator_user_prompt(
            context_content=context_content,
            recent_features_1m=recent_features_1m,
            retrieved_knowledge=retrieved_knowledge
        )
        messages.append({"role": "user", "content": user_context_prompt})
        
        # Assistant Acknowledgment (기존 simulator.py 흐름 유지)
        messages.append({
            "role": "assistant",
            "content": get_assistant_acknowledgment()
        })
        
        # Historical turns (clean past references to prevent source mixing)
        import re
        for turn in history:
            cleaned_turn = turn.copy()
            if turn["role"] == "assistant":
                # Remove [참조 출처: ...] or 참조 출처: ... from past history turns
                cleaned_turn["content"] = re.sub(r'\n\n\[참조 출처: .*?\]', '', turn["content"])
                cleaned_turn["content"] = re.sub(r'\[참조 출처: .*?\]', '', cleaned_turn["content"])
                cleaned_turn["content"] = re.sub(r'참조 출처: .*', '', cleaned_turn["content"])
            messages.append(cleaned_turn)
            
        # Current PB query
        messages.append({"role": "user", "content": question})
        
        # LLM Call
        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.4, api_key=api_key)
        response = llm.invoke(messages)
        answer = response.content
        
        # Clean any self-generated reference formatting LLM produced
        answer = re.sub(r'\n\n\[참조 출처: .*?\]', '', answer)
        answer = re.sub(r'\[참조 출처: .*?\]', '', answer)
        answer = re.sub(r'참조 출처: .*', '', answer)
        
        # Strip markdown formatting for terminal compatibility (removes **, *, _, #)
        answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)
        answer = re.sub(r'\*([^*]+)\*', r'\1', answer)
        answer = re.sub(r'__([^_]+)__', r'\1', answer)
        answer = re.sub(r'_([^_]+)_', r'\1', answer)
        answer = re.sub(r'(?m)^#+\s+', '', answer)
        
        answer = answer.strip()
        
        # Append references strictly based on system-retrieved knowledge
        if retrieved_knowledge.strip():
            sources = extract_sources_from_knowledge(retrieved_knowledge)
            if sources:
                source_suffix = "\n\n[참조 출처: " + ", ".join(sources) + "]"
                answer += source_suffix
        
        # Save updated history back to file
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        
        data_dir = os.path.join(current_dir, "data")
        history_path = os.path.join(data_dir, f"customer_{customer_id}_history.json")
        
        os.makedirs(data_dir, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
        return {"answer": answer}
    except Exception as e:
        return {"errors": [f"generate_answer failed: {str(e)}"]}


# 5. Compiled State Graph for Simulator Agent
workflow = StateGraph(SimulatorState)

workflow.add_node("load_context", load_context_node)
workflow.add_node("route_intent", route_intent_node)
workflow.add_node("knowledge", knowledge_node)
workflow.add_node("generate_answer", generate_answer_node)

workflow.set_entry_point("load_context")

workflow.add_edge("load_context", "route_intent")
workflow.add_conditional_edges(
    "route_intent",
    route_conditional_edge,
    {
        "knowledge": "knowledge",
        "generate_answer": "generate_answer"
    }
)
workflow.add_edge("knowledge", "generate_answer")
workflow.add_edge("generate_answer", END)

compiled_app = workflow.compile()


class SimulatorAgent:
    """
    SimulatorAgent (WM 상담 시뮬레이터 에이전트 - LangGraph 버전)
    Handles loading context, intent-based RAG search, 1-month features query, and generating
    accurate pitching advice for PBs.
    """
    def __init__(self, model_name: str = None):
        global DEFAULT_MODEL
        if model_name:
            DEFAULT_MODEL = model_name
        self.app = compiled_app

    def run(self, customer_id: int, question: str) -> Dict[str, Any]:
        initial_state: SimulatorState = {
            "customer_id": customer_id,
            "question": question,
            "context_content": "",
            "history": [],
            "intent": "",
            "retrieved_knowledge": "",
            "recent_features_1m": "",
            "answer": "",
            "errors": []
        }
        
        final_state = self.app.invoke(
            initial_state,
            config={"run_name": "SimulatorAgent", "tags": ["simulator_agent"]}
        )
        
        if final_state.get("errors"):
            raise RuntimeError(f"LangGraph execution encountered errors in SimulatorAgent: {final_state['errors']}")
            
        return final_state


def run_simulation(customer_id: str, question: str) -> str:
    """
    Legacy wrapper to maintain compatibility with the original script interface.
    """
    agent = SimulatorAgent()
    result = agent.run(int(customer_id), question)
    return result["answer"]


if __name__ == "__main__":
    # Ensure Windows console uses UTF-8
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stdin.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    if len(sys.argv) < 2:
        sys.stderr.write(json.dumps({"error": "Missing customer_id argument"}))
        sys.exit(1)

    customer_id = sys.argv[1]
    
    try:
        # Read the question from stdin
        question = sys.stdin.read().strip()
        if not question:
            sys.stdout.write(json.dumps({"answer": "질문이 입력되지 않았습니다. 궁금하신 내용을 질문해 주세요."}))
            sys.exit(0)

        answer = run_simulation(customer_id, question)
        sys.stdout.write(json.dumps({"answer": answer}))
        sys.stdout.flush()
    except Exception as e:
        sys.stderr.write(json.dumps({"error": str(e)}))
        sys.stderr.flush()
        sys.exit(1)
