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

# Map LANGSMITH_ environment variables to LANGCHAIN_ standard tracing variables
if os.getenv("LANGSMITH_TRACING") == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY").strip('"\'')
if os.getenv("LANGSMITH_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT").strip('"\'')
if os.getenv("LANGSMITH_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT").strip('"\'')

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

class SubQuery(BaseModel):
    query: str = Field(description="RAG 검색을 위해 구체적으로 분할 및 구체화된 검색 키워드 쿼리")
    asset_category: str = Field(description="이 쿼리가 타겟팅하는 금융 분야. '세무', '매크로', '금융상품', '컴플라이언스', '공통' 중 하나")
    
class QueryDecomposition(BaseModel):
    sub_queries: List[SubQuery] = Field(description="원본 질문을 분할한 1~3개의 RAG 검색 쿼리 리스트")

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

def get_query_decomposer_system_prompt() -> str:
    """Read the query decomposer prompt from prompt/query_decomposer_system_prompt.md."""
    prompt_path = os.path.join(current_dir, "prompt", "query_decomposer_system_prompt.md")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Query decomposer prompt file not found: {prompt_path}")
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

    # S3에서 프로필 로드
    context_content = (
        s3_read_text(f"simulator/history/customer_{customer_id}.md")
        or s3_read_text(f"simulator/history/customer_{customer_id}.txt")
        or "고객 정보가 존재하지 않습니다. 기본적인 금융 상담으로 대응해 주세요."
    )

    # S3에서 히스토리 로드
    history = s3_read_json(f"simulator/history/customer_{customer_id}_history.json") or []

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
    history = state.get("history", [])
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Format recent history for router context
    history_str = ""
    if history:
        history_lines = []
        for turn in history[-5:]:
            role_label = "PB" if turn["role"] == "user" else "AI"
            content = turn["content"]
            history_lines.append(f"- {role_label}: {content}")
        history_str = "\n".join(history_lines)
        
    try:
        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.1, api_key=api_key)
        structured_llm = llm.with_structured_output(IntentRoute)
        
        system_prompt = get_intent_router_system_prompt()
        
        user_content = f"PB 질문: {question}"
        if history_str:
            user_content = f"[이전 대화 이력]\n{history_str}\n\nPB 질문: {question}"
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{user_content}")
        ])
        
        chain = prompt | structured_llm
        result: IntentRoute = chain.invoke({"user_content": user_content})
        
        sys.stderr.write(f"[Router] 대화 의도 판별 완료: '{result.intent}' (사유: {result.reason})\n")
        return {"intent": result.intent.lower()}
    except Exception as e:
        sys.stderr.write(f"[Router 오류] 의도 분류 실패: {str(e)}. 'general'로 대체합니다.\n")
        return {"intent": "general"}


def knowledge_node(state: SimulatorState) -> Dict[str, Any]:
    """
    Node 3: Unified Knowledge Node
    Decomposes the user query into 1~3 sub-queries, predicts metadata tags (asset_category, target_segment)
    for each using the customer profile and history, performs parallel pre-filtered RAG searches,
    and merges the results to form a clean unified context.
    """
    if state.get("errors"):
        return {}
        
    question = state["question"]
    history = state.get("history", [])
    customer_id = state["customer_id"]
    context_content = state.get("context_content", "")
    chroma_db_dir = os.path.join(current_dir, "data", "chroma_db")
    THRESHOLD = 0.50
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    retrieved_knowledge_parts = []
    
    # 1. Decompose query and predict filter tags using structured LLM output
    try:
        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.1, api_key=api_key)
        structured_llm = llm.with_structured_output(QueryDecomposition)
        
        system_prompt = get_query_decomposer_system_prompt()
        
        # Format conversation history for context
        history_lines = []
        for turn in history[-5:]:
            role_label = "PB" if turn["role"] == "user" else "AI"
            content = turn["content"]
            history_lines.append(f"{role_label}: {content}")
        history_str = "\n".join(history_lines)
        
        user_content = (
            f"[고객 정보]\n{context_content}\n\n"
            f"[이전 대화 이력]\n{history_str if history_str else '이전 이력 없음'}\n\n"
            f"[PB 현재 질문]\n{question}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{user_content}")
        ])
        
        chain = prompt | structured_llm
        decomposition: QueryDecomposition = chain.invoke({"user_content": user_content})
        
        sys.stderr.write(f"[Query Decomposer] 분할 완료: {len(decomposition.sub_queries)}개 서브 쿼리 생성\n")
        
        # 2. Run query_knowledge_base for each sub-query with target metadata filtering
        import re
        rag_chunks = []
        seen_chunks = set()
        has_valid_rag_query = False # RAG 대상 검색이 가동되었는지 확인하는 플래그
        
        for idx, sq in enumerate(decomposition.sub_queries, 1):
            sys.stderr.write(f"  - 서브 쿼리 {idx}: '{sq.query}' | 카테고리: {sq.asset_category}\n")
            
            # [최적화] '금융상품'은 RAG 문서가 없고 MySQL DB에서 실시간으로 스펙 정보를 가져오므로
            # 무의미한 RAG 검색 및 Tavily 웹 검색 Fallback을 사전에 건너뜁니다.
            if sq.asset_category == "금융상품":
                sys.stderr.write(f"  - [Skip] '금융상품' 카테고리는 RAG 검색을 건너뛰고 실시간 MySQL 상품 DB 데이터로 대체합니다.\n")
                continue
                
            has_valid_rag_query = True
            
            # Query vector database with native filters
            rag_res = query_knowledge_base(
                question=sq.query,
                chroma_db_dir=chroma_db_dir,
                threshold=THRESHOLD,
                asset_category=sq.asset_category
            )
            
            if rag_res:
                # Split retrieved results by double newlines and deduplicate
                for chunk in rag_res.split("\n\n"):
                    chunk = chunk.strip()
                    if chunk:
                        # Deduplicate by document content to avoid identical chunks from multiple queries
                        norm_chunk = re.sub(r'\s+', ' ', chunk)
                        if norm_chunk not in seen_chunks:
                            seen_chunks.add(norm_chunk)
                            rag_chunks.append(chunk)
                            
        # 3. Format and join retrieved RAG chunks
        if rag_chunks:
            formatted_chunks = []
            for c_idx, chunk in enumerate(rag_chunks, 1):
                # Replace original [1], [2] numbering with a continuous index [1]~[N]
                cleaned_chunk = re.sub(r'^\[\d+\]', f"[{c_idx}]", chunk)
                formatted_chunks.append(cleaned_chunk)
            retrieved_knowledge_parts.append("\n\n".join(formatted_chunks))
        elif has_valid_rag_query:
            # RAG 탐색 대상이 있었음에도 임계값 만족 청크가 없을 때만 Tavily Fallback 가동
            fallback_query = decomposition.sub_queries[0].query if decomposition.sub_queries else question
            sys.stderr.write(f"[RAG->Tavily] 임계값({THRESHOLD}) 만족 지식 없음. Tavily 웹 검색 실행 (쿼리: '{fallback_query}')\n")
            web_search = fetch_from_tavily(fallback_query)
            retrieved_knowledge_parts.append(web_search)
            
    except Exception as e:
        sys.stderr.write(f"[Advanced RAG 분석 오류] {str(e)}. 기본 원본 검색 및 Tavily 폴백을 시도합니다.\n")
        try:
            rag_res = query_knowledge_base(question, chroma_db_dir, THRESHOLD)
            if rag_res:
                retrieved_knowledge_parts.append(rag_res)
            else:
                web_search = fetch_from_tavily(question)
                retrieved_knowledge_parts.append(web_search)
        except Exception:
            web_search = fetch_from_tavily(question)
            retrieved_knowledge_parts.append(web_search)
            
    # 4. Retrieve MySQL Financial Products & matching information
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
        
    combined_knowledge = "\n\n".join(retrieved_knowledge_parts)
    
    # 5. Retrieve customer's features in the last 1 month from MySQL
    recent_features_str = ""
    try:
        from agent.feature.tools import get_customer_features
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
                if " (코사인 유사도:" in source_val:
                    source_val = source_val.split(" (코사인 유사도:")[0].strip()
                elif " (유사 거리:" in source_val:
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
        messages = []
        messages.append({"role": "system", "content": get_simulator_system_prompt()})
        user_context_prompt = get_simulator_user_prompt(
            context_content=context_content,
            recent_features_1m=recent_features_1m,
            retrieved_knowledge=retrieved_knowledge
        )
        messages.append({"role": "user", "content": user_context_prompt})
        messages.append({
            "role": "assistant",
            "content": get_assistant_acknowledgment()
        })
        
        import re
        for turn in history:
            cleaned_turn = turn.copy()
            if turn["role"] == "assistant":
                cleaned_turn["content"] = re.sub(r'\n\n\[참조 출처: .*?\]', '', turn["content"])
                cleaned_turn["content"] = re.sub(r'\[참조 출처: .*?\]', '', cleaned_turn["content"])
                cleaned_turn["content"] = re.sub(r'참조 출처: .*', '', cleaned_turn["content"])
            messages.append(cleaned_turn)
            
        messages.append({"role": "user", "content": question})
        
        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.4, api_key=api_key)
        response = llm.invoke(messages)
        answer = response.content
        
        answer = re.sub(r'\n\n\[참조 출처: .*?\]', '', answer)
        answer = re.sub(r'\[참조 출처: .*?\]', '', answer)
        answer = re.sub(r'참조 출처: .*', '', answer)
        answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)
        answer = re.sub(r'\*([^*]+)\*', r'\1', answer)
        answer = re.sub(r'__([^_]+)__', r'\1', answer)
        answer = re.sub(r'_([^_]+)_', r'\1', answer)
        answer = re.sub(r'(?m)^#+\s+', '', answer)
        answer = answer.strip()
        
        if retrieved_knowledge.strip():
            sources = extract_sources_from_knowledge(retrieved_knowledge)
            if sources:
                source_suffix = "\n\n[참조 출처: " + ", ".join(sources) + "]"
                answer += source_suffix
        
        # S3에 히스토리 저장
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        s3_write_json(f"simulator/history/customer_{customer_id}_history.json", history)
            
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
            config={"run_name": "Simulator-Agent", "tags": ["simulator_agent"]}
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
