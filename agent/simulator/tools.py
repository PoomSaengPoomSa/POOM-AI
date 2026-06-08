import os
import sys
import requests
from typing import Optional, List
from openai import OpenAI

def fetch_from_tavily(query: str) -> str:
    """
    Fetch web search results using Tavily Search API wrapper.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        sys.stderr.write("[Tavily 경고] TAVILY_API_KEY 환경변수가 설정되지 않아 웹 검색을 생략합니다.\n")
        return "Tavily API 키가 설정되지 않아 실시간 웹 검색을 수행하지 못했습니다."
    
    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 3
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        if not results:
            return "Tavily 웹 검색 결과가 없습니다."
            
        search_parts = []
        for idx, item in enumerate(results, 1):
            title = item.get("title", "제목 없음")
            url_link = item.get("url", "")
            content = item.get("content", "")
            search_parts.append(f"[{idx}] 출처 (웹): {title} ({url_link})\n내용: {content.strip()}")
        return "\n\n".join(search_parts)
    except Exception as e:
        sys.stderr.write(f"[Tavily 오류] 웹 검색 실패: {str(e)}\n")
        return f"Tavily 웹 검색 중 오류가 발생했습니다: {str(e)}"


def query_knowledge_base(question: str, chroma_db_dir: str, threshold: float = 0.6) -> Optional[str]:
    """
    Query ChromaDB vector database across all document sources (excluding db_product metadata).
    Returns formatted knowledge string if within similarity threshold, else None.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not os.path.exists(chroma_db_dir):
        sys.stderr.write("[RAG 경고] ChromaDB 디렉토리가 존재하지 않습니다.\n")
        return None
        
    try:
        import chromadb
        chroma_client = chromadb.PersistentClient(path=chroma_db_dir)
        collection = chroma_client.get_collection(name="poom_knowledge")
        
        # 1. 질문 임베딩 생성
        openai_client = OpenAI(api_key=api_key)
        emb_response = openai_client.embeddings.create(
            input=[question],
            model="text-embedding-3-small"
        )
        query_embedding = emb_response.data[0].embedding
        
        # 2. ChromaDB 쿼리 (where 조건을 활용하여 db_product 제외 필터를 네이티브 적용)
        # 처음부터 db_product가 아닌 문서 중에서 최적의 3개 검색
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where={"source": {"$ne": "db_product"}}
        )
        
        # 3. 문자열 포맷팅 및 임계값 필터링
        knowledge_parts = []
        collected_count = 0
        if results and results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0], 
                results["metadatas"][0], 
                results["distances"][0]
            ):
                if dist <= threshold:
                    source = meta.get("source", "알 수 없음")
                    page = meta.get("page", "?")
                    collected_count += 1
                    knowledge_parts.append(f"[{collected_count}] 출처: {source} (Page {page}) (유사 거리: {dist:.4f})\n내용: {doc.strip()}")
        
        if knowledge_parts:
            sys.stderr.write(f"[RAG] 임계값({threshold}) 만족 지식 {len(knowledge_parts)}개 검색 완료 (전체 문서 대상)\n")
            return "\n\n".join(knowledge_parts)
        else:
            sys.stderr.write(f"[RAG] 임계값({threshold}) 만족 지식 없음 (전체 문서 대상)\n")
            return None
            
    except Exception as e:
        sys.stderr.write(f"[RAG 오류] 지식 검색 에러: {str(e)}\n")
        raise e


def get_customer_held_products(customer_id: int) -> List[dict]:
    """
    Fetch all financial products currently held by the customer from MySQL.
    """
    query = """
        SELECT p.pd_id, p.name, p.type, p.explanation, p.expected_return, p.return_type
        FROM customer_product cp
        JOIN product p ON cp.pd_id = p.pd_id
        WHERE cp.c_id = %s
    """
    try:
        from agent.customer.db import get_db_cursor
        with get_db_cursor() as cursor:
            cursor.execute(query, (customer_id,))
            return cursor.fetchall()
    except Exception as e:
        sys.stderr.write(f"[DB 오류] 고객 보유 상품 조회 실패: {str(e)}\n")
        return []


def get_all_products() -> List[dict]:
    """
    Fetch all financial products from MySQL.
    """
    query = """
        SELECT pd_id, name, type, explanation, features, target_customer, expected_return, return_type
        FROM product
    """
    try:
        from agent.customer.db import get_db_cursor
        with get_db_cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    except Exception as e:
        sys.stderr.write(f"[DB 오류] 전체 금융 상품 조회 실패: {str(e)}\n")
        return []


def get_customer_product_matching(customer_id: int) -> List[dict]:
    """
    Fetch suitability and matching recommendations for the customer from MySQL.
    """
    query = """
        SELECT pd_id, is_suitable, reason
        FROM product_matching
        WHERE c_id = %s
    """
    try:
        from agent.customer.db import get_db_cursor
        with get_db_cursor() as cursor:
            cursor.execute(query, (customer_id,))
            return cursor.fetchall()
    except Exception as e:
        sys.stderr.write(f"[DB 오류] 상품 매칭 정보 조회 실패: {str(e)}\n")
        return []


def format_products_context(held_products: List[dict], all_products: List[dict], matching_records: List[dict]) -> str:
    """
    Assemble and format MySQL product information into a structured text context.
    """
    held_ids = {p["pd_id"] for p in held_products}
    matching_map = {m["pd_id"]: (m["is_suitable"], m["reason"]) for m in matching_records}
    
    result_parts = []
    
    # 1. Held Products
    result_parts.append("[고객 보유 금융 상품 현황]")
    if not held_products:
        result_parts.append("현재 보유 중인 행내 금융 상품이 없습니다.")
    else:
        for idx, p in enumerate(held_products, 1):
            name = p.get("name") or "이름 없음"
            ptype = p.get("type") or "기타"
            expected_return = p.get("expected_return")
            return_type = p.get("return_type") or "수익률"
            return_str = f"{return_type} {expected_return}%" if expected_return is not None else "정보 없음"
            result_parts.append(f"{idx}. {name} (유형: {ptype} | 금리/수익률: {return_str})")
            
    result_parts.append("")
    
    # 2. Recommended & All Products
    result_parts.append("[고객 맞춤 금융 상품 추천 및 안내]")
    recommend_idx = 1
    for p in all_products:
        pd_id = p["pd_id"]
        name = p.get("name") or "이름 없음"
        ptype = p.get("type") or "기타"
        explanation = p.get("explanation") or "설명 없음"
        features = p.get("features") or "특징 없음"
        target = p.get("target_customer") or "모든 고객"
        expected_return = p.get("expected_return")
        return_type = p.get("return_type") or "수익률"
        return_str = f"{return_type} {expected_return}%" if expected_return is not None else "정보 없음"
        
        matching_info = matching_map.get(pd_id)
        suitability_str = ""
        rec_reason = ""
        
        if matching_info:
            is_suitable, reason = matching_info
            if is_suitable == 2:
                suitability_str = " (추천 강도: 낮음 - 현재 보유 중인 상품)"
            elif is_suitable == 1:
                suitability_str = " (추천 강도: 높음 - AI 매칭 추천 상품)"
            elif is_suitable == 0:
                suitability_str = " (추천 강도: 제외 - 투자 부적합 상품)"
            rec_reason = reason.strip()
        else:
            if pd_id in held_ids:
                suitability_str = " (현재 보유 중)"
            else:
                suitability_str = " (일반 추천 상품)"
        
        result_parts.append(f"[{recommend_idx}] 상품명: {name}{suitability_str}")
        result_parts.append(f"- 상품 유형: {ptype}")
        result_parts.append(f"- 금리/수익률: {return_str}")
        result_parts.append(f"- 상품 설명: {explanation}")
        result_parts.append(f"- 상품 특징: {features}")
        result_parts.append(f"- 가입 대상: {target}")
        if rec_reason:
            result_parts.append(f"- AI 추천 분석 사유: {rec_reason}")
        result_parts.append("")
        recommend_idx += 1
        
    sys.stderr.write(f"[DB] 금융 상품 매칭 분석 완료. 전체 상품 수: {len(all_products)}개 (매칭 기록 수: {len(matching_records)}개)\n")
    return "\n".join(result_parts)
