import os
import sys
from openai import OpenAI
import chromadb
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Environment variable OPENAI_API_KEY is not defined. Please check your .env file.")
    return OpenAI(api_key=api_key)

def query_vector_db(query_text: str, n_results: int = 3):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_db_dir = os.path.join(current_dir, "data", "chroma_db")
    
    print(f"=== Vector DB 검색 테스트 ===")
    print(f"검색 쿼리: '{query_text}'")
    print(f"DB 경로: {chroma_db_dir}")
    
    # OpenAI 및 ChromaDB 클라이언트 생성
    openai_client = get_openai_client()
    chroma_client = chromadb.PersistentClient(path=chroma_db_dir)
    
    try:
        collection = chroma_client.get_collection(name="poom_knowledge")
    except Exception as e:
        print(f"[오류] 콜렉션을 찾을 수 없습니다. 적재를 먼저 실행해주세요. 에러: {e}")
        return

    # 1. 쿼리 텍스트 임베딩 생성
    try:
        response = openai_client.embeddings.create(
            input=[query_text],
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        print(f"[오류] OpenAI 임베딩 생성 실패: {e}")
        return

    # 2. 유사도 검색 수행
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    # 3. 결과 출력
    if not results or not results["documents"] or not results["documents"][0]:
        print("검색 결과가 없습니다.")
        return
        
    print(f"\n총 {len(results['documents'][0])}개의 결과를 찾았습니다:\n")
    for idx, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], 
        results["metadatas"][0], 
        results["distances"][0]
    ), 1):
        print(f"[{idx}] 출처: {meta.get('source')} (Page: {meta.get('page')}) (유사 거리: {dist:.4f})")
        print(f"내용 요약:\n{doc.strip()[:300]}...")
        print("-" * 50)

if __name__ == "__main__":
    # Windows 콘솔 인코딩 에러 방지
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
            
    query = sys.argv[1] if len(sys.argv) > 1 else "부동산 시장 전망"
    query_vector_db(query)
