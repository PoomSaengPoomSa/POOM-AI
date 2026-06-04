import os
import sys
import glob
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import pypdf
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add project root to sys.path to allow relative imports
# Note: Since this file is in 'agent/simulator/utils', the project root is 3 levels up.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Load environment variables
load_dotenv(find_dotenv())

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Environment variable OPENAI_API_KEY is not defined. Please check your .env file.")
    return OpenAI(api_key=api_key)



def process_pdfs_to_chroma():
    # Base paths relative to the simulator directory (parent of utils)
    simulator_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    raw_data_dir = os.path.join(simulator_dir, "data", "raw_data")
    chroma_db_dir = os.path.join(simulator_dir, "data", "chroma_db")
    
    print("=== [1. 환경 설정 및 클라이언트 초기화] ===")
    print(f"원천 데이터 디렉토리: {raw_data_dir}")
    print(f"Vector DB 저장 경로: {chroma_db_dir}")
    
    # OpenAI 및 ChromaDB 클라이언트 생성
    openai_client = get_openai_client()
    chroma_client = chromadb.PersistentClient(path=chroma_db_dir)
    
    # Recreate the collection to ensure a clean build
    try:
        chroma_client.delete_collection(name="poom_knowledge")
        print("기존 ChromaDB 콜렉션 'poom_knowledge' 삭제 완료.")
    except Exception:
        pass
    collection = chroma_client.create_collection(name="poom_knowledge")
    print("ChromaDB 콜렉션 'poom_knowledge' 신규 생성 완료.")

    # PDF 파일 목록 탐색 및 텍스트 추출 불가 이미지 스캔본 필터링
    pdf_pattern = os.path.join(raw_data_dir, "*.pdf")
    all_pdf_files = glob.glob(pdf_pattern)
    pdf_files = [f for f in all_pdf_files if "세금절약" not in os.path.basename(f)]
    
    if not pdf_files:
        print(f"[오류] 적재 가능한 PDF 파일이 {raw_data_dir}에 존재하지 않습니다.")
        sys.exit(1)
    
    print(f"발견된 적재 대상 PDF 파일 개수: {len(pdf_files)}개")
    for pdf_path in pdf_files:
        print(f"  - {os.path.basename(pdf_path)}")
 
    # Text Splitter 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []
    print("\n=== [2. PDF 텍스트 추출 및 청크 분할] ===")
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\n파일 처리 중: {filename}")
        
        try:
            reader = pypdf.PdfReader(pdf_path)
            total_pages = len(reader.pages)
            print(f"총 페이지 수: {total_pages}장")
            
            for page_idx, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if not text or len(text.strip()) < 10:
                    # 텍스트가 거의 없거나 추출되지 않은 경우 (스캔된 이미지 가능성)
                    continue
                
                # 텍스트 분할 실행
                chunks = text_splitter.split_text(text)
                for chunk_idx, chunk_content in enumerate(chunks):
                    chunk_id = f"{filename}_p{page_idx}_c{chunk_idx}"
                    all_chunks.append({
                        "id": chunk_id,
                        "document": chunk_content,
                        "metadata": {
                            "source": filename,
                            "page": page_idx
                        }
                    })
                    
            print(f"-> {filename} 파싱 완료 (누적 청크 수: {len(all_chunks)})")
            
        except Exception as e:
            print(f"[경고] {filename} 파싱 실패: {str(e)}")



    if not all_chunks:
        print("[오류] 적재할 데이터를 추출하지 못했습니다.")
        sys.exit(1)
        
    print(f"\n추출 완료: 총 {len(all_chunks)}개의 데이터 청크가 생성되었습니다.")
    print("\n=== [3. OpenAI 임베딩 및 ChromaDB 적재 (배치 처리)] ===")
    
    # OpenAI Rate Limit 및 네트워크 안정을 위한 100개 단위 배치 처리
    batch_size = 100
    total_chunks = len(all_chunks)
    
    for i in range(0, total_chunks, batch_size):
        batch = all_chunks[i:i + batch_size]
        batch_ids = [item["id"] for item in batch]
        batch_texts = [item["document"] for item in batch]
        batch_metadatas = [item["metadata"] for item in batch]
        
        print(f"배치 적재 중: {i+1} ~ {min(i + batch_size, total_chunks)} / {total_chunks}")
        
        try:
            # 배치 임베딩 호출
            response = openai_client.embeddings.create(
                input=batch_texts,
                model="text-embedding-3-small"
            )
            embeddings = [emb.embedding for emb in response.data]
            
            # ChromaDB 적재
            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                metadatas=batch_metadatas,
                documents=batch_texts
            )
        except Exception as e:
            print(f"[오류] 배치 {i+1} 적재 실패: {str(e)}")
            continue
            
    print("\n=== [4. 적재 완료 및 검증] ===")
    print(f"ChromaDB 적재 완료! 총 적재 항목 수: {collection.count()}")

if __name__ == "__main__":
    # Windows 콘솔 인코딩 에러 방지
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
            
    process_pdfs_to_chroma()
