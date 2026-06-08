import os
import sys
import glob
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import pdfplumber
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



def format_table_as_markdown(table) -> str:
    """
    Format a 2D table array extracted by pdfplumber into a Markdown table string.
    """
    if not table or not any(table):
        return ""
        
    markdown_lines = []
    
    # Process header
    header = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in table[0]]
    # Handle completely empty header cases
    if not any(header):
        header = [f"Col {i}" for i in range(1, len(table[0]) + 1)]
    markdown_lines.append("| " + " | ".join(header) + " |")
    
    # Separator
    separators = ["---" for _ in header]
    markdown_lines.append("| " + " | ".join(separators) + " |")
    
    # Rows
    for row in table[1:]:
        if not any(row):
            continue
        row_str = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
        if len(row_str) < len(header):
            row_str += [""] * (len(header) - len(row_str))
        markdown_lines.append("| " + " | ".join(row_str[:len(header)]) + " |")
        
    return "\n".join(markdown_lines)


def process_pdfs_to_chroma():
    # Base metadata tagging taxonomy for the raw files
    BASE_TAGS = {
        "1. 2026년 6월 House View.pdf": {
            "asset_category": "매크로",
            "data_lifecycle": "주기적 변경",
            "target_segment": "공통"
        },
        "2025 우리금융 트렌드 보고서1.pdf": {
            "asset_category": "매크로",
            "data_lifecycle": "주기적 변경",
            "target_segment": "공통"
        },
        "2025 한국 부자 보고서.pdf": {
            "asset_category": "매크로",
            "data_lifecycle": "주기적 변경",
            "target_segment": "공통"
        },
        "2026 대한민국 웰스 리포트_하나금융연구소.pdf": {
            "asset_category": "매크로",
            "data_lifecycle": "주기적 변경",
            "target_segment": "공통"
        },
        "2026년 개정세법 해설.pdf": {
            "asset_category": "세무",
            "data_lifecycle": "영속 가이드",
            "target_segment": "공통"
        },
        "230919_금융소비자보호법 설명자료_f.pdf": {
            "asset_category": "컴플라이언스",
            "data_lifecycle": "영속 가이드",
            "target_segment": "공통"
        }
    }

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
    collection = chroma_client.create_collection(name="poom_knowledge", metadata={"hnsw:space": "cosine"})
    print("ChromaDB 콜렉션 'poom_knowledge' 신규 생성 완료 (코사인 유사도 설정).")

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
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"총 페이지 수: {total_pages}장")
                
                for page_idx, page in enumerate(pdf.pages, 1):
                    # 1. 감지된 표 영역(bbox) 확보
                    tables = page.find_tables()
                    table_bboxes = [t.bbox for t in tables]
                    
                    # 2. 표 영역 외부의 텍스트만 추출 (중복 방지)
                    if table_bboxes:
                        def is_outside_tables(obj):
                            if "x0" not in obj or "top" not in obj or "x1" not in obj or "bottom" not in obj:
                                return True
                            x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
                            for t_x0, t_top, t_x1, t_bottom in table_bboxes:
                                if not (x1 <= t_x0 or x0 >= t_x1 or bottom <= t_top or top >= t_bottom):
                                    return False
                            return True
                        non_table_page = page.filter(is_outside_tables)
                        non_table_text = non_table_page.extract_text()
                    else:
                        non_table_text = page.extract_text()
                    
                    # 3. 표 데이터 추출 및 마크다운 변환
                    markdown_tables = []
                    if tables:
                        extracted_tables = page.extract_tables()
                        for raw_table in extracted_tables:
                            md_table = format_table_as_markdown(raw_table)
                            if md_table:
                                markdown_tables.append(md_table)
                                
                    # 4. 결합 및 정제
                    content_parts = []
                    if non_table_text and len(non_table_text.strip()) >= 10:
                        content_parts.append(non_table_text.strip())
                    if markdown_tables:
                        content_parts.append("\n\n" + "\n\n".join(markdown_tables))
                        
                    page_content = "\n\n".join(content_parts).strip()
                    if len(page_content) < 10:
                        continue
                    
                    # 5. 텍스트 분할 실행
                    chunks = text_splitter.split_text(page_content)
                    
                    base_tags = BASE_TAGS.get(filename, {
                        "asset_category": "공통",
                        "data_lifecycle": "영속 가이드",
                        "target_segment": "공통"
                    })
                    
                    for chunk_idx, chunk_content in enumerate(chunks):
                        chunk_id = f"{filename}_p{page_idx}_c{chunk_idx}"
                        
                        # Dynamic target segment assignment based on content
                        target_segment = base_tags.get("target_segment", "공통")
                        lower_content = chunk_content.lower()
                        
                        if filename == "2026 대한민국 웰스 리포트_하나금융연구소.pdf":
                            if "영리치" in lower_content or "young rich" in lower_content:
                                target_segment = "영리치"
                            elif "시니어" in lower_content or "고령" in lower_content or "올드리치" in lower_content:
                                target_segment = "시니어"
                        elif filename == "2026년 개정세법 해설.pdf":
                            corp_keywords = ["가업상속", "가업승계", "법인세", "배당소득", "최대주주", "가업 상속", "가업 승계"]
                            if any(kw in chunk_content for kw in corp_keywords):
                                target_segment = "기업인"
                                
                        all_chunks.append({
                            "id": chunk_id,
                            "document": chunk_content,
                            "metadata": {
                                "source": filename,
                                "page": page_idx,
                                "asset_category": base_tags.get("asset_category", "공통"),
                                "data_lifecycle": base_tags.get("data_lifecycle", "영속 가이드"),
                                "target_segment": target_segment
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
