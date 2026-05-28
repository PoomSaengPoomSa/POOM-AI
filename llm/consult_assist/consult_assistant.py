import os
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# 현재 파일 기준 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path)

class ConsultationReport(BaseModel):
    key_contents: List[str] = Field(
        ..., 
        description="상담의 주요 핵심 내용을 요약한 리스트 (예: 내방 목적, 관심 상품 및 주요 대화 주제)"
    )
    special_notes: List[str] = Field(
        ..., 
        description="고객의 성향, 자금 운용 제약 조건, 가족 배경, 우려 사항 등의 특이사항 리스트"
    )
    follow_up_actions: List[str] = Field(
        ..., 
        description="상담 이후 진행되어야 할 To-Do 리스트 (예: 세무 상담 예약, 필요 서류 안내, 다음 방문 일정 등)"
    )
    summary: str = Field(
        ...,
        description="상담 전체의 핵심 맥락과 결론을 요약한 친절한 1문장 내외의 요약문"
    )

def get_prompt_template() -> str:
    """prompt/memo_report_prompt.md 파일의 내용을 읽어옵니다."""
    prompt_path = os.path.join(current_dir, "prompt", "memo_report_prompt.md")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def structure_consultation_memo(memo_text: str) -> ConsultationReport:
    """
    상담 메모를 입력받아 정해진 구조(주요 내용, 특이사항, 후속 조치)로 구조화하여 반환합니다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Environment variable OPENAI_API_KEY is not defined. Please check your .env file.")
    
    # OpenAI 클라이언트 초기화
    client = OpenAI(api_key=api_key)
    
    # 시스템 프롬프트 로드
    system_prompt = get_prompt_template()
    
    # GPT-4o-mini 모델을 사용하여 구조화된 출력(Structured Outputs) 획득
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"상담 메모:\n{memo_text}"}
        ],
        response_format=ConsultationReport,
        temperature=0.1,  # 사실성 보장 및 일관성을 위해 낮은 온도로 설정
    )
    
    # 파싱된 결과 반환
    return completion.choices[0].message.parsed

if __name__ == "__main__":
    import sys
    import json
    
    # Windows 콘솔 인코딩 에러 방지
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stdin.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
            
    try:
        # stdin에서 메모 읽기
        memo_input = sys.stdin.read().strip()
        if not memo_input:
            print(json.dumps({"error": "No input provided"}), file=sys.stderr)
            sys.exit(1)
            
        report_data = structure_consultation_memo(memo_input)
        # JSON 형태로 표준 출력에 인쇄
        sys.stdout.write(report_data.model_dump_json())
        sys.stdout.flush()
    except Exception as e:
        sys.stderr.write(json.dumps({"error": str(e)}))
        sys.stderr.flush()
        sys.exit(1)
