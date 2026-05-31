import os
import sys
from datetime import datetime, timedelta
from typing import TypedDict, List

# Windows Console UTF-8 Reconfiguration to prevent CP949 codec errors
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# 1. sys.path 및 백엔드 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.abspath(os.path.join(current_dir, "..")), # POOM-AI 루트
    os.path.abspath(os.path.join(current_dir, "..", "..")), # poom 루트
    os.path.abspath(os.path.join(current_dir, "..", "..", "POOM-BACK")), # Docker POOM-BACK
    os.path.abspath(os.path.join(current_dir, "..", "..", "back")), # 윈도우 로컬
]
back_path = None
for p in possible_paths:
    if os.path.exists(os.path.join(p, "app", "database.py")):
        back_path = p
        break
if not back_path:
    back_path = os.path.abspath(os.path.join(current_dir, "..", "..", "back")) # Fallback

if back_path not in sys.path:
    sys.path.insert(0, back_path)

poom_ai_path = os.path.abspath(os.path.join(current_dir, ".."))
if poom_ai_path not in sys.path:
    sys.path.insert(0, poom_ai_path)

# 2. .env 환경변수 로드
from dotenv import load_dotenv, find_dotenv
# poom 루트에 있는 .env 로드
env_file_path = os.path.abspath(os.path.join(back_path, "..", ".env"))
if os.path.exists(env_file_path):
    load_dotenv(env_file_path)
else:
    load_dotenv(find_dotenv())

# Pydantic Settings ValidationError 방지 처리를 적용하며 SQLAlchemy 임포트
from agent.todo.tools.db_helper import get_db_session
from app.models import Customer, ChurnLevel, Schedule, ConsultationMemo

# LangGraph 임포트
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 상태 정의
class AgentState(TypedDict):
    high_risk_c_ids: List[int]
    long_term_no_visit_c_ids: List[int]
    today_scheduled_c_ids: List[int]
    candidates_info: dict
    report: str

def extract_customer_node(state: AgentState) -> dict:
    """
    1단계: 데이터베이스를 개별적으로 조회하여 각 조건에 맞는 고객 c_id들을 추출하는 Node.
    조건:
      1. 이탈 위험 수준이 '위험' (ChurnLevel.grade == '위험')
      2. 마지막 방문(상담)일이 30일 이상 경과했거나 아예 상담 이력이 없음 (ConsultationMemo)
      3. 오늘 상담이 예정된 고객 (Schedule.category == '상담' 이고 execution_date가 오늘)
    """
    print("\n🔍 [Node 1] 고객 조건별 개별 추출 노드 구동 시작...")
    
    today = datetime.now()
    start_of_today = datetime.combine(today.date(), datetime.min.time())
    end_of_today = datetime.combine(today.date(), datetime.max.time())
    cutoff_date = today - timedelta(days=30)
    
    with get_db_session() as db:
        # A. 조건 1: 이탈 위험 등급이 '위험'인 고객 조회 (각 고객별 가장 최근 ChurnLevel 기준)
        from sqlalchemy import func
        subq = db.query(
            ChurnLevel.c_id,
            func.max(ChurnLevel.created_date).label('max_date')
        ).group_by(ChurnLevel.c_id).subquery()

        high_risk_records = db.query(ChurnLevel).join(
            subq,
            (ChurnLevel.c_id == subq.c.c_id) & (ChurnLevel.created_date == subq.c.max_date)
        ).filter(ChurnLevel.grade == '위험').all()
        
        high_risk_c_ids = [r.c_id for r in high_risk_records]
        
        # 상세 데이터 로드 (보고서용)
        high_risk_details = []
        for r in high_risk_records:
            cust = db.query(Customer).filter(Customer.c_id == r.c_id).first()
            if cust:
                high_risk_details.append({
                    "c_id": r.c_id,
                    "name": cust.name,
                    "grade": cust.grade,
                    "total_assets": f"{cust.total_assets / 100000000:.1f}억" if cust.total_assets else "0원",
                    "churn_reason": r.reason
                })

        # B. 조건 2: 마지막 상담일이 30일 이상 경과했거나 아예 상담 이력이 없는 고객 조회
        # 최근 30일 이내에 상담한 고객 c_id 조회
        recent_consulted_records = db.query(ConsultationMemo.c_id).filter(
            ConsultationMemo.consult_date > cutoff_date
        ).distinct().all()
        recent_consulted_c_ids = {r[0] for r in recent_consulted_records}
        
        # 전체 고객 c_id 조회
        all_customers = db.query(Customer.c_id).all()
        all_c_ids = {c.c_id for c in all_customers}
        
        # 마지막 상담이 30일 이상 지난 고객 = (전체 고객 - 최근 30일 이내 상담한 고객)
        long_term_no_visit_c_ids = list(all_c_ids - recent_consulted_c_ids)

        # C. 조건 3: 오늘 상담이 예정된 고객 조회
        today_scheduled_records = db.query(Schedule).filter(
            Schedule.category == '상담',
            Schedule.c_id.isnot(None),
            Schedule.execution_date >= start_of_today,
            Schedule.execution_date <= end_of_today
        ).all()
        
        today_scheduled_c_ids = list({s.c_id for s in today_scheduled_records})
        
        # 상세 데이터 로드 (보고서용)
        today_scheduled_details = []
        for s in today_scheduled_records:
            cust = db.query(Customer).filter(Customer.c_id == s.c_id).first()
            if cust:
                today_scheduled_details.append({
                    "c_id": s.c_id,
                    "name": cust.name,
                    "grade": cust.grade,
                    "scheduled_time": s.execution_date.strftime("%H:%M"),
                    "scheduled_title": s.title
                })
        
        # 디버그/리포팅 정보 통합
        candidates_info = {
            "high_risk_c_ids": high_risk_c_ids,
            "high_risk_details": high_risk_details,
            "long_term_no_visit_c_ids": sorted(long_term_no_visit_c_ids),
            "today_scheduled_c_ids": today_scheduled_c_ids,
            "today_scheduled_details": today_scheduled_details
        }
        
        print(f"📊 [중간 분석 현황]")
        print(f"  - 조건 1 (이탈 위험 '위험'): {high_risk_c_ids}")
        print(f"  - 조건 2 (마지막 방문 30일 이상 경과, 총 {len(long_term_no_visit_c_ids)}명 c_id 일부): {sorted(long_term_no_visit_c_ids)[:15]}...")
        print(f"  - 조건 3 (오늘 상담 예정): {today_scheduled_c_ids}")
        
        return {
            "high_risk_c_ids": high_risk_c_ids,
            "long_term_no_visit_c_ids": sorted(long_term_no_visit_c_ids),
            "today_scheduled_c_ids": today_scheduled_c_ids,
            "candidates_info": candidates_info
        }

def llm_agent_node(state: AgentState) -> dict:
    """
    2단계: 추출된 고객 ID 리스트와 데이터를 바탕으로 LLM을 활용해 분석 및 결과 보고서를 작성하는 Node.
    """
    print("\n🤖 [Node 2] LLM 에이전트 분석 노드 구동 시작...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    candidates_info = state.get("candidates_info", {})
    
    high_risk_c_ids = state.get("high_risk_c_ids", [])
    high_risk_details = candidates_info.get("high_risk_details", [])
    long_term_no_visit_c_ids = state.get("long_term_no_visit_c_ids", [])
    today_scheduled_c_ids = state.get("today_scheduled_c_ids", [])
    today_scheduled_details = candidates_info.get("today_scheduled_details", [])
    
    prompt = f"""당신은 자산관리 부문의 핵심 VIP 고객 케어를 전담하는 **고객 분석 전문 AI 에이전트**입니다.
데이터베이스에서 각각의 필터 조건으로 추출된 고객 ID(c_id) 목록과 정보를 분석하고, 담당 PB가 미팅 및 스케줄 관리에 참고할 수 있도록 깔끔한 **[조건별 고객 추출 보고서]**를 작성하십시오.

### [추출된 각 조건별 고객 데이터]
1. **조건 1: 이탈 위험 수준이 '위험'인 고객 (c_id 리스트 및 상세)**
   - c_id 목록: {high_risk_c_ids}
   - 상세 정보: {high_risk_details}

2. **조건 2: 마지막 방문(상담) 이력이 30일 이상 경과(혹은 없음)한 고객 (c_id 리스트)**
   - c_id 목록 (총 {len(long_term_no_visit_c_ids)}명): {long_term_no_visit_c_ids}

3. **조건 3: 오늘 상담이 예정된 고객 (c_id 리스트 및 상세)**
   - c_id 목록: {today_scheduled_c_ids}
   - 상세 정보: {today_scheduled_details}

### [작성 가이드라인]
1. 리포트에는 각 조건별로 추출된 **실제 c_id 목록**을 마크다운 표나 불릿 포인트 형식으로 **명확하게 명시**해야 합니다. 절대 인원수만 서술하고 넘어가면 안 됩니다.
2. 각 조건별 대상 고객들의 이름과 정보가 있다면 함께 기술하여 PB가 실질적으로 고객을 인지하고 식별할 수 있도록 해 주십시오.
3. 조건 1의 이탈 위험 사유, 조건 3의 오늘 상담 정보(시간, 제목)를 예리하게 짚고 각각에 맞는 PB 대처안을 서술해 주십시오.
4. 보고서는 논리적이고 세련된 마크다운 형식으로 작성해 주십시오.
"""

    report_content = ""
    
    if api_key and not api_key.startswith("your-") and len(api_key) > 10:
        try:
            print("  - OpenAI API Key가 감지되어 ChatOpenAI(gpt-4o) 분석을 요청합니다.")
            llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=api_key)
            messages = [
                SystemMessage(content="당신은 VIP 자산관리 전문 분석 에이전트입니다. 항상 c_id 리스트를 전면에 내세운 예리하고 구체적인 마크다운 보고서를 제공합니다."),
                HumanMessage(content=prompt)
            ]
            response = llm.invoke(messages)
            report_content = response.content
        except Exception as e:
            print(f"  - OpenAI API 호출 실패로 인한 Heuristic Fallback 가동: {e}")
            report_content = generate_heuristic_report(candidates_info)
    else:
        print("  - OpenAI API Key가 유효하지 않아 Heuristic Fallback 보고서로 가동합니다.")
        report_content = generate_heuristic_report(candidates_info)

    return {
        "report": report_content
    }

def generate_heuristic_report(info: dict) -> str:
    """API Key 부재 시 초정밀 Heuristic Fallback 보고서를 작성합니다."""
    high_risk_c_ids = info.get("high_risk_c_ids", [])
    high_risk_details = info.get("high_risk_details", [])
    long_term_no_visit_c_ids = info.get("long_term_no_visit_c_ids", [])
    today_scheduled_c_ids = info.get("today_scheduled_c_ids", [])
    today_scheduled_details = info.get("today_scheduled_details", [])
    
    report = f"""# 📊 [조건별 VIP 고객 추출 보고서 (Heuristic Fallback)]

이탈 위험, 상담 공백, 오늘 일정에 따른 각 조건별 고객 추출 결과입니다. (각 조건별 c_id 목록 포함)

---

### 🔥 1. 이탈 위험 수준이 높은 고객 ('위험' 등급)
- **추출된 c_id 목록**: {high_risk_c_ids if high_risk_c_ids else "없음"}
- **상세 고객 정보**:
"""
    if not high_risk_details:
        report += "  - 현재 이탈 위험 '위험' 단계인 고객이 없습니다.\n"
    else:
        for c in high_risk_details:
            report += f"  - **{c['name']}** 고객님 (c_id: **{c['c_id']}**) | 등급: {c['grade']} | 자산: {c['total_assets']} \n    * ⚠️ 이탈 위험 사유: {c['churn_reason']}\n"

    report += f"""
---

### ⏳ 2. 마지막 방문(상담)이 30일 이상 경과했거나 없는 고객
- **추출된 c_id 목록 (총 {len(long_term_no_visit_c_ids)}명)**:
  - {long_term_no_visit_c_ids if long_term_no_visit_c_ids else "없음"}
- **케어 공백 분석**:
  - 위 고객들은 최근 30일 이내에 상담을 진행한 적이 없는 관리 공백 상태입니다. 장기 방치 고객이 되지 않도록 PB의 모니터링이 필요합니다.

---

### 📅 3. 오늘 상담이 예정된 고객
- **추출된 c_id 목록**: {today_scheduled_c_ids if today_scheduled_c_ids else "없음"}
- **상세 일정 정보**:
"""
    if not today_scheduled_details:
        report += "  - 오늘 예정된 상담 일정이 존재하지 않습니다.\n"
    else:
        for c in today_scheduled_details:
            report += f"  - **{c['name']}** 고객님 (c_id: **{c['c_id']}**) | 예정 시간: {c['scheduled_time']} | 일정명: {c['scheduled_title']} ({c['grade']} 등급)\n"

    return report

def build_extract_agent():
    """LangGraph 워크플로우를 구성하고 컴파일합니다."""
    workflow = StateGraph(AgentState)
    
    # 노드 등록
    workflow.add_node("extractor", extract_customer_node)
    workflow.add_node("llm_agent", llm_agent_node)
    
    # 관계 연결
    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "llm_agent")
    workflow.add_edge("llm_agent", END)
    
    return workflow.compile()

if __name__ == "__main__":
    print("==================================================================")
    print("🚀 [AI Test Agent] 우선 케어 고객 선정 에이전트 가동 (extract_customer.py)")
    print("==================================================================")
    
    # API 키 로드 여부 검증
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"🔑 OPENAI_API_KEY 로드 성공: {api_key[:10]}... (총 {len(api_key)}자)")
    else:
        print("⚠️ Warning: OPENAI_API_KEY를 .env에서 찾지 못했습니다. Heuristic Fallback 모드로 실행합니다.")

    # 그래프 생성 및 실행
    agent = build_extract_agent()
    
    initial_state = {
        "high_risk_c_ids": [],
        "long_term_no_visit_c_ids": [],
        "today_scheduled_c_ids": [],
        "candidates_info": {},
        "report": ""
    }
    
    try:
        final_state = agent.invoke(initial_state)
        
        print("\n==================================================================")
        print("🎯 [FINAL REPORT] 생성 완료")
        print("==================================================================")
        print(final_state.get("report"))
        print("==================================================================")
        
        # 결과 파일 별도 기록 지원 (검증용)
        report_log_path = os.path.join(current_dir, "extract_report_output.md")
        with open(report_log_path, "w", encoding="utf-8") as f:
            f.write(final_state.get("report", ""))
        print(f"💾 보고서 원본이 {report_log_path} 에 성공적으로 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 에이전트 구동 중 오류 발생: {e}", exc_info=True)
