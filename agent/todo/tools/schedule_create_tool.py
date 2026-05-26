from datetime import datetime
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from .db_helper import get_db_session

# SQLAlchemy 모델 임포트
from app.models.ai_todo import AiTodo
from app.models.customer import Customer

def save_recommended_todo(
    u_id: str,
    c_id: Optional[int],
    execution_time_str: str,
    title: str,
    memo: str,
    category: str
) -> Dict[str, Any]:
    """
    추천 일정을 데이터베이스(ai_todo 테이블)에 저장합니다.
    """
    try:
        exec_date = datetime.strptime(execution_time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            exec_date = datetime.fromisoformat(execution_time_str.replace("T", " "))
        except ValueError:
            exec_date = datetime.now()

    # DB 카테고리 Check Constraint 정교한 매핑 및 예외 방지
    allowed_categories = ["KPI 기반", "상담 일정 제안", "안부 연락 제안", "신규 상품 분석"]
    if category not in allowed_categories:
        # 안전장치 매핑
        if "안부" in category or "생일" in category or "기념일" in category:
            category = "안부 연락 제안"
        elif "만기" in category or "상담" in category or "재가입" in category:
            category = "상담 일정 제안"
        elif "상품" in category or "투자" in category or "분석" in category:
            category = "신규 상품 분석"
        else:
            category = "KPI 기반"

    with get_db_session() as db:
        new_todo = AiTodo(
            title=title[:50],  # 50자 제한 방어
            memo=memo[:80] if memo else "AI 추천 일정",  # 80자 제한 방어
            category=category,
            create_date=datetime.now(),
            execution_date=exec_date,
            is_checked=False,
            u_id=u_id,
            c_id=c_id
        )
        db.add(new_todo)
        db.commit()
        db.refresh(new_todo)

        # 고객명 조회
        cust_name = "없음"
        if c_id:
            customer = db.query(Customer).filter(Customer.c_id == c_id).first()
            if customer:
                cust_name = customer.name

        return {
            "at_id": new_todo.at_id,
            "title": new_todo.title,
            "memo": new_todo.memo,
            "category": new_todo.category,
            "execution_date": new_todo.execution_date.strftime("%Y-%m-%d %H:%M:%S"),
            "u_id": new_todo.u_id,
            "c_id": new_todo.c_id,
            "customer_name": cust_name,
            "status": "success"
        }

@tool
def CreateScheduleTool(
    u_id: str,
    c_id: Optional[int],
    execution_time: str,
    title: str,
    memo: str,
    category: str
) -> str:
    """
    Create Schedule Tool.
    에이전트가 도출해 낸 최적의 추천 일정(AI To-Do) 후보를 DB(`ai_todo` 테이블)에 적재하여 저장합니다.
    - u_id: PB ID
    - c_id: 추천 상담 대상 고객 ID (int, 없거나 일반적인 할 일인 경우 null 허용)
    - execution_time: 추천 실행 날짜 및 시간 ('YYYY-MM-DD HH:MM:SS' 형식)
    - title: 추천 일정의 명확한 제목 (50자 이내)
    - memo: 추천 세부 사유 및 정보 기술 (80자 이내)
    - category: 카테고리 (반드시 'KPI 기반', '상담 일정 제안', '안부 연락 제안', '신규 상품 분석' 중 하나여야 합니다.)
    """
    try:
        res = save_recommended_todo(
            u_id=u_id,
            c_id=c_id,
            execution_time_str=execution_time,
            title=title,
            memo=memo,
            category=category
        )
        cust_info = f"{res['customer_name']} (c_id: {res['c_id']})" if res['c_id'] else "없음 (일반 업무)"
        return (
            f"[성공] 추천 일정(AI To-Do)이 DB에 성공적으로 저장되었습니다.\n"
            f"- 추천 ID: {res['at_id']}\n"
            f"- 대상 고객: {cust_info}\n"
            f"- 제목: {res['title']}\n"
            f"- 시간: {res['execution_date']}\n"
            f"- 카테고리: {res['category']}"
        )
    except Exception as e:
        return f"[실패] DB 적재 중 오류 발생: {str(e)}"
