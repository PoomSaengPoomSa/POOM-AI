import logging
from typing import Dict, Any
from graph.state import AgentState

# 도구들의 실제 파이썬 함수 및 도구 가져오기
from tools.calendar_tool import GetCalendarScheduleTool
from tools.customer_tool import (
    GetCustomerRiskTool,
    GetRecentConsultingHistoryTool,
    GetCustomerEventTool
)
from tools.notification_tool import GetNotificationTool
from tools.kpi_tool import GetKPIStatusTool

logger = logging.getLogger(__name__)

def state_analyzer_node(state: AgentState) -> Dict[str, Any]:
    """
    State Analyzer Node.
    PB 및 담당 고객의 모든 현황 데이터를 DB 툴들로 실시간 조회하여
    종합 컨텍스트 상황판(context_data)을 작성해 State에 저장합니다.
    """
    u_id = state.get("u_id")
    target_date = state.get("target_date")
    
    logger.info(f"[StateAnalyzer] u_id: {u_id}, target_date: {target_date} 데이터 수집 시작")

    # 1. 캘린더 기존 일정 수집
    calendar_data = GetCalendarScheduleTool.invoke({
        "u_id": u_id,
        "date_str": target_date
    })

    # 2. KPI 실적 데이터 수집
    kpi_data = GetKPIStatusTool.invoke({
        "u_id": u_id
    })

    # 3. 이탈 위험 고객 수집
    risks_data = GetCustomerRiskTool.invoke({
        "u_id": u_id
    })

    # 4. 30일 이내 만기 및 기념일 이벤트 수집
    events_data = GetCustomerEventTool.invoke({
        "u_id": u_id,
        "date_str": target_date
    })

    # 5. 최근 상담 이력 수집
    histories_data = GetRecentConsultingHistoryTool.invoke({
        "u_id": u_id
    })

    # 6. 기존 생성 알림 리스트 수집
    notifications_data = GetNotificationTool.invoke({
        "u_id": u_id
    })

    # 7. 이미 일정이 예약된 고객 및 과거 미등록(무시)된 추천 일정 히스토리 조회 (중요도 감쇠 및 중복 배제용)
    ignored_info_str = "무시된 과거 추천 히스토리 없음"
    scheduled_c_ids = []
    
    from tools.db_helper import get_db_session
    from app.models.ai_todo import AiTodo
    from app.models.schedule import Schedule
    from datetime import datetime
    
    try:
        target_dt_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        midnight_bound = datetime.combine(target_dt_obj, datetime.min.time())
        with get_db_session() as db:
            # (A) 과거에 AI 추천되었으나 PB가 무시(is_checked == False)한 리스트
            past_ignored = (
                db.query(AiTodo)
                .filter(AiTodo.u_id == u_id)
                .filter(AiTodo.is_checked == False)
                .filter(AiTodo.execution_date < midnight_bound)
                .all()
            )
            if past_ignored:
                ignored_list = []
                for t in past_ignored:
                    cust_name = t.customer.name if t.customer else f"ID {t.c_id}"
                    ignored_list.append(
                        f"- {cust_name} 고객(c_id: {t.c_id}): '{t.title}' "
                        f"({t.execution_date.strftime('%Y-%m-%d')} 추천되었으나 일정 등록하지 않음 -> 중요도 감쇠 대상)"
                    )
                ignored_info_str = "\n".join(ignored_list)
            
            # (B) 이미 미래 또는 현재 일정이 수립되어 상담 예약 완료된 고객 리스트
            active_schedules = (
                db.query(Schedule)
                .filter(Schedule.u_id == u_id)
                .filter(Schedule.c_id != None)
                .all()
            )
            scheduled_c_ids = list(set([s.c_id for s in active_schedules]))
            
    except Exception as e:
        logger.warning(f"[StateAnalyzer] 과거 무시 이력 및 기예약 고객 조회 실패: {e}")

    # 종합 컨텍스트 완성
    context_data = {
        "calendar": calendar_data,
        "kpi": kpi_data,
        "risks": risks_data,
        "events": events_data,
        "histories": histories_data,
        "notifications": notifications_data,
        "ignored_history": ignored_info_str,
        "scheduled_customers": scheduled_c_ids
    }

    logger.info("[StateAnalyzer] 모든 데이터 수집 및 context_data 구축 완료 (과거 이력 분석 포함)")
    
    return {
        "context_data": context_data
    }
