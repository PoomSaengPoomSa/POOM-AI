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

    # 종합 컨텍스트 완성
    context_data = {
        "calendar": calendar_data,
        "kpi": kpi_data,
        "risks": risks_data,
        "events": events_data,
        "histories": histories_data,
        "notifications": notifications_data
    }

    logger.info("[StateAnalyzer] 모든 데이터 수집 및 context_data 구축 완료")
    
    return {
        "context_data": context_data
    }
