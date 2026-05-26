from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    AI To Do Agent의 상태(State)를 추적하는 클래스입니다.
    LangGraph의 각 노드들은 이 상태를 읽고 업데이트하며 프로세스를 조율합니다.
    """
    u_id: str
    """분석 대상 PB(Private Banker)의 ID"""

    target_date: str
    """분석 기준일 (기본값: 오늘 날짜 YYYY-MM-DD)"""

    context_data: Dict[str, Any]
    """
    State Analyzer가 각 DB Tool을 가동하여 수집한 종합 상황 컨텍스트 정보
    구조 예시:
    {
      "calendar": [...],
      "kpi": {...},
      "risks": [...],
      "histories": [...],
      "features": {...},
      "events": [...],
      "notifications": [...]
    }
    """

    current_goal: Dict[str, Any]
    """
    Goal Selector Agent가 수집 데이터 분석을 바탕으로 수립한 오늘의 최우선 비즈니스 목표와 추천 사유
    구조 예시:
    {
      "goal": "VIP 고객 만기 예정 상품 재가입 상담",
      "reason": "오후 일정이 비어 있고, KPI AUM 달성률이 저조한 반면, 만기 도래 30일 이내인 VIP 고객 2명이 식별됨."
    }
    """

    plan_tools: List[str]
    """
    Planner Agent가 Goal 달성을 위해 동적으로 구성한 Tool 실행 계획 (Tool 명칭 리스트)
    구조 예시: ["GetCustomerEventTool", "GetCalendarScheduleTool", "CreateScheduleTool"]
    """

    execution_results: List[Dict[str, Any]]
    """
    Executor Agent가 Planner의 계획에 따라 Tool을 실행해 도출한 추천 일정(AI To-Do) 후보 목록
    구조 예시:
    [
      {
        "title": "[만기] 김OO 고객 상담 제안",
        "memo": "정기예금 만기 도래 15일 전. 재가입 및 포트폴리오 리밸런싱 상담 추천.",
        "category": "상담 일정 제안",
        "execution_date": "2026-06-01 14:00:00",
        "c_id": 123
      }
    ]
    """

    evaluation: Dict[str, Any]
    """
    Evaluator Agent가 추천 일정을 실제 캘린더 일정과 검증한 결과 리포트
    구조 예시:
    {
      "is_passed": False,
      "feedback": "14:00 ~ 15:00 시간대에 기존 캘린더 일정 '[상담] 강OO'과 겹쳐 일정 충돌이 발생했습니다."
    }
    """

    reflection_guidance: Optional[str]
    """
    Reflection Node가 Evaluator의 반려 원인을 정밀 분석하여Planner에게 피드백하는 재계획 가이드라인
    구조 예시: "기존 14:00~15:00 캘린더 일정 충돌이 감지되었으므로, 해당 고객의 빈 시간대인 16:00 이후로 일정을 제안하거나, 다른 VIP 고객을 선택하여 추천 일정을 잡으십시오."
    """

    retry_count: int
    """재계획 시도 횟수 (최대 3회 제한으로 무한 루프 방지)"""
