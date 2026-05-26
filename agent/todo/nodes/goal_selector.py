import json
import logging
from typing import Dict, Any
from graph.state import AgentState
from graph.llm import get_llm
from prompts.goal_prompt import GOAL_SELECTOR_SYSTEM_PROMPT, GOAL_SELECTOR_USER_PROMPT

logger = logging.getLogger(__name__)

def clean_json_string(text: str) -> str:
    """JSON 문자열 주변의 마크다운 백틱 등을 제거합니다."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def goal_selector_node(state: AgentState) -> Dict[str, Any]:
    """
    Goal Selector Node.
    종합 컨텍스트 데이터를 LLM에 전달하여 오늘의 최우선 비즈니스 목표를 수립합니다.
    """
    u_id = state.get("u_id")
    target_date = state.get("target_date")
    context_data = state.get("context_data", {})

    logger.info(f"[GoalSelector] u_id: {u_id} 목표 선정 분석을 진행합니다.")

    # 프롬프트 인자 준비
    calendar = context_data.get("calendar", "일정 없음")
    kpi = context_data.get("kpi", "KPI 정보 없음")
    risks = context_data.get("risks", "위험 고객 정보 없음")
    events = context_data.get("events", "이벤트 없음")
    histories = context_data.get("histories", "최근 상담 없음")
    notifications = context_data.get("notifications", "알림 없음")
    ignored_history = context_data.get("ignored_history", "무시된 과거 추천 히스토리 없음")
    scheduled_customers = str(context_data.get("scheduled_customers", []))

    # 시스템 프롬프트 포맷
    sys_prompt = GOAL_SELECTOR_SYSTEM_PROMPT.replace("{target_date}", target_date)
    
    # 사용자 프롬프트 포맷
    user_prompt = GOAL_SELECTOR_USER_PROMPT.format(
        target_date=target_date,
        calendar=calendar,
        kpi=kpi,
        risks=risks,
        events=events,
        histories=histories,
        notifications=notifications,
        ignored_history=ignored_history,
        scheduled_customers=scheduled_customers
    )

    messages = [
        ("system", sys_prompt),
        ("user", user_prompt)
    ]

    llm = get_llm()
    response = llm.invoke(messages)
    
    # JSON 파싱 시도
    try:
        clean_res = clean_json_string(response.content)
        parsed_result = json.loads(clean_res)
        logger.info(f"[GoalSelector] 오늘의 목표 선정 완료: {parsed_result.get('goal')}")
    except Exception as e:
        logger.error(f"[GoalSelector] LLM 응답 JSON 파싱 실패, Fallback 적용: {e}")
        # 방어적 Fallback 목표 설정
        parsed_result = {
            "goal": "VIP 고객 만기 예정 상품 재가입 유도 및 포트폴리오 리밸런싱 상담",
            "reason": f"AUM 실적 관리가 필요하며, {target_date} 기준 만기 및 기념일 이벤트가 도래한 고객 대상의 선제적 연락을 위함입니다."
        }

    return {
        "current_goal": parsed_result
    }
