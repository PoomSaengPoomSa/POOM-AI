import json
import logging
from typing import Dict, Any
from graph.state import AgentState
from graph.llm import get_llm
from prompts.reflection_prompt import REFLECTION_SYSTEM_PROMPT, REFLECTION_USER_PROMPT

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

def reflection_node(state: AgentState) -> Dict[str, Any]:
    """
    Reflection Node.
    캘린더 충돌이나 기타 반려 사유에 대해 원인을 정밀 분석하여
    Planner가 차후 루프에서 영리하게 충돌을 피해 계획을 보완할 수 있도록
    '반성 지침(reflection_guidance)'을 작성하고 재시도 횟수를 가산합니다.
    """
    goal = state.get("current_goal", {}).get("goal", "VIP 고객 마케팅 상담 유치")
    execution_results = state.get("execution_results", [])
    evaluation = state.get("evaluation", {})
    feedback = evaluation.get("feedback", "알 수 없는 일정 충돌 발생")
    context_data = state.get("context_data", {})
    calendar = context_data.get("calendar", "일정 없음")
    
    retry_count = state.get("retry_count", 0) + 1

    logger.warning(f"[Reflection] 일정 생성 반려 감지! 원인을 규명하고 Planner 반성 지침을 수립합니다. (Retry 횟수: {retry_count})")

    sys_prompt = REFLECTION_SYSTEM_PROMPT
    user_prompt = REFLECTION_USER_PROMPT.format(
        goal=goal,
        execution_results=json.dumps(execution_results, ensure_ascii=False),
        feedback=feedback,
        calendar=calendar
    )

    messages = [
        ("system", sys_prompt),
        ("user", user_prompt)
    ]

    llm = get_llm()
    response = llm.invoke(messages)

    try:
        clean_res = clean_json_string(response.content)
        parsed_result = json.loads(clean_res)
        guidance = parsed_result.get("reflection_guidance", "캘린더 일정이 비어 있는 다른 시간대(예: 16:00 또는 10:00)로 일정을 생성하십시오.")
        logger.info(f"[Reflection] 자가 반성 분석 완료: {parsed_result.get('feedback_analysis')}")
        logger.info(f"[Reflection] 수립된 재계획 지침: {guidance}")
    except Exception as e:
        logger.error(f"[Reflection] LLM 반성 프롬프트 파싱 실패, Fallback 반성 지침 적용: {e}")
        # 기본적 Heuristic 반성 가이드라인 구성
        if "14:00" in feedback:
            guidance = "직전 계획에서 14:00 시간대 기존 일정과 충돌이 발생했습니다. 14:00를 절대로 피하고, 오후의 또 다른 빈 시간대인 '16:00'에 일정을 수립해 주십시오."
        else:
            guidance = "기존 등록 캘린더 일정과의 충돌을 피하기 위해, 오전 10:00 또는 오후 16:00 시간대의 빈 슬롯에 추천 일정을 신규 수립해 주십시오."

    return {
        "reflection_guidance": guidance,
        "retry_count": retry_count
    }
