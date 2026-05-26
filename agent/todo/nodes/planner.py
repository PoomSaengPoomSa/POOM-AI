import json
import logging
from typing import Dict, Any
from graph.state import AgentState
from graph.llm import get_llm
from prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT

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

def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Planner Node.
    목표 및 반성 지침을 분석하여 실행할 Tool 실행 계획을 동적으로 작성합니다.
    """
    current_goal = state.get("current_goal", {})
    goal_title = current_goal.get("goal", "VIP 고객 마케팅 상담 유치")
    goal_reason = current_goal.get("reason", "비즈니스 기회 극대화")
    
    reflection_guidance = state.get("reflection_guidance") or "첫 계획 시도이므로 충돌 예방에 초점을 맞추십시오."

    logger.info(f"[Planner] 목표 '{goal_title}' 달성을 위한 동적 계획 수립 시작 (Retry 횟수: {state.get('retry_count', 0)})")

    sys_prompt = PLANNER_SYSTEM_PROMPT
    user_prompt = PLANNER_USER_PROMPT.format(
        goal=goal_title,
        reason=goal_reason,
        reflection_guidance=reflection_guidance
    )

    messages = [
        ("system", sys_prompt),
        ("user", user_prompt)
    ]

    llm = get_llm()
    response = llm.invoke(messages)

    try:
        clean_res = clean_json_string(response.content)
        plan_tools = json.loads(clean_res)
        if not isinstance(plan_tools, list):
            raise ValueError("Planner 응답이 JSON 리스트 형태가 아닙니다.")
        
        # 마지막은 항상 CreateScheduleTool이 들어오도록 방어
        if "CreateScheduleTool" not in plan_tools:
            plan_tools.append("CreateScheduleTool")
            
        logger.info(f"[Planner] 동적 계획 수립 완료: {plan_tools}")
    except Exception as e:
        logger.error(f"[Planner] LLM 계획 수립 파싱 실패, 기본 Heuristic 계획 적용: {e}")
        # 기본 Heuristic 계획 제공
        if "충돌" in reflection_guidance or "14:00" in reflection_guidance:
            plan_tools = ["GetCalendarScheduleTool", "CreateScheduleTool"]
        else:
            plan_tools = ["GetCustomerEventTool", "GetCalendarScheduleTool", "CreateScheduleTool"]

    return {
        "plan_tools": plan_tools
    }
