import logging
from langgraph.graph import StateGraph, END
from graph.state import AgentState
from nodes import (
    state_analyzer_node,
    goal_selector_node,
    planner_node,
    executor_node,
    evaluator_node,
    reflection_node
)

logger = logging.getLogger(__name__)

def condition_check(state: AgentState) -> str:
    """
    Evaluator 실행 후 진행 방향을 지시하는 조건부 엣지(Conditional Edge) 함수입니다.
    - 검증 통과(is_passed=True) 혹은 최대 재시도(3회) 도달 시 ➔ END로 종료
    - 검증 실패(is_passed=False) 시 ➔ Reflection(반성) 노드로 라우팅
    """
    evaluation = state.get("evaluation", {})
    retry_count = state.get("retry_count", 0)

    if evaluation.get("is_passed", False):
        logger.info("[Workflow] 검증이 정상적으로 통과되었습니다. 그래프를 종료합니다.")
        return "end"
    
    if retry_count >= 3:
        logger.warning(f"[Workflow] 최대 재시도 횟수({retry_count}회)에 도달하여 강제 종료합니다. (일부 일정 누락 가능)")
        return "end"

    logger.warning(f"[Workflow] 추천 일정 충돌로 인해 반성(Reflection) 노드로 전이합니다. (현재 재시도 횟수: {retry_count})")
    return "reflection"

def build_todo_agent():
    """
    LangGraph 기반 AI To Do Agent 워크플로우를 생성하고 컴파일하여 반환합니다.
    """
    # 1. 상태(State) 초기화
    workflow = StateGraph(AgentState)

    # 2. 노드(Nodes) 등록
    workflow.add_node("state_analyzer", state_analyzer_node)
    workflow.add_node("goal_selector", goal_selector_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("reflection", reflection_node)

    # 3. 엣지(Edges) 연결
    workflow.set_entry_point("state_analyzer")
    
    workflow.add_edge("state_analyzer", "goal_selector")
    workflow.add_edge("goal_selector", "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "evaluator")
    
    # Evaluator 이후 조건부 라우팅 설정
    workflow.add_conditional_edges(
        "evaluator",
        condition_check,
        {
            "end": END,
            "reflection": "reflection"
        }
    )
    
    # Reflection(반성) 완료 후 다시 Planner로 전이하여 교정된 계획 수립
    workflow.add_edge("reflection", "planner")

    # 4. 그래프 컴파일
    logger.info("[Workflow] LangGraph AI To Do Agent 워크플로우 빌드 및 컴파일 완료")
    return workflow.compile()
