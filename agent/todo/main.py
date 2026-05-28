import os
import sys
import argparse
import logging
from datetime import datetime

# sys.path 세팅 및 백엔드 연동 (윈도우 back 및 Docker poom 루트 동적 실존 검증 매핑 지원)
current_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.abspath(os.path.join(current_dir, "..", "..", "..")), # Docker poom 루트
    os.path.abspath(os.path.join(current_dir, "..", "..", "..", "POOM-BACK")), # Docker POOM-BACK
    os.path.abspath(os.path.join(current_dir, "..", "..", "..", "back")), # 윈도우 로컬
]
back_path = None
for p in possible_paths:
    # 해당 폴더 하위에 실제 데이터베이스 모듈이 실존하는지 검증
    if os.path.exists(os.path.join(p, "app", "database.py")):
        back_path = p
        break
if not back_path:
    back_path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "back")) # Fallback

if back_path not in sys.path:
    sys.path.insert(0, back_path)

from graph.graph_builder import build_todo_agent
from tools.db_helper import get_db_session
from app.models.account import PbUser

# 방문 예정 브리핑 및 알림 생성 파이프라인 연계
llm_brief_path = os.path.abspath(os.path.join(current_dir, "..", "..", "llm", "visit_brief"))
if llm_brief_path not in sys.path:
    sys.path.insert(0, llm_brief_path)
from visit_brief_generator import run_notification_generator


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AI_ToDo_Main")

def run_agent_for_pb(u_id: str, date_str: str):
    """
    특정 PB에 대해 AI To Do Agent를 가동하고 결과를 상세히 모니터링합니다.
    """
    logger.info(f"[START] [AI To Do Agent 구동] PB ID: '{u_id}', 기준일: '{date_str}'")

    # DB에 실제 등록된 PB인지 검증
    with get_db_session() as db:
        user = db.query(PbUser).filter(PbUser.u_id == u_id).first()
        if not user:
            logger.error(f"[ERROR] 데이터베이스에 존재하지 않는 PB ID입니다: '{u_id}'")
            # 등록된 PB 리스트 출력 지원
            active_pbs = db.query(PbUser).filter(PbUser.status == "재직").all()
            logger.info("현재 재직 중인 유효 PB ID 목록:")
            for p in active_pbs:
                logger.info(f" - {p.u_id} ({p.name} {p.position})")
            return

        logger.info(f"[MATCH] [PB 매칭 완료] {user.name} {user.position} ({user.branch_rel.name if user.branch_rel else '지점 정보 없음'})")

    # 그래프 빌드 및 실행
    agent = build_todo_agent()
    
    initial_state = {
        "u_id": u_id,
        "target_date": date_str,
        "retry_count": 0,
        "reflection_guidance": None,
        "evaluation": {"is_passed": False}
    }

    try:
        logger.info("[RUN] LangGraph State Machine 시작...")
        final_state = agent.invoke(initial_state)
        
        # 결과 리포트 출력
        logger.info("==================================================================")
        logger.info("[REPORT] AI To Do 에이전트 가동 완료 리포트")
        logger.info("==================================================================")
        logger.info(f"- 분석 기준일: {final_state.get('target_date')}")
        logger.info(f"- 총 재시도 횟수(Retry Count): {final_state.get('retry_count')}회")
        
        goal_info = final_state.get("current_goal", {})
        logger.info(f"- 선정된 최우선 목표 (Goal): {goal_info.get('goal')}")
        logger.info(f"  * 추천 근거: {goal_info.get('reason')}")
        
        plan_tools = final_state.get("plan_tools", [])
        logger.info(f"- 도출된 Tool 사용 흐름: {' -> '.join(plan_tools)}")

        eval_info = final_state.get("evaluation", {})
        logger.info(f"- 최종 검증 결과 (Evaluation): [{'PASS' if eval_info.get('is_passed') else 'FAIL'}]")
        logger.info(f"  * 검증 피드백: {eval_info.get('feedback')}")

        if final_state.get("reflection_guidance"):
            logger.info(f"- 보완된 반성 지침 (Reflection Guidance): {final_state.get('reflection_guidance')}")

        execution_results = final_state.get("execution_results", [])
        logger.info(f"- 생성되어 임시 조율된 일정 목록 ({len(execution_results)}건):")
        for idx, item in enumerate(execution_results, 1):
            logger.info(f"  {idx}) [{item.get('category')}] {item.get('title')}")
            logger.info(f"     * 시간: {item.get('execution_date')} | 대상고객 c_id: {item.get('c_id')}")
            logger.info(f"     * 메모: {item.get('memo')}")

        saved_info = eval_info.get("saved_info", [])
        if saved_info:
            logger.info("==================================================================")
            logger.info("[SAVE] 데이터베이스(ai_todo) 정식 적재 결과:")
            for s in saved_info:
                logger.info(f"  {s}")
            logger.info("==================================================================")

        # [AI 알림 및 방문 브리핑 실시간 적재 연계]
        try:
            logger.info("[PIPELINE] AI 알림 및 방문 예정 브리핑 생성 파이프라인 가동...")
            run_notification_generator(u_id, date_str)
            logger.info("[PIPELINE] AI 알림 및 방문 예정 브리핑 생성 파이프라인 성공 완료!")
        except Exception as ne:
            logger.error(f"[PIPELINE_ERROR] 알림 파이프라인 생성 실패: {ne}", exc_info=True)
            
    except Exception as e:
        logger.error(f"[ERROR] 에이전트 구동 실패: {str(e)}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI To Do Goal-driven Agent Runner")
    parser.add_argument("--u_id", type=str, help="분석 대상 PB ID (예: fisaai6 등)")
    parser.add_argument("--date", type=str, help="분석 기준일 YYYY-MM-DD (기본값: 오늘 날짜)")
    
    args = parser.parse_args()
    
    # 기본값 설정
    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    
    if not args.u_id:
        logger.info("PB ID가 지정되지 않았습니다. 기본 테스트 PB 'admin' 또는 첫 번째 재직 PB로 임시 조회를 시도합니다.")
        with get_db_session() as db:
            active_pb = db.query(PbUser).filter(PbUser.status == "재직").first()
            if active_pb:
                pb_id = active_pb.u_id
                logger.info(f"임시 조회 매칭 PB ID: '{pb_id}' ({active_pb.name})")
            else:
                logger.error("데이터베이스에 가용 재직 PB가 존재하지 않습니다. DB 연결을 다시 확인하세요.")
                sys.exit(1)
    else:
        pb_id = args.u_id

    run_agent_for_pb(pb_id, target_date)
