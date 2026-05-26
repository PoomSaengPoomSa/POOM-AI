import os
import sys
import logging
import time
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# DB 헬퍼 및 그래프 로더 추가
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

from tools.db_helper import get_db_session
from graph.graph_builder import build_todo_agent
from app.models.account import PbUser

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AI_ToDo_Scheduler")

def run_todo_agent_for_all_pbs():
    """
    모든 재직 중인 PB에 대해 AI To Do 추천 일정 생성 에이전트를 가동합니다.
    """
    logger.info("⏰ [Scheduler Trigger] 오전 06:00 AI To Do 생성 배치가 개시되었습니다.")
    
    agent = build_todo_agent()
    today_str = datetime.now().strftime("%Y-%m-%d")

    try:
        with get_db_session() as db:
            # 재직 중인 PB 리스트 조회
            pb_users = db.query(PbUser).filter(PbUser.status == "재직").all()
            logger.info(f"[Scheduler] 분석 대상 재직 PB 총 {len(pb_users)}명 검출")

            for pb in pb_users:
                logger.info(f"🚀 [PB 가동] u_id: {pb.u_id} ({pb.name} {pb.position}) 추천 일정 생성 에이전트 구동")
                
                # LangGraph 초기 상태 설정
                initial_state = {
                    "u_id": pb.u_id,
                    "target_date": today_str,
                    "retry_count": 0,
                    "reflection_guidance": None,
                    "evaluation": {"is_passed": False}
                }

                # 에이전트 실행
                try:
                    final_state = agent.invoke(initial_state)
                    eval_res = final_state.get("evaluation", {})
                    if eval_res.get("is_passed", False):
                        logger.info(f"✅ [성공] PB '{pb.name}'의 추천 일정이 최종 검증을 통과하여 DB에 저장되었습니다.")
                    else:
                        logger.warning(f"⚠️ [제한적 종료] PB '{pb.name}' 에이전트가 완벽히 검증되지 않고 종료되었습니다. (사유: {eval_res.get('feedback')})")
                except Exception as e:
                    logger.error(f"❌ [에러] PB '{pb.name}' 구동 중 치명적 오류 발생: {str(e)}", exc_info=True)
                    
        logger.info("⏰ [Scheduler 완료] 모든 PB 대상 AI To-Do 생성 배치가 성황리에 마무리되었습니다.")
    except Exception as e:
        logger.error(f"❌ [에러] 스케줄러 배치 구동 중 예외 발생: {str(e)}")

def start_scheduler():
    """
    APScheduler를 시작하여 매일 오전 6시에 run_todo_agent_for_all_pbs를 실행합니다.
    """
    scheduler = BlockingScheduler()
    # 매일 오전 06시 00분 Trigger 설정
    trigger = CronTrigger(hour=6, minute=0, second=0)
    
    scheduler.add_job(
        run_todo_agent_for_all_pbs,
        trigger=trigger,
        id="ai_todo_daily_batch",
        name="PB AI To-Do Daily Batch Job"
    )
    
    logger.info("🟢 [Scheduler 가동] AI To-Do 추천 생성 데몬 스케줄러가 대기 상태로 진입합니다. (매일 오전 06:00 실행)")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("🔴 [Scheduler 중지] AI To-Do 추천 데몬 스케줄러가 종료되었습니다.")

if __name__ == "__main__":
    # 로컬 수동 테스트로 직접 실행하려는 경우 처리
    if len(sys.argv) > 1 and sys.argv[1] == "--now":
        logger.info("⚡ [즉시 실행 모드] 스케줄러를 기다리지 않고 바로 배치를 가동합니다.")
        run_todo_agent_for_all_pbs()
    else:
        start_scheduler()
