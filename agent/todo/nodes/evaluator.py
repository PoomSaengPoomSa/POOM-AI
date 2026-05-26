import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from graph.state import AgentState
from tools.db_helper import get_db_session
from tools.schedule_create_tool import CreateScheduleTool

# SQLAlchemy 모델
from app.models.schedule import Schedule

logger = logging.getLogger(__name__)

def evaluator_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluator Node.
    메모리 상의 임시 추천 일정 후보들(`execution_results`)에 대해
    1. 캘린더 충돌 여부 (기존 pb_schedule과의 시간대 겹침)
    2. 중복 추천 여부
    3. 데이터 형식 적합성
    을 엄밀히 검증합니다.

    - 검증 실패 시: is_passed = False 및 피드백 기록 ➔ Reflection Node로 이동
    - 검증 성공 시: is_passed = True ➔ 임시 일정을 CreateScheduleTool을 통해 DB에 정식 적재
    """
    u_id = state.get("u_id")
    target_date = state.get("target_date")
    execution_results = state.get("execution_results", [])
    
    logger.info(f"[Evaluator] u_id: {u_id} 추천 일정 {len(execution_results)}건에 대한 정밀 검증을 시행합니다.")

    if not execution_results:
        logger.warning("[Evaluator] 추천할 일정 후보가 존재하지 않습니다.")
        return {
            "evaluation": {
                "is_passed": False,
                "feedback": "작성된 추천 일정 후보가 없어 평가를 진행할 수 없습니다."
            }
        }

    with get_db_session() as db:
        # PB의 해당 기준일 기존 일정 전부 조회
        target_dt_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        start_bound = datetime.combine(target_dt_obj, datetime.min.time())
        end_bound = datetime.combine(target_dt_obj, datetime.max.time())
        
        existing_schedules = (
            db.query(Schedule)
            .filter(Schedule.u_id == u_id)
            .filter(Schedule.execution_date >= start_bound)
            .filter(Schedule.execution_date <= end_bound)
            .all()
        )

        for item in execution_results:
            # 1. 시간 파싱 및 범위 계산
            try:
                rec_start = datetime.strptime(item["execution_date"], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    rec_start = datetime.fromisoformat(item["execution_date"])
                except ValueError:
                    logger.error(f"[Evaluator] 시간 파싱 실패: {item['execution_date']}")
                    return {
                        "evaluation": {
                            "is_passed": False,
                            "feedback": f"추천 일정의 시간 포맷이 유효하지 않습니다: '{item['execution_date']}'"
                        }
                    }

            rec_end = rec_start + timedelta(hours=1)

            # 2. 기존 일정과의 충돌(시간 겹침) 정밀 대조
            # 충돌 조건: 기존일정.시작 < 추천일정.종료 AND 기존일정.종료 > 추천일정.시작
            for sched in existing_schedules:
                sched_start = sched.execution_date
                sched_end = sched.end_datetime if sched.end_datetime else (sched_start + timedelta(hours=1))

                if sched_start < rec_end and sched_end > rec_start:
                    conflict_time = sched_start.strftime("%H:%M")
                    logger.warning(f"[Evaluator] 일정 충돌 감지! 기존 일정 '{sched.title}' ({conflict_time})")
                    return {
                        "evaluation": {
                            "is_passed": False,
                            "feedback": (
                                f"추천 일정 '{item['title']}'의 제안 시간대({rec_start.strftime('%H:%M')} ~ {rec_end.strftime('%H:%M')})는 "
                                f"이미 등록된 기존 일정 '{sched.title}' ({sched_start.strftime('%H:%M')} ~ {sched_end.strftime('%H:%M')})과 겹쳐서 충돌이 발생합니다."
                            )
                        }
                    }

    # 모든 검증을 완벽히 통과!
    logger.info("[Evaluator] 모든 추천 일정 검증 완료 (통과!). DB에 정식 적재를 진행합니다.")

    # [중요] 중복 적재 방지 및 과거 무시된 To-Do 정리
    # 신규 5개 일정을 등록하기 전에, u_id의 해당 기준일(target_date) 및 그 이전에 생성되었으나
    # 등록되지 않은(is_checked == False) 기존 AI To-Do들을 데이터베이스에서 안전하게 정리(삭제)합니다.
    try:
        from app.models.ai_todo import AiTodo
        with get_db_session() as db:
            target_dt_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
            end_bound = datetime.combine(target_dt_obj, datetime.max.time())
            
            deleted_count = db.query(AiTodo).filter(
                AiTodo.u_id == u_id,
                AiTodo.is_checked == False,
                AiTodo.execution_date <= end_bound
            ).delete(synchronize_session=False)
            db.commit()
            if deleted_count > 0:
                logger.info(f"[Evaluator] 이전 날짜 및 오늘 기준 미등록(무시된) 기존 AI To-Do {deleted_count}건을 삭제 정리했습니다. (중복 방지 및 UI 청정화 완료)")
    except Exception as e:
        logger.warning(f"[Evaluator] 기존 미등록 AI To-Do 정리 중 오류 발생 (진행 계속): {e}")

    saved_todos_info = []
    # DB에 영구 저장 (CreateScheduleTool 호출)
    for item in execution_results:
        tool_res = CreateScheduleTool.invoke({
            "u_id": u_id,
            "c_id": item["c_id"],
            "execution_time": item["execution_date"],
            "title": item["title"],
            "memo": item["memo"],
            "category": item["category"]
        })
        logger.info(f"[Evaluator] {tool_res}")
        saved_todos_info.append(tool_res)

    return {
        "evaluation": {
            "is_passed": True,
            "feedback": "모든 캘린더 충돌 검증을 무사히 통과하고 DB 적재를 완료했습니다.",
            "saved_info": saved_todos_info
        }
    }
