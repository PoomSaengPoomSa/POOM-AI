from datetime import datetime, time
from typing import List, Dict, Any
from langchain_core.tools import tool
from .db_helper import get_db_session

# sys.path에 back 경로가 보장되어 있으므로 import 가능
from app.models.schedule import Schedule
from app.models.customer import Customer

def get_calendar_schedules(u_id: str, date_str: str) -> List[Dict[str, Any]]:
    """
    특정 PB(u_id)의 지정된 날짜(date_str, YYYY-MM-DD)의 기존 캘린더 일정을 조회합니다.
    """
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        target_date = datetime.now().date()

    start_dt = datetime.combine(target_date, time.min)
    end_dt = datetime.combine(target_date, time.max)

    with get_db_session() as db:
        schedules = (
            db.query(Schedule)
            .outerjoin(Customer, Schedule.c_id == Customer.c_id)
            .filter(Schedule.u_id == u_id)
            .filter(Schedule.execution_date >= start_dt)
            .filter(Schedule.execution_date <= end_dt)
            .order_by(Schedule.execution_date.asc())
            .all()
        )

        results = []
        for s in schedules:
            results.append({
                "s_id": s.s_id,
                "title": s.title,
                "memo": s.memo,
                "category": s.category,
                "execution_date": s.execution_date.strftime("%Y-%m-%d %H:%M:%S"),
                "end_datetime": s.end_datetime.strftime("%Y-%m-%d %H:%M:%S") if s.end_datetime else None,
                "customer_name": s.customer.name if s.customer else None,
                "c_id": s.c_id
            })
        return results

@tool
def GetCalendarScheduleTool(u_id: str, date_str: str) -> str:
    """
    Get Calendar Schedule Tool.
    특정 PB(u_id)의 지정 날짜(date_str: YYYY-MM-DD 형식) 캘린더 기존 일정을 조회하여 겹치는 일정이 없는지 여유 일정을 확인합니다.
    """
    schedules = get_calendar_schedules(u_id, date_str)
    if not schedules:
        return f"[{date_str}] 해당 날짜에 등록된 PB 일정이 없습니다. 모든 시간대(오전 9시 ~ 오후 6시)가 비어 있어 일정을 자유롭게 생성할 수 있습니다."
    
    output = [f"[{date_str}] 기존 일정 리스트:"]
    for s in schedules:
        start_time = s["execution_date"].split(" ")[1][:5]
        end_time = s["end_datetime"].split(" ")[1][:5] if s["end_datetime"] else "종료시간 없음"
        cust_info = f" (고객: {s['customer_name']})" if s['customer_name'] else ""
        output.append(f"- [{start_time} ~ {end_time}] [{s['category']}] {s['title']} {cust_info} - 메모: {s['memo']}")
    return "\n".join(output)
