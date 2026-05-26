from typing import List, Dict, Any
from langchain_core.tools import tool
from .db_helper import get_db_session

# SQLAlchemy 모델 임포트
from app.models.notification import Notification

@tool
def GetNotificationTool(u_id: str) -> str:
    """
    Get Notification Tool.
    PB(u_id)에게 이미 생성/발송된 기존 알림(알림함 메시지) 내역을 최근 순으로 최대 10건 조회하여 중복 추천이나 알림 발송을 예방합니다.
    """
    with get_db_session() as db:
        notifications = (
            db.query(Notification)
            .filter(Notification.u_id == u_id)
            .order_by(Notification.created_time.desc())
            .limit(10)
            .all()
        )

        if not notifications:
            return "PB님에게 생성된 기존 알림 내역이 없습니다."

        output = ["### [최근 생성 알림함 내역 (중복 발송 방지용)]"]
        for idx, n in enumerate(notifications, 1):
            category_str = f"[{n.category}]" if n.category else "[일반]"
            created_str = n.created_time.strftime("%Y-%m-%d %H:%M") if n.created_time else "시간 정보 없음"
            state_str = "확인완료" if n.state_us == "checked" else "미확인"
            
            output.append(
                f"{idx}. {category_str} **{n.title}** ({created_str}) - 상태: {state_str}\n"
                f"   * 내용: {n.content}"
            )
        return "\n".join(output)
