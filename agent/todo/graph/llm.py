import os
import sys
import json
import logging
from typing import Any
from datetime import datetime
from dotenv import load_dotenv

# 백엔드 패키지 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
back_path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", "back"))
load_dotenv(os.path.join(back_path, ".env"))

logger = logging.getLogger(__name__)

class HeuristicFallbackLLM:
    """
    OpenAI API Key가 없거나 호출 실패 시 실행 흐름을 지탱하는
    정밀한 Heuristic/Rule-based Fallback LLM 엔진입니다.
    """
    def __init__(self):
        logger.warning("[LLM] OpenAI API Key가 존재하지 않거나 연동 오류가 있어 Heuristic Fallback 모드를 가동합니다.")

    def invoke(self, messages) -> Any:
        # Prompt 내용을 분석하여 맞춤형 Heuristic 결과 도출
        system_content = ""
        user_content = ""

        # 메시지 리스트에서 내용 병합
        for msg in messages:
            if hasattr(msg, "content"):
                text = msg.content
            elif isinstance(msg, tuple) and len(msg) == 2:
                text = msg[1]
            else:
                text = str(msg)
            
            if "최우수 PB" in text or "전략가" in text:
                system_content = text
            else:
                user_content += text

        # 1. Reflection Prompt 감지 시 피드백 생성
        if "반성" in system_content or "Reflection" in system_content or "반려" in user_content:
            logger.info("[LLM Fallback] Reflection 지침 생성 요청 감지")
            
            # 반려 원인 파악
            feedback_reason = "14:00 충돌"
            if "14:00" in user_content:
                feedback_reason = "14:00 시간대 캘린더 기존 일정과 충돌 발생"
            elif "중복" in user_content:
                feedback_reason = "이미 등록된 추천 일정과의 중복 감지"

            guidance = (
                f"14:00 시간대의 캘린더 충돌이 감지되었습니다. 캘린더 상의 빈 시간대인 '16:00' 또는 '10:00'로 "
                f"시간을 조정하여 CreateScheduleTool을 호출하도록 계획을 변경하십시오."
            )
            
            result = {
                "feedback_analysis": feedback_reason,
                "reflection_guidance": guidance
            }
            return FallbackMessage(json.dumps(result, ensure_ascii=False))

        # 1.5. Executor Prompt 감지 시 5개 추천 일정 JSON 반환
        if "실행 엔진" in system_content or "Executor" in system_content or "실행 엔진" in user_content:
            logger.info("[LLM Fallback] Executor 추천 일정 수립 요청 감지 (5개 생성)")
            
            # 사용자 프롬프트에서 유효한 c_ids 리스트 파싱 시도
            import re
            c_ids = [1001]
            match = re.search(r"리스트:\s*\[(.*?)\]", user_content)
            if match:
                try:
                    c_ids = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
                except Exception:
                    pass
            if not c_ids:
                c_ids = [1001]
            
            p_id = c_ids[0]
            s_id = c_ids[1] if len(c_ids) > 1 else p_id
            
            # target_date 파싱 시도
            target_date = datetime.now().strftime("%Y-%m-%d")
            date_match = re.search(r"기준일:\s*([\d-]+)", user_content)
            if date_match:
                target_date = date_match.group(1)

            t1, t2, t3, t4, t5 = "10:00:00", "11:00:00", "14:00:00", "15:00:00", "16:00:00"
            if "14:00" in user_content:
                t3 = "17:00:00"

            fallback_results = [
                {
                    "title": "[만기] 우량 고객 정기예금 재가입 상담 제안",
                    "memo": "예금 만기 15일 전. AUM 사수를 위한 고액 예금 재유치 상담 추천.",
                    "category": "상담 일정 제안",
                    "execution_date": f"{target_date} {t1}",
                    "c_id": p_id
                },
                {
                    "title": "[생일] 생일 맞이 축하 감사 연락",
                    "memo": "오늘 생일 도래. 모바일 커피 쿠폰 발송 및 감사 안부 연락.",
                    "category": "안부 연락 제안",
                    "execution_date": f"{target_date} {t2}",
                    "c_id": p_id
                },
                {
                    "title": "[기념일] 결혼기념일 기념 축하 안부 연락",
                    "memo": "결혼기념일 3일 전. VIP 밀착 관리용 축하 기프티콘 및 전화 안부.",
                    "category": "안부 연락 제안",
                    "execution_date": f"{target_date} {t3}",
                    "c_id": s_id
                },
                {
                    "title": "[포폴] 글로벌 채권 포트폴리오 리밸런싱 제안",
                    "memo": "고객 투자 성향 맞춤형 글로벌 하이일드 채권 상품 설명안 전달.",
                    "category": "신규 상품 분석",
                    "execution_date": f"{target_date} {t4}",
                    "c_id": p_id
                },
                {
                    "title": "[KPI] IRP 신규 고객 유치 상담 제안",
                    "memo": "연말 정산 세액공제 극대화를 위한 IRP 신규 개설 마케팅 상담.",
                    "category": "KPI 기반",
                    "execution_date": f"{target_date} {t5}",
                    "c_id": s_id
                }
            ]
            return FallbackMessage(json.dumps(fallback_results, ensure_ascii=False))

        # 2. Planner Prompt 감지 시 Tool 실행 계획 수립
        if "Planner" in system_content or "태스크 플래너" in system_content:
            logger.info("[LLM Fallback] Planner 실행 계획 생성 요청 감지")
            
            # 반성 가이드라인이 있는 경우, 빈 시간대 우선 탐색을 위해 캘린더 조회 등을 포함
            if "guidance" in user_content or "16:00" in user_content or "Reflective" in user_content:
                plan = [
                    "GetCalendarScheduleTool",
                    "CreateScheduleTool"
                ]
            else:
                plan = [
                    "GetCustomerEventTool",
                    "GetCalendarScheduleTool",
                    "CreateScheduleTool"
                ]
            return FallbackMessage(json.dumps(plan, ensure_ascii=False))

        # 3. Goal Selector Prompt 감지 시 오늘의 업무 목표 생성
        logger.info("[LLM Fallback] Goal Selector 최우선 목표 선정 요청 감지")
        
        # 위험도 높은 고객 검출
        goal = "VIP 고객 만기 예정 상품 재가입 유도 및 포트폴리오 리밸런싱 상담"
        reason = "AUM 달성률이 저조한 상황에서 30일 이내 고액의 예금 만기가 예정된 우량 고객의 이탈 위험을 선제 방어하고, 14:00~16:00 사이의 캘린더 유동성이 확보되어 있기 때문입니다."

        if "위험" in user_content or "주의" in user_content:
            goal = "이탈 우려 VIP 고객 선제 대면 상담 및 위험 방어 일정 수립"
            reason = "담당 고객 중 이탈 위험 등급이 '위험'인 고자산 VIP 고객이 식별되었으며, PB 캘린더 내 빈 시간대에 상담을 유치하여 이탈 우려를 신속히 해소하고 신뢰를 구축하기 위함입니다."

        result = {
            "goal": goal,
            "reason": reason
        }
        return FallbackMessage(json.dumps(result, ensure_ascii=False))

class FallbackMessage:
    def __init__(self, content):
        self.content = content

def get_llm():
    """
    환경 변수의 API 키 유무에 맞춰 실제 ChatOpenAI 모델 또는 Fallback 모델을 반환합니다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and not api_key.startswith("your-") and len(api_key) > 10:
        try:
            from langchain_openai import ChatOpenAI
            logger.info("[LLM] 실제 OpenAI ChatOpenAI(gpt-4o) 모델을 가동합니다.")
            return ChatOpenAI(
                model="gpt-4o",
                temperature=0.2,
                api_key=api_key
            )
        except Exception as e:
            logger.error(f"[LLM] ChatOpenAI 초기화 중 오류 발생, Fallback 가동: {e}")
            return HeuristicFallbackLLM()
    else:
        return HeuristicFallbackLLM()
