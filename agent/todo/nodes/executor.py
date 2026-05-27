import json
import logging
from typing import Dict, Any, List
from graph.state import AgentState
from graph.llm import get_llm

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

def executor_node(state: AgentState) -> Dict[str, Any]:
    """
    Executor Node.
    Planner가 계획한 Tool 목록(`plan_tools`)을 실행한 것으로 시뮬레이션하여
    상황 데이터를 결합해 최적의 추천 일정을 메모리 상(execution_results)에 수립합니다.
    (실제 DB 적재는 Evaluator가 최종 통과시킨 후 안전하게 수행됩니다.)
    """
    u_id = state.get("u_id")
    target_date = state.get("target_date")
    current_goal = state.get("current_goal", {})
    context_data = state.get("context_data", {})
    reflection_guidance = state.get("reflection_guidance") or ""

    logger.info(f"[Executor] u_id: {u_id} 계획된 도구를 바탕으로 임시 추천 일정 조율을 시작합니다.")

    # 0. PB의 실제 담당 고객 ID들을 최대 5명 조회하여 가이드 및 Fallback용으로 확보합니다.
    from tools.db_helper import get_db_session
    from app.models.in_charge import InCharge
    
    valid_c_ids = []
    try:
        with get_db_session() as db:
            charges = db.query(InCharge).filter(InCharge.u_id == u_id).limit(5).all()
            valid_c_ids = [c.c_id for c in charges]
    except Exception as e:
        logger.warning(f"[Executor] 담당 고객 조회 실패: {e}")

    # 가이드용 primary ID 확보
    primary_c_id = valid_c_ids[0] if valid_c_ids else None
    secondary_c_id = valid_c_ids[1] if len(valid_c_ids) > 1 else primary_c_id

    # 1. 상황 데이터 분석 및 고객 선정용 LLM 프롬프트
    prompt = f"""당신은 PB AI To-Do 에이전트의 **실행 엔진(Executor)**입니다.
업무 목표(Goal)와 상황 데이터를 바탕으로, PB가 오늘 하루 동안 처리하고 선택할 수 있는 **총 5가지의 다양하고 구체적인 추천 일정(AI To-Do) 후보**를 매칭하여 제안해 주십시오.

### [목표 및 상황 데이터]
- 업무 목표: {current_goal.get('goal')} (추천 사유: {current_goal.get('reason')})
- 분석 기준일: {target_date}

### [수집된 컨텍스트 상황 데이터]
- 캘린더 현황: {context_data.get('calendar')}
- 이탈 위험 정보: {context_data.get('risks')}
- 만기 및 생일 이벤트: {context_data.get('events')} (생일, 상품 만기, 결혼기념일 등 관계 형성 기념일을 적극 포착하십시오.)
- 최근 상담 이력: {context_data.get('histories')}
- 이미 발송된 알림: {context_data.get('notifications')}
- 과거 무시(미등록)했던 AI To-Do 히스토리: {context_data.get('ignored_history')}
- 이미 캘린더에 일정이 확보된 고객 ID 목록: {context_data.get('scheduled_customers')}
- 추천 가능한 유효 담당 고객 ID 리스트: {valid_c_ids} (일정 생성 시 이 리스트 내의 c_id를 최우선 매칭하십시오.)

### [재계획 및 반성 지침 (Reflection Guidance)]
- 지침: {reflection_guidance} (반성 지침에서 제안된 시간 회피 지시가 있다면 철저히 준수하십시오.)

[생성 및 시간 조율 규칙]
2. 제안하는 5개 일정은 **서로 시간이 겹치지 않도록 조율**하고, **업무의 성격에 따라 오전/오후 시간대(Time Slot)에 정교하게 분배**해 주십시오:
   - **오전 시간대 (오전 10:00 ~ 12:00)**: **`안부 연락 제안` (생일 축하 안부, 결혼기념일 축하 연락 등)**을 전면 배치하십시오. (기념일 축하 연락은 하루의 시작 시점인 오전에 전달되어야 고객에게 정성스럽고 세련되게 느껴지기 때문입니다.)
   - **오후 시간대 (오후 14:00 ~ 17:00)**: **`상담 일정 제안`, `신규 상품 분석`, `KPI 기반` (정기예금 만기 재가입 상담, 포트폴리오 리밸런싱 대면 미팅, 은퇴 노후 설계 상담 등)**을 집중 배치하십시오. (심도 깊은 상품 매칭 및 자산 상담은 PB가 오전 동안 철저하게 사전 분석과 준비를 마치고, 고객도 비교적 시간적 유동성이 확보되는 오후 시간대에 진행하는 것이 정석이기 때문입니다.)
   - 각 추천 일정에는 반드시 위 원칙에 맞춰 `10:00:00`, `11:00:00`, `14:00:00`, `15:00:00`, `16:00:00` 등 1시간 단위의 고유 시간대를 부여하되, PB의 기존 캘린더 일정과 겹치지 않는 빈 슬롯을 우선적으로 활용하십시오.
3. **기념일(생일, 결혼기념일) 및 상품 만기** 등의 일정을 골고루 배정하십시오:
   - 생일, 결혼기념일 등 관계 관리가 필요한 일정은 카테고리를 `'안부 연락 제안'`으로 설정하고 친근한 축하 연락 일정을 수립하십시오.
   - 상품 만기가 예정된 고객은 `'상담 일정 제안'`으로 배정하십시오.
   - AUM 관련 목표 등은 `'KPI 기반'`으로 배정하십시오.
4. 카테고리는 반드시 `'KPI 기반'`, `'상담 일정 제안'`, `'안부 연락 제안'`, `'신규 상품 분석'` 중 하나여야 합니다. (check constraint 제약조건 철저 준수)
5. 각 일정의 제목은 50자 이내, 메모는 80자 이내로 정밀히 작성하십시오.
6. **[중요] 이미 일정이 수립된 고객 리스트({context_data.get('scheduled_customers')})에 포함된 고객은 추가로 일정을 잡지 않도록 이번 추천에서 배제하십시오.**
7. **[중요 - 스마트 중요도 감쇠 (Smart Decay)]**: 
   - 과거 추천 중 날짜가 이미 기준일 이전으로 완전히 지나갔음에도 PB가 일정 등록을 하지 않은 고객과 제안({context_data.get('ignored_history')})은 **진짜 원하지 않는 것(거절)**으로 판단되므로 강력히 배제 및 후순위 감쇠 처리하십시오.
   - 반면, 추천된 지 얼마 되지 않았거나 아직 실행 예정 기한이 많이 남아있어 단순 보류 상태인 미래의 추천들은 **시기상조 보류(Deferred)**된 것으로 아직 매우 신선한 추천이므로 정상적으로 재추천 대상으로 분류하십시오.
8. **[중요 - 추천 다양성 및 카테고리 분배 규칙]**: 생성하는 5개 추천 일정에는 반드시 4가지 카테고리('KPI 기반', '상담 일정 제안', '안부 연락 제안', '신규 상품 분석')가 최소 1회 이상 골고루 포함되어 균형 있는 일정이 되도록 보장하십시오. 특정 카테고리가 5개 전체를 지배하는 쏠림 현상을 방지하십시오.
9. **[중요 - 중요도 순서 배치 규칙]**: 생성하는 5개의 일정은 **비즈니스적 중요도와 시급성이 높은 순서(1위부터 5위까지)**대로 정렬하여 JSON 배열의 앞쪽(Index 0)부터 차례대로 배치해 주십시오. (예: 가장 중요한 자산 이탈 위험 VVIP 고객 대면 상담 등이 배열의 1~2번에 위치해야 하며, 덜 긴급한 축하 안부 등은 뒤쪽에 배치됩니다.)

[출력 형식]
반드시 다음 **JSON 배열 형식**으로만 출력해 주세요.

```json
[
  {{
    "title": "[카테고리별 머리글] 추천 일정 제목 (예: [만기] 김OO 고객 예적금 재가입 상담)",
    "memo": "구체적인 제안 메모 (예: 정기예금 만기 도래 10일 전. 포트폴리오 리밸런싱 제안 추천)",
    "category": "상담 일정 제안",
    "execution_date": "YYYY-MM-DD HH:MM:SS 형식 (날짜는 반드시 {target_date}로 지정)",
    "c_id": {primary_c_id or 'null'}
  }},
  ... (총 5개 생성) ...
]
```
"""
    
    llm = get_llm()
    messages = [
        ("system", "당신은 영리하게 상황 데이터를 조율하고 여러 개의 일정을 매칭하는 AI 실행가입니다."),
        ("user", prompt)
    ]
    response = llm.invoke(messages)

    try:
        clean_res = clean_json_string(response.content)
        execution_results = json.loads(clean_res)
        if not isinstance(execution_results, list):
            raise ValueError("Executor 응답이 JSON 배열 형태가 아닙니다.")
        
        # 날짜 포맷 방어 처리
        for item in execution_results:
            if target_date not in item["execution_date"]:
                # 날짜 강제 보정
                time_part = item["execution_date"].split(" ")[-1]
                if ":" not in time_part:
                    time_part = "14:00:00"
                item["execution_date"] = f"{target_date} {time_part}"
                
        logger.info(f"[Executor] 임시 추천 일정 후보 작성 완료 (총 {len(execution_results)}건): {execution_results}")
    except Exception as e:
        logger.error(f"[Executor] LLM 추천 일정 수립 파싱 실패, 5개 Fallback 적용: {e}")
        
        # 반성 지침에 따른 시간 조정 지원
        t1, t2, t3, t4, t5 = "10:00:00", "11:00:00", "14:00:00", "15:00:00", "16:00:00"
        if "14:00" in reflection_guidance:
            t3 = "17:00:00"
        if "10:00" in reflection_guidance:
            t1 = "09:00:00"

        # Heuristic Fallback 데이터 작성 (기념일은 오전, 상담/KPI는 오후 배치 및 중요도 정렬 준수)
        execution_results = [
            {
                "title": "[생일] 생일 맞이 축하 감사 연락",
                "memo": "오늘 생일 도래. 모바일 커피 쿠폰 발송 및 감사 안부 연락.",
                "category": "안부 연락 제안",
                "execution_date": f"{target_date} {t1}",  # 오전 10:00 또는 09:00
                "c_id": primary_c_id
            },
            {
                "title": "[기념일] 결혼기념일 기념 축하 안부 연락",
                "memo": "결혼기념일 3일 전. VIP 밀착 관리용 축하 기프티콘 및 전화 안부.",
                "category": "안부 연락 제안",
                "execution_date": f"{target_date} {t2}",  # 오전 11:00
                "c_id": secondary_c_id
            },
            {
                "title": "[만기] 우량 고객 정기예금 재가입 상담 제안",
                "memo": "예금 만기 15일 전. AUM 사수를 위한 고액 예금 재유치 상담 추천.",
                "category": "상담 일정 제안",
                "execution_date": f"{target_date} {t3}",  # 오후 14:00 또는 17:00
                "c_id": primary_c_id
            },
            {
                "title": "[포폴] 글로벌 채권 포트폴리오 리밸런싱 제안",
                "memo": "고객 투자 성향 맞춤형 글로벌 하이일드 채권 상품 설명안 전달.",
                "category": "신규 상품 분석",
                "execution_date": f"{target_date} {t4}",  # 오후 15:00
                "c_id": primary_c_id
            },
            {
                "title": "[KPI] 퇴직연금 IRP 신규 유치 상담 유치",
                "memo": "소득 세액공제 한도 극대화를 위한 IRP 계좌 개설 안내 마케팅.",
                "category": "KPI 기반",
                "execution_date": f"{target_date} {t5}",  # 오후 16:00
                "c_id": secondary_c_id
            }
        ]

    # 2. 추천 일정 제목에 실제 고객 이름 정밀 결합 (PB 시인성 극대화 및 이전 DB 적재 데이터와 일관성 유지)
    try:
        from app.models.customer import Customer
        with get_db_session() as db:
            for item in execution_results:
                c_id = item.get("c_id")
                if c_id:
                    cust = db.query(Customer).filter(Customer.c_id == c_id).first()
                    if cust:
                        name = cust.name
                        title = item.get("title", "")
                        
                        # 김OO 고객, 우량 고객 등을 실제 데이터베이스의 실명으로 조율 및 교체
                        # 기존 [카테고리] 대괄호 등 제거하고 `{고객명} 고객({c_id}) {Title}` 형태로 전면 통일
                        import re
                        cleaned_title = re.sub(r"\[.*?\]", "", title).strip() # 대괄호 머리글 제거 (예: [만기])
                        cleaned_title = re.sub(r"[\wO]+ 고객", "", cleaned_title).strip() # 기존 임시 고객명 제거
                        cleaned_title = re.sub(r"\(\d+\)", "", cleaned_title).strip() # 기존 괄호 ID 제거
                        cleaned_title = cleaned_title.strip()
                        
                        title = f"{name} 고객({c_id}) {cleaned_title}"
                        item["title"] = title[:50]  # VARCHAR(50) 크래시 방지 방어
    except Exception as e:
        logger.warning(f"[Executor] 추천명 고객 실명 치환 중 오류 발생: {e}")

    return {
        "execution_results": execution_results
    }
