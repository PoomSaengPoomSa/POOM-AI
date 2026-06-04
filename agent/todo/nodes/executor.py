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
업무 목표(Goal)와 상황 데이터를 바탕으로, PB가 오늘 하루 동안 처리하고 선택할 수 있는 **최소 2가지 ~ 최대 5가지의 다양하고 구체적인 추천 일정(AI To-Do) 후보**를 매칭하여 제안해 주십시오.

### [목표 및 상황 데이터]
- 업무 목표: {current_goal.get('goal')} (추천 사유: {current_goal.get('reason')})
- 분석 기준일: {target_date}

### [수집된 컨텍스트 상황 데이터]
- 캘린더 현황: {context_data.get('calendar')}
- 이탈 위험 정보: {context_data.get('risks')}
- 만기 이벤트: {context_data.get('events')} (주로 30일 이내 도래하는 상품 만기 예정 정보를 바탕으로 분석 및 상담 일정을 구상하십시오.)
- 최근 상담 이력: {context_data.get('histories')}
- 60일 이상 미상담 고객 정보: {context_data.get('unconsulted_customers')} (이 정보를 적극 활용하여, 오래 접촉하지 않은 VIP 고객을 위한 사전 분석/준비 일정을 구성하십시오.)
- 이미 발송된 알림: {context_data.get('notifications')}
- 과거 무시(미등록)했던 AI To-Do 히스토리: {context_data.get('ignored_history')}
- 이미 캘린더에 일정이 확보된 고객 ID 목록: {context_data.get('scheduled_customers')}
- 추천 가능한 유효 담당 고객 ID 리스트: {valid_c_ids} (일정 생성 시 이 리스트 내의 c_id를 최우선 매칭하십시오.)

### [재계획 및 반성 지침 (Reflection Guidance)]
- 지침: {reflection_guidance} (반성 지침에서 제안된 시간 회피 지시가 있다면 철저히 준수하십시오.)

[생성 및 시간 조율 규칙]
2. 제안하는 일정들은 **서로 시간이 겹치지 않도록 조율**하고, PB의 하루 일정 흐름에 자연스럽게 배치하십시오:
   - 추천 카테고리(`상담 일정 제안`, `신규 상품 분석`, `KPI 기반`)의 일정들을 하루 일정 흐름에 자연스럽게 배치하십시오.
   - 각 추천 일정에는 반드시 `10:00:00`, `11:00:00`, `13:00:00`, `14:00:00`, `15:00:00`, `16:00:00` 등 1시간 단위의 고유 시간대를 부여하되, PB의 기존 캘린더 일정과 겹치지 않는 빈 슬롯을 우선적으로 활용하십시오.
   - **[유연한 추천 개수]**: 기존 일정이 많아 비어 있는 시간대(빈 슬롯)가 부족한 날에는 억지로 5개를 채우지 말고, **비어 있는 시간대에 맞추어 최소 2개 ~ 최대 5개 이내로 유연하게 추천 일정을 구성**하십시오.
3. **업무 유형의 유연화 및 사전 준비/분석 업무 적극 권장**:
   - 당일 고객과 즉시 상담 약속을 잡기 어려울 수 있음을 고려하여, 단순 '대면 상담/미팅' 외에도 **'고객 자산 현황 분석', '이탈 위험 요인 사전 분석', '맞춤형 포트폴리오 리밸런싱 제안서 작성', 'KPI 달성을 위한 타겟 고객 선별 및 분석' 등의 사전 준비/분석 업무**를 추천 일정으로 적극 포함시키십시오.
   - 상품 만기가 예정된 고객은 `'상담 일정 제안'`으로 배정하여 자산 유치 상담 준비 및 미팅을 기획하십시오.
   - AUM 관련 목표 및 마케팅 등은 `'KPI 기반'`으로 배정하십시오.
   - 포트폴리오 분석 및 상품 연구 등은 `'신규 상품 분석'`으로 배정하십시오.
   - **생일 축하, 결혼기념일 등 고객 관계 관리 차원의 축하/안부 목적의 일정은 To-Do로 생성하지 마십시오. (To-Do에서는 제외)**
   - **[중요 - 목표 기반 자율적 고객 선택 (Goal-Client Alignment)]**: 오늘의 핵심 업무 목표(`Goal`)의 성격에 따라 고객들을 현명하게 매칭하십시오.
     * 오늘 목표가 **실적 증대(AUM 확보/IRP/신규 상품 유치 등)**와 직관되는 경우 ➔ 미상담 VIP 고객 중 **총자산이 높은 고객**들을 매칭하여 자산 분석 및 제안서 작성 일정을 잡으십시오.
     * 오늘 목표가 **고객 이탈 예방 및 관계 유지(밀착 관리 등)**와 직관되는 경우 ➔ 미상담 VIP 고객 중 **최종 상담일이 가장 오래되었거나 없는(미접촉 기간이 가장 긴) 고객**을 우선 매칭하여 현황 분석 및 관리 일정을 세우십시오.
4. 카테고리는 반드시 `'KPI 기반'`, `'상담 일정 제안'`, `'신규 상품 분석'` 중 하나여야 합니다. (안부 연락 제안 카테고리는 절대 생성하지 마십시오. check constraint 제약조건 철저 준수)
5. 각 일정의 제목은 50자 이내, 메모는 80자 이내로 정밀히 작성하십시오.
6. **[초비상 중요 제약조건 - 중복 추천 절대 금지]**: 이미 예약된 고객 ID 리스트({context_data.get('scheduled_customers')})는 이미 내일 일정 혹은 최근에 상담이 완료된 고객들입니다. 결과로 생성하는 JSON 배열 내의 어떤 객체도 이 리스트에 포함된 ID를 c_id로 가져서는 **절대 안 됩니다.** (예: 1002번 박수진 고객 등이 기예약 리스트에 들어있다면, 이번 추천 To-Do의 c_id에 1002를 지정하는 것은 엄격하게 금지됩니다. 이 규칙을 어길 시 에이전트 구동이 완전히 실패합니다.)
   - **[중요 - 추천 고객 다양성 및 쏠림 방지]**: 특정 1~2명의 고객(예: 이종혁 고객 등)에게 전체 추천 목록이 과도하게 쏠리지 않도록 분산해 주십시오. 동일한 고객 ID(`c_id`)는 전체 추천 목록에서 최대 1~2회까지만 포함될 수 있으며, 오랫동안 상담하지 않은 미상담 고객 리스트에 있는 다른 다양한 고객들을 고르게 골라 추천에 반영해 주십시오.
7. **[중요 - 스마트 중요도 감쇠 (Smart Decay)]**: 
   - 과거 추천 중 날짜가 이미 기준일 이전으로 완전히 지나갔음에도 PB가 일정 등록을 하지 않은 고객과 제안({context_data.get('ignored_history')})은 **진짜 원하지 않는 것(거절)**으로 판단되므로 강력히 배제 및 후순위 감쇠 처리하십시오.
   - 반면, 추천된 지 얼마 되지 않았거나 아직 실행 예정 기한이 많이 남아있어 단순 보류 상태인 미래의 추천들은 **시기상조 보류(Deferred)**된 것으로 아직 매우 신선한 추천이므로 정상적으로 재추천 대상으로 분류하십시오.
8. **[중요 - 추천 다양성 및 카테고리 분배 규칙]**: 생성하는 추천 일정들에는 최대한 3가지 카테고리('KPI 기반', '상담 일정 제안', '신규 상품 분석')가 골고루 포함되도록 노력하십시오. (개수가 적은 날에는 카테고리 균등 분포보다 충돌 회피와 비즈니스 정합성을 우선하십시오.)
9. **[중요 - 중요도 순서 배치 규칙]**: 생성하는 일정들은 **비즈니스적 중요도와 시급성이 높은 순서(1위부터 순서대로)**대로 정렬하여 JSON 배열의 앞쪽(Index 0)부터 차례대로 배치해 주십시오. (예: 가장 중요한 자산 이탈 위험 VVIP 고객 사전 자산 분석 등이 배열의 앞쪽에 위치하며, 상대적으로 덜 긴급한 업무는 뒤쪽에 배치됩니다.)

[출력 형식]
반드시 다음 **JSON 배열 형식**으로만 출력해 주세요.

```json
[
  {{
    "title": "[카테고리별 머리글] 추천 일정 제목 (예: [만기] 김OO 고객 예적금 재가입 상담 준비)",
    "memo": "구체적인 제안 메모 (예: 정기예금 만기 도래 10일 전. 포트폴리오 리밸런싱 설명안 작성)",
    "category": "상담 일정 제안",
    "execution_date": "YYYY-MM-DD HH:MM:SS 형식 (날짜는 반드시 {target_date}로 지정)",
    "c_id": {primary_c_id or 'null'}
  }},
  ... (조건에 맞게 최소 2개 ~ 최대 5개 생성) ...
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
        logger.error(f"[Executor] LLM 추천 일정 수립 파싱 실패, Fallback 적용: {e}")
        
        # 반성 지침에 따른 시간 조정 지원
        t1, t2, t3, t4, t5 = "10:00:00", "11:00:00", "14:00:00", "15:00:00", "16:00:00"
        if "14:00" in reflection_guidance:
            t3 = "17:00:00"
        if "10:00" in reflection_guidance:
            t1 = "09:00:00"

        # 기존 일정의 시간 파싱하여 빈 시간 슬롯(9:00 ~ 18:00 중) 추출
        busy_hours = []
        try:
            import re
            time_matches = re.findall(r"(\d{2}):(\d{2})", str(context_data.get('calendar', '')))
            for match in time_matches:
                busy_hours.append(int(match[0]))
        except Exception:
            pass

        # 9시부터 18시 중 비어 있는 시간대 후보 추출
        available_hours = [h for h in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18] if h not in busy_hours]
        if not available_hours:
            available_hours = [9, 10, 11, 13, 14, 15, 16, 17, 18] # 최소 방어

        # Heuristic Fallback 데이터 작성 (비즈니스 중심, 상담/분석/KPI 위주 및 중요도 정렬 준수)
        all_candidates = [
            {
                "title": "[만기] 우량 고객 정기예금 만기 분석 및 재가입 안내 준비",
                "memo": "예금 만기 15일 전. AUM 사수를 위한 고액 예금 재유치 상담 설명안 준비.",
                "category": "상담 일정 제안",
                "c_id": primary_c_id
            },
            {
                "title": "[분석] 신규 자산 포트폴리오 다각화 제안서 작성",
                "memo": "금리 변동성에 대응하기 위한 신규 펀드 및 채권 자산 리밸런싱 포트폴리오 분석.",
                "category": "신규 상품 분석",
                "c_id": secondary_c_id
            },
            {
                "title": "[상담] 우량 고객 투자성향 만료 현황 사전 검토",
                "memo": "투자성향 만료 임박 고객 대상 성향 재진단 및 안정성 향상 자산 다각화 제안 준비.",
                "category": "상담 일정 제안",
                "c_id": primary_c_id
            },
            {
                "title": "[포폴] 글로벌 채권 포트폴리오 리밸런싱 상품 연구",
                "memo": "고객 투자 성향 맞춤형 글로벌 하이일드 채권 상품 설명안 전달 대비 연구.",
                "category": "신규 상품 분석",
                "c_id": primary_c_id
            },
            {
                "title": "[KPI] 퇴직연금 IRP 마케팅 타겟 고객 선별",
                "memo": "소득 세액공제 한도 극대화를 위한 IRP 계좌 개설 안내 마케팅 준비.",
                "category": "KPI 기반",
                "c_id": secondary_c_id
            }
        ]

        # 빈 시간 슬롯 개수만큼만 유연하게 일정을 매칭하여 생성 (최소 2개 ~ 최대 5개)
        execution_results = []
        limit = min(5, len(available_hours))
        if limit < 2:
            limit = 2
            if len(available_hours) < 2:
                available_hours = [9, 11] # 강제 방어
        
        for idx in range(limit):
            hour = available_hours[idx]
            cand = all_candidates[idx % len(all_candidates)]
            execution_results.append({
                "title": cand["title"],
                "memo": cand["memo"],
                "category": cand["category"],
                "execution_date": f"{target_date} {hour:02d}:00:00",
                "c_id": cand["c_id"]
            })

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
