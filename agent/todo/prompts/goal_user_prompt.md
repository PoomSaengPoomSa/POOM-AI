다음은 현재 PB와 담당 고객들의 데이터입니다. 이를 종합적으로 분석하여 오늘의 핵심 업무 목표(Goal)를 도출하십시오.

### 1. PB 일정 정보 (Target Date: {target_date})
{calendar}

### 2. PB KPI 상태
{kpi}

### 3. 담당 고객 이탈 위험 상태
{risks}

### 4. 30일 이내 상품 만기 및 중요 기념일 이벤트
{events}

### 5. 최근 상담 타임라인 정보
{histories}

### 5.5. 최근 60일 이상 미상담한 담당 고자산 VIP 고객 목록
{unconsulted_customers}

### 6. 기존 알림 상태
{notifications}

### 7. 과거 무시(미등록)했던 AI To-Do 히스토리 (날짜가 완전히 지나갔음에도 일정을 잡지 않아 진짜 안 원해 거절된 목록 - 배제 대상)
{ignored_history}

### 8. 이미 캘린더에 일정이 수립되어 상담 예약 완료된 고객 ID 리스트 (추천에서 가급적 제외할 대상)
{scheduled_customers}

오늘의 최우선 업무 목표와 사유를 담은 JSON 객체를 반환하십시오.
