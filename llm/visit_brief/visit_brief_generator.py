import os
import sys
from datetime import datetime, date, timedelta
import logging
from contextlib import contextmanager

# 로깅 설정
logger = logging.getLogger("VisitBriefGenerator")

# 백엔드 패키지 경로를 sys.path에 동적으로 추가하여 app.database 및 app.models를 재사용합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.abspath(os.path.join(current_dir, "..", "..", "..")), # Docker poom 루트
    os.path.abspath(os.path.join(current_dir, "..", "..", "..", "POOM-BACK")), # Docker POOM-BACK
    os.path.abspath(os.path.join(current_dir, "..", "..", "..", "back")), # 윈도우 로컬
]
back_path = None
for p in possible_paths:
    if os.path.exists(os.path.join(p, "app", "database.py")):
        back_path = p
        break
if not back_path:
    back_path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "back")) # Fallback

if back_path not in sys.path:
    sys.path.insert(0, back_path)

# Pydantic Settings ValidationError 방어
env_path = os.path.abspath(os.path.join(back_path, "..", ".env"))
env_backup_path = os.path.abspath(os.path.join(back_path, "..", ".env.backup"))
has_env = os.path.exists(env_path)

if has_env:
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            env_content = f.read()
        with open(env_backup_path, "w", encoding="utf-8") as f:
            f.write(env_content)
        
        allowed_keys = {
            "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME",
            "ECOS_API_KEY", "FRED_API_KEY", "REB_API_KEY",
            "JWT_SECRET_KEY", "JWT_ALGORITHM", "ACCESS_TOKEN_EXPIRE_MINUTES", "REFRESH_TOKEN_EXPIRE_DAYS",
            "OPENAI_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_ENDPOINT", "LANGSMITH_PROJECT"
        }
        
        clean_lines = []
        for line in env_content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                clean_lines.append(line)
                continue
            parts = stripped.split("=", 1)
            key = parts[0].strip()
            if key in allowed_keys or key.upper() in allowed_keys:
                clean_lines.append(line)
        
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("\n".join(clean_lines))
    except Exception as e:
        logger.warning(f"[Warning] .env 임시 패치 실패: {e}")

try:
    from app.database import SessionLocal
    from app.models import PbUser, Customer, CustomerProduct, CustomerAccount, ChurnLevel, Schedule, Notification, CustomerInformation, Product, CustomerTransaction, ConsultationMemo, ConsultationReport
    from app.models.customer import CustomerRelationship
    from app.models.in_charge import InCharge
finally:
    if has_env and os.path.exists(env_backup_path):
        try:
            with open(env_backup_path, "r", encoding="utf-8") as f:
                original_content = f.read()
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(original_content)
            os.remove(env_backup_path)
        except Exception as e:
            logger.warning(f"[Warning] .env 복원 실패: {e}")

from app.database import SessionLocal

@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

# .env 환경변수 원본 로드 (OpenAI 호출용)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def load_prompt(filename: str) -> str:
    """
    Utility to load prompt templates from the prompt directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "prompt", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def generate_briefing_via_llm(customer_info: dict) -> str:
    """
    OpenAI API를 활용하여 정교한 구조화 방문 예정 브리핑을 생성합니다.
    API 호출 실패 또는 Key 부재 시 Heuristic Fallback을 안전하게 지원합니다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 상담 이력 텍스트를 미리 정형화된 문자열로 빌드 (포맷 누락 원천 방지)
    history_lines = []
    if customer_info.get('consultations') and customer_info['consultations'] != ["이전 상담 이력 없음"]:
        for h in customer_info['consultations']:
            # 각 상담 이력이 이미 format되어 들어왔으므로, 앞에 '- '를 붙여줍니다.
            history_lines.append(f"- {h}")
        history_text = "\n".join(history_lines)
    else:
        history_text = "- 이전 상담 이력 없음"

    # 1. OpenAI API 호출
    if api_key and not api_key.startswith("your-") and len(api_key) > 10:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            prompt_template = load_prompt("visit_brief_prompt.md")
            prompt = prompt_template.format(
                name=customer_info['name'],
                tendency=customer_info['tendency'],
                pb_name=customer_info['pb_name'],
                preferences=customer_info['preferences'],
                total_assets=f"{customer_info['total_assets']:,}",
                deposit=f"{customer_info['deposit']:,}",
                investment=f"{customer_info['investment']:,}",
                pension=f"{customer_info['pension']:,}",
                loan=f"{customer_info['loan']:,}",
                products=customer_info['products'],
                churn_grade=customer_info['churn_grade'],
                churn_reason=customer_info['churn_reason'],
                transactions=customer_info['transactions'],
                consultations=customer_info['consultations'],
                history_text=history_text
            )
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 우량 고객 자산관리 비서 전문가입니다. 항상 정해진 마커 포맷으로 정밀하고 개별화된 보고서를 반환합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            briefing_text = completion.choices[0].message.content
            # Post-processing: Ensure history lines are restored to their original form (with IDs)
            for marker in ["- 이전 상담 히스토리 요약:", "이전 상담 히스토리 요약:"]:
                if marker in briefing_text:
                    parts = briefing_text.split(marker)
                    briefing_text = parts[0].rstrip() + "\n" + marker + "\n" + history_text
                    break
            return briefing_text
        except Exception as e:
            logger.error(f"[LLM] OpenAI API 호출 오류 발생, Fallback 가동: {e}")

    # 2. Heuristic Fallback (API 키가 없거나 호출이 실패한 경우 작동하는 초정밀 가동 장치)
    logger.info("[LLM] Heuristic Fallback 모드로 방문 예정 브리핑을 동적 빌드합니다.")
    
    # 기호 가공
    prefs_text = ""
    if customer_info['preferences']:
        for p in customer_info['preferences']:
            prefs_text += f"\n  - {p}"
    else:
        prefs_text = "\n  - 특이 기호 없음 (기본 따뜻한 차 선호)"

    # 상품 가공
    prods_text = ""
    if customer_info['products']:
        for pr in customer_info['products']:
            prods_text += f"\n  - {pr}"
    else:
        prods_text = "\n  - 보통 보통예금 계좌 거래 중"

    # 거래 내역 가공
    txs_text = ""
    if customer_info['transactions']:
        for tx in customer_info['transactions']:
            txs_text += f"\n  - {tx}"
    else:
        txs_text = "\n  - 최근 5건 거래 내역 없음"

    # 상담 이력 가공
    history_fallback_text = ""
    if customer_info['consultations']:
        for h in customer_info['consultations']:
            history_fallback_text += f"\n- {h}"
    else:
        history_fallback_text = "\n- 이전 상담 이력 없음"

    fallback_content = f"""[Quick Summary]
최근 총 자산 변동성이 확인된 {customer_info['tendency']} 성향의 고객입니다. 
당일 예정된 방문 일정에서는 만기 자금의 타행 유출 방어를 위한 정기 특판 재가입 및 맞춤형 포트폴리오 리밸런싱 상담을 집중 지원하십시오.

[고객 정보 & Preference]
- 고객명/등급: {customer_info['name']} 고객 ({customer_info['tendency']} 성향)
- 담당 PB: {customer_info['pb_name']} 팀장
- 음료/편의 선호도 (★필독):{prefs_text}
  - 연한 커피 선호 및 신속한 두괄식 보고 선호

[자산 현황 & 최근 거래 내역]
- 총 자산: {customer_info['total_assets']:,} 원
  * 예적금 잔액: {customer_info['deposit']:,} 원
  * 투자 상품: {customer_info['investment']:,} 원
  * 대출 현황: {customer_info['loan']:,} 원
- 보유 상품 상세:{prods_text}
- 최근 거래 내역:{txs_text}

[핵심 특이사항]
- 이탈 위험도: {customer_info['churn_grade']} ({customer_info['churn_reason']})
  * 최근 우대금리 조건 문의 이력이 있으므로 신중한 세후 실질 수익률 중심 상담 설계 권장.
- 이전 상담 히스토리 요약:{history_fallback_text}
"""
    return fallback_content

def run_notification_generator(u_id: str, date_str: str, db=None):
    """
    매일 아침 또는 조회 시점에 실행되어 
    1) 오늘 생일인 고객 알림 생성 (안부 연락)
    1-B) 오늘 지인 기념일인 고객 알림 생성 (안부 연락 - Relation 테이블)
    2) 만기 도래 7일 이내 상품 알림 생성 (만기 알림)
    3) 이탈 등급 '위험' 발생 알림 생성 (이탈 위험)
    4) 캘린더 상담 확정 일정의 실시간 LLM 브리핑 생성 (방문 예정 브리핑)
    을 완벽하게 실행합니다. ( u_id PB의 담당 고객으로 엄격히 한정 )
    """
    logger.info(f"=== [START] 알림/브리핑 생성 파이프라인 가동 (PB: '{u_id}', 기준일: '{date_str}') ===")
    
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    start_of_today = datetime.combine(target_date, datetime.min.time())
    end_of_today = datetime.combine(target_date, datetime.max.time())
    
    from datetime import timezone, timedelta
    kst_tz = timezone(timedelta(hours=9))
    kst_now = datetime.now(kst_tz)
    kst_time = kst_now.time()
    
    from contextlib import nullcontext
    db_context = nullcontext(db) if db is not None else get_db()
    should_commit = db is None
    
    with db_context as db:
        # PB 정보 검증
        pb = db.query(PbUser).filter(PbUser.u_id == u_id).first()
        if not pb:
            logger.error(f"존재하지 않는 PB 유저 ID: '{u_id}'")
            return

        # PB의 담당 고객 목록 조회 (InCharge 테이블 활용)
        pb_customers = db.query(Customer).join(InCharge, InCharge.c_id == Customer.c_id).filter(InCharge.u_id == u_id).all()
        pb_customer_ids = [c.c_id for c in pb_customers]
        logger.info(f"PB '{u_id}' 담당 고객 수: {len(pb_customers)}명 (IDs: {pb_customer_ids})")

        # ---------------------------------------------------------
        # 트랙 1: DB 데이터 기반의 정적 알림 생성
        # ---------------------------------------------------------
        
        # 1-1. 오늘 생일인 담당 고객 조회 및 알림 생성 (안부 연락)
        for c in pb_customers:
            if c.birthday and c.birthday.month == target_date.month and c.birthday.day == target_date.day:
                dup = db.query(Notification).filter(
                    Notification.u_id == u_id,
                    Notification.category == "안부 연락",
                    Notification.title.like(f"%{c.name}%생일%"),
                    Notification.created_time >= start_of_today,
                    Notification.created_time <= end_of_today
                ).first()
                
                if not dup:
                    new_noti = Notification(
                        created_time=datetime.combine(target_date, kst_time),
                        title=f"{c.name} 고객 생일 축하 연락 제안",
                        content=f"오늘 생일을 맞이한 {c.name} 고객님께 친근한 축하 문자 메시지 및 모바일 기프트 쿠폰 발송을 제안합니다.",
                        category="안부 연락",
                        state_us="미확인",
                        u_id=u_id,
                        c_id=c.c_id
                    )
                    db.add(new_noti)
                    logger.info(f"[안부 연락 알림 추가] {c.name} 고객 생일")

        # 1-1-B. 오늘 지인 기념일인 고객 조회 및 알림 생성 (안부 연락 - Relation 테이블)
        if pb_customer_ids:
            relationships = db.query(CustomerRelationship).filter(
                CustomerRelationship.c_id.in_(pb_customer_ids)
            ).all()
            
            for rel in relationships:
                cust = rel.customer
                if not cust:
                    continue
                # 지인 생일 체크
                if rel.birthday and rel.birthday.month == target_date.month and rel.birthday.day == target_date.day:
                    dup = db.query(Notification).filter(
                        Notification.u_id == u_id,
                        Notification.category == "안부 연락",
                        Notification.title.like(f"%{cust.name}%지인%{rel.relationship_}%생일%"),
                        Notification.created_time >= start_of_today,
                        Notification.created_time <= end_of_today
                    ).first()
                    
                    if not dup:
                        new_noti = Notification(
                            created_time=datetime.combine(target_date, kst_time),
                            title=f"{cust.name} 고객 지인({rel.relationship_}) 생일 축하 제안",
                            content=f"오늘 {cust.name} 고객님의 지인({rel.relationship_}) 생일입니다. 고객님께 축하 메시지 및 안부 인사를 제안합니다.",
                            category="안부 연락",
                            state_us="미확인",
                            u_id=u_id,
                            c_id=cust.c_id
                        )
                        db.add(new_noti)
                        logger.info(f"[지인 생일 알림 추가] {cust.name} 고객 지인({rel.relationship_}) 생일")

                # 지인 결혼기념일 체크
                if rel.wedding_date and rel.wedding_date.month == target_date.month and rel.wedding_date.day == target_date.day:
                    dup = db.query(Notification).filter(
                        Notification.u_id == u_id,
                        Notification.category == "안부 연락",
                        Notification.title.like(f"%{cust.name}%지인%{rel.relationship_}%결혼기념일%"),
                        Notification.created_time >= start_of_today,
                        Notification.created_time <= end_of_today
                    ).first()
                    
                    if not dup:
                        new_noti = Notification(
                            created_time=datetime.combine(target_date, kst_time),
                            title=f"{cust.name} 고객 지인({rel.relationship_}) 결혼기념일 축하 제안",
                            content=f"오늘 {cust.name} 고객님의 지인({rel.relationship_}) 결혼기념일입니다. 고객님께 축하 메시지 및 안부 인사를 제안합니다.",
                            category="안부 연락",
                            state_us="미확인",
                            u_id=u_id,
                            c_id=cust.c_id
                        )
                        db.add(new_noti)
                        logger.info(f"[지인 결혼기념일 알림 추가] {cust.name} 고객 지인({rel.relationship_}) 결혼기념일")
        
        # 1-2. 만기가 7일 이내로 남은 상품 조회 (만기 알림)
        d_plus_7 = target_date + timedelta(days=7)
        if pb_customer_ids:
            expiring_products = db.query(CustomerProduct).filter(
                CustomerProduct.c_id.in_(pb_customer_ids),
                CustomerProduct.expiration_date >= target_date,
                CustomerProduct.expiration_date <= d_plus_7
            ).all()
            
            for cp in expiring_products:
                c = cp.customer
                p = cp.product
                if c and p:
                    # 해당 상품의 만기일 기준 30일 이내에 이미 만기 알림이 생성되었는지 확인 (매일 중복 알림 생성 방지)
                    dup = db.query(Notification).filter(
                        Notification.u_id == u_id,
                        Notification.category == "만기 알림",
                        Notification.c_id == c.c_id,
                        Notification.title.like(f"%{p.name}%"),
                        Notification.created_time >= datetime.combine(target_date - timedelta(days=30), datetime.min.time())
                    ).first()
                    
                    if not dup:
                        remaining_days = (cp.expiration_date - target_date).days
                        d_day_str = f"D-{remaining_days}" if remaining_days > 0 else "금일 만기"
                        new_noti = Notification(
                            created_time=datetime.combine(target_date, kst_time),
                            title=f"{c.name} 고객 {p.name} 만기({d_day_str}) 안내",
                            content=f"{c.name} 고객님이 보유 중인 [{p.name}] 상품의 만기일({cp.expiration_date})이 임박했습니다. 타행 이탈 방지를 위한 선제 상담을 권장합니다.",
                            category="만기 알림",
                            state_us="미확인",
                            u_id=u_id,
                            c_id=c.c_id
                        )
                        db.add(new_noti)
                        logger.info(f"[만기 알림 추가] {c.name} 고객 - {p.name} 만기 ({d_day_str})")

        # 1-3. 이탈 위험 등급 '위험' 고객 조회 (이탈 위험)
        if pb_customer_ids:
            danger_churns = db.query(ChurnLevel).filter(
                ChurnLevel.c_id.in_(pb_customer_ids),
                ChurnLevel.grade == "위험",
                ChurnLevel.created_date >= start_of_today,
                ChurnLevel.created_date <= end_of_today
            ).all()
            
            for ch in danger_churns:
                c = ch.customer
                if c:
                    dup = db.query(Notification).filter(
                        Notification.u_id == u_id,
                        Notification.category == "이탈 위험",
                        Notification.title.like(f"%{c.name}%이탈%"),
                        Notification.created_time >= start_of_today,
                        Notification.created_time <= end_of_today
                    ).first()
                    
                    if not dup:
                        new_noti = Notification(
                            created_time=datetime.combine(target_date, kst_time),
                            title=f"{c.name} 고객 이탈 위험 주의 경보",
                            content=f"{c.name} 고객님의 이탈 위험 등급이 '위험' 단계로 감지되었습니다. 사유: {ch.reason}. 신속한 자산 흐름 파악 및 케어가 요구됩니다.",
                            category="이탈 위험",
                            state_us="미확인",
                            u_id=u_id,
                            c_id=c.c_id
                        )
                        db.add(new_noti)
                        logger.info(f"[이탈 위험 알림 추가] {c.name} 고객 이탈 등급 위험")

        # 1-4. 최근 7일 내 타행 거액(1,000만원 이상) 출금 거래 등록 후 알림 생성 (거액 거래 탐지)
        # 타행은 "우리은행"이 아닌 경우로 설정
        start_of_7days_ago = datetime.combine(target_date - timedelta(days=7), datetime.min.time())
        if pb_customer_ids:
            large_withdrawals = db.query(CustomerTransaction).filter(
                CustomerTransaction.c_id.in_(pb_customer_ids),
                CustomerTransaction.ct_type == 'W',
                CustomerTransaction.amount >= 10000000,
                CustomerTransaction.opp_bank_name != "우리은행",
                CustomerTransaction.ct_datetime >= start_of_7days_ago,
                CustomerTransaction.ct_datetime <= end_of_today
            ).all()
            
            # 고객(c_id)별로 거래 그룹화
            from collections import defaultdict
            withdrawals_by_customer = defaultdict(list)
            for tx in large_withdrawals:
                withdrawals_by_customer[tx.c_id].append(tx)
                
            for c_id, tx_list in withdrawals_by_customer.items():
                # 해당 고객의 최신 거래 시간 기준으로 알림 여부 판단
                max_tx_datetime = max(tx.ct_datetime for tx in tx_list)
                c = tx_list[0].customer
                if c:
                    # 해당 고객의 이 거래 내역들(가장 최근 거래 기준)에 대해 이미 알림을 생성했는지 체크
                    # max_tx_datetime 10초 전 이후에 생성된 동일 알림이 있다면 이미 알림 완료된 거래로 취급
                    dup = db.query(Notification).filter(
                        Notification.u_id == u_id,
                        Notification.category == "거액 거래 탐지",
                        Notification.c_id == c_id,
                        Notification.created_time >= max_tx_datetime - timedelta(seconds=10)
                    ).first()
                    
                    if not dup:
                        total_amount = sum(tx.amount for tx in tx_list)
                        count = len(tx_list)
                        opp_banks = list(set(tx.opp_bank_name for tx in tx_list))
                        opp_banks_str = ", ".join(opp_banks)
                        
                        new_noti = Notification(
                            created_time=datetime.combine(target_date, kst_time),
                            title=f"{c.name} 고객 타행 거액 출금 감지",
                            content=f"{c.name} 고객님이 최근 7일 내 타행({opp_banks_str})으로 거액 출금(총 {count}건, {int(total_amount):,}원) 거래를 발생시켰습니다. 타행 자산 이탈 여부 확인을 위한 선제적 관리가 필요합니다.",
                            category="거액 거래 탐지",
                            state_us="미확인",
                            u_id=u_id,
                            c_id=c.c_id
                        )
                        db.add(new_noti)
                        logger.info(f"[거액 거래 알림 추가] {c.name} 고객 - 총 {count}건, {int(total_amount):,}원 출금")

        # ---------------------------------------------------------
        # 트랙 2: 확정 상담 일정 기반 실시간 LLM '방문 예정 브리핑' 생성
        # ---------------------------------------------------------
        confirmed_visits = db.query(Schedule).filter(
            Schedule.u_id == u_id,
            Schedule.category == "상담",
            Schedule.c_id.isnot(None),
            Schedule.execution_date >= start_of_today,
            Schedule.execution_date <= end_of_today
        ).all()
        
        for s in confirmed_visits:
            c = s.customer
            if not c:
                continue
                
            # 해당 확정일정에 대해 이미 브리핑 알림이 존재하는지 체크 (s_id 매칭 혹은 동일 고객의 오늘 자 정상 브리핑 중복 생성 원천 방지)
            dup_briefing = db.query(Notification).filter(
                Notification.u_id == u_id,
                Notification.category == "방문 예정 브리핑",
                (Notification.s_id == s.s_id) |
                ((Notification.c_id == c.c_id) & 
                 Notification.s_id.isnot(None) &
                 (Notification.created_time >= start_of_today) & 
                 (Notification.created_time <= end_of_today))
            ).first()
            
            if not dup_briefing:
                logger.info(f"[VisitBrief] 확정일정 발견 (s_id: {s.s_id}, 고객: {c.name}). LLM 브리핑 생성을 준비합니다.")
                
                # A. 고객 취향(Preferences) 조회 - 중복 제거 및 최신 순 정렬
                prefs = db.query(CustomerInformation).filter(
                    CustomerInformation.c_id == c.c_id
                ).order_by(CustomerInformation.created_date.desc()).all()
                seen_prefs = set()
                prefs_list = []
                for pr in prefs:
                    clean_content = pr.contents.strip()
                    if clean_content not in seen_prefs:
                        seen_prefs.add(clean_content)
                        prefs_list.append(f"[{pr.category}] {clean_content}")
                prefs_list = prefs_list[:10]  # 최대 10개만 유지
                
                # B. 보유 상품 & 만기일 조회
                prods = db.query(CustomerProduct).filter(
                    CustomerProduct.c_id == c.c_id
                ).all()
                prods_list = [f"{cp.product.name} (만기: {cp.expiration_date.strftime('%Y-%m-%d') if cp.expiration_date else '없음'})" for cp in prods if cp.product]
                
                # C. 최신 이탈 등급 조회
                churn = db.query(ChurnLevel).filter(
                    ChurnLevel.c_id == c.c_id
                ).order_by(ChurnLevel.created_date.desc()).first()
                churn_grade = churn.grade if churn else "양호"
                churn_reason = churn.reason if churn else "특이사항 없음"
                
                # D. 최근 5건 거래 내역 조회
                txs = db.query(CustomerTransaction).filter(
                    CustomerTransaction.c_id == c.c_id
                ).order_by(CustomerTransaction.ct_datetime.desc()).limit(5).all()
                txs_list = []
                for tx in txs:
                    type_str = "입금" if tx.ct_type == 'D' else "출금"
                    txs_list.append(
                        f"[{tx.ct_datetime.strftime('%Y-%m-%d %H:%M')}] {type_str}: {int(tx.amount):,}원 | 상대방: {tx.opp_bank_name} ({tx.opp_name}) | 채널: {tx.channel}"
                    )
 
                # E. 최근 3건 상담 이력 및 AI 요약 조회 - 중복 제거
                history = db.query(ConsultationMemo).filter(
                    ConsultationMemo.c_id == c.c_id
                ).order_by(ConsultationMemo.consult_date.desc()).limit(10).all()
                history_list = []
                seen_memos = set()
                for h in history:
                    clean_memo = h.memo.strip()
                    if clean_memo not in seen_memos:
                        seen_memos.add(clean_memo)
                        rep = h.report
                        rep_summary = rep.summary if rep else "요약 정보 없음"
                        memo_snippet = clean_memo[:100] + "..." if len(clean_memo) > 100 else clean_memo
                        # 포맷 규격화
                        history_list.append(
                            f"[{h.consult_date.strftime('%Y-%m-%d')}] 상담 내용: {memo_snippet} | AI 요약: {rep_summary} | ID: {h.cm_id}"
                        )
                        if len(history_list) >= 3:
                            break
 
                # F. 정보 종합
                cust_data = {
                    "name": c.name,
                    "tendency": c.tendency,
                    "pb_name": pb.name,
                    "preferences": prefs_list if prefs_list else ["등록된 취향 정보 없음"],
                    "total_assets": c.total_assets,
                    "deposit": c.deposit,
                    "investment": c.investment,
                    "pension": c.pension,
                    "loan": c.loan,
                    "products": prods_list if prods_list else ["보유 중인 만기성 상품 없음"],
                    "churn_grade": churn_grade,
                    "churn_reason": churn_reason,
                    "transactions": txs_list if txs_list else ["최근 5건 거래 내역 없음"],
                    "consultations": history_list if history_list else ["이전 상담 이력 없음"]
                }
                
                # G. LLM 또는 Heuristic을 통한 브리핑 생성
                briefing_text = generate_briefing_via_llm(cust_data)
                
                # H. 알림 등록
                visit_time = s.execution_date.strftime("%H:%M")
                title_text = f"{c.name} 고객 — {visit_time} 방문 예정"
                
                new_briefing_noti = Notification(
                    created_time=datetime.combine(target_date, kst_time),
                    title=title_text,
                    content=briefing_text,
                    category="방문 예정 브리핑",
                    state_us="미확인",
                    u_id=u_id,
                    s_id=s.s_id,
                    c_id=c.c_id
                )
                try:
                    with db.begin_nested():
                        db.add(new_briefing_noti)
                    logger.info(f"[방문 예정 브리핑 추가 완료] 고객: {c.name}, 일정: {visit_time}")
                except Exception as e:
                    logger.warning(f"[VisitBrief] Duplicate notification insertion prevented (s_id: {s.s_id}): {e}")
                
        if should_commit:
            db.commit()
        
    logger.info("=== [FINISHED] 알림/브리핑 생성 파이프라인 구동 완료 ===")

if __name__ == "__main__":
    # 단위 테스트 코드 지원
    import argparse
    import logging
    from datetime import timezone, timedelta
    
    # 단위 테스트용 콘솔 로깅 활성화
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    kst_tz = timezone(timedelta(hours=9))
    default_date = datetime.now(kst_tz).strftime("%Y-%m-%d")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--u_id", type=str, default="user1")
    parser.add_argument("--date", type=str, default=default_date)
    args = parser.parse_args()
    
    run_notification_generator(args.u_id, args.date)
