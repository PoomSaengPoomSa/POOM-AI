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
env_path = os.path.join(back_path, ".env")
env_backup_path = os.path.join(back_path, ".env.backup")
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
            "JWT_SECRET_KEY", "JWT_ALGORITHM", "ACCESS_TOKEN_EXPIRE_MINUTES", "REFRESH_TOKEN_EXPIRE_DAYS"
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
    from app.models import PbUser, Customer, CustomerProduct, CustomerAccount, ChurnLevel, Schedule, Notification, CustomerInformation, Product
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
from dotenv import load_dotenv
load_dotenv(env_path)

def generate_briefing_via_llm(customer_info: dict) -> str:
    """
    OpenAI API를 활용하여 정교한 구조화 방문 예정 브리핑을 생성합니다.
    API 호출 실패 또는 Key 부재 시 Heuristic Fallback을 안전하게 지원합니다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 1. OpenAI API 호출
    if api_key and not api_key.startswith("your-") and len(api_key) > 10:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            prompt = f"""당신은 우량 고객 자산관리 부문의 전문 수석 비서 AI입니다. 
아래 제공되는 해당 고객의 실시간 정량 및 정성 정보를 바탕으로, 담당 PB가 미팅 전에 한눈에 숙지할 수 있도록 실용적이고 예리한 **'방문 예정 브리핑'**을 정교하게 생성하십시오.

### [고객 실시간 데이터]
- 고객명: {customer_info['name']} (성향: {customer_info['tendency']})
- 담당 PB: {customer_info['pb_name']} 팀장
- 기호 및 선호도: {customer_info['preferences']}
- 자산 현황: 총 자산 {customer_info['total_assets']:,}원 (보통예금/예적금: {customer_info['deposit']:,}원, 투자상품: {customer_info['investment']:,}원, 퇴직연금: {customer_info['pension']:,}원, 대출: {customer_info['loan']:,}원)
- 가입 중인 상품 및 만기: {customer_info['products']}
- 이탈 등급: {customer_info['churn_grade']} (사유: {customer_info['churn_reason']})

### [작성 규칙]
1. 반드시 아래의 **[구조화 마커 포맷]**을 정확하게 지켜 출력하십시오. 마커 괄호명(`[...]`)을 그대로 유지해 주셔야 프론트엔드가 올바르게 인식합니다.
2. **[Quick Summary]**: 금리 추이, 만기 여부, 이탈 위험 사유를 바탕으로 당일 상담에서 달성해야 할 최우선 목표 2가지를 명확하게 요약 제시하십시오.
3. **[고객 정보 & Preference]**: 선호 기호(커피 타입 등), 기피하는 것(예: 비타500 싫어함 등)을 꼼꼼하게 정리하여 PB 필독 사항으로 기록하십시오.
4. **[자산 현황 & 최근 거래 내역]**: 총 자산과 상품별 비중, 만기일 등을 눈에 띄게 숫자로 표현하십시오.
5. **[핵심 특이사항]**: 이탈 가능성 또는 최근 상담 메모에서 포착된 투자 심리 변화를 날카롭게 지적해 상담 전략을 제공하십시오.

### [구조화 마커 포맷]
[Quick Summary]
(요약 내용 작성)

[고객 정보 & Preference]
- 고객명/등급: {customer_info['name']} 고객 ({customer_info['tendency']} 성향)
- 담당 PB: {customer_info['pb_name']} 팀장
- 음료/편의 선호도 (★필독):
(선호도 내용 불릿 포인트로 작성)

[자산 현황 & 최근 거래 내역]
- 총 자산: {customer_info['total_assets']:,} 원
- 보유 상품 상세:
(보유 상품 명칭 및 만기 현황 상세 작성)

[핵심 특이사항]
- 이탈 위험도: {customer_info['churn_grade']} ({customer_info['churn_reason']})
(기타 우대금리 민감 성향 및 필수 체크 사안 작성)
"""
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 우량 고객 자산관리 비서 전문가입니다. 항상 정해진 마커 포맷으로 정밀한 보고서를 반환합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return completion.choices[0].message.content
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

    fallback_content = f"""[Quick Summary]
금리 추이에 비교적 민감한 {customer_info['tendency']} 성향의 자산가 고객입니다. 
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

[핵심 특이사항]
- 이탈 위험도: {customer_info['churn_grade']} ({customer_info['churn_reason']})
  * 최근 우대금리 조건 문의 이력이 있으므로 신중한 세후 실질 수익률 중심 상담 설계 권장.
"""
    return fallback_content

def run_notification_generator(u_id: str, date_str: str):
    """
    매일 아침 실행되어 
    1) 오늘 생일인 고객 알림 생성 (안부 연락)
    2) 만기 도래 7일 이내 상품 알림 생성 (만기 알림)
    3) 이탈 등급 '위험' 발생 알림 생성 (이탈 위험)
    4) 캘린더 상담 확정 일정의 실시간 LLM 브리핑 생성 (방문 예정 브리핑)
    을 완벽하게 실행합니다.
    """
    logger.info(f"=== [START] 알림/브리핑 생성 파이프라인 가동 (PB: '{u_id}', 기준일: '{date_str}') ===")
    
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    start_of_today = datetime.combine(target_date, datetime.min.time())
    end_of_today = datetime.combine(target_date, datetime.max.time())
    
    with get_db() as db:
        # PB 정보 검증
        pb = db.query(PbUser).filter(PbUser.u_id == u_id).first()
        if not pb:
            logger.error(f"존재하지 않는 PB 유저 ID: '{u_id}'")
            return

        # ---------------------------------------------------------
        # 트랙 1: DB 데이터 기반의 정적 알림 생성
        # ---------------------------------------------------------
        
        # 1-1. 오늘 생일인 고객 조회 (안부 연락)
        birthday_customers = db.query(Customer).filter(
            Customer.birthday.isnot(None)
        ).all()
        
        for c in birthday_customers:
            # 월/일이 오늘과 매칭되는지 체크
            if c.birthday.month == target_date.month and c.birthday.day == target_date.day:
                # 이미 오늘 등록된 생일 알림이 있는지 검증
                dup = db.query(Notification).filter(
                    Notification.u_id == u_id,
                    Notification.category == "안부 연락",
                    Notification.title.like(f"%{c.name}%생일%"),
                    Notification.created_time >= start_of_today,
                    Notification.created_time <= end_of_today
                ).first()
                
                if not dup:
                    new_noti = Notification(
                        created_time=datetime.combine(target_date, datetime.now().time()),
                        title=f"{c.name} 고객 생일 축하 연락 제안",
                        content=f"오늘 생일을 맞이한 {c.name} 고객님께 친근한 축하 문자 메시지 및 모바일 기프트 쿠폰 발송을 제안합니다.",
                        category="안부 연락",
                        state_us="미확인",
                        u_id=u_id
                    )
                    db.add(new_noti)
                    logger.info(f"[안부 연락 알림 추가] {c.name} 고객 생일")
        
        # 1-2. 만기가 7일 이내로 남은 상품 조회 (만기 알림)
        d_plus_7 = target_date + timedelta(days=7)
        expiring_products = db.query(CustomerProduct).filter(
            CustomerProduct.expiration_date >= target_date,
            CustomerProduct.expiration_date <= d_plus_7
        ).all()
        
        for cp in expiring_products:
            c = cp.customer
            p = cp.product
            if c:
                dup = db.query(Notification).filter(
                    Notification.u_id == u_id,
                    Notification.category == "만기 알림",
                    Notification.title.like(f"%{c.name}%만기%"),
                    Notification.created_time >= start_of_today,
                    Notification.created_time <= end_of_today
                ).first()
                
                if not dup:
                    remaining_days = (cp.expiration_date - target_date).days
                    d_day_str = f"D-{remaining_days}" if remaining_days > 0 else "금일 만기"
                    new_noti = Notification(
                        created_time=datetime.combine(target_date, datetime.now().time()),
                        title=f"{c.name} 고객 {p.name} 만기({d_day_str}) 안내",
                        content=f"{c.name} 고객님이 보유 중인 [{p.name}] 상품의 만기일({cp.expiration_date})이 임박했습니다. 타행 이탈 방지를 위한 선제 상담을 권장합니다.",
                        category="만기 알림",
                        state_us="미확인",
                        u_id=u_id
                    )
                    db.add(new_noti)
                    logger.info(f"[만기 알림 추가] {c.name} 고객 - {p.name} 만기 ({d_day_str})")

        # 1-3. 이탈 위험 등급 '위험' 고객 조회 (이탈 위험)
        danger_churns = db.query(ChurnLevel).filter(
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
                        created_time=datetime.combine(target_date, datetime.now().time()),
                        title=f"{c.name} 고객 이탈 위험 주의 경보",
                        content=f"{c.name} 고객님의 이탈 위험 등급이 '위험' 단계로 감지되었습니다. 사유: {ch.reason}. 신속한 자산 흐름 파악 및 케어가 요구됩니다.",
                        category="이탈 위험",
                        state_us="미확인",
                        u_id=u_id
                    )
                    db.add(new_noti)
                    logger.info(f"[이탈 위험 알림 추가] {c.name} 고객 이탈 등급 위험")

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
                
            # 해당 확정일정에 대해 이미 브리핑 알림이 존재하는지 체크
            dup_briefing = db.query(Notification).filter(
                Notification.u_id == u_id,
                Notification.category == "방문 예정 브리핑",
                Notification.s_id == s.s_id
            ).first()
            
            if not dup_briefing:
                logger.info(f"[VisitBrief] 확정일정 발견 (s_id: {s.s_id}, 고객: {c.name}). LLM 브리핑 생성을 준비합니다.")
                
                # A. 고객 취향(Preferences) 조회
                prefs = db.query(CustomerInformation).filter(
                    CustomerInformation.c_id == c.c_id
                ).all()
                prefs_list = [f"[{pr.category}] {pr.contents}" for pr in prefs]
                
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
                
                # D. 정보 종합
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
                    "churn_reason": churn_reason
                }
                
                # E. LLM 또는 Heuristic을 통한 브리핑 생성
                briefing_text = generate_briefing_via_llm(cust_data)
                
                # F. 알림 등록
                visit_time = s.execution_date.strftime("%H:%M")
                title_text = f"{c.name} 고객 — {visit_time} 방문 예정"
                
                new_briefing_noti = Notification(
                    created_time=datetime.combine(target_date, datetime.now().time()),
                    title=title_text,
                    content=briefing_text,
                    category="방문 예정 브리핑",
                    state_us="미확인",
                    u_id=u_id,
                    s_id=s.s_id
                )
                db.add(new_briefing_noti)
                logger.info(f"[방문 예정 브리핑 추가 완료] 고객: {c.name}, 일정: {visit_time}")
                
        db.commit()
        
    logger.info("=== [FINISHED] 알림/브리핑 생성 파이프라인 구동 완료 ===")

if __name__ == "__main__":
    # 단위 테스트 코드 지원
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--u_id", type=str, default="user1")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()
    
    run_notification_generator(args.u_id, args.date)
