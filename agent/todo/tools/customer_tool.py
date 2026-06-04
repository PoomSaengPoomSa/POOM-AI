from datetime import datetime, timedelta
from typing import List, Dict, Any
from langchain_core.tools import tool
from .db_helper import get_db_session

# SQLAlchemy 모델 임포트
from app.models.customer import Customer, CustomerInformation, CustomerRelationship
from app.models.in_charge import InCharge
from app.models.churn_level import ChurnLevel
from app.models.consultation import ConsultationMemo, ConsultationReport
from app.models.product import CustomerProduct, Product

@tool
def GetCustomerRiskTool(u_id: str) -> str:
    """
    Get Customer Risk Tool.
    특정 PB(u_id)가 담당하는 고객들 중 이탈 위험 등급이 '위험' 또는 '주의'인 고객 목록과 이탈 위험 사유를 조회합니다.
    """
    with get_db_session() as db:
        # PB가 담당하는 고객 ID 목록 조회
        c_ids = [r.c_id for r in db.query(InCharge).filter(InCharge.u_id == u_id).all()]
        if not c_ids:
            return "담당하는 고객이 존재하지 않습니다."

        # 해당 고객들의 최신 이탈 위험 조회 (주의, 위험 위주)
        risks = (
            db.query(ChurnLevel)
            .join(Customer, ChurnLevel.c_id == Customer.c_id)
            .filter(ChurnLevel.c_id.in_(c_ids))
            .filter(ChurnLevel.grade.in_(["주의", "위험"]))
            .order_by(ChurnLevel.created_date.desc())
            .all()
        )

        # 가장 최근 레코드만 필터링 (고객당 1개 최신값 유지)
        latest_risks = {}
        for r in risks:
            if r.c_id not in latest_risks:
                latest_risks[r.c_id] = r

        if not latest_risks:
            return "현재 담당 고객 중 이탈 위험 등급이 '주의' 또는 '위험'인 고객이 존재하지 않습니다. 자산 상태가 안정적입니다."

        output = ["### [이탈 위험 고객 목록]"]
        for c_id, r in latest_risks.items():
            customer = db.query(Customer).filter(Customer.c_id == c_id).first()
            name = customer.name if customer else f"고객ID:{c_id}"
            grade = customer.grade if customer else "VIP"
            assets = f"{customer.total_assets / 100000000:.1f}억" if customer and customer.total_assets else "0원"
            output.append(
                f"- **{name}** ({grade} 등급, 총자산: {assets}) - **위험도: [{r.grade}]**\n"
                f"  * 이탈 우려 사유: {r.reason} (최근 분석일: {r.created_date.strftime('%Y-%m-%d')})"
            )
        return "\n".join(output)

@tool
def GetRecentConsultingHistoryTool(u_id: str, c_id: int = None) -> str:
    """
    Get Recent Consulting History Tool.
    특정 PB(u_id)가 담당하는 특정 고객(c_id) 또는 전체 고객의 최근 상담 내역(최대 5건)을 조회합니다.
    """
    with get_db_session() as db:
        query = db.query(ConsultationMemo).filter(ConsultationMemo.u_id == u_id)
        if c_id is not None:
            query = query.filter(ConsultationMemo.c_id == c_id)
        
        memos = query.order_by(ConsultationMemo.consult_date.desc()).limit(5).all()
        if not memos:
            return "최근 진행된 상담 기록이 존재하지 않습니다."

        output = ["### [최근 상담 이력]"]
        for m in memos:
            customer = db.query(Customer).filter(Customer.c_id == m.c_id).first()
            cust_name = customer.name if customer else f"고객ID:{m.c_id}"
            
            # 리포트가 있는지 확인
            report = db.query(ConsultationReport).filter(ConsultationReport.cm_id == m.cm_id).first()
            report_summary = f"\n  * 요약 리포트: {report.summary or report.key_contents}" if report else ""

            output.append(
                f"- **{cust_name}** 고객 상담 ({m.consult_date.strftime('%Y-%m-%d')}): {m.memo}{report_summary}"
            )
        return "\n".join(output)

@tool
def GetCustomerFeatureTool(c_id: int) -> str:
    """
    Get Customer Feature Tool.
    특정 고객(c_id)의 성향, 기호, 관계, 자산 관리 목표 등 정성적인 분석 정보(메모 기반 고객 특징)를 조회합니다.
    """
    with get_db_session() as db:
        customer = db.query(Customer).filter(Customer.c_id == c_id).first()
        if not customer:
            return f"c_id = {c_id}인 고객이 존재하지 않습니다."

        features = (
            db.query(CustomerInformation)
            .filter(CustomerInformation.c_id == c_id)
            .order_by(CustomerInformation.created_date.desc())
            .all()
        )

        output = [f"### [{customer.name} 고객 특징 및 선호 정보]"]
        output.append(f"- **고객명**: {customer.name} ({customer.grade} 등급, 투자성향: {customer.tendency})")
        output.append(f"- **총 자산**: {customer.total_assets / 100000000:.1f}억 (예금: {customer.deposit/100000000:.1f}억, 투자: {customer.investment/100000000:.1f}억, 연금: {customer.pension/100000000:.1f}억, 대출: {customer.loan/100000000:.1f}억)")
        output.append(f"- **직업**: {customer.job or '미지정'} | **이메일**: {customer.email}")

        if not features:
            output.append("- 등록된 세부 고객 선호/특징 태그 정보가 없습니다.")
        else:
            output.append("- **카테고리별 정성 특징 요약**:")
            for f in features:
                output.append(f"  * [{f.category}] {f.contents} (등록일: {f.created_date.strftime('%Y-%m-%d')})")
        return "\n".join(output)

@tool
def GetCustomerEventTool(u_id: str, date_str: str) -> str:
    """
    Get Customer Event Tool.
    PB가 담당하는 고객들 중 분석 기준일(date_str, YYYY-MM-DD)로부터 30일 이내에 도래하는 '상품 만기 예정일' 등의 주요 비즈니스 이벤트를 조회합니다. (생일/결혼기념일 등 관계 안부성 이벤트는 제외)
    """
    try:
        base_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        base_date = datetime.now().date()

    end_period = base_date + timedelta(days=30)

    with get_db_session() as db:
        # PB의 담당 고객 ID 조회
        c_ids = [r.c_id for r in db.query(InCharge).filter(InCharge.u_id == u_id).all()]
        if not c_ids:
            return "담당하는 고객이 존재하지 않습니다."

        customers = db.query(Customer).filter(Customer.c_id.in_(c_ids)).all()

        output = [f"### [{date_str} 기준 30일 이내 주요 상품 만기 예정 현황]"]
        events_found = False

        # 1. 상품 만기 조회 (오늘부터 30일 이내 만기)
        cust_products = (
            db.query(CustomerProduct)
            .join(Product, CustomerProduct.pd_id == Product.pd_id)
            .filter(CustomerProduct.c_id.in_(c_ids))
            .filter(CustomerProduct.expiration_date >= base_date)
            .filter(CustomerProduct.expiration_date <= end_period)
            .all()
        )

        if cust_products:
            output.append("\n[1. 상품 만기 도래 예정 안내]")
            for cp in cust_products:
                events_found = True
                cust = next((c for c in customers if c.c_id == cp.c_id), None)
                cust_name = cust.name if cust else f"고객ID:{cp.c_id}"
                output.append(
                    f"- **{cust_name}** 고객: 보유 상품 **'{cp.product.name}'** 만기 예정 (**만기일: {cp.expiration_date.strftime('%Y-%m-%d')}**)"
                )

        if not events_found:
            return f"[{date_str} 기준] 30일 이내에 예정된 상품 만기 이벤트가 전혀 없습니다."

        return "\n".join(output)

@tool
def GetUnconsultedCustomersTool(u_id: str, date_str: str) -> str:
    """
    Get Unconsulted Customers Tool.
    PB(u_id)가 담당하는 고객들 중 분석 기준일(date_str, YYYY-MM-DD)로부터 최근 60일 동안 상담 기록이 없거나 
    상담 기록이 전혀 없는 고자산 VIP 고객 목록(최대 5명)을 총자산 순으로 조회합니다.
    """
    try:
        base_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        base_date = datetime.now().date()

    limit_date = base_date - timedelta(days=60)

    with get_db_session() as db:
        # PB의 담당 고객 ID 조회
        c_ids = [r.c_id for r in db.query(InCharge).filter(InCharge.u_id == u_id).all()]
        if not c_ids:
            return "담당하는 고객이 존재하지 않습니다."

        # 담당 고객들의 정보 조회
        customers = db.query(Customer).filter(Customer.c_id.in_(c_ids)).all()
        if not customers:
            return "담당 고객 정보가 존재하지 않습니다."

        unconsulted_list = []
        for c in customers:
            # 해당 고객의 최신 상담 기록 1건 조회
            latest_memo = (
                db.query(ConsultationMemo)
                .filter(ConsultationMemo.c_id == c.c_id)
                .order_by(ConsultationMemo.consult_date.desc())
                .first()
            )

            # 60일 이전 상담이거나 상담이 아예 없는 경우 대상에 포함
            if not latest_memo:
                unconsulted_list.append({
                    "customer": c,
                    "last_consult_date": "상담 기록 없음"
                })
            elif latest_memo.consult_date.date() < limit_date:
                unconsulted_list.append({
                    "customer": c,
                    "last_consult_date": latest_memo.consult_date.strftime("%Y-%m-%d")
                })

        if not unconsulted_list:
            return "최근 60일 이내에 모든 담당 고객과 상담을 마쳤습니다. 밀착 관리가 잘 유지되고 있습니다."

        # 정렬 기준:
        # 1차: 상담 기록이 아예 없는 경우(가장 우선순위가 높음)와 상담 기록이 오래된 순서
        # 2차: 총자산이 높은 순서
        # 이를 위해 커스텀 정렬 키 적용
        def sorting_key(x):
            # 상담일이 없는 경우 매우 과거의 날짜(1970년)로 지정하여 정렬 우선순위 극대화
            if x["last_consult_date"] == "상담 기록 없음":
                consult_val = datetime(1970, 1, 1).date()
            else:
                consult_val = datetime.strptime(x["last_consult_date"], "%Y-%m-%d").date()
            # 1차 올림차순(consult_val), 2차 내림차순(total_assets)
            return (consult_val, -(x["customer"].total_assets or 0))

        unconsulted_list.sort(key=sorting_key)
        # 최대 15명까지 넉넉하게 context 제공
        top_unconsulted = unconsulted_list[:15]

        output = ["### [장기 미상담 고자산 고객 리스트 (60일 이상 미접촉, 최대 15명)]"]
        for idx, item in enumerate(top_unconsulted, 1):
            c = item["customer"]
            assets = f"{c.total_assets / 100000000:.1f}억" if c.total_assets else "0원"
            output.append(
                f"{idx}. **{c.name}** (c_id: {c.c_id}, {c.grade} 등급, 자산: {assets}) - **최종 상담일: {item['last_consult_date']}**"
            )
        return "\n".join(output)
