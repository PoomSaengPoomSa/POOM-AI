from typing import Dict, Any
from langchain_core.tools import tool
from .db_helper import get_db_session

# SQLAlchemy 모델 임포트
from app.models.kpi import Kpi
from app.models.account import PbUser
from app.models.branch import Branch

def format_money(val: int) -> str:
    """원화 금액 포맷터 (억 원 단위 표시)"""
    if val is None:
        return "0원"
    if val >= 100000000:
        return f"{val / 100000000:.1f}억 원"
    return f"{val / 10000:,}만 원"

@tool
def GetKPIStatusTool(u_id: str) -> str:
    """
    Get KPI Status Tool.
    특정 PB(u_id) 및 그 소속 지점의 목표(Target) 대비 현재 실적(Current) KPI 달성 상태를 정밀 분석하여 어떤 지표가 부진하고 채워야 하는지 분석합니다.
    """
    with get_db_session() as db:
        # PB 정보 및 소속 지점 식별
        user = db.query(PbUser).filter(PbUser.u_id == u_id).first()
        if not user:
            return f"u_id = '{u_id}'인 PB 정보를 찾을 수 없습니다."

        branch = db.query(Branch).filter(Branch.b_id == user.branch).first()
        branch_name = branch.name if branch else "소속 지점 없음"

        # 1. PB 개인 최신 KPI 조회
        pb_kpi = (
            db.query(Kpi)
            .filter(Kpi.u_id == u_id)
            .filter(Kpi.kpi_type == "PB")
            .order_by(Kpi.recorded_date.desc())
            .first()
        )

        # 2. 소속 지점 최신 KPI 조회
        branch_kpi = None
        if user.branch:
            branch_kpi = (
                db.query(Kpi)
                .filter(Kpi.b_id == user.branch)
                .filter(Kpi.kpi_type == "BRANCH")
                .order_by(Kpi.recorded_date.desc())
                .first()
            )

        output = [f"### [{user.name} PB & 소속 {branch_name} KPI 달성 현황]"]

        # PB 개인 KPI 요약
        if pb_kpi:
            aum_rate = (pb_kpi.current_aum / pb_kpi.target_aum * 100) if pb_kpi.target_aum else 0
            non_int_rate = (pb_kpi.current_non_interest / pb_kpi.target_non_interest * 100) if pb_kpi.target_non_interest else 0
            new_cust_rate = (pb_kpi.current_new_customer / pb_kpi.target_new_customer * 100) if pb_kpi.target_new_customer else 0

            output.append("\n[1. PB 개인 실적 상황]")
            output.append(f"- **자산 관리 규모 (AUM)**: 목표 {format_money(pb_kpi.target_aum)} | 현재 {format_money(pb_kpi.current_aum)} (**달성률: {aum_rate:.1f}%**)")
            output.append(f"- **비이자 수익**: 목표 {format_money(pb_kpi.target_non_interest)} | 현재 {format_money(pb_kpi.current_non_interest)} (**달성률: {non_int_rate:.1f}%**)")
            output.append(f"- **신규 우량 고객 유치**: 목표 {pb_kpi.target_new_customer}명 | 현재 {pb_kpi.current_new_customer}명 (**달성률: {new_cust_rate:.1f}%**)")
            output.append(f"  *(기준일: {pb_kpi.recorded_date.strftime('%Y-%m-%d')})*")
        else:
            output.append("\n[1. PB 개인 실적 상황]\n- 개인 KPI 정보가 아직 등록되지 않았습니다.")

        # 지점 KPI 요약
        if branch_kpi:
            b_aum_rate = (branch_kpi.current_aum / branch_kpi.target_aum * 100) if branch_kpi.target_aum else 0
            b_non_int_rate = (branch_kpi.current_non_interest / branch_kpi.target_non_interest * 100) if branch_kpi.target_non_interest else 0
            b_new_cust_rate = (branch_kpi.current_new_customer / branch_kpi.target_new_customer * 100) if branch_kpi.target_new_customer else 0

            output.append(f"\n[2. 소속 지점 ({branch_name}) 실적 상황]")
            output.append(f"- **지점 AUM**: 목표 {format_money(branch_kpi.target_aum)} | 현재 {format_money(branch_kpi.current_aum)} (**달성률: {b_aum_rate:.1f}%**)")
            output.append(f"- **지점 비이자 수익**: 목표 {format_money(branch_kpi.target_non_interest)} | 현재 {format_money(branch_kpi.current_non_interest)} (**달성률: {b_non_int_rate:.1f}%**)")
            output.append(f"- **지점 신규 고객**: 목표 {branch_kpi.target_new_customer}명 | 현재 {branch_kpi.current_new_customer}명 (**달성률: {b_new_cust_rate:.1f}%**)")
            output.append(f"  *(기준일: {branch_kpi.recorded_date.strftime('%Y-%m-%d')})*")
        else:
            output.append(f"\n[2. 소속 지점 ({branch_name}) 실적 상황]\n- 지점 KPI 정보가 등록되지 않았습니다.")

        # 보완이 필요한 비즈니스 목표 조언 추가
        warnings = []
        if pb_kpi:
            if aum_rate < 85:
                warnings.append("PB 개인 AUM(수신 자산 유치) 실적이 다소 부진합니다. 만기 도래 상품을 보유한 고자산 고객 대상 리밸런싱 및 재예치 설득이 필요합니다.")
            if non_int_rate < 80:
                warnings.append("PB 개인 비이자 수익(펀드, 신탁, 보험 판매 수수료 등) 실적이 낮습니다. 투자성향이 공격적이거나 수수료 기반 투자상품에 적합한 우량 고객군 상담이 시급합니다.")
            if new_cust_rate < 80:
                warnings.append("신규 우량 고객 유치가 미진합니다. 가망 고객 상담 및 다른 채널 연계 마케팅 일정이 권장됩니다.")

        if warnings:
            output.append("\n[3. KPI 부진에 따른 비즈니스 우선 제안]")
            for idx, w in enumerate(warnings, 1):
                output.append(f" {idx}) {w}")
        else:
            output.append("\n[3. KPI 부진에 따른 비즈니스 우선 제안]\n- 현재 모든 비즈니스 지표가 양호하며 목표 달성 페이스가 우수합니다. VIP 관계 강화에 초점을 맞추는 것이 좋습니다.")

        return "\n".join(output)
