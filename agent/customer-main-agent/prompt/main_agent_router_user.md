## 분석 대상 고객 데이터 요약

- **고객 기본 정보**:
  * 고객명: {name}
  * 고객 등급: {grade}
  * 투자 성향: {tendency}
- **자산 구성 현황**:
  * 총자산: {total_assets:,} 원
  * 예금 잔액: {deposit:,} 원
  * 투자 금액: {investment:,} 원
  * 연금 자산: {pension:,} 원
  * 대출 잔액(부채): {loan:,} 원
  * 순자산: {net_worth:,} 원
- **최근 7일간 타행 거액 이출금 내역**:
{large_withdrawals_str}
- **최근 상담 보고서 존재 여부**:
  * 존재 여부: {has_consultation_report}

상기 지표 데이터와 서브 에이전트 판정 기준을 바탕으로 오늘 작동시킬 서브 에이전트를 정확히 판단하여 JSON 결과를 도출하십시오.
