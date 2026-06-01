## 1. 분석 대상 고객 기본 프로필 데이터
- 고객명: {name} (등급: {grade})
- 투자 성향: {tendency}
- 총자산 정보: {total_assets:,} 원 (예금: {deposit:,} 원, 대출: {loan:,} 원)

## 2. 최근 1개월간 축적된 고객 정성 행동 특징
{features_str}

## 3. 본점 주력 금융 상품 명세서 (Active Main Products)
{products_str}

## 4. [추가 수집 맥락 정보]
- **보유/가입 중인 금융 상품 목록**:
{active_products_str}

- **가족 관계 및 가구 구성 정보**:
{relationship_str}

- **계좌유형별 잔액 및 유동 자금 현황**:
{accounts_str}

---

제공된 모든 주력 상품에 대해 **개별적으로 적합 여부를 전부 평가**하여, 지정된 출력 JSON 형식으로 결과를 생성해 주십시오. (is_suitable: 0=부적합, 1=적합, 2=기보유)
