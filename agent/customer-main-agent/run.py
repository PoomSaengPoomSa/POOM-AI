import argparse
import logging
import sys
from datetime import datetime
from .db import get_db_cursor
from .agent import MainAgent

# 로깅 환경 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("IntegratedCustomerAgent")

def fetch_batch_target_c_ids() -> list:
    """
    고객 선정 Node: DB 단일 스캔 쿼리를 통해 분석이 시급한 VVIP 고객 ID 추출
    1. 총자산 1억 이상 우량 고객 중 자산분석(llm_insight)이 없는 고객
    2. 최근 7일 내 타행 고액 이출금(1천만 원 이상)이 발생한 고객 (W = Withdrawal)
    3. 30일 이내 만기 예정 금융 상품을 보유한 고객
    4. 오늘 상담 예약이 확정되어 내방하는 고객
    """
    query = """
        -- (1) 예금 1억 이상 우량 고객 중 llm_insight가 비어있는 고객
        SELECT c_id FROM customer WHERE total_assets >= 100000000 AND (llm_insight IS NULL OR llm_insight = '')
        UNION
        -- (2) 최근 7일 내 타행 고액 이체 출금이 감지된 고객
        SELECT DISTINCT c_id FROM customer_transaction 
        WHERE opp_bank_name != '품' AND ct_type = 'W' AND amount >= 10000000 
          AND ct_datetime >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        UNION
        -- (3) 30일 이내에 도래하는 만기 예적금 상품 보유 고객
        SELECT DISTINCT c_id FROM customer_product 
        WHERE expiration_date >= CURDATE() AND expiration_date <= DATE_ADD(CURDATE(), INTERVAL 30 DAY)
        UNION
        -- (4) 오늘 상담 예약이 확정되어 내방하는 고객
        SELECT DISTINCT c_id FROM schedule 
        WHERE category = '상담' AND DATE(execution_date) = CURDATE() AND c_id IS NOT NULL
    """
    try:
        with get_db_cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            c_ids = [row["c_id"] for row in results]
            logger.info(f"[고객 선정 Node] 자동 스캔 완료. 총 {len(c_ids)}명의 분석 대상 선별: {c_ids}")
            return c_ids
    except Exception as e:
        logger.error(f"[고객 선정 Node] 대상 조회 실패 (Fallback 적용): {e}")
        return []

def run_integrated_batch(specified_c_ids: list = None):
    logger.info("==========================================================")
    logger.info("🤖 POOM-AI 초경량 고객분석 배치 에이전트 구동 개시")
    logger.info("==========================================================")
    
    # 1단계: 분석 대상 c_id 리스트 수집
    if specified_c_ids:
        target_c_ids = specified_c_ids
        logger.info(f"[1단계] 지정된 수동 고객 분석 실행: {target_c_ids}")
    else:
        target_c_ids = fetch_batch_target_c_ids()
        
    if not target_c_ids:
        logger.info("[배치 중단] 오늘 분석을 수행할 대상 고객이 한 명도 존재하지 않습니다.")
        logger.info("==========================================================")
        return

    # 2단계: 에이전트 인스턴스 초기화 (4개 독립 SubAgent 포함)
    logger.info("[2단계] 통합 Main Agent 및 4대 SubAgent 모듈 초기화 중...")
    main_agent = MainAgent()

    # 3단계: 순차 및 독립적 분석 루프 실행
    logger.info(f"[3단계] 총 {len(target_c_ids)}명 고객 대상 독립 분석 순회 루프 시작")
    success_count = 0
    failure_count = 0

    for idx, c_id in enumerate(target_c_ids, 1):
        logger.info(f"\n({idx}/{len(target_c_ids)}) [고객 ID: {c_id}] 4대 핵심 분석(SubAgent 1, 2, 3, 4) 실행")
        
        results = main_agent.run_for_customer(customer_id=c_id)
        
        # 4개 세부 서브 태스크가 모두 성공했는지 여부 판단
        # (3, 4는 상담 기록 미존재 시 스킵되므로, 로직상 성공으로 카운트됩니다)
        is_all_success = (
            results["sub1_success"] and 
            results["sub2_success"] and 
            results["sub3_success"] and 
            results["sub4_success"]
        )
        
        if is_all_success:
            success_count += 1
            logger.info(f" -> [고객 ID: {c_id}] 모든 분석 및 DB 적재 완료 (SUCCESS)")
        else:
            failure_count += 1
            logger.info(f" -> [고객 ID: {c_id}] 일부 분석 실패 감지 (FAILURE)")

    logger.info("==========================================================")
    logger.info("📊 배치 분석 완료 보고서")
    logger.info("==========================================================")
    logger.info(f"- 분석 기준일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"- 총 분석 대상 고객: {len(target_c_ids)}명")
    logger.info(f"- 분석 성공 고객: {success_count}명")
    logger.info(f"- 분석 실패 고객: {failure_count}명")
    logger.info("==========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POOM-AI Batch Customer Analysis Runner")
    parser.add_argument(
        "--c_ids", 
        type=str, 
        help="수동 분석을 수행할 특정 고객 ID 목록 (예: 1001,1002)"
    )
    
    args = parser.parse_args()
    
    # CLI 인코딩 방어 처리
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stdin.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    specified_ids = None
    if args.c_ids:
        specified_ids = [int(i.strip()) for i in args.c_ids.split(",") if i.strip()]

    run_integrated_batch(specified_ids)
