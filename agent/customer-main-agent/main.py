import argparse
import logging
import sys
import os

# Ensure customer-main-agent directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from agent.main_agent import MainAgent

# 로깅 환경 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POOM-AI Batch Customer Analysis Runner")
    parser.add_argument(
        "--c_id", 
        type=str, 
        help="수동 분석을 수행할 특정 고객 ID 목록 (예: 1,2)"
    )
    parser.add_argument(
        "--u_id",
        type=str,
        default="pb_b1_1",
        help="특정 담당 PB의 관리 고객들만 분석하도록 제한하는 PB 사용자 ID (예: pb01)"
    )
    parser.add_argument(
        "--sub1",
        action="store_true",
        help="자산 리밸런싱 인사이트 에이전트(SubAgent 1) 강제 구동 여부"
    )
    parser.add_argument(
        "--sub2",
        action="store_true",
        help="이탈 위험 분석 에이전트(SubAgent 2) 강제 구동 여부"
    )
    parser.add_argument(
        "--sub3",
        action="store_true",
        help="주력 금융 상품 매칭 에이전트(SubAgent 3) 강제 구동 여부"
    )
    
    args = parser.parse_args()
    
    # CLI 인코딩 방어 및 출력 버퍼링 비활성화 (실시간 로그 출력 보장)
    try:
        sys.stdout.reconfigure(encoding='utf-8', write_through=True)
        sys.stdin.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    specified_ids = None
    if args.c_id:
        specified_ids = [int(i.strip()) for i in args.c_id.split(",") if i.strip()]

    # Main Agent 초기화 및 배치 실행 (오케스트레이션 수행)
    main_agent = MainAgent()
    main_agent.run_batch(
        specified_c_ids=specified_ids, 
        u_id=args.u_id,
        force_sub1=args.sub1, 
        force_sub2=args.sub2, 
        force_sub3=args.sub3
    )
