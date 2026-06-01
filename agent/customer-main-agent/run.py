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

    # Main Agent 초기화 및 배치 실행 (오케스트레이션 수행)
    main_agent = MainAgent()
    main_agent.run_batch(specified_ids)
