import os
import sys

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from agent.simulator.simulator import run_simulation

def main():
    # Ensure Windows console uses UTF-8
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stdin.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    if len(sys.argv) < 2:
        print("=" * 60)
        print("❌ 사용법 안내")
        print("  python agent/simulator/test_simulator.py <customer_id>")
        print("  예시: python agent/simulator/test_simulator.py 1")
        print("=" * 60)
        sys.exit(1)
        
    customer_id = sys.argv[1]
    
    # Load profile details to greet the user
    md_path = os.path.join(current_dir, "data", f"customer_{customer_id}.md")
    
    customer_info = f"고객 ID: {customer_id}"
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            for line in f:
                if "고객명" in line:
                    raw_name = line.replace("-", "").replace("*", "").replace("고객명(등급):", "").replace("고객명", "").replace("등급", "").replace(":", "").strip()
                    customer_info = f"{raw_name} (ID: {customer_id})"
                    break
                    
    print("=" * 60)
    print(f"🤖 POOM-AI PB 상담 시뮬레이터 에이전트 터미널 테스트")
    print(f"  - 로드된 대화 고객: {customer_info}")
    print("  - 종료하려면 'exit' 또는 'quit'을 입력하세요.")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n[PB 입력] > ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                print("시뮬레이터를 종료합니다.")
                break
                
            print("\n[AI 에이전트 분석 중... (지식 DB 검색 및 의도 파악)]")
            
            # Call run_simulation directly
            answer = run_simulation(customer_id, question)
            
            print(f"\n[AI 답변]\n{answer}")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n시뮬레이터를 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ [에러 발생] {e}")
            print("=" * 60)

if __name__ == "__main__":
    main()
