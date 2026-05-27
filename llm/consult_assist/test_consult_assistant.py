import json
import sys
from consult_assistant import structure_consultation_memo

# Windows 콘솔 한글 깨짐 방지 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # python 버전에 따라 reconfigure가 없을 수 있으므로 예외처리
        pass

def run_test():
    test_memo = (
        "강준혁 고객님께서 지점에 내방하셔서 정기 상담을 나누었다. "
        "최근 하반기에 예정된 사옥 이전 건 때문에 현금 유동성을 일정 부분 묶이지 않고 확보해 두는 쪽에 신경을 엄청 쓰고 계신 상황이다. "
        "또한 가족 얘기가 나왔는데, 내년에 아들(27세)이 대학원에 진학하거나 해외 유학을 갈 예정이라 교육비나 정착 자금으로 목돈이 나갈 일이 생겨서 고민이 많으시다고 하신다. "
        "예금 이외에 단기로 굴리면서 리스크가 없는 곳을 원하셔서 골드 바 매입이나 하루만 맡겨도 이자가 나오는 파킹 통장 상품에 큰 관심을 보이셨다. "
        "보유하고 계신 기존 펀드 외에, 나중에 아들에게 자산을 합법적으로 증여하고 상속하기에 괜찮은 장기 연금보험이 있는지 문의하셨다. "
        "다음 달 즈음에 아드님이랑 같이 지점에 다시 와서 본격적인 증여 세무 상담도 받고 상품도 하나 신규 가입하시겠다고 예약하고 돌아가셨다."
    )
    
    print("=" * 60)
    print("1. 입력 상담 메모:")
    print("-" * 60)
    print(test_memo)
    print("=" * 60)
    
    print("\n[AI 상담 보고서 구조화 진행 중...]\n")
    
    try:
        report = structure_consultation_memo(test_memo)
        
        print("=" * 60)
        print("2. 구조화 결과 (AI 상담 보고서):")
        print("-" * 60)
        
        print("\n■ 주요 내용 (Key Contents)")
        for idx, item in enumerate(report.key_contents, 1):
            print(f"  {idx}. {item}")
            
        print("\n■ 특이사항 (Special Notes)")
        for idx, item in enumerate(report.special_notes, 1):
            print(f"  {idx}. {item}")
            
        print("\n■ 후속 조치 (Follow-up Actions)")
        for idx, item in enumerate(report.follow_up_actions, 1):
            print(f"  {idx}. {item}")
            
        print("=" * 60)
        
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    run_test()
