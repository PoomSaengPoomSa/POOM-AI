import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from db import get_db_cursor

def check_results(customer_id):
    print(f"\n==================== [고객 ID: {customer_id}] 결과 확인 ====================")
    
    with get_db_cursor() as cursor:
        # 1. 자산분석 결과 확인 (customer.llm_insight)
        cursor.execute("SELECT name, total_assets, net_worth, llm_insight, analysis_time FROM customer WHERE c_id = %s", (customer_id,))
        cust = cursor.fetchone()
        if cust:
            print(f"[고객명]: {cust['name']} | [총자산]: {cust['total_assets']:,}원 | [순자산]: {cust['net_worth']:,}원")
            print(f"[분석 시각]: {cust['analysis_time']}")
            print(f"[자산 분석 인사이트 (150자 이내)]:\n{cust['llm_insight']}")
            print(f"-> 글자 수 (공백 포함): {len(cust['llm_insight']) if cust['llm_insight'] else 0}")
        else:
            print("고객 정보를 찾을 수 없습니다.")
            
        # 2. 이탈위험 수준 결과 확인 (churn_level)
        cursor.execute("SELECT grade, reason, created_date FROM churn_level WHERE c_id = %s ORDER BY created_date DESC LIMIT 1", (customer_id,))
        churn = cursor.fetchone()
        if churn:
            print(f"\n[이탈 위험 등급]: {churn['grade']}")
            print(f"[판정 사유 (80자 이내)]: {churn['reason']}")
            print(f"-> 글자 수 (공백 포함): {len(churn['reason'])}")
            print(f"[판정 시각]: {churn['created_date']}")
        else:
            print("\n이탈 위험 분석 결과가 없습니다.")
            
        # 3. 상품 매칭 결과 확인 (product_matching)
        cursor.execute("""
            SELECT pm.pd_id, p.name, pm.is_suitable, pm.reason, pm.created_date 
            FROM product_matching pm
            JOIN product p ON pm.pd_id = p.pd_id
            WHERE pm.c_id = %s
            ORDER BY pm.pd_id
        """, (customer_id,))
        matchings = cursor.fetchall()
        if matchings:
            print("\n[주력 금융 상품 매칭]:")
            for m in matchings:
                status_map = {1: "적합(1)", 0: "부적합(0)", 2: "보유 중(2)"}
                status = status_map.get(m['is_suitable'], str(m['is_suitable']))
                print(f"  - [{m['pd_id']}] {m['name']} -> {status}")
                print(f"    * 사유: {m['reason']}")
        else:
            print("\n상품 매칭 결과가 없습니다.")

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
            
    if len(sys.argv) > 1:
        c_ids = [int(i.strip()) for i in sys.argv[1].split(",")]
    else:
        c_ids = [1001, 1004]
        
    for c_id in c_ids:
        check_results(c_id)
