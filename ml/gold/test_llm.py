"""
LLM 보고서 생성 단독 테스트 스크립트
GENERATE_REPORT = True 상태에서 generate_and_save_gold_report() 직접 실행
"""
import os
import sys

# GENERATE_REPORT 강제 True
import train as t
t.GENERATE_REPORT = True

print("=" * 50)
print("LLM 보고서 생성 단독 테스트")
print("=" * 50)

# 직접 함수 호출 (실제 prob 값은 임의)
t.generate_and_save_gold_report(prob_rise=0.65, prob_fall=0.35, run_id="test_debug_run")

print("=" * 50)
print("테스트 완료")
