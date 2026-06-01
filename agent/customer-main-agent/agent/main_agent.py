import logging
import os
import sys
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

# 서브 에이전트 및 DB 임포트
from agent.asset_insight_agent import AssetInsightAgent
from agent.churn_risk_agent import ChurnRiskAgent
from agent.product_matching_agent import ProductMatchingAgent
from tool import tools

logger = logging.getLogger("IntegratedCustomerAgent")

DEFAULT_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 프롬프트 동적 로드 헬퍼 함수
def load_prompt(filename: str) -> str:
    """
    Utility to load prompt templates from the local prompt directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(os.path.dirname(current_dir), "prompt", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# 1. Pydantic 라우팅 구조 정의
class SubAgentRouting(BaseModel):
    run_asset_insight: bool = Field(description="자산 리밸런싱 인사이트 에이전트(Sub Agent 1) 구동 여부. 고객 정보(자산 등)가 수정된 이후에 AI 분석이 없었거나, 1억 이상 우량 고객인 경우 등.")
    run_churn_risk: bool = Field(description="이탈 위험 수준 분석 에이전트(Sub Agent 2) 구동 여부. 최근 거액 거래(출금)가 발생했거나, 이탈 징후가 있을 때.")
    run_product_matching: bool = Field(description="주력 금융 상품 적합성 평가 에이전트(Sub Agent 3) 구동 여부. 신규 상담 기록이 있고 추천 가능한 상품 매칭이 필요할 때. 단, 최근 상담 보고서 존재 여부가 False이면 무조건 False여야 함.")
    reason: str = Field(description="각 서브 에이전트 선택 여부에 대한 분석적 판단 근거 (한 문장)")

class MainAgent:
    """
    POOM-AI 차세대 고객 분석 통합 Main Agent
    Orchestrates SubAgent execution dynamically using an LLM Router.
    """
    def __init__(self, model_name: str = None):
        global DEFAULT_MODEL
        if model_name:
            DEFAULT_MODEL = model_name
            
        self.sub1 = AssetInsightAgent(model_name=model_name)
        self.sub2 = ChurnRiskAgent(model_name=model_name)
        self.sub3 = ProductMatchingAgent(model_name=model_name)

    @traceable(name="MainAgent.run_for_customer", run_type="chain")
    def run_for_customer(self, customer_id: int) -> dict:
        """
        특정 고객 ID에 대해 dynamic routing을 판단하여 선택적으로 서브 에이전트를 호출합니다.
        """
        results = {
            "c_id": customer_id,
            "routing_reason": "",
            "sub1_called": False, "sub1_success": True,
            "sub2_called": False, "sub2_success": True,
            "sub3_called": False, "sub3_success": True
        }

        logger.info(f" -> [Main Router] 고객 {customer_id}번 dynamic routing 분석 시작...")

        # 1. 고객 현황 및 거래/상담 정보 수집
        try:
            portfolio = tools.get_portfolio_weight(customer_id)
            if not portfolio:
                raise ValueError(f"Customer with ID {customer_id} not found in database.")
            
            # 최근 7일간 타행 출금(1천만원 이상) 거래 수집
            large_withdrawals = tools.get_large_external_transactions(customer_id, threshold_amount=10000000.0)
            withdrawals_list = []
            for tx in large_withdrawals:
                # get_large_external_transactions는 이미 ct_type='W'인 것만 반환함
                withdrawals_list.append(
                    f"- {tx['ct_datetime'].strftime('%Y-%m-%d')}: {tx['amount']:,}원 (상대행: {tx['opp_bank_name']}, 잔액: {tx['balance_after']:,}원)"
                )
            large_withdrawals_str = "\n".join(withdrawals_list) if withdrawals_list else "감지된 타행 거액 이출금 내역 없음."

            # 상담 보고서 존재 여부 파악
            recent_report = tools.get_recent_consultation_report(customer_id)
            has_report = "있음 (True)" if recent_report else "없음 (False)"
            
        except Exception as e:
            logger.error(f" -> [Main Router ERROR] 고객 {customer_id} 정보 수집 실패: {e}")
            results["routing_reason"] = f"정보 수집 오류: {e}"
            results["sub1_success"] = results["sub2_success"] = results["sub3_success"] = False
            return results

        # 2. LLM 라우터 구동을 위한 프롬프트 바인딩 및 의사결정
        try:
            system_prompt = load_prompt("main_agent_router_system.md")
            user_prompt_template = load_prompt("main_agent_router_user.md")

            user_prompt = user_prompt_template.format(
                name=portfolio["name"],
                grade=portfolio["grade"],
                tendency=portfolio["tendency"],
                total_assets=portfolio["total_assets"],
                deposit=portfolio["deposit"],
                investment=portfolio["investment"],
                pension=portfolio["pension"],
                loan=portfolio["loan"],
                net_worth=portfolio["net_worth"],
                large_withdrawals_str=large_withdrawals_str,
                has_consultation_report=has_report
            )

            llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.1, api_key=OPENAI_API_KEY)
            structured_llm = llm.with_structured_output(SubAgentRouting)
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "{user_content}")
            ])
            chain = prompt | structured_llm
            routing: SubAgentRouting = chain.invoke({"user_content": user_prompt})
            
            logger.info(f" -> [Main Router 결정] 자산분석={routing.run_asset_insight}, 이탈분석={routing.run_churn_risk}, 상품매칭={routing.run_product_matching}")
            logger.info(f" -> [Main Router 사유] {routing.reason}")
            results["routing_reason"] = routing.reason

        except Exception as e:
            logger.error(f" -> [Main Router LLM ERROR] 라우터 의사결정 실패: {e}")
            results["routing_reason"] = f"라우터 구동 오류: {e}"
            results["sub1_success"] = results["sub2_success"] = results["sub3_success"] = False
            return results

        # 3. 판정에 따른 선택적 서브 에이전트 샌드박스 구동
        
        # 3-1. Sub Agent 1: 자산 보유 현황 분석
        if routing.run_asset_insight:
            results["sub1_called"] = True
            try:
                logger.info(f"   -> [Sub Agent 1] 자산 리밸런싱 인사이트 분석 시작...")
                self.sub1.run(customer_id)
                results["sub1_success"] = True
            except Exception as e:
                logger.error(f"   -> [Sub Agent 1 ERROR] 고객 {customer_id} 분석 중 오류: {e}")
                results["sub1_success"] = False
        else:
            logger.info(f"   -> [Sub Agent 1 SKIP] 라우터의 배제 판단으로 구동을 건너뜁니다.")

        # 3-2. Sub Agent 2: 이탈 위험 수준 분석
        if routing.run_churn_risk:
            results["sub2_called"] = True
            try:
                logger.info(f"   -> [Sub Agent 2] 이탈 위험 등급 평가 시작...")
                self.sub2.run(customer_id)
                results["sub2_success"] = True
            except Exception as e:
                logger.error(f"   -> [Sub Agent 2 ERROR] 고객 {customer_id} 분석 중 오류: {e}")
                results["sub2_success"] = False
        else:
            logger.info(f"   -> [Sub Agent 2 SKIP] 라우터의 배제 판단으로 구동을 건너뜁니다.")

        # (이전 Sub Agent 3 상담 특징 추출 부분은 제거되었습니다.)

        # 3-3. Sub Agent 3: 주력 금융 상품 적합성 평가
        if routing.run_product_matching:
            results["sub3_called"] = True
            try:
                logger.info(f"   -> [Sub Agent 3] 주력 금융 상품 적합성 평가 시작...")
                self.sub3.run(customer_id)
                results["sub3_success"] = True
            except Exception as e:
                logger.error(f"   -> [Sub Agent 3 ERROR] 고객 {customer_id} 상품 적합성 평가 중 오류: {e}")
                results["sub3_success"] = False
        else:
            logger.info(f"   -> [Sub Agent 3 SKIP] 라우터의 배제 판단으로 구동을 건너뜁니다.")

        return results
