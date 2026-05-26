import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

def run_interpret():
    # 1. 환경변수 및 기본 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # back 폴더의 .env를 직접 경유하여 동기화 (3레벨 위 부모 디렉토리인 POOM 기준)
    back_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../back/.env'))
    load_dotenv(dotenv_path=back_env_path)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] 오류: .env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    prompt_path = os.path.join(base_dir, 'prompt', 'interpret_prompt.md')

    if not os.path.exists(prompt_path):
        print(f"[ERROR] 오류: 프롬프트 파일이 존재하지 않습니다: {prompt_path}")
        return

    print("[XAI] 필요한 리소스 읽어오는 중...")

    # 프롬프트 로드
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    # 1. 모델 평가지표 CSV 로드
    metrics_path = os.path.join(results_dir, 'evaluation_metrics.csv')
    metrics_text = "데이터 없음"
    if os.path.exists(metrics_path):
        try:
            metrics_df = pd.read_csv(metrics_path)
            metrics_text = "\n[모델별 평가지표 비교 (MAE, RMSE, R2)]\n" + metrics_df.to_csv(index=False)
        except Exception as e:
            print(f"[ERROR] 평가지표 CSV 로드 실패: {e}")

    # 2. 예측 및 실제 오차 분석 CSV 로드
    predictions_path = os.path.join(results_dir, 'predictions.csv')
    predictions_text = ""
    if os.path.exists(predictions_path):
        try:
            pred_df = pd.read_csv(predictions_path)
            # 최대 오차가 발생한 아웃라이어 월 분석용 상위 5건 추출
            worst_df = pred_df.sort_values(by='abs_error_ensemble', ascending=False).head(5)
            predictions_text = "\n\n[예측 오차가 가장 심했던 아웃라이어 시점 (Top 5 Worst)]\n" + worst_df.to_csv(index=False)
        except Exception as e:
            print(f"[ERROR] 예측 오차 CSV 로드 실패: {e}")

    # 3. SHAP 기여도 CSV 로드 및 평균 기여도 요약
    shap_path = os.path.join(results_dir, 'shap_values.csv')
    shap_text = ""
    if os.path.exists(shap_path):
        try:
            shap_df = pd.read_csv(shap_path)
            shap_cols = [c for c in shap_df.columns if c != 'date_ym']
            mean_abs_shap = shap_df[shap_cols].abs().mean().sort_values(ascending=False)
            
            ko_names = {
                'house_price_idx': '이번달 매매가격지수',
                'kr_cpi': '한국 소비자물가지수 (CPI)',
                'kr_unemployment': '한국 실업률',
                'kr_base_rate': '한국 기준금리',
                'kr_mortgage_rate': '주택담보대출 금리',
                'kospi200': 'KOSPI200 지수',
                'apt_trade_count': '아파트 거래량',
                'kr_m2': '한국 M2 통화량',
                'buyer_dominance': '매수우위지수'
            }
            
            summary_list = []
            for col, val in mean_abs_shap.items():
                original_feat = col.replace('shap_', '')
                k_name = ko_names.get(original_feat, original_feat)
                summary_list.append({
                    'feature': original_feat,
                    'feature_kr': k_name,
                    'mean_abs_shap': val
                })
            
            summary_df = pd.DataFrame(summary_list)
            shap_text = "\n\n[부동산 모델 피처별 SHAP 글로벌 기여도 평균 절대값 순위]\n" + summary_df.to_csv(index=False)
        except Exception as e:
            print(f"[ERROR] SHAP CSV 로드 및 요약 실패: {e}")

    # 2. OpenAI API 요청 메시지 구성
    print(f"[XAI] OpenAI GPT-4o 로 부동산 XAI 분석 보고서 생성 요청 중...")
    
    client = OpenAI(api_key=api_key)

    user_content_text = (
        "다음은 부동산 가격지수 ML 모델 성능 및 SHAP 분석 결과에서 추출된 정량 데이터입니다:\n" 
        + metrics_text
        + predictions_text
        + shap_text
        + "\n\n위의 모델 성능 표와 변수별 SHAP 기여도 순위, 최대 오차 시점 데이터를 종합적으로 참조하여 대한민국 부동산 가격지수의 동역학적 메커니즘을 규명하는 심층적인 XAI 분석 보고서를 작성해 주세요."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_content_text}
        ]}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=3000,
            temperature=0.3
        )
        result_text = response.choices[0].message.content

        output_path = os.path.join(results_dir, 'interpret_result.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_text)

        print(f"\n[OK] 분석 완료! 파일이 성공적으로 저장되었습니다: {output_path}")

    except Exception as e:
        print(f"[ERROR] OpenAI API 호출 중 오류 발생: {e}")

if __name__ == '__main__':
    run_interpret()
