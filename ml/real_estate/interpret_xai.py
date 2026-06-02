import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

def run_interpret():
    # 1. 환경변수 및 기본 경로 설정
    load_dotenv(find_dotenv())
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

    # 2. 예측 및 실제 오차 분석 CSV 로드 (아웃라이어 5건 추출)
    predictions_path = os.path.join(results_dir, 'predictions.csv')
    predictions_text = ""
    if os.path.exists(predictions_path):
        try:
            pred_df = pd.read_csv(predictions_path)
            worst_df = pred_df.sort_values(by='abs_error', ascending=False).head(5)
            predictions_text = "\n\n[예측 오차가 가장 심했던 아웃라이어 시점 (Top 5 Worst)]\n" + worst_df.to_csv(index=False)
        except Exception as e:
            print(f"[ERROR] 예측 오차 CSV 로드 실패: {e}")

    # 3. 중요도 CSV 로드
    importance_path = os.path.join(results_dir, 'feature_importance_regressor.csv')
    importance_text = ""
    if os.path.exists(importance_path):
        try:
            imp_df = pd.read_csv(importance_path)
            importance_text = "\n\n[피처별 글로벌 중요도 순위]\n" + imp_df.to_csv(index=False)
        except Exception as e:
            print(f"[ERROR] 중요도 CSV 로드 실패: {e}")

    # 4. Beeswarm CSV 로드 및 요약
    beeswarm_csv_path = os.path.join(results_dir, 'shap_beeswarm.csv')
    beeswarm_text = ""
    if os.path.exists(beeswarm_csv_path):
        try:
            beeswarm_df = pd.read_csv(beeswarm_csv_path)
            summary_df = beeswarm_df.groupby(['class', 'feature_kr']).apply(
                lambda x: pd.Series({
                    'mean_abs_shap': x['shap_value'].abs().mean(),
                    'corr_feature_shap': x['feature_value'].corr(x['shap_value'])
                })
            ).reset_index().sort_values(by=['class', 'mean_abs_shap'], ascending=[True, False])
            
            beeswarm_text = "\n\n[Beeswarm 분석 (피처별 SHAP 기여도 요약 및 피처값과 SHAP값의 상관계수)]\n" + summary_df.to_csv(index=False)
        except Exception as e:
            print(f"[ERROR] Beeswarm CSV 요약 실패: {e}")

    # 2. OpenAI API 요청 메시지 구성
    print(f"[XAI] OpenAI GPT-4o 로 부동산 XAI 분석 보고서 생성 요청 중...")
    
    client = OpenAI(api_key=api_key)

    user_content_text = (
        "다음은 부동산 가격지수 ML 모델 성능 및 SHAP 분석 결과에서 추출된 정량 데이터입니다:\n" 
        + metrics_text
        + predictions_text
        + importance_text
        + beeswarm_text
        + "\n\n위의 모델 성능 지표, 오차 시점 데이터, 변수별 중요도와 기여도 상관관계 데이터를 종합적으로 참조하여 대한민국 부동산 가격지수의 동역학적 메커니즘을 규명하는 심층적인 XAI 분석 보고서를 작성해 주세요."
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
