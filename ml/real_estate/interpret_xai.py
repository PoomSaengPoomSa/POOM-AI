import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

def save_report_to_mysql(content, summary, report_type):
    import pymysql
    
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("[Warning] Missing DB credentials. Skipping DB save for LLM report.")
        return
        
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=int(DB_PORT),
            charset='utf8mb4'
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS trend_llm_report (
                    report_id INT AUTO_INCREMENT PRIMARY KEY,
                    type VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                sql = """
                INSERT INTO trend_llm_report (report_id, type, content, summary)
                VALUES (2, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    content = VALUES(content),
                    summary = VALUES(summary),
                    created_at = CURRENT_TIMESTAMP
                """
                cursor.execute(sql, (report_type, content, summary))
            connection.commit()
            print(f"[DB] Successfully saved {report_type} XAI report and summary to MySQL trend_llm_report table.")
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to save {report_type} XAI report to MySQL: {e}")

def run_interpret(valid_mode=False):
    # 1. 환경변수 및 기본 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 로컬 .env 또는 상위 폴더 탐색을 통한 통합 .env 로드
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

    eval_name = "Validation" if valid_mode else "Test"

    # 1. 모델 평가지표 CSV 로드
    metrics_path = os.path.join(results_dir, f'{eval_name.lower()}_metrics.csv')
    metrics_text = "데이터 없음"
    if os.path.exists(metrics_path):
        try:
            metrics_df = pd.read_csv(metrics_path)
            metrics_text = f"\n[{eval_name} 모델별 평가지표 비교 (MAE, RMSE, R2)]\n" + metrics_df.to_csv(index=False)
        except Exception as e:
            print(f"[ERROR] 평가지표 CSV 로드 실패: {e}")

    # 2. 예측 및 실제 오차 분석 CSV 로드
    predictions_path = os.path.join(results_dir, f'{eval_name.lower()}_predictions.csv')
    predictions_text = ""
    if os.path.exists(predictions_path):
        try:
            pred_df = pd.read_csv(predictions_path)
            # 최대 오차가 발생한 아웃라이어 월 분석용 상위 5건 추출
            worst_df = pred_df.sort_values(by='abs_error_ensemble', ascending=False).head(5)
            predictions_text = f"\n\n[{eval_name} 예측 오차가 가장 심했던 아웃라이어 시점 (Top 5 Worst)]\n" + worst_df.to_csv(index=False)
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
            # 동적 SHAP 기여도 순위 문자열 생성 (Top 4)
            top4 = summary_df.head(4)
            total_imp = top4['mean_abs_shap'].sum()
            shap_rank_str = ", ".join(
                f"{row['feature_kr']} ({row['feature']}, {int(round(row['mean_abs_shap'] / total_imp * 100))}%)"
                for _, row in top4.iterrows()
            )
        except Exception as e:
            print(f"[ERROR] SHAP CSV 로드 및 요약 실패: {e}")
            shap_rank_str = "데이터 없음"
    else:
        shap_rank_str = "데이터 없음"

    # 2. OpenAI API 요청 메시지 구성
    print(f"[XAI] OpenAI GPT-4o 로 부동산 XAI 분석 보고서 생성 요청 중...")
    
    client = OpenAI(api_key=api_key)

    user_content_text = (
        f"다음은 부동산 가격지수 ML 모델 성능 및 SHAP 분석 결과에서 추출된 정량 데이터 ({eval_name} 세트)입니다:\n" 
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
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=3000,
            #temperature=0.3
        )
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```"):
            lines = result_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            result_text = "\n".join(lines).strip()

        output_path = os.path.join(results_dir, 'interpret_result.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_text)

        print(f"\n[OK] 분석 완료! 파일이 성공적으로 저장되었습니다: {output_path}")
        
        # 5. Fetch predictions and actual rates from MySQL for summary report
        predicted_value = 0.0
        predicted_index = None
        re_today = None
        
        DB_USER = os.getenv('DB_USER')
        DB_PASSWORD = os.getenv('DB_PASSWORD')
        DB_HOST = os.getenv('DB_HOST')
        DB_PORT = os.getenv('DB_PORT')
        DB_NAME = os.getenv('DB_NAME')
        
        if all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
            import pymysql
            try:
                connection = pymysql.connect(
                    host=DB_HOST,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    database=DB_NAME,
                    port=int(DB_PORT),
                    charset='utf8mb4'
                )
                try:
                    with connection.cursor() as cursor:
                        # 1. Fetch latest predictions
                        cursor.execute("SELECT predicted_value, predicted_index FROM realestate_predictions ORDER BY created_at DESC LIMIT 1")
                        res_pred = cursor.fetchone()
                        if res_pred:
                            predicted_value = float(res_pred[0])
                            if res_pred[1] is not None:
                                predicted_index = float(res_pred[1])
                            
                        # 2. Fetch latest actual realestate index
                        cursor.execute("SELECT house_price_idx FROM ml_realestate_preprocessed ORDER BY date_ym DESC LIMIT 1")
                        res_val = cursor.fetchone()
                        if res_val and res_val[0] is not None:
                            re_today = float(res_val[0])
                finally:
                    connection.close()
            except Exception as e:
                print(f"[Warning] Failed to fetch realestate values from DB: {e}")

        # 6. Load summary_prompt.md dynamically and generate summary report
        summary_prompt_path = os.path.join(base_dir, 'prompt', 'summary_prompt.md')
        if os.path.exists(summary_prompt_path):
            with open(summary_prompt_path, 'r', encoding='utf-8') as f:
                summary_template = f.read()
        else:
            summary_template = (
                "부동산 가격지수 AI 예측 모델 분석 결과:\n"
                "- 이번달 실제 가격지수(re_today): {latest_re_val_str}\n"
                "- 다음달 예측 변동률: {predicted_value:.2f}%\n"
                "- 다음달 예측 환산 가격지수(predicted_index): {predicted_index_str}\n"
                "위 예측 데이터를 바탕으로 한국어 리포트를 마크다운 형식으로 작성해주세요."
            )
            
        latest_re_val_str = f"{re_today:.2f}" if re_today is not None else "데이터 없음"
        predicted_index_str = f"{predicted_index:.2f}" if predicted_index is not None else "데이터 없음"
        
        prompt = summary_template.format(
            latest_re_val_str=latest_re_val_str,
            predicted_value=predicted_value,
            predicted_index_str=predicted_index_str,
            shap_rank_str=shap_rank_str
        )
        
        summary_messages = [
            {"role": "system", "content": "You are a professional economic analyst. Always respond in Korean markdown format. Keep it concise, engaging, and professional."},
            {"role": "user", "content": prompt}
        ]
        
        print(f"[XAI] OpenAI gpt-4o-mini 로 부동산 요약 보고서 생성 요청 중...")
        response_sum = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=summary_messages,
            temperature=0.7
        )
        summary_text = response_sum.choices[0].message.content
        
        # 7. Save both reports to MySQL DB
        save_report_to_mysql(result_text, summary_text, "real_estate")

    except Exception as e:
        print(f"[ERROR] OpenAI API 호출 중 오류 발생: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid', action='store_true', help='Validation mode')
    args = parser.parse_known_args()[0]
    run_interpret(valid_mode=args.valid)
