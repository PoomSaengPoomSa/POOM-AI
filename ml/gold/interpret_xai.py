import os
import base64
import pandas as pd
import pymysql
from dotenv import load_dotenv, find_dotenv

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

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
                VALUES (1, %s, %s, %s)
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

def interpret_xai():
    # 1. Environment & Path Setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(find_dotenv())
    
    api_key = os.getenv("OPENAI_API_KEY")
    results_dir = os.path.join(base_dir, 'results')
    
    if not api_key or not HAS_OPENAI:
        if not api_key:
            print("[XAI] [Warning] OpenAI API Key is missing in .env file. Skipping LLM interpretation.")
        else:
            print("[XAI] [Warning] 'openai' library is not installed in the python environment. Skipping LLM interpretation.")
            
        # Write a friendly user guide into interpret_result.md
        output_path = os.path.join(results_dir, 'interpret_result.md')
        os.makedirs(results_dir, exist_ok=True)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# [XAI 리포트] LLM 경제 분석 및 해석 가이드\n\n")
                f.write("> [!NOTE]\n")
                f.write("> OpenAI API Key가 설정되지 않았거나 `openai` 라이브러리가 미설치되어 머신러닝 모델 가중치에 대한 LLM XAI 분석 보고서 생성이 생략되었습니다.\n\n")
                f.write("## 분석 보고서 활성화 방법\n")
                f.write("금값 예측 모델의 SHAP XAI 데이터 경제학적 해석 리포트를 자동으로 받아보려면 아래 단계를 완료하세요:\n\n")
                f.write("1. **OpenAI 라이브러리 설치**:\n")
                f.write("   현재 Python 가상환경(`c:/ITStudy/poom/.venv`)에서 다음 명령어를 실행하세요:\n")
                f.write("   ```bash\n")
                f.write("   c:/ITStudy/poom/.venv/Scripts/pip.exe install openai\n")
                f.write("   ```\n\n")
                f.write("2. **OpenAI API Key 추가**:\n")
                f.write("   `c:/ITStudy/poom/ai/.env` 파일에 아래 환경 변수를 정의하고 발급받은 API Key 값을 지정하세요:\n")
                f.write("   ```env\n")
                f.write("   OPENAI_API_KEY=sk-proj-...\n")
                f.write("   ```\n\n")
                f.write("3. **파이프라인 재실행**:\n")
                f.write("   `c:/ITStudy/poom/.venv/Scripts/python.exe gold/run.py`를 실행하면 모델 결과 분석을 바탕으로 AI 경제학자가 작성한 종합 해석 리포트(`gold/results/interpret_result.md`)가 즉시 자동 생성됩니다.\n")
            print(f"[XAI] Created LLM interpretation setup guide at: {output_path}")
        except Exception as e:
            print(f"[Warning] Failed to write guide markdown: {e}")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    prompt_path = os.path.join(base_dir, 'prompt', 'interpret_prompt.md')

    print("[XAI] Reading resources for OpenAI GPT-4o analysis...")

    # Load System Prompt
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    # Load Feature Importance CSV
    csv_path = os.path.join(results_dir, 'feature_importance_classifier.csv')
    try:
        df = pd.read_csv(csv_path)
        csv_text = df.head(15).to_csv(index=False)
        # 동적 SHAP 기여도 순위 문자열 생성 (Top 4)
        top4 = df.head(4)
        total_imp = top4['importance'].sum()
        shap_rank_str = ", ".join(
            f"{row['feature_kr']} ({row['feature']}, {int(round(row['importance'] / total_imp * 100))}%)"
            for _, row in top4.iterrows()
        )
    except Exception as e:
        print(f"[ERROR] Failed to load feature importance CSV: {e}")
        csv_text = "데이터 없음"
        shap_rank_str = "데이터 없음"

    # Load Misclassification Analysis CSV
    misclass_csv_path = os.path.join(results_dir, 'misclassification_analysis.csv')
    misclass_text = ""
    if os.path.exists(misclass_csv_path):
        try:
            misclass_df = pd.read_csv(misclass_csv_path)
            # Select first 5 cases to avoid bloating prompt tokens
            misclass_text = "\n\n[대표 오분류 케이스 별 SHAP 가중치 (텍스트 요약)]\n" + misclass_df.head(5).to_csv(index=False)
        except Exception as e:
            print(f"[ERROR] Failed to load misclassification CSV: {e}")

    # Load Beeswarm coordinates CSV
    beeswarm_csv_path = os.path.join(results_dir, 'shap_beeswarm.csv')
    beeswarm_text = ""
    if os.path.exists(beeswarm_csv_path):
        try:
            beeswarm_df = pd.read_csv(beeswarm_csv_path)
            # Aggregate correlation and absolute mean SHAP values for cleaner LLM injection
            summary_df = beeswarm_df.groupby('feature_kr').apply(
                lambda x: pd.Series({
                    'mean_abs_shap': x['shap_value'].abs().mean(),
                    'corr_feature_shap': x['feature_value'].corr(x['shap_value'])
                })
            ).reset_index().sort_values(by='mean_abs_shap', ascending=False)
            
            beeswarm_text = "\n\n[Beeswarm 분석 (피처별 SHAP 기여도 요약 - 피처값과 SHAP값의 상관계수)]\n" + summary_df.to_csv(index=False)
        except Exception as e:
            print(f"[ERROR] Failed to summarize Beeswarm CSV: {e}")

    # 2. Construct API Payload
    print(f"[XAI] Requesting economic interpretation from OpenAI GPT-4o...")
    
    client = OpenAI(api_key=api_key)

    # Load user_xai_prompt.md dynamically
    user_xai_prompt_path = os.path.join(base_dir, 'prompt', 'user_xai_prompt.md')
    if os.path.exists(user_xai_prompt_path):
        with open(user_xai_prompt_path, 'r', encoding='utf-8') as f:
            user_xai_template = f.read()
    else:
        user_xai_template = (
            "다음은 금값 상승/하락 예측 모델에 대한 SHAP 분석 결과에서 추출된 가중치 요약 데이터입니다:\n" 
            + "[1. 상위 15개 중요 피처 목록]\n"
            + "{csv_text}" 
            + "{misclass_text}"
            + "{beeswarm_text}"
            + "\n\n위 데이터들을 종합적으로 참고하여 금값 예측 모델이 어떠한 금융 시장 패러다임과 동학에 근거해 동작하는지 경제학적 관점으로 분석해주세요."
        )

    user_content_text = user_xai_template.format(
        csv_text=csv_text,
        misclass_text=misclass_text,
        beeswarm_text=beeswarm_text
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_content_text}]}
    ]

    # 3. Call API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=3000,
            temperature=0.3
        )
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```"):
            lines = result_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            result_text = "\n".join(lines).strip()

        # 4. Save results to markdown
        output_path = os.path.join(results_dir, 'interpret_result.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_text)

        print(f"\n[OK] GPT-4o analysis completed successfully! Saved to: {output_path}")

        # 5. Fetch predictions and actual rates from MySQL for summary report
        prob_rise = 0.0
        prob_fall = 0.0
        latest_gold_val = None
        
        DB_USER = os.getenv('DB_USER')
        DB_PASSWORD = os.getenv('DB_PASSWORD')
        DB_HOST = os.getenv('DB_HOST')
        DB_PORT = os.getenv('DB_PORT')
        DB_NAME = os.getenv('DB_NAME')
        
        if all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
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
                        cursor.execute("SELECT prob_rise, prob_fall FROM gold_predictions ORDER BY created_at DESC LIMIT 1")
                        res_pred = cursor.fetchone()
                        if res_pred:
                            prob_rise = float(res_pred[0])
                            prob_fall = float(res_pred[1])
                            
                        # 2. Fetch latest actual gold price
                        cursor.execute("SELECT value FROM economic_indicator_history WHERE type = 'gold' ORDER BY recorded_at DESC LIMIT 1")
                        res_val = cursor.fetchone()
                        if res_val:
                            latest_gold_val = float(res_val[0])
                finally:
                    connection.close()
            except Exception as e:
                print(f"[Warning] Failed to fetch gold values from DB: {e}")

        # 6. Load summary_prompt.md dynamically and generate summary report
        summary_prompt_path = os.path.join(base_dir, 'prompt', 'summary_prompt.md')
        if os.path.exists(summary_prompt_path):
            with open(summary_prompt_path, 'r', encoding='utf-8') as f:
                summary_template = f.read()
        else:
            summary_template = (
                "금값 AI 예측 모델 분석 결과:\n"
                "- 내일 상승 확률: {prob_rise_pct:.1f}%\n"
                "- 내일 하락 확률: {prob_fall_pct:.1f}%\n"
                "- 최신 금값 실제 가격: {latest_gold_val_str}\n"
                "위 예측 데이터를 바탕으로 한국어 리포트를 마크다운 형식으로 작성해주세요."
            )
            
        prob_rise_pct = prob_rise * 100.0 if prob_rise <= 1.0 else prob_rise
        prob_fall_pct = prob_fall * 100.0 if prob_fall <= 1.0 else prob_fall
        latest_gold_val_str = f"{latest_gold_val:,.2f}" if latest_gold_val is not None else "데이터 없음"
        
        prompt = summary_template.format(
            prob_rise_pct=prob_rise_pct,
            prob_fall_pct=prob_fall_pct,
            latest_gold_val_str=latest_gold_val_str,
            shap_rank_str=shap_rank_str
        )
        
        summary_messages = [
            {"role": "system", "content": "You are a professional economic analyst. Always respond in Korean markdown format. Keep it concise, engaging, and professional."},
            {"role": "user", "content": prompt}
        ]
        
        print(f"[XAI] OpenAI GPT-4o 로 금값 요약 보고서 생성 요청 중...")
        response_sum = client.chat.completions.create(
            model="gpt-4o",
            messages=summary_messages,
            temperature=0.7
        )
        summary_text = response_sum.choices[0].message.content
        
        # 7. Save both reports to MySQL DB
        save_report_to_mysql(result_text, summary_text, "gold")

    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")

if __name__ == "__main__":
    interpret_xai()
