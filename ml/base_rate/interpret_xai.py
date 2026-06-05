import os
import base64
import glob  # 동적으로 파일을 찾기 위해 추가
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
                VALUES (3, %s, %s, %s)
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

def encode_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def interpret_xai():
    # 1. 환경변수 및 기본 경로 설정
    load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] 오류: .env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    prompt_path = os.path.join(base_dir, 'prompt', 'interpret_prompt.md')

    print("[XAI] 필요한 리소스 읽어오는 중...")

    # 프롬프트 로드
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    # CSV 데이터 준비
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
        print(f"[ERROR] 중요도 CSV 로드 실패: {e}")
        csv_text = "데이터 없음"
        shap_rank_str = "데이터 없음"

    misclass_csv_path = os.path.join(results_dir, 'misclassification_analysis.csv')
    misclass_text = ""
    if os.path.exists(misclass_csv_path):
        try:
            misclass_df = pd.read_csv(misclass_csv_path)
            misclass_text = "\n\n[오분류 케이스 별 SHAP 가중치 (텍스트 요약)]\n" + misclass_df.to_csv(index=False)
        except Exception as e:
            print(f"[ERROR] 오분류 CSV 로드 실패: {e}")

    beeswarm_csv_path = os.path.join(results_dir, 'shap_beeswarm.csv')
    beeswarm_text = ""
    if os.path.exists(beeswarm_csv_path):
        try:
            # 전체를 넘기면 토큰이 길어질 수 있으므로, 각 (클래스, 피처) 별 집계 정보로 줄이거나 상위 데이터만 넘깁니다.
            beeswarm_df = pd.read_csv(beeswarm_csv_path)
            # 데이터를 요약해서 넣습니다 (피처별 평균 SHAP 값, 피처값과 SHAP값의 상관관계 등)
            # 여기서는 프롬프트 길이를 고려해 앞부분 일부 혹은 집계치를 텍스트로 추가합니다.
            summary_df = beeswarm_df.groupby(['class', 'feature_kr']).apply(
                lambda x: pd.Series({
                    'mean_abs_shap': x['shap_value'].abs().mean(),
                    'corr_feature_shap': x['feature_value'].corr(x['shap_value'])
                })
            ).reset_index().sort_values(by=['class', 'mean_abs_shap'], ascending=[True, False])
            
            beeswarm_text = "\n\n[Beeswarm 분석 (클래스 및 피처별 SHAP 기여도 요약 - 피처값과 SHAP값의 상관계수)]\n" + summary_df.to_csv(index=False)
        except Exception as e:
            print(f"[ERROR] Beeswarm CSV 요약 실패: {e}")

    # 2. OpenAI API 요청 메시지 구성
    print(f"[XAI] OpenAI GPT-4o 로 XAI 분석 요청 중 (CSV 데이터만 사용)...")
    
    client = OpenAI(api_key=api_key)

    # Load user_xai_prompt.md dynamically
    user_xai_prompt_path = os.path.join(base_dir, 'prompt', 'user_xai_prompt.md')
    if os.path.exists(user_xai_prompt_path):
        with open(user_xai_prompt_path, 'r', encoding='utf-8') as f:
            user_xai_template = f.read()
    else:
        user_xai_template = (
            "다음은 SHAP 분석 결과에서 추출된 데이터입니다 (이미지 없이 텍스트로 제공됨):\n" 
            + "[1. 상위 15개 중요도 표 및 클래스별 중요도]\n"
            + "{csv_text}" 
            + "{misclass_text}"
            + "{beeswarm_text}"
            + "\n\n위 데이터들을 종합적으로 참고하여 금리 예측 모델이 이 피처들을 어떻게 활용하는지 분석해주세요."
        )

    user_content_text = user_xai_template.format(
        csv_text=csv_text,
        misclass_text=misclass_text,
        beeswarm_text=beeswarm_text
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_content_text}
        ]}
    ]

    # 3. API 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=3000, # 워터폴 분석까지 포함되므로 토큰 여유를 조금 더 줍니다.
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

        # 4. 결과 저장
        output_path = os.path.join(results_dir, 'interpret_result.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_text)

        print(f"\n[OK] 분석 완료! 파일이 저장되었습니다: {output_path}")

        # 5. Fetch predictions and actual rates from MySQL for summary report
        prob_hike = 0.0
        prob_freeze = 0.0
        prob_cut = 0.0
        latest_br_val = None
        
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
                        cursor.execute("SELECT prob_hike, prob_freeze, prob_cut FROM baserate_predictions ORDER BY created_at DESC LIMIT 1")
                        res_pred = cursor.fetchone()
                        if res_pred:
                            prob_hike = float(res_pred[0])
                            prob_freeze = float(res_pred[1])
                            prob_cut = float(res_pred[2])
                            
                        # 2. Fetch latest actual base rate
                        cursor.execute("SELECT value FROM economic_indicator_history WHERE type = 'base_rate' ORDER BY recorded_at DESC LIMIT 1")
                        res_val = cursor.fetchone()
                        if res_val:
                            latest_br_val = float(res_val[0])
                finally:
                    connection.close()
            except Exception as e:
                print(f"[Warning] Failed to fetch base rate values from DB: {e}")

        # 6. Load summary_prompt.md dynamically and generate summary report
        summary_prompt_path = os.path.join(base_dir, 'prompt', 'summary_prompt.md')
        if os.path.exists(summary_prompt_path):
            with open(summary_prompt_path, 'r', encoding='utf-8') as f:
                summary_template = f.read()
        else:
            summary_template = (
                "한국은행 기준금리 AI 예측 모델 분석 결과:\n"
                "- 금리 인하 확률: {prob_cut_pct:.1f}%\n"
                "- 금리 동결 확률: {prob_freeze_pct:.1f}%\n"
                "- 금리 인상 확률: {prob_hike_pct:.1f}%\n"
                "- 최신 실제 기준금리: {latest_br_val_str}\n"
                "위 예측 데이터를 바탕으로 한국어 리포트를 마크다운 형식으로 작성해주세요."
            )
            
        prob_cut_pct = prob_cut * 100.0 if prob_cut <= 1.0 else prob_cut
        prob_freeze_pct = prob_freeze * 100.0 if prob_freeze <= 1.0 else prob_freeze
        prob_hike_pct = prob_hike * 100.0 if prob_hike <= 1.0 else prob_hike
        latest_br_val_str = f"{latest_br_val:.2f}%" if latest_br_val is not None else "데이터 없음"
        
        prompt = summary_template.format(
            prob_cut_pct=prob_cut_pct,
            prob_freeze_pct=prob_freeze_pct,
            prob_hike_pct=prob_hike_pct,
            latest_br_val_str=latest_br_val_str,
            shap_rank_str=shap_rank_str
        )
        
        summary_messages = [
            {"role": "system", "content": "You are a professional economic analyst. Always respond in Korean markdown format. Keep it concise, engaging, and professional."},
            {"role": "user", "content": prompt}
        ]
        
        print(f"[XAI] OpenAI GPT-4o 로 기준금리 요약 보고서 생성 요청 중...")
        response_sum = client.chat.completions.create(
            model="gpt-4o",
            messages=summary_messages,
            temperature=0.7
        )
        summary_text = response_sum.choices[0].message.content
        
        # 7. Save both reports to MySQL DB
        save_report_to_mysql(result_text, summary_text, "base_rate")

    except Exception as e:
        print(f"[ERROR] OpenAI API 호출 중 오류 발생: {e}")

if __name__ == "__main__":
    interpret_xai()