import os
import base64
import pandas as pd
from dotenv import load_dotenv

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

def interpret_xai():
    # 1. Environment & Path Setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.abspath(os.path.join(base_dir, '../../.env'))
    load_dotenv(dotenv_path=env_path)
    
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
    except Exception as e:
        print(f"[ERROR] Failed to load feature importance CSV: {e}")
        csv_text = "데이터 없음"

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

    user_content_text = (
        "다음은 금값 상승/하락 예측 모델에 대한 SHAP 분석 결과에서 추출된 가중치 요약 데이터입니다:\n" 
        + "[1. 상위 15개 중요 피처 목록]\n"
        + csv_text 
        + misclass_text
        + beeswarm_text
        + "\n\n위 데이터들을 종합적으로 참고하여 금값 예측 모델이 어떠한 금융 시장 패러다임과 동학에 근거해 동작하는지 경제학적 관점으로 분석해주세요."
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
        result_text = response.choices[0].message.content

        # 4. Save results to markdown
        output_path = os.path.join(results_dir, 'interpret_result.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_text)

        print(f"\n[OK] GPT-4o analysis completed successfully! Saved to: {output_path}")

    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")

if __name__ == "__main__":
    interpret_xai()
