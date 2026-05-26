import os
import base64
import glob  # 동적으로 파일을 찾기 위해 추가
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

def encode_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def interpret_xai():
    # 1. 환경변수 및 기본 경로 설정
    load_dotenv()
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
    except Exception as e:
        print(f"[ERROR] 중요도 CSV 로드 실패: {e}")
        csv_text = "데이터 없음"

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

    # 시각화 이미지 없이 CSV 데이터만 종합적으로 참조하도록 지시
    user_content_text = (
        "다음은 SHAP 분석 결과에서 추출된 데이터입니다 (이미지 없이 텍스트로 제공됨):\n" 
        + "[1. 상위 15개 중요도 표 및 클래스별 중요도]\n"
        + csv_text 
        + misclass_text
        + beeswarm_text
        + "\n\n위 데이터들을 종합적으로 참고하여 금리 예측 모델이 이 피처들을 어떻게 활용하는지 분석해주세요."
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
        result_text = response.choices[0].message.content

        # 4. 결과 저장
        output_path = os.path.join(results_dir, 'interpret_result.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_text)

        print(f"\n[OK] 분석 완료! 파일이 저장되었습니다: {output_path}")

    except Exception as e:
        print(f"[ERROR] OpenAI API 호출 중 오류 발생: {e}")

if __name__ == "__main__":
    interpret_xai()