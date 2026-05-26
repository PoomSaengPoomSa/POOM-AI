# [XAI 리포트] LLM 경제 분석 및 해석 가이드

> [!NOTE]
> OpenAI API Key가 설정되지 않았거나 `openai` 라이브러리가 미설치되어 머신러닝 모델 가중치에 대한 LLM XAI 분석 보고서 생성이 생략되었습니다.

## 분석 보고서 활성화 방법
금값 예측 모델의 SHAP XAI 데이터 경제학적 해석 리포트를 자동으로 받아보려면 아래 단계를 완료하세요:

1. **OpenAI 라이브러리 설치**:
   현재 Python 가상환경(`c:/ITStudy/poom/.venv`)에서 다음 명령어를 실행하세요:
   ```bash
   c:/ITStudy/poom/.venv/Scripts/pip.exe install openai
   ```

2. **OpenAI API Key 추가**:
   `c:/ITStudy/poom/ai/.env` 파일에 아래 환경 변수를 정의하고 발급받은 API Key 값을 지정하세요:
   ```env
   OPENAI_API_KEY=sk-proj-...
   ```

3. **파이프라인 재실행**:
   `c:/ITStudy/poom/.venv/Scripts/python.exe gold/run.py`를 실행하면 모델 결과 분석을 바탕으로 AI 경제학자가 작성한 종합 해석 리포트(`gold/results/interpret_result.md`)가 즉시 자동 생성됩니다.
