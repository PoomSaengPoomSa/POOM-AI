# 🤖 POOM-AI

> **POOM** — PB(Private Banker) 업무 지원 AI Assistant 플랫폼의 AI 서버

---

## 📌 개요

POOM 플랫폼의 AI 서버 레포지토리입니다.
LangGraph 기반 멀티 에이전트, LLM 파이프라인, 경제지표 머신러닝 예측 모델을 담당하며
FastAPI를 통해 백엔드 서버(POOM-BACK)와 통신합니다.

---

## 🗂 프로젝트 구조
POOM-AI/

├── agent/          # LangGraph 멀티 에이전트

├── app/            # FastAPI 애플리케이션 (라우터, 서비스)

├── llm/            # LLM 파이프라인 (LangChain, RAG, 보고서 생성)

├── ml/             # 경제지표 예측 모델 (학습, 추론, SHAP)

├── sql/            # DB 스키마 및 쿼리

├── .github/workflows/  # GitHub Actions CI/CD

├── Dockerfile

└── requirements.txt

---

## ⚙️ 기술 스택

| 분류 | 기술 |
|---|---|
| **Framework** | FastAPI, Uvicorn |
| **에이전트** | LangGraph, LangChain, LangSmith |
| **LLM** | OpenAI GPT-4o-mini |
| **벡터 DB** | ChromaDB |
| **ML 모델** | CatBoost, XGBoost, LightGBM, scikit-learn |
| **모델 설명** | SHAP |
| **실험 관리** | MLflow, WandB |
| **PDF 처리** | pdfplumber, pypdf |
| **스토리지** | AWS S3 (boto3) |
| **인프라** | Docker (멀티스테이지 빌드, port 8001), GitHub Actions |

---

## 💡 주요 기능

| 기능 | 설명 |
|---|---|
| 멀티 에이전트 | LangGraph 기반 상담 시뮬레이터, 메모 어시스턴트, AI To-Do 에이전트 |
| LLM 파이프라인 | LangChain + ChromaDB RAG 기반 고객 맞춤 응답 생성 |
| 경제지표 예측 | 금값·기준금리·매매가격지수 회귀 예측 |
| SHAP 설명 | 예측 결과에 대한 feature importance 기반 근거 제공 |
| 시황 보고서 | LLM 기반 경제 시황 보고서 자동 생성 |
| 실험 관리 | MLflow + WandB를 통한 모델 버전 및 실험 이력 관리 |

---

## 🚀 실행 방법

### 로컬

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8001
```

## 🔗 연관 레포지토리

| 레포 | 역할 |
|---|---|
| [POOM-BACK](https://github.com/PoomSaengPoomSa/POOM-BACK) | FastAPI 백엔드 서버 |
| [POOM-FRONT](https://github.com/PoomSaengPoomSa/POOM-FRONT) | React 프론트엔드 |
| [POOM-AIRFLOW](https://github.com/PoomSaengPoomSa/POOM-AIRFLOW) | MLOps 데이터 파이프라인 |
| [POOM-MLFLOW](https://github.com/PoomSaengPoomSa/POOM-MLFLOW) | 모델 실험 관리 |
| [POOM-ELK](https://github.com/PoomSaengPoomSa/POOM-ELK) | 로그 모니터링 |

---

> 우리FISA AI 엔지니어링 1팀 | POOM 프로젝트
