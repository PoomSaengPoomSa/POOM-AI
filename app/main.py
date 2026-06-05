import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# POOM-AI 루트 디렉토리를 path에 추가하여 내부 모듈 참조 가능하게 설정
POOM_AI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if POOM_AI_DIR not in sys.path:
    sys.path.insert(0, POOM_AI_DIR)

# AI 모듈이 백엔드의 DB 모델 및 app 모듈을 참조하기 위한 임포트 경로 매핑
POOM_BACK_DIR = "/POOM-BACK"
if os.path.exists(POOM_BACK_DIR) and POOM_BACK_DIR not in sys.path:
    sys.path.insert(0, POOM_BACK_DIR)

app = FastAPI(title="POOM AI Agent Server")

class ConsultAssistantRequest(BaseModel):
    memo: str

class CustomerFeatureRequest(BaseModel):
    c_id: int

class SimulatorChatRequest(BaseModel):
    c_id: int
    question: str

class AiTodoRequest(BaseModel):
    u_id: str
    date: str

@app.post("/api/v1/consult-assistant")
def consult_assistant(req: ConsultAssistantRequest):
    from llm.consult_assist.consult_assistant import structure_consultation_memo
    try:
        report = structure_consultation_memo(req.memo)
        return report.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/customer-feature")
def customer_feature(req: CustomerFeatureRequest):
    from agent.customer.feature_agent import CustomerFeatureAgent
    try:
        agent = CustomerFeatureAgent()
        result = agent.run(req.c_id)
        return {
            "status": "success", 
            "extracted_features": len(result.get("extracted_features", [])),
            "refined_decisions": len(result.get("refined_decisions", []))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/simulator/chat")
def simulator_chat(req: SimulatorChatRequest):
    from agent.simulator.simulator import SimulatorAgent
    try:
        agent = SimulatorAgent()
        result = agent.run(req.c_id, req.question)
        return {"answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ai-todo/run")
def run_ai_todo(req: AiTodoRequest):
    try:
        if req.u_id == "all":
            from agent.todo.scheduler import run_todo_agent_for_all_pbs
            run_todo_agent_for_all_pbs(req.date)
        else:
            from agent.todo.main import run_agent_for_pb
            run_agent_for_pb(req.u_id, req.date)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
