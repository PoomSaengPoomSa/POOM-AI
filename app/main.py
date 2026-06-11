import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# POOM-AI 루트 디렉토리를 path에 추가하여 내부 모듈 참조 가능하게 설정
POOM_AI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if POOM_AI_DIR not in sys.path:
    sys.path.insert(0, POOM_AI_DIR)

# AI 모듈이 백엔드의 DB 모델 및 app 모듈을 참조하기 위한 임포트 경로 매핑
POOM_BACK_DIR = "/POOM-BACK"
if os.path.exists(POOM_BACK_DIR):
    if POOM_BACK_DIR not in sys.path:
        sys.path.insert(0, POOM_BACK_DIR)
    try:
        import app
        back_app_path = os.path.join(POOM_BACK_DIR, "app")
        ai_app_path = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(back_app_path):
            app.__path__ = [ai_app_path, back_app_path]
    except Exception as e:
        print(f"[Warning] app 패키지 경로 병합 실패: {e}")

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
    from agent.feature.feature_agent import CustomerFeatureAgent
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
    import sys
    import os
    todo_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent", "todo"))
    if todo_dir not in sys.path:
        sys.path.insert(0, todo_dir)
    try:
        if req.u_id == "all":
            from agent.todo.scheduler import run_todo_agent_for_all_pbs
            results = run_todo_agent_for_all_pbs(req.date)
            return {"status": "success", "results": results}
        else:
            from agent.todo.main import run_agent_for_pb
            result = run_agent_for_pb(req.u_id, req.date)
            return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CustomerMainRequest(BaseModel):
    u_id: Optional[str] = "pb_b1_1"
    c_ids: Optional[str] = None
    force_sub1: Optional[bool] = False
    force_sub2: Optional[bool] = False
    force_sub3: Optional[bool] = False

@app.post("/api/v1/customer-main/run")
def run_customer_main(req: CustomerMainRequest):
    import sys
    import os
    main_agent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent", "customer-main-agent"))
    if main_agent_dir not in sys.path:
        sys.path.insert(0, main_agent_dir)
        
    # 패키지 캐시 충돌 방지: agent 패키지 경로에 customer-main-agent/agent 경로 병합
    try:
        import agent
        sub_agent_path = os.path.join(main_agent_dir, "agent")
        if os.path.exists(sub_agent_path) and sub_agent_path not in agent.__path__:
            agent.__path__.append(sub_agent_path)
    except Exception as e:
        print(f"[Warning] agent 패키지 경로 병합 실패: {e}")
        
    try:
        from agent.main_agent import MainAgent
        agent = MainAgent()
        
        specified_ids = None
        if req.c_ids:
            specified_ids = [int(i.strip()) for i in req.c_ids.split(",") if i.strip()]
            
        agent.run_batch(
            specified_c_ids=specified_ids,
            u_id=req.u_id,
            force_sub1=req.force_sub1,
            force_sub2=req.force_sub2,
            force_sub3=req.force_sub3
        )
        return {"status": "success", "message": f"Customer Main Agent batch analysis completed for PB {req.u_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ai-todo/debug/paths")
def debug_paths():
    import os
    import sys
    try:
        poom_back_exists = os.path.exists("/POOM-BACK")
        poom_back_app_exists = os.path.exists("/POOM-BACK/app")
        database_file_exists = os.path.exists("/POOM-BACK/app/database.py")
        root_dir_contents = os.listdir("/") if os.path.exists("/") else []
        app_dir_contents = os.listdir("/app") if os.path.exists("/app") else []
        poom_back_contents = os.listdir("/POOM-BACK") if os.path.exists("/POOM-BACK") else []
        
        # Test imports
        import_database_success = False
        import_error = None
        try:
            from app.database import SessionLocal
            import_database_success = True
        except Exception as e:
            import_error = str(e)
            
        return {
            "poom_back_exists": poom_back_exists,
            "poom_back_app_exists": poom_back_app_exists,
            "database_file_exists": database_file_exists,
            "sys_path": sys.path,
            "root_dir_contents": root_dir_contents,
            "app_dir_contents": app_dir_contents,
            "poom_back_contents": poom_back_contents,
            "import_database_success": import_database_success,
            "import_error": import_error
        }
    except Exception as e:
        return {"error": str(e)}
