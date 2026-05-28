import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Set paths and load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    # Look in the parent/root directories
    load_dotenv(find_dotenv())

def get_simulator_system_prompt() -> str:
    """Read the simulator system prompt from prompt/simulator_system_prompt.md."""
    prompt_path = os.path.join(current_dir, "prompt", "simulator_system_prompt.md")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def get_simulator_user_prompt(context_content: str) -> str:
    """Read the simulator user prompt template and fill in variables."""
    prompt_path = os.path.join(current_dir, "prompt", "simulator_user_prompt.md")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"User prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.replace("{context_content}", context_content)

def run_simulation(customer_id: str, question: str) -> str:
    """Run LLM chatbot simulation based on customer markdown and history."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Environment variable OPENAI_API_KEY is not defined. Please check your .env file.")
    
    # 1. Load customer context markdown/text
    data_dir = os.path.join(current_dir, "data")
    md_path = os.path.join(data_dir, f"customer_{customer_id}.md")
    txt_path = os.path.join(data_dir, f"customer_{customer_id}.txt")
    
    context_content = ""
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            context_content = f.read()
    elif os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            context_content = f.read()
    else:
        context_content = "고객 정보가 존재하지 않습니다. 기본적인 금융 상담으로 대응해 주세요."

    # 2. Load conversation history
    history_path = os.path.join(data_dir, f"customer_{customer_id}_history.json")
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    # Limit history to last 10 messages (5 turns) to prevent context bloat
    history = history[-10:]

    # 3. Construct messages
    messages = []
    # System Instruction
    messages.append({"role": "system", "content": get_simulator_system_prompt()})
    
    # Context (Customer Info + PB Notes)
    messages.append({
        "role": "user",
        "content": get_simulator_user_prompt(context_content)
    })
    messages.append({
        "role": "assistant",
        "content": "확인했습니다, PB님. 제공해주신 고객 정보와 추가 요청사항을 바탕으로 자산 시뮬레이션 및 맞춤형 자산 관리 상담 시나리오/대응 화법 분석을 준비했습니다. PB님의 상담을 지원하기 위한 추가적인 질의나 시뮬레이션 요청사항을 편하게 말씀해 주십시오."
    })

    # Historical turns
    for turn in history:
        messages.append(turn)

    # Current user query
    messages.append({"role": "user", "content": question})

    # 4. Request OpenAI Chat Completion
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
    )
    answer = completion.choices[0].message.content

    # 5. Save updated conversation history
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    
    os.makedirs(data_dir, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return answer

if __name__ == "__main__":
    # Ensure Windows console uses UTF-8
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stdin.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    if len(sys.argv) < 2:
        sys.stderr.write(json.dumps({"error": "Missing customer_id argument"}))
        sys.exit(1)

    customer_id = sys.argv[1]
    
    try:
        # Read the question from stdin
        question = sys.stdin.read().strip()
        if not question:
            sys.stdout.write(json.dumps({"answer": "질문이 입력되지 않았습니다. 궁금하신 내용을 질문해 주세요."}))
            sys.exit(0)

        answer = run_simulation(customer_id, question)
        sys.stdout.write(json.dumps({"answer": answer}))
        sys.stdout.flush()
    except Exception as e:
        sys.stderr.write(json.dumps({"error": str(e)}))
        sys.stderr.flush()
        sys.exit(1)
