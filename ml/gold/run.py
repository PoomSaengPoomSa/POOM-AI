import subprocess
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import os
from dotenv import load_dotenv

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"[START] 실행 시작: {script_name}")
    print(f"{'='*60}")
    
    python_executable = sys.executable
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    
    # Pass current environment (with loaded .env variables) to the child process
    env = os.environ.copy()
    
    try:
        process = subprocess.Popen(
            [python_executable, "-X", "utf8", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            env=env
        )
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[ERROR] 오류 발생: {script_name} 실행 중 문제 발생 (반환 코드: {process.returncode})")
            sys.exit(process.returncode)
        else:
            print(f"\n[OK] 정상 완료: {script_name}")
            
    except Exception as e:
        print(f"\n[ERROR] 실행 예외 발생: {script_name} ({e})")
        sys.exit(1)

def main():
    # Load .env at the orchestrator level
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.abspath(os.path.join(base_dir, '../../.env'))
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        print(f"[ENV] Loaded environment variables from: {env_path}")
    else:
        print(f"[ENV] Warning: .env file not found at: {env_path}")

    pipeline_scripts = [
        'utils/get_data.py',
        'utils/preprocess.py',
        'train.py',
        'test.py',
        'explain.py',
        'interpret_xai.py'
    ]
    
    print("[PIPELINE] 전체 Gold ML 파이프라인 연속 실행 시작 (Collect -> Preprocess -> Train -> Test -> Explain -> Interpret)\n")
    
    for script in pipeline_scripts:
        run_script(script)
        
    print(f"\n{'='*60}")
    print("[SUCCESS] Gold 파이프라인 전체 프로세스가 성공적으로 완료되었습니다!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
