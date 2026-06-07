import subprocess
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import os
from dotenv import load_dotenv, find_dotenv

def run_script(script_name, forward_args=[]):
    print(f"\n{'='*60}")
    print(f"[START] 실행 시작: {script_name} {' '.join(forward_args)}")
    print(f"{'='*60}")
    
    python_executable = sys.executable
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    
    # Pass current environment (with loaded .env variables) to the child process
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    try:
        process = subprocess.Popen(
            [python_executable, "-X", "utf8", script_path] + forward_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            env=env
        )
        
        for line in process.stdout:
            print(line, end='', flush=True)
            
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid', action='store_true', help='Validation mode')
    args = parser.parse_known_args()[0]
    
    forward_args = []
    if args.valid:
        forward_args = ['--valid']

    # Load .env at the orchestrator level
    base_dir = os.path.dirname(os.path.abspath(__file__))
    found_env = find_dotenv()
    if found_env:
        load_dotenv(dotenv_path=found_env)
        print(f"[ENV] Loaded environment variables from: {found_env}")
    else:
        print(f"[ENV] Warning: .env file not found via find_dotenv()")

    pipeline_scripts = [
        'utils/get_data.py',
        'utils/preprocess.py',
        'train.py',
        'test.py',
        'explain.py',
        'interpret_xai.py'
    ]
    
    print(f"[PIPELINE] 전체 Gold ML 파이프라인 연속 실행 시작 (Valid Mode: {args.valid})\n")
    
    for script in pipeline_scripts:
        if script in ['train.py', 'test.py', 'explain.py']:
            run_script(script, forward_args)
        else:
            run_script(script, [])
        
    print(f"\n{'='*60}")
    print("[SUCCESS] Gold 파이프라인 전체 프로세스가 성공적으로 완료되었습니다!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
