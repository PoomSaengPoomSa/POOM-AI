import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"[START] 실행 시작: {script_name}")
    print(f"{'='*60}")
    
    # 현재 Python 인터프리터를 사용하여 실행 (콘솔 인코딩 문제 방지를 위해 -X utf8 옵션 추가)
    python_executable = sys.executable
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    
    try:
        process = subprocess.Popen(
            [python_executable, "-X", "utf8", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        
        # 프로세스 출력을 실시간으로 터미널에 표시
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
    # 파이프라인 순서
    pipeline_scripts = [
        'utils/get_data.py',
        'utils/preprocess.py',
        'train.py',
        'test.py',
        'explain.py',
        'interpret_xai.py'
    ]
    
    print("[PIPELINE] 전체 ML 파이프라인 연속 실행 시작 (Train -> Test -> Explain)\n")
    
    for script in pipeline_scripts:
        run_script(script)
        
    print(f"\n{'='*60}")
    print("[SUCCESS] 파이프라인 전체 프로세스가 성공적으로 완료되었습니다!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
