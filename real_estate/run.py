import os
import sys

# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.get_data import collect_all
from train import run_train
from test import run_test
from explain import run_explain
from interpret_xai import run_interpret

def main():
    print("=" * 60)
    print("STARTING REAL ESTATE ML & XAI PIPELINE WORKFLOW")
    print("=" * 60)
    
    # 1. Collect Data
    collect_all()
    
    # 2. Train Model
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING MODEL...")
    print("=" * 60)
    run_train()
    
    # 3. Test & Evaluate Model
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATING MODEL...")
    print("=" * 60)
    run_test()
    
    # 4. Explain Model (SHAP)
    print("\n" + "=" * 60)
    print("STEP 4: GENERATING EXPLANATIONS (SHAP)...")
    print("=" * 60)
    run_explain()
    
    # 5. Generate Markdown XAI Narrative Report
    print("\n" + "=" * 60)
    print("STEP 5: GENERATING NARRATIVE XAI REPORT...")
    print("=" * 60)
    run_interpret()
    
    print("\n" + "=" * 60)
    print("PIPELINE WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    main()
