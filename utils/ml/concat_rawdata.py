import os
import pandas as pd
import numpy as np

def concat_rawdata():
    # 1. 경로 설정 (utils/ml/concat_rawdata.py 기준 상위 2단계 위가 프로젝트 루트)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(base_dir, 'data', 'ml')
    
    print(f"📂 데이터 디렉토리 확인: {data_dir}")

    # 대상 파일 리스트 정의
    monthly_files = [f for f in os.listdir(data_dir) if f.endswith('_m.csv') and not f.startswith('rawdata')]
    daily_files = [f for f in os.listdir(data_dir) if f.endswith('_d.csv') and not f.startswith('rawdata')]

    def merge_logic(file_list, freq, output_filename):
        if not file_list:
            print(f"⚠ {output_filename} 생성을 위한 대상 파일이 없습니다.")
            return

        dfs = []
        all_dates = []

        # 데이터 로드 및 날짜 수집
        for f in file_list:
            path = os.path.join(data_dir, f)
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            dfs.append(df)
            all_dates.extend(df['date'].tolist())

        # 전체 기간 마스터 생성
        start_date = min(all_dates)
        end_date = max(all_dates)
        
        if freq == 'M': # 월별 (Month Start 기준 생성 후 포맷팅)
            master_date = pd.date_range(start=start_date, end=end_date, freq='MS')
            date_format = '%Y-%m'
        else: # 일별
            master_date = pd.date_range(start=start_date, end=end_date, freq='D')
            date_format = '%Y-%m-%d'

        master_df = pd.DataFrame({'date': master_date})

        # 순차 병합
        for df in dfs:
            # 중복 컬럼 방지를 위해 'date' 외에 겹치는 컬럼이 있는지 확인 (있다면 제거/수정 가능)
            master_df = pd.merge(master_df, df, on='date', how='left')

        # 날짜 포맷팅 변환 (저장용)
        master_df['date'] = master_df['date'].dt.strftime(date_format)
        
        # 저장
        output_path = os.path.join(data_dir, output_filename)
        master_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 통합 완료: {output_filename} ({len(master_df)}건, 기간: {master_df['date'].iloc[0]} ~ {master_df['date'].iloc[-1]})")

    # 2. 월별 데이터 통합 수행
    print("[진행] 월별 데이터 통합 중...")
    merge_logic(monthly_files, 'M', 'rawdata_m.csv')

    # 3. 일별 데이터 통합 수행
    print("[진행] 일별 데이터 통합 중...")
    merge_logic(daily_files, 'D', 'rawdata_d.csv')

if __name__ == '__main__':
    concat_rawdata()
