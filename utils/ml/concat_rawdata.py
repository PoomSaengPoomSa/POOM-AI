import os
import pandas as pd
import numpy as np

def concat_rawdata():
    # 1. 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(base_dir, 'data', 'ml')
    
    print(f"📂 데이터 디렉토리 확인: {data_dir}")

    # 대상 파일 리스트 정의
    monthly_files = [f for f in os.listdir(data_dir) if f.endswith('_m.csv') and not f.startswith('rawdata')]
    daily_files = [f for f in os.listdir(data_dir) if f.endswith('_d.csv') and not f.startswith('rawdata')]

    def merge_logic(file_list, freq, output_filename, monthly_df_for_daily=None):
        if not file_list:
            print(f"⚠ {output_filename} 생성을 위한 대상 파일이 없습니다.")
            return None

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
        
        if freq == 'M': 
            master_date = pd.date_range(start=start_date, end=end_date, freq='MS')
            date_format = '%Y-%m'
        else: 
            master_date = pd.date_range(start=start_date, end=end_date, freq='D')
            date_format = '%Y-%m-%d'

        master_df = pd.DataFrame({'date': master_date})

        # 순차 병합
        for df in dfs:
            master_df = pd.merge(master_df, df, on='date', how='left')

        # 🌟 [추가 로직] 월별 데이터 처리 시, 분기별 데이터(GDP)를 해당 분기의 모든 월에 채움 (ffill)
        if freq == 'M' and 'kr_gdp' in master_df.columns:
            # 1월(Q1), 4월(Q2), 7월(Q3), 10월(Q4)의 데이터를 이어지는 2달에 복사
            master_df['kr_gdp'] = master_df['kr_gdp'].ffill(limit=2)

        # [기존 로직] 일별 데이터 처리 시, CPI 등 월별 지표를 일별로 매핑
        if freq == 'D' and monthly_df_for_daily is not None:
            # 일별 날짜에서 'YYYY-MM' 형태의 병합 키 추출
            master_df['year_month'] = master_df['date'].dt.strftime('%Y-%m')
            
            # 월별 데이터에서 필요한 컬럼(kr_cpi)만 가져옴
            if 'kr_cpi' in monthly_df_for_daily.columns:
                monthly_subset = monthly_df_for_daily[['date', 'kr_cpi']].copy()
                
                monthly_subset.rename(columns={'date': 'year_month'}, inplace=True)
                
                # 일별 마스터 데이터에 병합
                master_df = pd.merge(master_df, monthly_subset, on='year_month', how='left')
            
            # 병합 후 임시 키 삭제
            master_df.drop(columns=['year_month'], inplace=True)

        # 날짜 포맷팅 변환 (저장용)
        master_df['date'] = master_df['date'].dt.strftime(date_format)
        
        # 저장
        output_path = os.path.join(data_dir, output_filename)
        master_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 통합 완료: {output_filename} ({len(master_df)}건, 기간: {master_df['date'].iloc[0]} ~ {master_df['date'].iloc[-1]})")
        
        return master_df

    # 2. 월별 데이터 통합 우선 수행 (결과물 Dataframe 반환)
    print("\n[진행] 월별 데이터 통합 중...")
    master_m_df = merge_logic(monthly_files, 'M', 'rawdata_m.csv')

    # 3. 일별 데이터 통합 수행 (월별 Dataframe을 인자로 전달하여 CPI 병합)
    print("\n[진행] 일별 데이터 통합 중...")
    merge_logic(daily_files, 'D', 'rawdata_d.csv', monthly_df_for_daily=master_m_df)

if __name__ == '__main__':
    concat_rawdata()