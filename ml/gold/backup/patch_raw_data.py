import os
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gold_raw_path = os.path.join(base_dir, 'data', 'raw_data.csv')
    baserate_raw_path = os.path.join(base_dir, '..', 'base_rate', 'data', 'raw_data.csv')
    
    if not os.path.exists(gold_raw_path):
        print(f"[ERROR] Gold raw data not found at: {gold_raw_path}")
        return
    if not os.path.exists(baserate_raw_path):
        print(f"[ERROR] Base rate raw data not found at: {baserate_raw_path}")
        return
        
    print(f"Reading gold raw: {gold_raw_path}")
    df_gold = pd.read_csv(gold_raw_path)
    
    print(f"Reading base rate raw: {baserate_raw_path}")
    df_base = pd.read_csv(baserate_raw_path)
    
    # loaded_date -> YYYYMM 포맷 컬럼 임시 생성
    df_gold['ym'] = pd.to_datetime(df_gold['loaded_date']).dt.strftime('%Y%m').astype(int)
    
    # base rate에서 필요한 금리 데이터 매핑 딕셔너리 생성
    rate_map = {}
    for _, row in df_base.iterrows():
        rate_map[int(row['date_ym'])] = {
            'us_fed_rate': row['us_fed_rate'],
            'kr_base_rate': row['kr_base_rate']
        }
        
    # 금리 컬럼 보간/매핑
    us_rates = []
    kr_rates = []
    
    # 맵에 없는 경우를 위한 정렬된 키
    available_yms = sorted(list(rate_map.keys()))
    
    for _, row in df_gold.iterrows():
        ym = int(row['ym'])
        if ym in rate_map:
            rate = rate_map[ym]
        else:
            # 범위 밖인 경우 가장 가까운/마지막 값을 사용
            nearest = available_yms[-1]
            rate = rate_map[nearest]
            
        us_rates.append(rate['us_fed_rate'])
        kr_rates.append(rate['kr_base_rate'])
        
    df_gold['us_fed_rate'] = us_rates
    df_gold['kr_base_rate'] = kr_rates
    
    # 결측치 보간
    df_gold['us_fed_rate'] = df_gold['us_fed_rate'].ffill().bfill()
    df_gold['kr_base_rate'] = df_gold['kr_base_rate'].ffill().bfill()
    
    # 임시 ym 컬럼 삭제
    df_gold = df_gold.drop(columns=['ym'])
    
    # 파일 오버라이트 저장
    df_gold.to_csv(gold_raw_path, index=False, encoding='utf-8-sig')
    print("[SUCCESS] Successfully patched gold raw_data.csv with us_fed_rate and kr_base_rate!")

if __name__ == '__main__':
    main()
