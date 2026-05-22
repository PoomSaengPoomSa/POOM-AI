import os
import pandas as pd
import numpy as np


def preprocess():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')

    # ═══════════════════════════════════════════════
    # 1. 원천 데이터 로드
    # ═══════════════════════════════════════════════
    raw_path = os.path.join(data_dir, 'raw_data.csv')
    print("📂 원천 데이터 로드 중...")
    df = pd.read_csv(raw_path, encoding='utf-8-sig')
    print(f"   로드 완료: {df.shape[0]}행 × {df.shape[1]}열")
    print(f"   기간: {df['date_ym'].min()} ~ {df['date_ym'].max()}")

    df['date_ym'] = df['date_ym'].astype(str).str.strip()
    df = df.sort_values('date_ym').reset_index(drop=True)

    # ═══════════════════════════════════════════════
    # 2. 결측치 처리 (IterativeImputer 적용)
    # ═══════════════════════════════════════════════
    print("\n🔧 결측치 처리 중 (머신러닝 기반 보완 적용)...")
    before_na = df.isna().sum().sum()

    if 'kr_gdp_index' in df.columns:
        df['kr_gdp_index'] = df['kr_gdp_index'].interpolate(method='linear')

    # 수치형 컬럼 추출
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if df[numeric_cols].isna().sum().sum() > 0:
        # fancyimpute는 scikit-learn 최신버전과 호환되지 않아
        # 더욱 안정적인 scikit-learn의 IterativeImputer(MICE) 사용
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
        
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # 범주형이나 식별자 컬럼 등에 남은 결측치는 ffill/bfill로 처리
    df = df.ffill().bfill()
    
    after_na = df.isna().sum().sum()
    print(f"   결측치: {before_na}건 → {after_na}건")

    # ═══════════════════════════════════════════════
    # 3. 파생 변수 생성 (강화 버전)
    # ═══════════════════════════════════════════════
    print("\n✨ 파생 변수 생성 중...")

    # ── 타겟 변수: 전월 대비 기준금리 변동폭 ──
    df['kr_base_rate_change'] = df['kr_base_rate'].diff()

    # ── 3-1) 전월 대비 변화율/변동폭 ──
    change_pairs = [
        ('kr_cpi',          'kr_cpi_change',       'pct'),
        ('kr_m2',           'kr_m2_change',        'pct'),
        ('kr_usd_exchange', 'kr_exchange_change',  'pct'),
        ('us_fed_rate',     'us_fed_rate_change',  'diff'),
        ('us_cpi',          'us_cpi_change',       'pct'),
        ('wti_oil',         'wti_change',          'pct'),
        ('vix',             'vix_change',          'pct'),
        ('us_treasury_10y', 'us_10y_change',       'diff'),
    ]
    for src, dst, method in change_pairs:
        if src in df.columns:
            if method == 'pct':
                df[dst] = df[src].pct_change() * 100
            else:
                df[dst] = df[src].diff()

    # ── 3-2) YoY (전년동월대비) 변화율 — 계절성 반영 ──
    yoy_targets = ['kr_cpi', 'kr_usd_exchange', 'us_cpi', 'wti_oil']
    for col in yoy_targets:
        if col in df.columns:
            df[f'{col}_yoy'] = df[col].pct_change(periods=12) * 100

    # ── 3-3) 한미 금리 스프레드 + 스프레드 변화 ──
    if 'kr_base_rate' in df.columns and 'us_fed_rate' in df.columns:
        df['rate_spread'] = df['kr_base_rate'] - df['us_fed_rate']
        df['rate_spread_change'] = df['rate_spread'].diff()

    # ── 3-4) 이동평균 (3개월, 6개월) ──
    ma_targets = [
        'kr_base_rate', 'kr_cpi', 'kr_unemployment',
        'kr_usd_exchange', 'us_fed_rate', 'vix', 'wti_oil'
    ]
    for col in ma_targets:
        if col in df.columns:
            df[f'{col}_ma3'] = df[col].rolling(window=3, min_periods=1).mean()
            df[f'{col}_ma6'] = df[col].rolling(window=6, min_periods=1).mean()

    # ── 3-5) 3개월/6개월 누적 변화 (모멘텀) ──
    momentum_targets = ['kr_cpi', 'kr_usd_exchange', 'us_fed_rate',
                        'vix', 'wti_oil', 'kr_unemployment']
    for col in momentum_targets:
        if col in df.columns:
            df[f'{col}_mom3'] = df[col].diff(3)
            df[f'{col}_mom6'] = df[col].diff(6)

    # ── 3-6) 래그 변수 (1개월, 2개월, 3개월) ──
    lag_targets = ['kr_cpi_change', 'kr_unemployment', 'us_fed_rate_change',
                   'kr_exchange_change', 'vix', 'rate_spread']
    for col in lag_targets:
        if col in df.columns:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag2'] = df[col].shift(2)
            df[f'{col}_lag3'] = df[col].shift(3)

    # ── 3-7) 금리 동결 유지 기간 (중요 피처) ──
    hold_months = []
    count = 0
    for val in df['kr_base_rate_change'].values:
        if pd.isna(val) or val == 0:
            count += 1
        else:
            count = 0
        hold_months.append(count)
    df['hold_duration'] = hold_months

    # ── 3-8) 금리 사이클 방향 ──
    # 최근 변동이 인상이었으면 1, 인하였으면 -1, 없으면 0
    df['last_change_dir'] = df['kr_base_rate_change'].apply(
        lambda x: np.sign(x) if pd.notna(x) and x != 0 else np.nan
    )
    df['last_change_dir'] = df['last_change_dir'].ffill().fillna(0)

    print(f"   파생 변수 포함 총 컬럼: {len(df.columns)}개")

    # ═══════════════════════════════════════════════
    # 4. label 컬럼 추가 (금리 방향성)
    # ═══════════════════════════════════════════════
    print("\n🏷️  label(금리 방향) 컬럼 생성 중...")

    def classify_direction(change):
        if pd.isna(change):
            return None
        elif change > 0:
            return '인상'
        elif change < 0:
            return '인하'
        else:
            return '동결'

    df['label'] = df['kr_base_rate_change'].apply(classify_direction)

    # 숫자 라벨 (분류 모델용): 인하=0, 동결=1, 인상=2
    label_map = {'인하': 0, '동결': 1, '인상': 2}
    df['label_encoded'] = df['label'].map(label_map)

    label_counts = df['label'].value_counts()
    print("   [라벨 분포]")
    for lbl in ['인상', '동결', '인하']:
        cnt = label_counts.get(lbl, 0)
        ratio = cnt / len(df) * 100
        print(f"     {lbl}: {cnt}건 ({ratio:.1f}%)")

    # ═══════════════════════════════════════════════
    # 5. 미래 예측(Forecasting) 프레임워크: Target 1개월 Shift 적용
    # ═══════════════════════════════════════════════
    print("\n🔮 시계열 예측(Forecasting) 프레임워크를 위해 Target 변수 1달씩 당기기 (Shift -1)...")
    id_cols = ['date_ym']
    target_cols = ['kr_base_rate_change', 'label', 'label_encoded']
    feature_cols = [c for c in df.columns if c not in id_cols + target_cols]
    
    # 201501 데이터로 201502 금리 결정을 예측하도록 타겟 변수들을 위로 1칸 당깁니다.
    df[target_cols] = df[target_cols].shift(-1)

    # ═══════════════════════════════════════════════
    # 6. NaN 행 정리
    # ═══════════════════════════════════════════════
    # diff/shift로 인한 초기 NaN 행 제거 (최대 12행: YoY 때문)
    first_valid = 12  # YoY 12개월 래그
    # 마지막 행은 다음 달의 Target이 없으므로 제외합니다.
    df = df.iloc[first_valid:-1].reset_index(drop=True)
    df = df.ffill().bfill()

    # ═══════════════════════════════════════════════
    # 7. 컬럼 순서 정리 및 저장
    # ═══════════════════════════════════════════════
    final_order = id_cols + feature_cols + target_cols
    df = df[final_order]

    save_path = os.path.join(data_dir, 'final_dataset.csv')
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 55)
    print("✅ 전처리 완료!")
    print("=" * 55)
    print(f"   저장 경로  : {save_path}")
    print(f"   최종 크기  : {df.shape[0]}행 × {df.shape[1]}열")
    import sys
    sys.path.insert(0, base_dir)
    from model import InterestRateEnsembleModel
    cfg = InterestRateEnsembleModel
    print(f"   학습 기간  : {df[df['date_ym'] <= cfg.TRAIN_END].shape[0]}개월 (~{cfg.TRAIN_END})")
    print(f"   테스트 기간: {df[df['date_ym'] >= cfg.TEST_START].shape[0]}개월 ({cfg.TEST_START}~)")
    print(f"\n   컬럼 수: {len(df.columns)}개")

    # 데이터 미리보기
    print(f"\n   📋 데이터 미리보기 (마지막 5행):")
    pd.set_option('display.max_columns', 10)
    print(df.tail().to_string(index=False))


if __name__ == '__main__':
    preprocess()
