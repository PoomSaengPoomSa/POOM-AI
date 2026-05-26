from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import statsmodels.api as sm



def preprocess_and_calculate_vif(df, n_features, threshold=10.0):

    df_filtered = df.copy()
    feature_cols = df_filtered.columns.tolist()

    initial_excluded = []

    for col in feature_cols:
        if (df_filtered[col] == 0).mean() > 0.5:  # 0 값이 50% 이상인 열 제외
            initial_excluded.append(col)

    df_filtered = df_filtered.drop(columns=initial_excluded)
    print(f"초기에 제거된 변수(0값 50% 이상): {initial_excluded}")

    # 초기 필터링 후 이미 특징 개수가 목표치 이하이면 VIF 제거를 수행하지 않음
    if len(df_filtered.columns) <= n_features:
        print(f"초기 필터링 후 특징 개수가 {len(df_filtered.columns)}개로, 목표치 {n_features}개 이하이므로 VIF 제거를 중단합니다.")
        return initial_excluded, pd.DataFrame()

    # 2. VIF를 한 번만 계산
    X = sm.add_constant(df_filtered)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print("\n--- 전체 VIF 계산 결과 ---")
    print(vif_data.sort_values('VIF', ascending=False).to_string())

    # 3. 제거할 후보 변수 목록 생성 (VIF > threshold)
    vif_candidates = vif_data[vif_data["Variable"] != "const"]
    high_vif_features = vif_candidates[vif_candidates["VIF"] > threshold]

    # 4. 제거 가능한 최대 변수 개수 계산
    num_current_features = len(df_filtered.columns)
    max_removals = num_current_features - n_features

    # 5. 최종 제거할 변수 목록 결정
    remove_features = []
    if max_removals > 0:
        # VIF가 높은 순서대로 후보 정렬
        sorted_high_vif = high_vif_features.sort_values('VIF', ascending=False)

        # 제거할 개수는 '제거 가능 최대 개수'와 'VIF 높은 변수 개수' 중 작은 값
        num_to_remove = min(len(sorted_high_vif), max_removals)

        # 최종 제거 목록 확정
        remove_features = sorted_high_vif['Variable'].head(num_to_remove).tolist()

    final_removed_list = initial_excluded + remove_features
    print(f"\n제거 가능한 최대 변수 수: {max_removals}개")
    print(f"VIF > {threshold} 인 변수들 중 상위 {len(remove_features)}개를 제거합니다: {remove_features}")

    return final_removed_list, vif_data


# 기존 preprocess_and_calculate_vif 함수를 아래 함수로 교체 또는 새로 추가
def calculate_vif_iteratively(df, n_features, threshold=10.0):
    """
    VIF를 반복적으로 계산하여 임계값을 넘는 변수를 하나씩 제거하는 함수

    :param df: VIF를 계산할 데이터프레임 (특징 변수들만 포함)
    :param threshold: VIF 임계값
    :return: 제거된 변수 리스트, 최종 VIF 데이터프레임
    """

    # 원본 데이터프레임 복사
    df_filtered = df.copy()
    vif_data = pd.DataFrame()

    # 0값이 50% 이상인 열은 미리 제거
    excluded_cols_for_vif = []
    for col in df_filtered.columns.tolist():
        if (df_filtered[col] == 0).mean() > 0.5:
            excluded_cols_for_vif.append(col)
    df_filtered = df_filtered.drop(columns=excluded_cols_for_vif)
    print(f"초기에 제거된 변수(0값 50% 이상): {excluded_cols_for_vif}")

    # 제거된 변수들을 기록할 리스트
    removed_features = excluded_cols_for_vif.copy()

    while len(df_filtered.columns)-1 > n_features:
        # 상수항 추가
        X = sm.add_constant(df_filtered)

        # VIF 계산
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        # 상수항(const)을 제외하고 VIF가 가장 높은 변수 찾기
        vif_without_const = vif_data[vif_data["Variable"] != "const"]
        max_vif = vif_without_const['VIF'].max()

        # 가장 높은 VIF 값이 임계값보다 낮으면 반복 중단
        if max_vif < threshold:
            print("모든 변수의 VIF가 임계값보다 낮아져서 반복을 중단합니다.")
            break

        # VIF가 가장 높은 변수 이름 찾기
        feature_to_remove = vif_without_const.sort_values('VIF', ascending=False)['Variable'].iloc[0]

        # 해당 변수 제거
        df_filtered = df_filtered.drop(columns=[feature_to_remove])
        removed_features.append(feature_to_remove)
        print(f"VIF 값 {max_vif:.2f} (으)로 인해 '{feature_to_remove}' 변수를 제거합니다.")

    print(f"\n최종적으로 제거된 변수들: {removed_features}")
    return removed_features, vif_data

def remove_highly_correlated_features(df, correlation_threshold):
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = df_numeric.corr()

    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                colname = corr_matrix.columns[i]
                to_drop.add(colname)

    print(f"제거할 변수 목록 (상관계수 {correlation_threshold} 이상):")
    print(to_drop)


    return to_drop, corr_matrix
