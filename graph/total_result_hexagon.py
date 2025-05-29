import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# total_result.csv 파일 읽기
try:
    df = pd.read_csv("result_csv/total_result.csv")
except Exception as e:
    print(f"Error reading total_result.csv: {e}")
    exit()

# 데이터셋 컬럼 (MedQA는 medqa_5_options로 매핑)
dataset_columns = ['Doctor', 'Nurse', 'Pharm', 'Dentist', 'MedQA']
datasets_labels = ['kormedmcqa\ndoctor', 'kormedmcqa\nnurse', 'kormedmcqa\npharm', 'kormedmcqa\ndentist', 'medqa\n5_options']

# 선택할 모델들 찾기
selected_models = set()

# 1. Proprietary Models (GPT, Claude, Gemini 등) 중 최고점 모델 찾기
proprietary_df = df[df['Tag'] == 'Proprietary Models']
if not proprietary_df.empty:
    proprietary_df = proprietary_df.copy()
    proprietary_df['avg_score'] = proprietary_df[dataset_columns].mean(axis=1, skipna=True)
    best_proprietary = proprietary_df.loc[proprietary_df['avg_score'].idxmax(), 'Model']
    selected_models.add(best_proprietary)
    print(f"Proprietary Models 최고 모델: {best_proprietary}")

# 2. 파라미터 크기별로 최고점 모델 찾기 (Proprietary Models 제외)
df_with_params = df[(df['Parameters'].notna()) & (df['Tag'] != 'Proprietary Models')].copy()
df_with_params['param_size'] = df_with_params['Parameters'].astype(float)

# 파라미터 크기별 그룹 정의
param_groups = {
    '2B-': df_with_params[df_with_params['param_size'] < 3],
    '3-4B': df_with_params[(df_with_params['param_size'] >= 3) & 
                           (df_with_params['param_size'] <= 4)],
    '5-9B': df_with_params[(df_with_params['param_size'] >= 5) & 
                           (df_with_params['param_size'] <= 9)],
    '10-50B': df_with_params[(df_with_params['param_size'] >= 10) & 
                            (df_with_params['param_size'] <= 50)],
    '50-100B': df_with_params[(df_with_params['param_size'] > 50) & 
                             (df_with_params['param_size'] <= 100)],
    '100B+': df_with_params[df_with_params['param_size'] > 100]
}

for group_name, group_df in param_groups.items():
    if not group_df.empty:
        group_df = group_df.copy()
        group_df['avg_score'] = group_df[dataset_columns].mean(axis=1, skipna=True)
        best_model = group_df.loc[group_df['avg_score'].idxmax(), 'Model']
        selected_models.add(best_model)
        print(f"{group_name} 최고 모델: {best_model}")

print(f"\n선택된 모델들: {selected_models}")

# 선택된 모델들만 필터링
df_filtered = df[df['Model'].isin(selected_models)]

num_vars = len(datasets_labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

# 선택된 모델들만 그리기
for idx, row in df_filtered.iterrows():
    model_name = row['Model']
    
    # 점수 추출 (NaN 값은 0으로 채움)
    scores = [row[col] if not pd.isna(row[col]) else 0 for col in dataset_columns]
    
    # 차트를 닫기 위해 시작점을 다시 추가
    scores_plot = scores + scores[:1]
    angles_plot = angles + angles[:1]
    
    ax.plot(angles_plot, scores_plot, linewidth=2, linestyle="solid", label=model_name, alpha=0.8)
    ax.fill(angles_plot, scores_plot, alpha=0.2)

# x축 라벨 설정
ax.set_thetagrids(np.degrees(angles), [])
for i, (angle, label) in enumerate(zip(angles, datasets_labels)):
    ax.text(angle, 1.15, label, ha='center', va='center', fontsize=12)

# y축 눈금 및 범위 설정
yticks = [0.2, 0.4, 0.6, 0.8]
ax.set_ylim(0, 1.0)

# 기본 격자 및 원형 테두리 완전 비활성화
ax.grid(False)
ax.spines['polar'].set_visible(False)

# y축 라벨 설정
ax.set_rgrids(yticks, labels=[str(tick) for tick in yticks])
ax.set_rlabel_position(0)

# 중심에서 각 축으로 향하는 직선 그리기
for angle in angles:
    ax.plot([angle, angle], [0, 1.0], 'k-', alpha=0.3, linewidth=0.5)

# 육각형 격자 그리기
for ytick in yticks:
    hexagon_angles = angles + [angles[0]]
    hexagon_radii = [ytick] * len(hexagon_angles)
    ax.plot(hexagon_angles, hexagon_radii, 'k-', alpha=0.3, linewidth=0.5)

plt.title("Model Performance Comparison - Accuracy Scores", size=16, y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=8)
plt.tight_layout()
plt.show()
