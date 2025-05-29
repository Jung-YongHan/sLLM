import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

result_csvs = os.listdir("result_csv")
datasets = [
    "medqa_5_options", "medqa_4_options", "kormedmcqa_dentist",
    "kormedmcqa_doctor", "kormedmcqa_nurse", "kormedmcqa_pharm"
]

for result_csv in result_csvs:
    if not result_csv.endswith(".csv"):
        continue
    try:
        df = pd.read_csv(f"result_csv/{result_csv}")
    except Exception as e:
        print(f"Error reading {result_csv}: {e}")
        continue
        
    metric = "acc" #"f1(macro)"
    # CSV 파일의 열 이름과 정확히 일치해야 합니다.
    options = {
        "option_finetuning": False,
        "option_BitsAndBytes": False,
        "option_CoT": False,
        "option_LoRA(r=32 a=64)": False,
    }

    # 데이터 필터링
    df_filtered = df[df["metric"] == metric]
    for option_key, option_value in options.items():
        if option_key in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[option_key] == option_value]
        else:
            print(f"Warning: Option key '{option_key}' not found in columns of {result_csv}")

    if df_filtered.empty:
        print(f"No data found for metric '{metric}' and options in {result_csv}")
        continue

    # 데이터셋 순서대로 점수 추출
    # 'data' 열을 인덱스로 설정하고, 'datasets' 리스트 순서대로 재정렬 후 'score' 선택
    # 누락된 데이터셋은 0으로 채움
    scores_series = df_filtered.set_index("data").reindex(datasets)["score"].fillna(0)
    scores = scores_series.tolist()

    if not scores:
        print(f"No scores to plot for {result_csv} with metric {metric} and options {options}")
        continue

    num_vars = len(datasets)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # 차트를 닫기 위해 시작점을 다시 추가
    scores_plot = scores + scores[:1]
    angles_plot = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 옵션 문자열 생성 (범례용)
    true_options = [key for key, value in options.items() if value]
    if not true_options:
        legend_label = "Base Model"
    else:
        legend_label = ", ".join(true_options)
    
    ax.plot(angles_plot, scores_plot, linewidth=2, linestyle="solid", label=legend_label)
    ax.fill(angles_plot, scores_plot, alpha=0.25)
    
    # 각 데이터 포인트에 점수 표시
    for angle, score in zip(angles, scores):
        ax.text(angle, score + 0.05, f'{score:.3f}', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # x축 라벨 설정 및 줄바꿈
    ax.set_thetagrids(np.degrees(angles), [])  # 빈 라벨로 설정
    # 수동으로 라벨 추가 (특별 처리)
    for i, (angle, dataset) in enumerate(zip(angles, datasets)):
        if dataset == "medqa_4_options":
            label_text = "medqa\n4_options"
        elif dataset == "medqa_5_options":
            label_text = "medqa\n5_options"
        else:
            label_text = dataset.replace("_", "\n")  # 일반적인 줄바꿈 적용
        ax.text(angle, 1.15, label_text, ha='center', va='center', fontsize=10)
    
    # y축 눈금 및 범위 설정
    yticks = [0.2, 0.4, 0.6, 0.8]
    ax.set_ylim(0, 1.0)
    
    # 기본 격자 및 원형 테두리 완전 비활성화
    ax.grid(False)
    ax.spines['polar'].set_visible(False)  # 가장 바깥쪽 원 제거
    
    # y축 라벨 설정
    ax.set_rgrids(yticks, labels=[str(tick) for tick in yticks])
    ax.set_rlabel_position(0)  # y축 라벨 위치 조정 (첫 번째 축 방향에 표시)
    
    # 중심에서 각 축으로 향하는 직선 그리기
    for angle in angles:
        ax.plot([angle, angle], [0, 1.0], 'k-', alpha=0.3, linewidth=0.5)
    
    # 육각형 격자 그리기 (6개 점만 사용)
    for ytick in yticks:
        hexagon_angles = angles + [angles[0]]  # 시작점으로 돌아가서 육각형 완성
        hexagon_radii = [ytick] * len(hexagon_angles)
        ax.plot(hexagon_angles, hexagon_radii, 'k-', alpha=0.3, linewidth=0.5)

    plt.title(f"{result_csv.replace('.csv', '')} - {metric.upper()}", size=16, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()