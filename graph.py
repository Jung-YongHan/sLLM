import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sota_1b_model_id_list = [
    "Qwen3-1.7B.csv",
    "Qwen2.5-1.5B-Instruct.csv",
    "HyperCLOVAX-SEED-Text-Instruct-1.5B.csv",
    "gemma-3-1b-it.csv",
    "Llama-3.2-1B-Instruct.csv",
]

result_csvs = [pd.read_csv(f"result_csv/{dir}") for dir in sota_1b_model_id_list]

# Extract model names from file names
model_names = ["Qwen3 1.7B", "Qwen2.5 1.5B", "HyperCLOVAX-SEED 1.5B", "Gemma3 1B", "Llama3.2 1B"]

# Add model_name column to each dataframe
for i, df in enumerate(result_csvs):
    df['model_name'] = model_names[i]

# Concatenate all dataframes
all_results_df = pd.concat(result_csvs, ignore_index=True)

# Melt the DataFrame to have a single 'score' column
# Identify id_vars and value_vars correctly
id_vars = ['data_metric', 'model_name']
value_vars = [col for col in all_results_df.columns if col not in id_vars]

all_results_df_melted = pd.melt(all_results_df, 
                                id_vars=id_vars, 
                                value_vars=value_vars, 
                                var_name='experiment_type', 
                                value_name='score')

# Drop rows where score is NaN, as these cannot be plotted
all_results_df_melted.dropna(subset=['score'], inplace=True)
# Ensure score is numeric
all_results_df_melted['score'] = pd.to_numeric(all_results_df_melted['score'], errors='coerce')
all_results_df_melted.dropna(subset=['score'], inplace=True)

# Get unique data_metrics for creating subplots
unique_metrics = all_results_df_melted['data_metric'].unique()
n_metrics = len(unique_metrics)

# Determine the layout of subplots (e.g., 2 columns)
ncols = 2
nrows = (n_metrics + ncols - 1) // ncols # Ceiling division

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows * 6), sharex=True)
axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

# --- Create global legend handles and labels ---
# Get all unique experiment types and sort them to ensure consistent order for hue
all_experiment_types = sorted(all_results_df_melted['experiment_type'].unique())
# Create a palette for these types
palette = sns.color_palette(n_colors=len(all_experiment_types))
# Create a mapping from experiment type to color
color_map = dict(zip(all_experiment_types, palette))
# Create custom legend handles
legend_handles = [plt.Rectangle((0,0),1,1, color=color_map[exp_type]) for exp_type in all_experiment_types]
# --- End of global legend setup ---

for i, metric in enumerate(unique_metrics):
    ax = axes[i]
    metric_df = all_results_df_melted[all_results_df_melted['data_metric'] == metric]
    
    sns.barplot(x='score', y='model_name', hue='experiment_type', data=metric_df, ax=ax, 
                legend=False, orient='h', 
                hue_order=all_experiment_types, # Ensure consistent hue order
                palette=color_map)             # Ensure consistent colors
    ax.set_title(metric, fontsize=14)
    ax.set_xlabel('Score', fontsize=12)
    
    if i % ncols == 0: 
        ax.set_ylabel('') 
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([]) 

    ax.tick_params(axis='x', labelsize=10) 
    ax.tick_params(axis='y', labelsize=10) 
    ax.set_xlim(0, 1) 
    
    # No longer need to get handles from the first plot:
    # if i == 0: 
    #     handles, labels = ax.get_legend_handles_labels()

# Remove any unused subplots
for j in range(i + 1, len(axes)): # Corrected loop start index
    fig.delaxes(axes[j])

# Add a single legend for the entire figure using global handles and labels
if legend_handles:
    fig.legend(legend_handles, all_experiment_types, 
               loc='upper right', fontsize=10) # Added title_fontsize

# Adjust layout to prevent labels and legend from overlapping, and ensure titles are clear
plt.subplots_adjust(hspace=0.415, wspace=0.1, top=0.965, bottom=0.05, left=0.1, right=0.965) 
plt.show()