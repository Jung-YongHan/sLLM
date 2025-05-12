import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

result_df = pd.read_csv("result_csv/total_result.csv")
result_df["Doctor"] = pd.to_numeric(result_df["Doctor"], errors='coerce')
result_df.dropna(subset=["Doctor"], inplace=True)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=result_df, x="Parameters", y="Doctor", hue="Tag", style="Tag", palette="deep", s=100, alpha=0.7)
plt.xscale("log")
plt.xlabel("Parameters (Log Scale)")
plt.ylabel("Doctor Score")
plt.title("Doctor Score vs. Parameters by Tag")

if not result_df.empty:
    # Ensure Parameters column is numeric for min/max
    result_df["Parameters"] = pd.to_numeric(result_df["Parameters"], errors='coerce')
    # Drop rows where Parameters could not be converted, if any, before trying to use min/max
    result_df.dropna(subset=["Parameters"], inplace=True)
    
    if not result_df.empty: # Check again if dataframe became empty after dropping NaNs from Parameters
        plt.hlines(y=result_df.iloc[0]["Doctor"], xmin=result_df["Parameters"].min(), xmax=result_df["Parameters"].max(), color="red", linestyle="--")
        plt.text(
            # Adjust x position if necessary, e.g., use a value from Parameters or a relative position
            result_df["Parameters"].min() if not result_df["Parameters"].empty else 1.5, 
            result_df.iloc[0]["Doctor"] + 0.02,
            "Human Performance",
            horizontalalignment="center",
            verticalalignment="bottom",
            color="red",
        )
    else:
        print("DataFrame is empty after attempting to convert 'Parameters' column to numeric and dropping NaNs.")
else:
    print("DataFrame is empty after attempting to convert 'Doctor' column to numeric and dropping NaNs.")

plt.ylim(0, 1)
plt.legend(title="Tag")
plt.show()