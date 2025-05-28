import json

import pandas as pd
import os
        
def generate_kormedmcqa_prompt(x) -> dict[str, str]:
    x["question"] = f'''**다음 질문을 읽고, 주어진 선택지 중에서 가장 적절한 답을 하나만 선택하세요.**
질문: {x["question"]}
- A: {x["A"]}
- B: {x["B"]}
- C: {x["C"]}
- D: {x["D"]}
- E: {x["E"]}

정답: '''
    return x
    
def generate_kormedmcqa_completion(x) -> str:
    return f"{x["answer"]}: {x["answer_text"]}"

def generate_kormedmcqa_cot_prompt(x) -> dict[str, str]:
    x["cot"] = f'''**다음 질문을 읽고, 주어진 선택지 중에서 가장 적절한 답을 하나만 선택하세요.**
질문: {x["question"]}
- A: {x["A"]}
- B: {x["B"]}
- C: {x["C"]}
- D: {x["D"]}
- E: {x["E"]}

정답: {x["cot"]}'''
    return x
    
def generate_medqa_5_options_prompt(x) -> dict[str, str]:
    x["question"] = f'''**Read the following question and select only the most appropriate answer from the given options.**
Question: {x["question"]}
- A: {x["options"]["A"]}
- B: {x["options"]["B"]}
- C: {x["options"]["C"]}
- D: {x["options"]["D"]}
- E: {x["options"]["E"]}

Answer: '''
    return x
    
def generate_medqa_4_options_prompt(x) -> dict[str, str]:
    x["question"] = f'''**Read the following question and select the most appropriate answer from the given options.**
Question: {x["question"]}
- A: {x["options"]["A"]}
- B: {x["options"]["B"]}
- C: {x["options"]["C"]}
- D: {x["options"]["D"]}

Answer: '''
    return x

def generate_medqa_finetuning_completion(x) -> str:
    return f"{x["answer_idx"]}: {x["answer"]}"

def organize_train_data():
    for dataset_folder in os.listdir("raw_data"):
        if dataset_folder == "KorMedMCQA":
            for data in os.listdir(f"raw_data/{dataset_folder}"):
                df = pd.read_csv(f"raw_data/{dataset_folder}/{data}/train.csv", index_col=0)
                df["answer_text"] = [df[answer_idx].values[0] for answer_idx in df["answer"]]
                df["question"] = df.apply(generate_kormedmcqa_prompt, axis=1)["question"]
                df["answer"] = df.apply(generate_kormedmcqa_completion, axis=1)
                finetuning_df = pd.concat((df["question"], df["answer"]), axis=1)
                with open(f"data/{dataset_folder}/{data}/train.jsonl", "w", encoding="utf-8") as f:
                    for row in finetuning_df.itertuples(index=False):
                        f.write(json.dumps({"question": row.prompt, "answer": row.completion}, ensure_ascii=False) + "\n")
        elif dataset_folder == "MedQA":
            for data in os.listdir(f"raw_data/{dataset_folder}"):
                with open(f"raw_data/{dataset_folder}/{data}/train.jsonl", "r", encoding="utf-8") as f:
                    df = pd.DataFrame([json.loads(line) for line in f])
                if data == "4_options":
                    df["question"] = df.apply(generate_medqa_4_options_prompt, axis=1)["question"]
                elif data == "5_options":
                    df["question"] = df.apply(generate_medqa_5_options_prompt, axis=1)["question"]
                df["answer"] = df.apply(generate_medqa_finetuning_completion, axis=1)
                finetuning_df = pd.concat((df["question"], df["answer"]), axis=1)
                with open(f"data/{dataset_folder}/{data}/train.jsonl", "w", encoding="utf-8") as f:
                    for row in finetuning_df.itertuples(index=False):
                        f.write(json.dumps({"question": row.prompt, "answer": row.completion}, ensure_ascii=False) + "\n")

def organize_valid_data():
    for dataset_folder in os.listdir("raw_data"):
        if dataset_folder == "KorMedMCQA":
            for data in os.listdir(f"raw_data/{dataset_folder}"):
                df = pd.read_csv(f"raw_data/{dataset_folder}/{data}/valid.csv", index_col=0)
                df["question"] = df.apply(generate_kormedmcqa_prompt, axis=1)["question"]
                finetuning_df = pd.concat((df["question"], df["answer"]), axis=1)
                with open(f"data/{dataset_folder}/{data}/valid.jsonl", "w", encoding="utf-8") as f:
                    for row in finetuning_df.itertuples(index=False):
                        f.write(json.dumps({"question": row.question, "answer": row.answer}, ensure_ascii=False) + "\n")
        elif dataset_folder == "MedQA":
            for data in os.listdir(f"raw_data/{dataset_folder}"):
                with open(f"raw_data/{dataset_folder}/{data}/dev.jsonl", "r", encoding="utf-8") as f:
                    df = pd.DataFrame([json.loads(line) for line in f])
                if data == "4_options":
                    df["question"] = df.apply(generate_medqa_4_options_prompt, axis=1)["question"]
                elif data == "5_options":
                    df["question"] = df.apply(generate_medqa_5_options_prompt, axis=1)["question"]
                finetuning_df = pd.concat((df["question"], df["answer"]), axis=1)
                with open(f"data/{dataset_folder}/{data}/dev.jsonl", "w", encoding="utf-8") as f:
                    for row in finetuning_df.itertuples(index=False):
                        f.write(json.dumps({"question": row.question, "answer": row.answer}, ensure_ascii=False) + "\n")

def organize_test_data():
    for dataset_folder in os.listdir("raw_data"):
        if dataset_folder == "KorMedMCQA":
            for data in os.listdir(f"raw_data/{dataset_folder}"):
                df = pd.read_csv(f"raw_data/{dataset_folder}/{data}/test.csv", index_col=0)
                df["question"] = df.apply(generate_kormedmcqa_prompt, axis=1)["question"]
                finetuning_df = pd.concat((df["question"], df["answer"]), axis=1)
                with open(f"data/{dataset_folder}/{data}/test.jsonl", "w", encoding="utf-8") as f:
                    for row in finetuning_df.itertuples(index=False):
                        f.write(json.dumps({"question": row.question, "answer": row.answer}, ensure_ascii=False) + "\n")
        elif dataset_folder == "MedQA":
            for data in os.listdir(f"raw_data/{dataset_folder}"):
                with open(f"raw_data/{dataset_folder}/{data}/test.jsonl", "r", encoding="utf-8") as f:
                    df = pd.DataFrame([json.loads(line) for line in f])
                if data == "4_options":
                    df["question"] = df.apply(generate_medqa_4_options_prompt, axis=1)["question"]
                elif data == "5_options":
                    df["question"] = df.apply(generate_medqa_5_options_prompt, axis=1)["question"]
                finetuning_df = pd.concat((df["question"], df["answer"]), axis=1)
                with open(f"data/{dataset_folder}/{data}/test.jsonl", "w", encoding="utf-8") as f:
                    for row in finetuning_df.itertuples(index=False):
                        f.write(json.dumps({"question": row.question, "answer": row.answer}, ensure_ascii=False) + "\n")

def organize_cot_data():
    for dataset_folder in os.listdir("raw_data/KorMedMCQA"):
        for data in os.listdir(f"raw_data/KorMedMCQA/{dataset_folder}"):
            df = pd.read_csv(f"raw_data/KorMedMCQA/{dataset_folder}/fewshot.csv", index_col=0)
            df["cot"] = df.apply(generate_kormedmcqa_cot_prompt, axis=1)["cot"]
            with open(f"data/KorMedMCQA/{dataset_folder}/fewshot.jsonl", "w", encoding="utf-8") as f:
                for row in df.itertuples(index=False):
                    f.write(json.dumps({"cot": row.cot}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    organize_train_data()
    organize_valid_data()
    organize_test_data()
    organize_cot_data()