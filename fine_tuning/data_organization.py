import json
import pandas as pd
import os
from pathlib import Path

# Constants
RAW_DATA_DIR = "raw_data"
OUTPUT_DATA_DIR = "fine_tuning/data"
KORMEDMCQA = "KorMedMCQA"
MEDQA = "MedQA"

def ensure_dir_exists(path: str):
    """디렉토리가 존재하지 않으면 생성"""
    Path(path).mkdir(parents=True, exist_ok=True)

def write_jsonl(data: pd.DataFrame, filepath: str, columns: list):
    """DataFrame을 JSONL 형식으로 저장"""
    ensure_dir_exists(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        for row in data[columns].itertuples(index=False):
            row_dict = {columns[i]: getattr(row, columns[i]) for i in range(len(columns))}
            f.write(json.dumps(row_dict, ensure_ascii=False) + "\n")

def generate_kormedmcqa_prompt(x) -> dict[str, str]:
    x["question"] = f'''질문: {x["question"]}
- A: {x["A"]}
- B: {x["B"]}
- C: {x["C"]}
- D: {x["D"]}
- E: {x["E"]}

정답: '''
    return x
    
def generate_kormedmcqa_answer(x) -> str:
    return f"{x["answer"]}: {x["answer_text"]}"

def generate_kormedmcqa_cot_prompt(x) -> dict[str, str]:
    x["cot"] = f'''질문: {x["question"]}
- A: {x["A"]}
- B: {x["B"]}
- C: {x["C"]}
- D: {x["D"]}
- E: {x["E"]}

정답: {x["cot"]}'''
    return x
    
def generate_medqa_5_options_prompt(x) -> dict[str, str]:
    x["question"] = f'''Question: {x["question"]}
- A: {x["options"]["A"]}
- B: {x["options"]["B"]}
- C: {x["options"]["C"]}
- D: {x["options"]["D"]}
- E: {x["options"]["E"]}

Answer: '''
    return x
    
def generate_medqa_4_options_prompt(x) -> dict[str, str]:
    x["question"] = f'''Question: {x["question"]}
- A: {x["options"]["A"]}
- B: {x["options"]["B"]}
- C: {x["options"]["C"]}
- D: {x["options"]["D"]}

Answer: '''
    return x

def generate_medqa_answer(x) -> str:
    return f"{x["answer_idx"]}: {x["answer"]}"

def korean_chat_template_train(question: str, answer: str) -> str:
    """한국어 훈련용 chat template"""
    messages = [
        {
            "role": "system",
            "content": "다음 질문을 읽고, 주어진 선택지 중에서 가장 적절한 답을 하나만 선택하세요.\n반드시 '정답: <선택지>' 형식으로 답을 먼저 작성한 후 설명을 작성하세요.",
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    return messages

def korean_chat_template_eval(question: str) -> str:
    """한국어 평가용 chat template"""
    messages = [
        {
            "role": "system",
            "content": "다음 질문을 읽고 '정답: <선택지>' 형식으로 답을 작성해주세요. 선택지는 반드시 A, B, C, D, E 중 하나를 선택하세요.",
        },
        {"role": "user", "content": question},
    ]
    return messages

def english_chat_template_train(question: str, answer: str) -> str:
    """영어 훈련용 chat template"""
    messages = [
        {
            "role": "system",
            "content": "Read the following question and select the most appropriate answer from the given options.\nYou must first write the answer in the format 'Answer: <option>' and then provide an explanation.",
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    return messages

def english_chat_template_eval(question: str) -> str:
    """영어 평가용 chat template"""
    messages = [
        {
            "role": "system",
            "content": "Read the following question and write the answer in the format 'Answer: <option>'. You must choose the one in A, B, C, D, E as the option.",
        },
        {"role": "user", "content": question},
    ]
    return messages

def write_jsonl_with_templates(data: pd.DataFrame, filepath: str, columns: list, is_korean: bool = True, is_train: bool = True):
    """DataFrame을 chat template이 적용된 JSONL 형식으로 저장"""
    ensure_dir_exists(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        for row in data[columns].itertuples(index=False):
            if is_train and len(columns) == 2:  # train/valid with question and answer
                question, answer = getattr(row, columns[0]), getattr(row, columns[1])
                if is_korean:
                    messages = korean_chat_template_train(question, answer)
                else:
                    messages = english_chat_template_train(question, answer)
                row_dict = {"messages": messages}
            else:  # test with question only or other formats
                if len(columns) == 2:
                    question, answer = getattr(row, columns[0]), getattr(row, columns[1])
                    if is_korean:
                        messages = korean_chat_template_eval(question)
                    else:
                        messages = english_chat_template_eval(question)
                    row_dict = {"messages": messages, "answer": answer}
                else:
                    row_dict = {columns[i]: getattr(row, columns[i]) for i in range(len(columns))}
            
            f.write(json.dumps(row_dict, ensure_ascii=False) + "\n")

def process_kormedmcqa_data(dataset_folder: str, data: str, file_type: str, include_answer: bool = True):
    """KorMedMCQA 데이터 처리"""
    input_file = f"{RAW_DATA_DIR}/{dataset_folder}/{data}/{file_type}.csv"
    output_file = f"{OUTPUT_DATA_DIR}/{dataset_folder}/{data}/{file_type}.jsonl"
    
    df = pd.read_csv(input_file, index_col=0)
    
    if include_answer:
        df["answer_text"] = [df.loc[row_idx, answer_idx] for row_idx, answer_idx in enumerate(df["answer"])]
        df["answer"] = df.apply(generate_kormedmcqa_answer, axis=1)
    
    df["question"] = df.apply(generate_kormedmcqa_prompt, axis=1)["question"]
    
    columns = ["question", "answer"] if include_answer else ["question", "answer"]
    is_train = file_type in ["train", "valid"]
    write_jsonl_with_templates(df, output_file, columns, is_korean=True, is_train=is_train)

def process_medqa_data(dataset_folder: str, data: str, input_file: str, output_file: str, use_full_answer: bool = True):
    """MedQA 데이터 처리"""
    with open(f"{RAW_DATA_DIR}/{dataset_folder}/{data}/{input_file}", "r", encoding="utf-8") as f:
        df = pd.DataFrame([json.loads(line) for line in f])
    
    if data == "4_options":
        df["question"] = df.apply(generate_medqa_4_options_prompt, axis=1)["question"]
    elif data == "5_options":
        df["question"] = df.apply(generate_medqa_5_options_prompt, axis=1)["question"]
    
    if use_full_answer:
        df["answer"] = df.apply(generate_medqa_answer, axis=1)
        columns = ["question", "answer"]
    else:
        columns = ["question", "answer_idx"]
    
    output_path = f"{OUTPUT_DATA_DIR}/{dataset_folder}/{data}/{output_file}"
    is_train = "train" in output_file or "valid" in output_file
    write_jsonl_with_templates(df, output_path, columns, is_korean=False, is_train=is_train)

def organize_train_data():
    for dataset_folder in os.listdir(RAW_DATA_DIR):
        if dataset_folder == KORMEDMCQA:
            for data in os.listdir(f"{RAW_DATA_DIR}/{dataset_folder}"):
                process_kormedmcqa_data(dataset_folder, data, "train")
        elif dataset_folder == MEDQA:
            for data in os.listdir(f"{RAW_DATA_DIR}/{dataset_folder}"):
                process_medqa_data(dataset_folder, data, "train.jsonl", "train.jsonl")

def organize_valid_data():
    for dataset_folder in os.listdir(RAW_DATA_DIR):
        if dataset_folder == KORMEDMCQA:
            for data in os.listdir(f"{RAW_DATA_DIR}/{dataset_folder}"):
                process_kormedmcqa_data(dataset_folder, data, "valid", include_answer=False)
        elif dataset_folder == MEDQA:
            for data in os.listdir(f"{RAW_DATA_DIR}/{dataset_folder}"):
                process_medqa_data(dataset_folder, data, "dev.jsonl", "valid.jsonl")

def organize_test_data():
    for dataset_folder in os.listdir(RAW_DATA_DIR):
        if dataset_folder == KORMEDMCQA:
            for data in os.listdir(f"{RAW_DATA_DIR}/{dataset_folder}"):
                process_kormedmcqa_data(dataset_folder, data, "test", include_answer=False)
        elif dataset_folder == MEDQA:
            for data in os.listdir(f"{RAW_DATA_DIR}/{dataset_folder}"):
                process_medqa_data(dataset_folder, data, "test.jsonl", "test.jsonl", use_full_answer=False)

def organize_cot_data():
    for dataset_folder in os.listdir(f"{RAW_DATA_DIR}/{KORMEDMCQA}"):
        input_file = f"{RAW_DATA_DIR}/{KORMEDMCQA}/{dataset_folder}/fewshot.csv"
        output_file = f"{OUTPUT_DATA_DIR}/{KORMEDMCQA}/{dataset_folder}/fewshot.jsonl"
        
        df = pd.read_csv(input_file, index_col=0)
        df["cot"] = df.apply(generate_kormedmcqa_cot_prompt, axis=1)["cot"]
        
        write_jsonl(df, output_file, ["cot"])

if __name__ == "__main__":
    organize_train_data()
    organize_valid_data()
    organize_test_data()
    organize_cot_data()