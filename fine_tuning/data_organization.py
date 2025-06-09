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
'''
    return x
    
def generate_kormedmcqa_answer(x) -> str:
    return f"정답: {x['answer']}: {x['answer_text']}"

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
'''
    return x
    
def generate_medqa_4_options_prompt(x) -> dict[str, str]:
    x["question"] = f'''Question: {x["question"]}
- A: {x["options"]["A"]}
- B: {x["options"]["B"]}
- C: {x["options"]["C"]}
- D: {x["options"]["D"]}
'''
    return x

def generate_medqa_answer(x) -> str:
    return f"Answer: {x['answer_idx']}: {x['answer']}"

def korean_chat_template_train(question: str, answer: str) -> list[dict[str, str]]:
    """한국어 훈련용 chat template"""
    messages = [
        {
            "role": "system",
            "content": "다음 질문을 읽고, 주어진 선택지 중에서 가장 적절한 답을 하나만 선택하세요.\n반드시 '정답: <선택지>' 형식으로 답을 작성하세요.",
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    return messages

def korean_chat_template_eval(question: str) -> list[dict[str, str]]:
    """한국어 평가용 chat template"""
    messages = [
        {
            "role": "system",
            "content": "다음 질문을 읽고 '정답: <선택지>' 형식으로 답을 작성해주세요. 선택지는 반드시 A, B, C, D, E 중 하나를 선택하세요.",
        },
        {"role": "user", "content": question},
    ]
    return messages

def english_chat_template_train(question: str, answer: str) -> list[dict[str, str]]:
    """영어 훈련용 chat template"""
    messages = [
        {
            "role": "system",
            "content": "Read the following question and select the most appropriate answer from the given options.\nYou must write the answer in the format 'Answer: <option>'.",
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    return messages

def english_chat_template_eval(question: str) -> list[dict[str, str]]:
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
            question = getattr(row, columns[0])
            
            if is_train:
                # Train: 질문과 답변이 모두 포함된 chat template
                answer = getattr(row, columns[1]) if len(columns) > 1 else ""
                if is_korean:
                    messages = korean_chat_template_train(question, answer)
                else:
                    messages = english_chat_template_train(question, answer)
                row_dict = {"text": messages, "label": answer}
            else:
                # Valid/Test: 질문만 포함된 chat template
                if is_korean:
                    messages = korean_chat_template_eval(question)
                else:
                    messages = english_chat_template_eval(question)
                # 실제 정답을 label로 저장
                answer = getattr(row, columns[1]) if len(columns) > 1 else ""
                row_dict = {"text": messages, "label": answer}
            
            f.write(json.dumps(row_dict, ensure_ascii=False) + "\n")

def process_kormedmcqa_data(dataset_folder: str, data: str, file_type: str, include_answer: bool = True):
    """KorMedMCQA 데이터 처리"""
    input_file = f"{RAW_DATA_DIR}/{dataset_folder}/{data}/{file_type}.csv"
    output_file = f"{OUTPUT_DATA_DIR}/{dataset_folder}/{data}/{file_type}.jsonl"
    
    df = pd.read_csv(input_file, index_col=0)
    
    # 모든 경우에 answer_text와 answer 생성
    df["answer_text"] = [df.loc[row_idx, df.iloc[row_idx]["answer"]] for row_idx in range(len(df))]
    df["answer"] = df.apply(generate_kormedmcqa_answer, axis=1)
    
    if include_answer:
        columns = ["question", "answer"]
    else:
        columns = ["question", "answer"]  # valid/test에서도 label을 위해 answer 포함
    
    df["question"] = df.apply(generate_kormedmcqa_prompt, axis=1)["question"]
    
    is_train = file_type == "train"
    write_jsonl_with_templates(df, output_file, columns, is_korean=True, is_train=is_train)

def process_medqa_data(dataset_folder: str, data: str, input_file: str, output_file: str, use_full_answer: bool = True):
    """MedQA 데이터 처리"""
    with open(f"{RAW_DATA_DIR}/{dataset_folder}/{data}/{input_file}", "r", encoding="utf-8") as f:
        df = pd.DataFrame([json.loads(line) for line in f])
    
    if data == "4_options":
        df["question"] = df.apply(generate_medqa_4_options_prompt, axis=1)["question"]
    elif data == "5_options":
        df["question"] = df.apply(generate_medqa_5_options_prompt, axis=1)["question"]
    
    # 모든 경우에 answer 생성
    df["answer"] = df.apply(generate_medqa_answer, axis=1)
    
    if use_full_answer:
        columns = ["question", "answer"]
    else:
        columns = ["question", "answer"]  # valid/test에서도 label을 위해 answer 포함
    
    output_path = f"{OUTPUT_DATA_DIR}/{dataset_folder}/{data}/{output_file}"
    is_train = "train" in output_file
    write_jsonl_with_templates(df, output_path, columns, is_korean=False, is_train=is_train)

def process_datasets(kormedmcqa_config: dict, medqa_config: dict):
    """데이터셋을 처리하는 공통 함수"""
    for dataset_folder in os.listdir(RAW_DATA_DIR):
        if dataset_folder == KORMEDMCQA:
            for data in os.listdir(f"{RAW_DATA_DIR}/{dataset_folder}"):
                process_kormedmcqa_data(dataset_folder, data, **kormedmcqa_config)
        elif dataset_folder == MEDQA:
            for data in os.listdir(f"{RAW_DATA_DIR}/{dataset_folder}"):
                process_medqa_data(dataset_folder, data, **medqa_config)

def organize_train_data():
    kormedmcqa_config = {"file_type": "train"}
    medqa_config = {"input_file": "train.jsonl", "output_file": "train.jsonl"}
    process_datasets(kormedmcqa_config, medqa_config)

def organize_valid_data():
    kormedmcqa_config = {"file_type": "valid", "include_answer": True}  # include_answer를 True로 변경
    medqa_config = {"input_file": "dev.jsonl", "output_file": "valid.jsonl", "use_full_answer": True}  # use_full_answer를 True로 변경
    process_datasets(kormedmcqa_config, medqa_config)

def organize_test_data():
    kormedmcqa_config = {"file_type": "test", "include_answer": True}  # include_answer를 True로 변경
    medqa_config = {"input_file": "test.jsonl", "output_file": "test.jsonl", "use_full_answer": True}  # use_full_answer를 True로 변경
    process_datasets(kormedmcqa_config, medqa_config)

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