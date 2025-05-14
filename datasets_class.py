import json

from datasets import Dataset, load_dataset


class CustomDataset:
    def __init__(self, kormedmcqa_dir="sean0042/KorMedMCQA", medqa_5options_dir="US/", medqa_4options_dir="US/4_options/"):
        num_to_alpha = ["", "A", "B", "C", "D", "E"]
        
        self.kormedmcqa_datasets = {
            "dentist":
                {
                    "train":{
                        "X": load_dataset(kormedmcqa_dir, "dentist", split="train").map(self.generate_kormedmcqa_prompt)["question"],
                        "y": [num_to_alpha[i] for i in load_dataset(kormedmcqa_dir, "dentist", split="train")["answer"]]
                    },
                    "valid":{
                        "X": load_dataset(kormedmcqa_dir, "dentist", split="dev").map(self.generate_kormedmcqa_prompt)["question"],
                        "y": [num_to_alpha[i] for i in load_dataset(kormedmcqa_dir, "dentist", split="dev")["answer"]]
                    },
                    "test":{
                        "X": load_dataset(kormedmcqa_dir, "dentist", split="test").map(self.generate_kormedmcqa_prompt)["question"],
                        "y": [num_to_alpha[i] for i in load_dataset(kormedmcqa_dir, "dentist", split="test")["answer"]]
                    },
                },
            "doctor":
                {
                    "train":{
                        "X": load_dataset(kormedmcqa_dir, "doctor", split="train").map(self.generate_kormedmcqa_prompt)["question"],
                        "y": [num_to_alpha[i] for i in load_dataset(kormedmcqa_dir, "doctor", split="train")["answer"]]
                    },
                    "valid":{
                        "X": load_dataset(kormedmcqa_dir, "doctor", split="dev").map(self.generate_kormedmcqa_prompt)["question"],
                        "y": [num_to_alpha[i] for i in load_dataset(kormedmcqa_dir, "doctor", split="dev")["answer"]]
                    },
                    "test":{
                        "X": load_dataset(kormedmcqa_dir, "doctor", split="test").map(self.generate_kormedmcqa_prompt)["question"],
                        "y": [num_to_alpha[i] for i in load_dataset(kormedmcqa_dir, "doctor", split="test")["answer"]]
                    },
                },
            "nurse":
                {
                    "train":{
                        "X": load_dataset(kormedmcqa_dir, "nurse", split="train").map(self.generate_kormedmcqa_prompt)["question"],
                        "y": [num_to_alpha[i] for i in load_dataset(kormedmcqa_dir, "nurse", split="train")["answer"]]
                    },
                    "valid":{
                        "X": load_dataset(kormedmcqa_dir, "nurse", split="dev").map(self.generate_kormedmcqa_prompt)["question"],
                        "y": [num_to_alpha[i] for i in load_dataset(kormedmcqa_dir, "nurse", split="dev")["answer"]]
                    },
                    "test":{
                        "X": load_dataset(kormedmcqa_dir, "nurse", split="test").map(self.generate_kormedmcqa_prompt)["question"],
                        "y": [num_to_alpha[i] for i in load_dataset(kormedmcqa_dir, "nurse", split="test")["answer"]]
                    },
                },
            "pharm":
                {
                    "train":{
                        "X": load_dataset(kormedmcqa_dir, "pharm", split="train").map(self.generate_kormedmcqa_prompt)["question"],
                        "y": [num_to_alpha[i] for i in load_dataset(kormedmcqa_dir, "pharm", split="train")["answer"]]
                    },
                    "valid":{
                        "X": load_dataset(kormedmcqa_dir, "pharm", split="dev").map(self.generate_kormedmcqa_prompt)["question"],
                        "y": [num_to_alpha[i] for i in load_dataset(kormedmcqa_dir, "pharm", split="dev")["answer"]]
                    },
                    "test":{
                        "X": load_dataset(kormedmcqa_dir, "pharm", split="test").map(self.generate_kormedmcqa_prompt)["question"],
                        "y": [num_to_alpha[i] for i in load_dataset(kormedmcqa_dir, "pharm", split="test")["answer"]]
                    },
                },
        }
        
        self.kormedmcqa_cot_data = {
            "dentist": load_dataset(kormedmcqa_dir, "dentist", split="fewshot").map(self.generate_kormedmcqa_cot_prompt)["cot"],
            "doctor": load_dataset(kormedmcqa_dir, "dentist", split="fewshot").map(self.generate_kormedmcqa_cot_prompt)["cot"],
            "nurse": load_dataset(kormedmcqa_dir, "dentist", split="fewshot").map(self.generate_kormedmcqa_cot_prompt)["cot"],
            "pharm": load_dataset(kormedmcqa_dir, "dentist", split="fewshot").map(self.generate_kormedmcqa_cot_prompt)["cot"],
        }
        
        with (open(medqa_5options_dir+"train.jsonl", "r", encoding="utf-8") as f_train,
              open(medqa_5options_dir+"dev.jsonl", "r", encoding="utf-8") as f_valid,
              open(medqa_5options_dir+"test.jsonl", "r", encoding="utf-8") as f_test):
            f_train, f_valid, f_test = [json.loads(line) for line in f_train], [json.loads(line) for line in f_valid], [json.loads(line) for line in f_test]
            self.medqa_5options_datasets = {
                "train":
                    {
                        "X": Dataset.from_list(f_train).map(self.generate_medqa_5options_prompt)["question"],
                        "y": Dataset.from_list(f_train)["answer_idx"]
                    },
                "valid":
                    {
                        "X": Dataset.from_list(f_valid).map(self.generate_medqa_5options_prompt)["question"],
                        "y": Dataset.from_list(f_valid)["answer_idx"]
                    },
                "test":
                    {
                        "X": Dataset.from_list(f_test).map(self.generate_medqa_5options_prompt)["question"],
                        "y": Dataset.from_list(f_test)["answer_idx"]
                    }
            }
            
        with (open(medqa_4options_dir+"phrases_no_exclude_train.jsonl", "r", encoding="utf-8") as f_train,
                open(medqa_4options_dir+"phrases_no_exclude_dev.jsonl", "r", encoding="utf-8") as f_valid,
                open(medqa_4options_dir+"phrases_no_exclude_test.jsonl", "r", encoding="utf-8") as f_test):
            f_train, f_valid, f_test = [json.loads(line) for line in f_train], [json.loads(line) for line in f_valid], [json.loads(line) for line in f_test]
            self.medqa_4options_datasets = {
                "train":
                    {
                        "X": Dataset.from_list(f_train).map(self.generate_medqa_4options_prompt)["question"],
                        "y": Dataset.from_list(f_train)["answer_idx"]
                    },
                "valid":
                    {
                        "X": Dataset.from_list(f_valid).map(self.generate_medqa_4options_prompt)["question"],
                        "y": Dataset.from_list(f_valid)["answer_idx"]
                    },
                "test":
                    {
                        "X": Dataset.from_list(f_test).map(self.generate_medqa_4options_prompt)["question"],
                        "y": Dataset.from_list(f_test)["answer_idx"]
                    }
            }
        
    def generate_kormedmcqa_prompt(self, x) -> dict[str, str]:
        x["question"] = f'''# 다음 질문을 읽고, 주어진 선택지 중에서 가장 적절한 답을 하나만 선택하세요.
## 질문: {x["question"]}
- A: {x["A"]}
- B: {x["B"]}
- C: {x["C"]}
- D: {x["D"]}
- E: {x["E"]}

정답: 
'''
        return x
    
    def generate_kormedmcqa_finetuning_prompt(self, x) -> dict[str, str]:
        x["question"] = f'''# 다음 질문을 읽고, 주어진 선택지 중에서 가장 적절한 답을 하나만 선택하세요.
## 질문: {x["question"]}
- A: {x["A"]}
- B: {x["B"]}
- C: {x["C"]}
- D: {x["D"]}
- E: {x["E"]}

정답: {x["answer"]}
'''
        return x

    def generate_kormedmcqa_cot_prompt(self, x) -> dict[str, str]:
        x["cot"] = f'''# 다음 질문을 읽고, 주어진 선택지 중에서 가장 적절한 답을 하나만 선택하세요.
## 질문: {x["question"]}
- A: {x["A"]}
- B: {x["B"]}
- C: {x["C"]}
- D: {x["D"]}
- E: {x["E"]}

정답: =
{x["cot"]}
'''
        return x
    
    def generate_medqa_5options_prompt(self, x) -> dict[str, str]:
        x["question"] = f'''# Read the following question and select only the most appropriate answer from the given options.
## Question: {x["question"]}
- A: {x["options"]["A"]}
- B: {x["options"]["B"]}
- C: {x["options"]["C"]}
- D: {x["options"]["D"]}
- E: {x["options"]["E"]}

Answer: 
'''
        return x
    
    def generate_medqa_5options_finetuning_prompt(self, x) -> dict[str, str]:
        x["question"] = f'''# Read the following question and select only the most appropriate answer from the given options.
## Question: {x["question"]}
- A: {x["options"]["A"]}
- B: {x["options"]["B"]}
- C: {x["options"]["C"]}
- D: {x["options"]["D"]}
- E: {x["options"]["E"]}

Answer: {x["answer_idx"]}
'''
        return x
    
    def generate_medqa_4options_prompt(self, x) -> dict[str, str]:
        x["question"] = f'''# Read the following question and select the most appropriate answer from the given options.
## Question: {x["question"]}
- A: {x["options"]["A"]}
- B: {x["options"]["B"]}
- C: {x["options"]["C"]}
- D: {x["options"]["D"]}

Answer: 
'''
        return x
    
    def generate_medqa_4options_finetuning_prompt(self, x) -> dict[str, str]:
        x["question"] = f'''# Read the following question and select the most appropriate answer from the given options.
## Question: {x["question"]}
- A: {x["options"]["A"]}
- B: {x["options"]["B"]}
- C: {x["options"]["C"]}
- D: {x["options"]["D"]}

Answer: {x["answer_idx"]}
'''

        return x

# testcode
if __name__ == "__main__":
    dataset = CustomDataset()
    print(dataset.kormedmcqa_datasets["dentist"]["train"]["X"][0])
    print(dataset.medqa_5options_datasets["train"]["X"][0])
    print(dataset.medqa_4options_datasets["train"]["X"][0])
    print(dataset.kormedmcqa_datasets["dentist"]["train"]["y"][0])
    print(dataset.medqa_5options_datasets["train"]["y"][0])
    print(dataset.medqa_4options_datasets["train"]["y"][0])