import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import LoraConfig
from datasets import load_dataset
import json

sota_1b_model_id_list = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen2.5-1.5B-Instruct"
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
    "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
]

sota_3b_model_id_list = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen2.5-3B-Instruct",
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-3B",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    "LGAI-EXAONE/EXAONE-Deep-2.4B",
    #"microsoft/bitnet-b1.58-2B-4T"
]

sota_8b_model_id_list = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B-Instruct",
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "LGAI-EXAONE/EXAONE-Deep-7.8B",
]

sota_10b_model_id_list = [
    "Qwen/Qwen3-14B",
    "Qwen/Qwen2.5-14B-Instruct",
    "google/gemma-3-12b-it",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "microsoft/Phi-4",
]


class FineTuner:
    def __init__(self, model_id, is_quantization=False, is_lora=False, **kwargs):
        self.model_id = model_id
        
        if is_quantization:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else: self.quantization_config = None
        
        if is_lora:
            self.lora_config = LoraConfig(
                r=kwargs.get("lora_r", 16),
                lora_alpha=kwargs.get("lora_alpha", 32),
                task_type="CAUSAL_LM",
            )
        else: self.lora_config = None
        
    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=self.quantization_config,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.lora_config:
            self.model.add_adapter(self.lora_config)
            
    def train(self, train_dataset, eval_dataset, output_dir, **kwargs):
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=kwargs.get("train_batch_size", 8),
            per_device_eval_batch_size=kwargs.get("eval_batch_size", 8),
            num_train_epochs=kwargs.get("num_train_epochs", 3),
            logging_dir=f"{output_dir}/logs",
            logging_steps=kwargs.get("logging_steps", 10),
            save_steps=kwargs.get("save_steps", 500),
            eval_strategy="steps",
            eval_steps=kwargs.get("eval_steps", 500),
            save_total_limit=kwargs.get("save_total_limit", 2),
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        trainer.train()
        trainer.evaluate()
        trainer.save_model(output_dir)
        
    def test(self, test_dataset):
        self.model.eval()
        predictions = []
        for example in test_dataset:
            question = example["question"]
            A = example["A"]
            B = example["B"]
            C = example["C"]
            D = example["D"]
            E = example.get("E", None)
            
            prompt = self.generate_prompt(question, A, B, C, D, E)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(answer)
        
        return predictions

def generate_prompt(self, question: str, A: str, B: str, C: str, D: str, E: str | None, is_Korean=True) -> str:
    if is_Korean:
        prompt = f'''
# 다음 질문을 읽고, 주어진 선택지 중에서 가장 적절한 답을 하나만 선택하세요.
## 질문: {question}
### A: {A}
### B: {B}
### C: {C}
### D: {D}
### E: {E}

### 정답: 
'''
        return prompt
    else:
        if E is None:
            prompt = f'''
# Read the following question and select only the most appropriate answer from the given options.
## Question: {question}
### A: {A}
### B: {B}
### C: {C}
### D: {D}

### Answer: 
'''
            return prompt
        else:
                prompt = f'''
# Read the following question and select the most appropriate answer from the given options.
## Question: {question}
### A: {A}
### B: {B}
### C: {C}
### D: {D}
### E: {E}

### Answer: 
'''
        return prompt

if __name__ == "__main__":
    
    # kormedmcqa
    kormedmcqa_datasets = {
        "dentist_train": load_dataset("sean0042/KorMedMCQA", "dentist", split="train"),
        "dentist_eval": load_dataset("sean0042/KorMedMCQA", "dentist", split="dev"),
        "doctor_train": load_dataset("sean0042/KorMedMCQA", "doctor", split="train"),
        "doctor_eval": load_dataset("sean0042/KorMedMCQA", "doctor", split="dev"),
        "nurse_train": load_dataset("sean0042/KorMedMCQA", "nurse", split="train"),
        "nurse_eval": load_dataset("sean0042/KorMedMCQA", "nurse", split="dev"),
        "pharm_train": load_dataset("sean0042/KorMedMCQA", "pharm", split="train"),
        "pharm_eval": load_dataset("sean0042/KorMedMCQA", "pharm", split="dev"),
    }
    
    # 5 options medqa
    with open("US/train.jsonl", "r", encoding="utf-8") as f:
        medqa_train = [json.loads(line) for line in f.readlines()]
    with open("US/dev.jsonl", "r", encoding="utf-8") as f:
        medqa_eval = [json.loads(line) for line in f.readlines()]
        
    # 4 options medqa
    with open("US/4_options/phrases_no_exclude_train.jsonl", "r", encoding="utf-8") as f:
        medqa_train_4 = [json.loads(line) for line in f.readlines()]
    with open("US/4_options/phrases_no_exclude_dev.jsonl", "r", encoding="utf-8") as f:
        medqa_eval_4 = [json.loads(line) for line in f.readlines()]
    
    train_dataset, eval_dataset = kormedmcqa_datasets["dentist_train"], kormedmcqa_datasets["dentist_eval"]
    
    for model_id in sota_1b_model_id_list:
        fine_tuner = FineTuner(model_id, is_quantization=False, is_lora=False, lora_r=64, lora_alpha=64)
        fine_tuner.load_model_and_tokenizer()
        fine_tuner.train(train_dataset, eval_dataset, output_dir=f"./output/{model_id.split('/')[-1]}")
        fine_tuner.model.