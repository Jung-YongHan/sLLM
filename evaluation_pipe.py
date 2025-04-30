from datasets import load_dataset
from transformers.pipelines import pipeline
from tqdm import tqdm
import json

class EvaluationPipeline:
    def __init__(self, model_dir: str|None=None, model_id : str|None=None):
        if model_dir is not None:
            self.inference_pipeline = pipeline(
                task="text-generation",
                model=model_dir,
                config=model_dir,
                tokenizer=model_dir,
                device="auto",
                trust_remote_code=True,
            )
        elif model_id is not None:
            self.inference_pipeline = pipeline(
                task="text-generation",
                model=model_id,
                config=model_id,
                tokenizer=model_id,
                device="auto",
                trust_remote_code=True,
            )
        elif model_dir is not None and model_id is not None: raise ValueError("model_dir and model_id cannot be both set.")
        else: raise ValueError("Either model_dir or model_id must be set.")
        
    def generate_prompt(self, question: str, A: str, B: str, C: str, D: str, E: str | None) -> str:
        if E is None:
            prompt = f'''
            다음 질문을 읽고, 주어진 선택지 중에서 가장 적절한 답을 하나 선택하세요.
            질문: {question}
            A: {A}
            B: {B}
            C: {C}
            D: {D}
            정답:
            '''
        else:
            prompt = f'''
            다음 질문을 읽고, 주어진 선택지 중에서 가장 적절한 답을 하나 선택하세요.
            질문: {question}
            A: {A}
            B: {B}
            C: {C}
            D: {D}
            E: {E}
            정답:
            '''
        return prompt
        
    def evaluate_kormedmcqa(self) -> dict[str, int]:
        dentist_test = load_dataset("sean0042/KorMedMCQA", "dentis", split="test")
        doctor_test = load_dataset("sean0042/KorMedMCQA", "doctor", split="test")
        nurse_test = load_dataset("sean0042/KorMedMCQA", "nurse", split="test")
        pharm_test = load_dataset("sean0042/KorMedMCQA", "pharm", split="test")
        
        for dataset in [dentist_test, doctor_test, nurse_test, pharm_test]:
            correct = {"dentist":0, "doctor":0, "nurse":0, "pharm":0}
            for data in tqdm(dataset, desc=f"Evaluating {dataset.info.dataset_name}",
                             total=len(dataset)):
                answer = self.inference_pipeline(self.generate_prompt(
                    dataset["question"],
                    dataset["A"], dataset["B"], dataset["C"], dataset["D"], dataset["E"]
                ))
                # TODO: check the model output format
                if dataset["answer"] == answer[0]["generated_text"].strip():
                    correct[dataset.info.dataset_name] += 1
                # TODO: the type of answer is str? or int?
                    
        return correct
    
    # Perhaps evaluate_medqa_5options and evaluate_medqa_4options can be merged into one function
    def evaluate_medqa_5options(self):
        with open("US/test.jsonl", "r", encoding="utf-8") as f:
            medqa_test = [json.loads(line) for line in f.readlines()]
        correct = 0
        for data in tqdm(medqa_test, desc="Evaluating MedQA",
                         total=len(medqa_test)):
            answer = self.inference_pipeline(self.generate_prompt(
                data["question"],
                data["options"]["A"], data["options"]["B"],
                data["options"]["C"], data["options"]["D"],
                data["options"]["E"]
            ))
            # TODO: check the model output format
            if data["answer_idx"] == answer[0]["generated_text"].strip():
                correct += 1
            # TODO: the type of answer is str? or int?
                
        return correct
    
    def evaluate_medqa_4options(self):
        with open("US/4_options/phrases_no_exclude_test.jsonl", "r", encoding="utf-8") as f:
            medqa_test = [json.loads(line) for line in f.readlines()]
        correct = 0
        for data in tqdm(medqa_test, desc="Evaluating MedQA",
                         total=len(medqa_test)):
            answer = self.inference_pipeline(self.generate_prompt(
                data["question"],
                data["options"]["A"], data["options"]["B"],
                data["options"]["C"], data["options"]["D"], None
            ))
            # TODO: check the model output format
            if data["answer_idx"] == answer[0]["generated_text"].strip():
                correct += 1
            # TODO: the type of answer is str? or int?
        pass