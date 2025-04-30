import json

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers.pipelines import pipeline
from transformers.configuration_utils import PretrainedConfig
from transformers.utils.quantization_config import BitsAndBytesConfig


class EvaluationPipeline:
    def __init__(self, model_dir: str|None=None, model_id : str|None=None, quantization: bool=False):
        if model_dir is not None:
            self.inference_pipeline = pipeline(
                task="text-generation",
                model=model_dir,
                config=model_dir,
                tokenizer=model_dir,
                device="cuda",
                trust_remote_code=True,
            )
        elif model_id is not None:
            if quantization:
                bitsandbytes_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype="bloat16",
                )
                config = PretrainedConfig.from_pretrained(model_id, quantization_config=bitsandbytes_config)
                self.inference_pipeline = pipeline(
                    task="text-generation",
                    model=model_id,
                    config=config,
                    tokenizer=model_id,
                    device="cuda",
                    trust_remote_code=True,
                )
            else:
                self.inference_pipeline = pipeline(
                    task="text-generation",
                    model=model_id,
                    config=model_id,
                    tokenizer=model_id,
                    device="cuda",
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
        
    def evaluate_kormedmcqa(self) -> dict[str, list[str]]:
        dentist_test = load_dataset("sean0042/KorMedMCQA", "dentist", split="test")
        doctor_test = load_dataset("sean0042/KorMedMCQA", "doctor", split="test")
        nurse_test = load_dataset("sean0042/KorMedMCQA", "nurse", split="test")
        pharm_test = load_dataset("sean0042/KorMedMCQA", "pharm", split="test")
        
        for dataset in [dentist_test, doctor_test, nurse_test, pharm_test]:
            answers = {"dentist":[], "doctor":[], "nurse":[], "pharm":[]}
            for data in tqdm(dataset, desc=f"Evaluating {dataset.info.dataset_name}",
                             total=len(dataset)):
                answer_list = []
                answer = self.inference_pipeline(self.generate_prompt(
                    data["question"], data["A"], data["B"], data["C"], data["D"], data["E"]
                ))
                # TODO: check the model output format
                answer_list.append(answer[0]["generated_text"].strip())
            answers[dataset.info.dataset_name].append(answer_list)
                    
        return answers
    
    def get_grountruth_kormedmcqa(self) -> dict[str, list[str]]:
        dentist_test = load_dataset("sean0042/KorMedMCQA", "dentis", split="test")
        doctor_test = load_dataset("sean0042/KorMedMCQA", "doctor", split="test")
        nurse_test = load_dataset("sean0042/KorMedMCQA", "nurse", split="test")
        pharm_test = load_dataset("sean0042/KorMedMCQA", "pharm", split="test")
        
        number_to_alpha = {1:"A", 2:"B", 3:"C", 4:"D", 5:"E"}
        groundtruths = {"dentist":dentist_test["answer"], "doctor":doctor_test["answer"], "nurse":nurse_test["answer"], "pharm":pharm_test["answer"]}
        for groundtruth in groundtruths.items():
            groundtruths[groundtruth[0]] = [number_to_alpha[int(answer)] for answer in groundtruth[1]]
            
        return groundtruths
    
    # Perhaps evaluate_medqa_5options and evaluate_medqa_4options can be merged into one function
    def evaluate_medqa_5options(self) -> list[str]:
        with open("US/test.jsonl", "r", encoding="utf-8") as f:
            medqa_test = [json.loads(line) for line in f.readlines()]
        answers  = []
        for data in tqdm(medqa_test, desc="Evaluating MedQA",
                         total=len(medqa_test)):
            answer = self.inference_pipeline(self.generate_prompt(
                data["question"],
                data["options"]["A"], data["options"]["B"],
                data["options"]["C"], data["options"]["D"],
                data["options"]["E"]
            ))
            # TODO: check the model output format
            answers.append(answer[0]["generated_text"].strip())
                
        return answers
    
    def get_groundtruth_medqa_5options(self) -> list[str]:
        with open("US/test.jsonl", "r", encoding="utf-8") as f:
            medqa_test = [json.loads(line) for line in f.readlines()]
        groundtruths = []
        for data in medqa_test:
            groundtruths.append(data["answer_idx"])
        
        return groundtruths
    
    def evaluate_medqa_4options(self) -> list[str]:
        with open("US/4_options/phrases_no_exclude_test.jsonl", "r", encoding="utf-8") as f:
            medqa_test = [json.loads(line) for line in f.readlines()]
        answers = []
        for data in tqdm(medqa_test, desc="Evaluating MedQA",
                         total=len(medqa_test)):
            answer = self.inference_pipeline(self.generate_prompt(
                data["question"],
                data["options"]["A"], data["options"]["B"],
                data["options"]["C"], data["options"]["D"], None
            ))
            # TODO: check the model output format
            answers.append(answer[0]["generated_text"].strip())
        
        return answers
    
    def get_groundtruth_medqa_4options(self) -> list[str]:
        with open("US/4_options/phrases_no_exclude_test.jsonl", "r", encoding="utf-8") as f:
            medqa_test = [json.loads(line) for line in f.readlines()]
        groundtruths = []
        for data in medqa_test:
            groundtruths.append(data["answer_idx"])
        
        return groundtruths
    
    def calculate_metrics(self, labels, predictions) -> tuple[float, float]:
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        return round(accuracy, 4), round(f1, 4)
    
if __name__ == "__main__":
    
    basemodel_id, option = "google/gemma-3-1b-it", "baseline(no-finetuning, no quantization)"
    
    evalation_pipeline = EvaluationPipeline(model_id=basemodel_id)
    
    result_kormedmcqa = evalation_pipeline.evaluate_kormedmcqa()
    result_medqa_5options = evalation_pipeline.evaluate_medqa_5options()
    result_medqa_4options = evalation_pipeline.evaluate_medqa_4options()
    
    labels_kormedmcqa = evalation_pipeline.get_grountruth_kormedmcqa()
    labels_medqa_5options = evalation_pipeline.get_groundtruth_medqa_5options()
    labels_medqa_4options = evalation_pipeline.get_groundtruth_medqa_4options()
    
    result_df = pd.read_csv(f"result_csv/{basemodel_id.split('/'[-1])}.csv", index_col=0)
    
    result_df.loc["medqa_5option_acc", option], result_df.loc["medqa_5option_f1(macro)", option] = evalation_pipeline.calculate_metrics(labels_medqa_5options, result_medqa_5options)
    result_df.loc["medqa_4option_acc", option], result_df.loc["medqa_4option_f1(macro)", option] = evalation_pipeline.calculate_metrics(labels_medqa_4options, result_medqa_4options)
    result_df.loc["kormedmcqa_dentist_acc", option], result_df.loc["kormedmcqa_dentist_f1(macro)", option] = evalation_pipeline.calculate_metrics(labels_kormedmcqa["dentist"], result_kormedmcqa["dentist"])
    result_df.loc["kormedmcqa_doctor_acc", option], result_df.loc["kormedmcqa_doctor_f1(macro)", option] = evalation_pipeline.calculate_metrics(labels_kormedmcqa["doctor"], result_kormedmcqa["doctor"])
    result_df.loc["kormedmcqa_nurse_acc", option], result_df.loc["kormedmcqa_nurse_f1(macro)", option] = evalation_pipeline.calculate_metrics(labels_kormedmcqa["nurse"], result_kormedmcqa["nurse"])
    result_df.loc["kormedmcqa_pharm_acc", option], result_df.loc["kormedmcqa_pharm_f1(macro)", option] = evalation_pipeline.calculate_metrics(labels_kormedmcqa["pharm"], result_kormedmcqa["pharm"])
    
    for row in result_df.loc["option"].itertuples():
        print(f"{row.Index}: {row[1]}")
    
    result_df.to_csv(f"result_csv/{basemodel_id.split('/'[-1])}.csv")