import json

import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers.configuration_utils import PretrainedConfig
from transformers.pipelines import pipeline
from transformers.utils.quantization_config import BitsAndBytesConfig


class EvaluationPipeline:
    def __init__(self, model_dir: str|None=None, model_id: str|None=None, quantization: bool=False):
        if quantization:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="bfloat16",
            )
        else:
            self.quantization_config = None
        
        assert model_dir is None or model_id is None, "Either model_dir or model_id must be set."
        assert model_dir is not None or model_id is not None, "Both model_dir and model_id cannot be set at the same time."
        
        if model_dir:
            self.inference_pipeline = pipeline(
                task="text-generation",
                model=model_dir,
                config=model_dir,
                tokenizer=model_dir,
                device_map="auto",
                return_full_text=False,
                max_new_tokens=100,
                
                trust_remote_code=True,
            )
        elif model_id:
            self.pretrained_config = PretrainedConfig.from_pretrained(
                model_id,
                quantization_config=self.quantization_config,
                do_sample=False,
                repetition_penalty=1.5)
            self.inference_pipeline = pipeline(
                task="text-generation",
                model=model_id,
                config=self.pretrained_config,
                tokenizer=model_id,
                device_map="auto",
                return_full_text=False,
                max_new_tokens=100,
                trust_remote_code=True,
            )
            
        self.inference_pipeline.model.generate

    def generate_answers(self, data: Dataset, cot: list[dict[str, str]] | bool=False, is_Korean=True) -> list[str]:
        if not is_Korean and cot:
            raise ValueError("CoT is not supported for MedQA.")
        
        def _korean_chat_template(question: str) -> list[dict[str, str]]:
            korean_chat_template = [
                {
                    "role": "system",
                    "content": "다음 질문을 읽고, 주어진 선택지 중에서 가장 적절한 답을 하나만 선택하세요.\n답을 표기할 때는 반드시 '답: 선택지' 형식으로 작성하세요."
                },
                {
                    "role": "user",
                    "content": f"{question}"
                }
            ]
            return korean_chat_template
        
        def _english_chat_template(question: str) -> list[dict[str, str]]:
            english_chat_template = [
                {
                    "role": "system",
                    "content": "Read the following question and select the most appropriate answer from the given options.\nYou must format your answer as 'Answer: Option'."
                },
                {
                    "role": "user",
                    "content": f"{question}"
                }
            ]
            return english_chat_template
        
        answers: list[str] = []
        for question in tqdm(data["question"], total=len(data["question"]), desc="Generating answers"):
            if cot:
                # TODO: Implement CoT for Korean datasets
                pass
            else:
                if is_Korean:
                    chat_template = _korean_chat_template(question)
                else:
                    chat_template = _english_chat_template(question)
                answers.append(self.inference_pipeline(chat_template)[0]["generated_text"][-1])
            
        return answers

    def calculate_metrics(self, labels: list[str], predictions: list[str]) -> tuple[float, float]:
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        return round(accuracy, 4), round(f1, 4)
    
    def preprocess_answer(self, answer: str) -> str | None:
        
        return None

if __name__ == "__main__":

    # Load models
    with open("models.json", "r", encoding="utf-8") as f:
        models = json.load(f)
        
    # TODO: cot 데이터 어떻게 불러올 것인지
    # Load Korean datasets
    kormedmcqa_dentist = load_dataset("json", data_dir="data/KorMedMCQA/dentist/", split=None)
    kormedmcqa_doctor = load_dataset("json", data_dir="data/KorMedMCQA/doctor/", split=None)
    kormedmcqa_nurse = load_dataset("json", data_dir="data/KorMedMCQA/nurse/", split=None)
    kormedmcqa_pharm = load_dataset("json", data_dir="data/KorMedMCQA/pharm/", split=None)
    
    # Load English datasets
    medqa_4options = load_dataset("json", data_dir="data/MedQA/4options/", split=None)
    medqa_5options = load_dataset("json", data_dir="data/MedQA/5options/", split=None)

    for basemodel_id in models["sota_1b_model_id_list"]:
        options = {
            "option_finetuning": 0,
            "option_BitsAndBytes": 0,
            "option_CoT": 1,
            "option_LoRA(r=32 a=64)": 0
        }
        evaluation_pipeline = EvaluationPipeline(model_id=basemodel_id)
        result_df: pd.DataFrame = pd.read_csv(f"result_csv/{basemodel_id.split('/')[-1]}.csv", index_col=0)
        
        for data in [kormedmcqa_dentist, kormedmcqa_doctor, kormedmcqa_nurse, kormedmcqa_pharm, medqa_4options, medqa_5options]:
            answer_texts = evaluation_pipeline.generate_answers(data["test"])
            labels = data["test"]["answer"]
            accuracy, f1 = evaluation_pipeline.calculate_metrics(labels, answer_texts)
            
            
        result_df.to_csv(f"result_csv/{basemodel_id.split('/')[-1]}.csv")