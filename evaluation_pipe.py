import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers.configuration_utils import PretrainedConfig
from transformers.pipelines import pipeline
from transformers.utils.quantization_config import BitsAndBytesConfig

from datasets_class import CustomDataset
from fine_tuning import *


class EvaluationPipeline:
    def __init__(self, model_dir: str|None=None, model_id : str|None=None, quantization: bool=False):
        if model_dir is not None:
            self.inference_pipeline = pipeline(
                task="text-generation",
                model=model_dir,
                config=model_dir,
                tokenizer=model_dir,
                device_map="auto",
                return_full_text=False,
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
                config = PretrainedConfig.from_pretrained(model_id, quantization_config=bitsandbytes_config,
                                                          do_sample=False, repeatition_penalty=1.5)
                self.inference_pipeline = pipeline(
                    task="text-generation",
                    model=model_id,
                    config=config,
                    tokenizer=model_id,
                    device_map="auto",
                    return_full_text=False,
                    trust_remote_code=True,
                )
            else:
                config = PretrainedConfig.from_pretrained(model_id, do_sample=False, repeatition_penalty=1.5)
                self.inference_pipeline = pipeline(
                    task="text-generation",
                    model=model_id,
                    config=config,
                    tokenizer=model_id,
                    device_map="auto",
                    return_full_text=False,
                    trust_remote_code=True,
                )
        elif model_dir is not None and model_id is not None: raise ValueError("model_dir and model_id cannot be both set.")
        else: raise ValueError("Either model_dir or model_id must be set.")
        
        self.datasets = CustomDataset()

    def evaluate_kormedmcqa(self, cot=False) -> dict[str, list[str]]:
        dentist_test = self.datasets.kormedmcqa_datasets["dentist"]["test"]
        doctor_test = self.datasets.kormedmcqa_datasets["doctor"]["test"]
        nurse_test = self.datasets.kormedmcqa_datasets["nurse"]["test"]
        pharm_test = self.datasets.kormedmcqa_datasets["pharm"]["test"]

        answers = {"dentist":[], "doctor":[], "nurse":[], "pharm":[]}
        for name, dataset in zip(["dentist","doctor","nurse","pharm"], [dentist_test, doctor_test, nurse_test, pharm_test]):
            answer_list = []
            for data in tqdm(iterable=dataset["X"], desc=f"Evaluating {name}",
                             total=len(dataset)):
                if cot:
                    fewshot_cot = "\n".join(self.datasets.kormedmcqa_cot_data[name])
                    answer = self.inference_pipeline(fewshot_cot + "\n" + data)
                else:
                    answer = self.inference_pipeline(data)
                preprocessed_answer = answer[0]["generated_text"].split("\n")[0]
                if "A" in preprocessed_answer:
                    answer_list.append("A")
                elif "B" in preprocessed_answer:
                    answer_list.append("B")
                elif "C" in preprocessed_answer:
                    answer_list.append("C")
                elif "D" in preprocessed_answer:
                    answer_list.append("D")
                elif "E" in preprocessed_answer:
                    answer_list.append("E")
                else:
                    answer_list.append("None")
            answers[name] = answer_list

        return answers

    # Perhaps evaluate_medqa_5options and evaluate_medqa_4options can be merged into one function
    def evaluate_medqa_5options(self) -> list[str]:
        medqa_test = self.datasets.medqa_5options_datasets["test"]
        answers  = []
        for data in tqdm(medqa_test["X"], desc="Evaluating MedQA",
                         total=len(medqa_test)):
            answer = self.inference_pipeline(data)
            preprocessed_answer = answer[0]["generated_text"].split("\n")[0]
            if "A" in preprocessed_answer:
                answers.append("A")
            elif "B" in preprocessed_answer:
                answers.append("B")
            elif "C" in preprocessed_answer:
                answers.append("C")
            elif "D" in preprocessed_answer:
                answers.append("D")
            elif "E" in preprocessed_answer:
                answers.append("E")
            else:
                answers.append("None")

        return answers

    def evaluate_medqa_4options(self) -> list[str]:
        medqa_test = self.datasets.medqa_4options_datasets["test"]
        answers = []
        for data in tqdm(medqa_test["X"], desc="Evaluating MedQA",
                         total=len(medqa_test)):
            answer = self.inference_pipeline(data)
            preprocessed_answer = answer[0]["generated_text"].split("\n")[0]
            if "A" in preprocessed_answer:
                answers.append("A")
            elif "B" in preprocessed_answer:
                answers.append("B")
            elif "C" in preprocessed_answer:
                answers.append("C")
            elif "D" in preprocessed_answer:
                answers.append("D")
            elif "E" in preprocessed_answer:
                answers.append("E")
            else:
                answers.append("None")

        return answers

    def calculate_metrics(self, labels, predictions) -> tuple[float, float]:
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        return round(accuracy, 4), round(f1, 4)

if __name__ == "__main__":

    for basemodel_id in sota_1b_model_id_list:
        option = "baseline(no finetuning/no quantization/cot)"

        evaluation_pipeline = EvaluationPipeline(model_id=basemodel_id)

        result_kormedmcqa = evaluation_pipeline.evaluate_kormedmcqa(True)
        result_medqa_5options = evaluation_pipeline.evaluate_medqa_5options()
        result_medqa_4options = evaluation_pipeline.evaluate_medqa_4options()

        result_df = pd.read_csv(f"result_csv/{basemodel_id.split('/')[-1]}.csv", index_col=0)

        result_df.loc["kormedmcqa_dentist_acc", option], result_df.loc["kormedmcqa_dentist_f1(macro)", option] = evaluation_pipeline.calculate_metrics(evaluation_pipeline.datasets.kormedmcqa_datasets["dentist"]["test"]["y"], result_kormedmcqa["dentist"])
        result_df.loc["kormedmcqa_doctor_acc", option], result_df.loc["kormedmcqa_doctor_f1(macro)", option] = evaluation_pipeline.calculate_metrics(evaluation_pipeline.datasets.kormedmcqa_datasets["doctor"]["test"]["y"], result_kormedmcqa["doctor"])
        result_df.loc["kormedmcqa_nurse_acc", option], result_df.loc["kormedmcqa_nurse_f1(macro)", option] = evaluation_pipeline.calculate_metrics(evaluation_pipeline.datasets.kormedmcqa_datasets["nurse"]["test"]["y"], result_kormedmcqa["nurse"])
        result_df.loc["kormedmcqa_pharm_acc", option], result_df.loc["kormedmcqa_pharm_f1(macro)", option] = evaluation_pipeline.calculate_metrics(evaluation_pipeline.datasets.kormedmcqa_datasets["pharm"]["test"]["y"], result_kormedmcqa["pharm"])
        result_df.loc["medqa_5option_acc", option], result_df.loc["medqa_5option_f1(macro)", option] = evaluation_pipeline.calculate_metrics(evaluation_pipeline.datasets.medqa_5options_datasets["test"]["y"], result_medqa_5options)
        result_df.loc["medqa_4option_acc", option], result_df.loc["medqa_4option_f1(macro)", option] = evaluation_pipeline.calculate_metrics(evaluation_pipeline.datasets.medqa_4options_datasets["test"]["y"], result_medqa_4options)

        for idx, row in result_df[option].items():
            print(f"{idx}: {row}")

        result_df.to_csv(f"result_csv/{basemodel_id.split('/')[-1]}.csv")