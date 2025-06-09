import json

import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers.generation.configuration_utils import GenerationConfig

from fine_tuning.common import load_model_and_tokenizer


class EvaluationPipeline:
    def __init__(
        self,
        model_dir: str | None = None,
        model_id: str | None = None,
        quantization: bool = False,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
    ):
        assert (
            not model_dir or not model_id
        ), "Only one of model_dir or model_id should be provided."

        self.model_source = model_dir or model_id
        self.is_quantization = quantization
        self.torch_dtype = torch_dtype

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_id=self.model_source,
            quantization=self.is_quantization,
            torch_dtype=self.torch_dtype,
            attn_implementation="flash_attention_2",
        )

        try:
            self.generation_configs = GenerationConfig.from_pretrained(
                self.model_source
            )
        except OSError:
            self.generation_configs = GenerationConfig()

    def generate_answers(self, data: Dataset, cot=False) -> list[str]:
        if cot:
            raise NotImplementedError("CoT is not yet implemented.")

        answers: list[str] = []
        for example in tqdm(data, desc="Generating answers"):
            formatted_chat = self.tokenizer.apply_chat_template(
                example["messages"],
                continue_final_message=True,
                thinking=False,
                return_tensors="pt",
            ).to(self.model.device)

            attention_mask = (
                (formatted_chat != self.tokenizer.eos_token_id)
                .long()
                .to(self.model.device)
            )

            generated_ids = self.model.generate(
                formatted_chat,
                attention_mask=attention_mask,
                generation_config=self.generation_configs,
                pad_token_id=self.tokenizer.pad_token_id,
            )[0]

            input_text_length = len(formatted_chat[0])
            generated_text = self.tokenizer.decode(generated_ids[input_text_length:])
            answers.append(self.preprocess_answer(generated_text))

        return answers

    def calculate_metrics(
        self, labels: list[str], predictions: list[str]
    ) -> tuple[float, float]:
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        return round(accuracy, 4), round(f1, 4)

    def preprocess_answer(self, answer: str) -> str:
        answer_cantidates = [
            str(line.split("Answer")[-1])
            for line in answer.strip().split("\n")
            if "정답" in line or "Answer" in line
        ]
        for line in answer_cantidates:
            if "A" in line or "1" in line:
                return "A"
            elif "B" in line or "2" in line:
                return "B"
            elif "C" in line or "3" in line:
                return "C"
            elif "D" in line or "4" in line:
                return "D"
            elif "E" in line or "5" in line:
                return "E"
        else:
            return "Z"


if __name__ == "__main__":

    # Load models and generation configurations
    with open("models.json", "r", encoding="utf-8") as f:
        models = json.load(f)

    # TODO: cot 데이터 어떻게 불러올 것인지
    # Load Korean datasets
    kormedmcqa_dentist = load_dataset(
        "json", data_dir="data/KorMedMCQA/dentist/", split=None
    )
    kormedmcqa_doctor = load_dataset(
        "json", data_dir="data/KorMedMCQA/doctor/", split=None
    )
    kormedmcqa_nurse = load_dataset(
        "json", data_dir="data/KorMedMCQA/nurse/", split=None
    )
    kormedmcqa_pharm = load_dataset(
        "json", data_dir="data/KorMedMCQA/pharm/", split=None
    )

    # Load English datasets
    medqa_4_options = load_dataset("json", data_dir="data/MedQA/4_options/", split=None)
    medqa_5_options = load_dataset("json", data_dir="data/MedQA/5_options/", split=None)

    for basemodel_id in models["sota_1b_model_id_list"]:
        print(f"Evaluating {basemodel_id}...")

        options = {
            "option_finetuning": False,
            "option_BitsAndBytes": False,
            "option_CoT": False,
            "option_LoRA(r=32 a=64)": False,
        }

        # float16 is recommended for AWQ
        evaluation_pipeline = EvaluationPipeline(
            model_id=basemodel_id,
            quantization=options["option_BitsAndBytes"],
            torch_dtype="bfloat16",
        )
        result_df: pd.DataFrame = pd.read_csv(
            f"result_csv/{basemodel_id.split('/')[-1]}.csv"
        )

        for data_name, data in zip(
            [
                "kormedmcqa_dentist",
                "kormedmcqa_doctor",
                "kormedmcqa_nurse",
                "kormedmcqa_pharm",
                "medqa_4_options",
                "medqa_5_options",
            ],
            [
                kormedmcqa_dentist,
                kormedmcqa_doctor,
                kormedmcqa_nurse,
                kormedmcqa_pharm,
                medqa_4_options,
                medqa_5_options,
            ],
        ):
            # If the pipeline with model_id is available, disable this line
            # evaluation_pipeline = EvaluationPipeline(model_dir=f"fine_tuned/{basemodel_id.split('/')[-1]}/{data_name}/")
            answer_texts = evaluation_pipeline.generate_answers(data["test"])
            labels = data["test"]["answer"]
            accuracy, f1 = evaluation_pipeline.calculate_metrics(labels, answer_texts)

            new_row = [
                data_name,
                "f1(macro)",
                f1,
                options["option_finetuning"],
                options["option_BitsAndBytes"],
                options["option_CoT"],
                options["option_LoRA(r=32 a=64)"],
            ]

            # Check if the row already exists in the DataFrame
            if (
                new_row[:2] + new_row[3:]
                not in result_df.drop("score", axis=1).values.tolist()
            ):
                result_df.loc[-1] = new_row
                result_df.index += 1
                result_df = result_df.sort_index()
            else:
                existed_idx = (
                    result_df.drop("score", axis=1)
                    .values.tolist()
                    .index(new_row[:2] + new_row[3:])
                )
                result_df.loc[existed_idx] = new_row

            new_row = [
                data_name,
                "acc",
                accuracy,
                options["option_finetuning"],
                options["option_BitsAndBytes"],
                options["option_CoT"],
                options["option_LoRA(r=32 a=64)"],
            ]

            if (
                new_row[:2] + new_row[3:]
                not in result_df.drop("score", axis=1).values.tolist()
            ):
                result_df.loc[-1] = new_row
                result_df.index += 1
                result_df = result_df.sort_index()
            else:
                existed_idx = (
                    result_df.drop("score", axis=1)
                    .values.tolist()
                    .index(new_row[:2] + new_row[3:])
                )
                result_df.loc[existed_idx] = new_row
            result_df.to_csv(
                f"result_csv/{basemodel_id.split('/')[-1]}.csv", index=False
            )
