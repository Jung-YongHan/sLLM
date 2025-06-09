import json

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers.generation.configuration_utils import GenerationConfig

from common import ModelHandler
from dataloader import DatasetLoader
from answer_process import AnswerProcessor
from configs import DatasetConfig


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
        self.model_handler = ModelHandler(
            model_source=self.model_source,
            quantization=quantization,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self.answer_processor = AnswerProcessor()

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
            formatted_chat = self.model_handler.tokenizer.apply_chat_template(
                example["messages"],
                continue_final_message=True,
                thinking=False,
                return_tensors="pt",
            ).to(self.model_handler.model.device)

            attention_mask = (
                (formatted_chat != self.model_handler.tokenizer.eos_token_id)
                .long()
                .to(self.model_handler.model.device)
            )

            generated_ids = self.model_handler.model.generate(
                formatted_chat,
                attention_mask=attention_mask,
                generation_config=self.generation_configs,
                pad_token_id=self.model_handler.tokenizer.pad_token_id,
            )[0]

            input_text_length = len(formatted_chat[0])
            generated_text = self.model_handler.tokenizer.decode(generated_ids[input_text_length:])
            answers.append(self.answer_processor.preprocess_answer(generated_text))

        return answers

    def calculate_metrics(
        self, labels: list[str], predictions: list[str]
    ) -> tuple[float, float]:
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        return round(accuracy, 4), round(f1, 4)


if __name__ == "__main__":

    # Load models and generation configurations
    with open("fine_tuning/models.json", "r", encoding="utf-8") as f:
        models = json.load(f)

    # Load datasets using DatasetLoader
    dataset_config = DatasetConfig()
    dataset_loader = DatasetLoader(dataset_config)
    korean_datasets = dataset_loader.load_korean_datasets()
    english_datasets = dataset_loader.load_english_datasets()

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

        all_datasets = {**korean_datasets, **english_datasets}
        
        for data_name, data in all_datasets.items():
            # If the pipeline with model_id is available, disable this line
            # evaluation_pipeline = EvaluationPipeline(model_dir=f"fine_tuned/{basemodel_id.split('/')[-1]}/{data_name}/")
            
            # test 데이터의 text 컬럼 사용 (chat template이 적용된 프롬프트)
            test_texts = [example["text"] for example in data["test"]]
            answer_texts = evaluation_pipeline.generate_answers(test_texts)
            
            # label 컬럼 사용 (실제 정답)
            labels = [example["label"] for example in data["test"]]
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
