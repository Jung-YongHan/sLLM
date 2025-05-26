import torch
from datasets import Dataset
from peft import LoraConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from transformers.utils.quantization_config import BitsAndBytesConfig
import json

from datasets_class import CustomDataset


class FineTuner:
    def __init__(self, model_id, is_quantization=False, is_lora=False, **kwargs):
        self.model_id = model_id
        self.tokenizer = None

        if is_quantization:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            self.quantization_config = None

        if is_lora:
            self.lora_config = LoraConfig(
                r=kwargs.get("lora_r", 16),
                lora_alpha=kwargs.get("lora_alpha", 32),
                target_modules=["q_proj", "v_proj"],
                # if the r is small enough to apply to all modules, you can use the following line instead:
                # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            )
        else:
            self.lora_config = None

    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            padding_side="left",  # TODO : 지금 활용할 모델들의 패딩 방향이 모두 left인가?
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=self.quantization_config,
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.lora_config:
            self.model.add_adapter(self.lora_config)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def train(self, train_dataset, eval_dataset, output_dir, **kwargs):

        stf_config = SFTConfig(
            output_dir=output_dir,
            metric_for_best_model="accuracy",
            max_seq_length=3500,
            packing=True,
            completion_only_loss=False,  # Set to True if you want to use only the completion part for loss calculation
            eos_token=self.tokenizer.eos_token,
            pad_token=self.tokenizer.pad_token,
            remove_unused_columns=True,
            eval_packing=False,
            **kwargs,
        )

        trainer = SFTTrainer(
            model=self.model,
            args=stf_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()

    def test(self, test_dataset):
        self.model.eval()
        predictions = []
        for example in test_dataset:
            inputs = self.tokenizer(
                example["prompt"],
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs)

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(answer)

        return predictions


if __name__ == "__main__":

    option = "baseline(finetuning/no quantization/no cot)"

    datasets = CustomDataset()

    # Original datasets
    raw_train_dataset = Dataset.from_dict(datasets.medqa_5options_datasets["train"])
    raw_eval_dataset = Dataset.from_dict(datasets.medqa_5options_datasets["valid"])

    # Load jsons
    
    models = 

    for model_id in sota_1b_model_id_list:
        fine_tuner = FineTuner(
            model_id, is_quantization=False, is_lora=False, lora_r=32, lora_alpha=64
        )
        fine_tuner.load_model_and_tokenizer()

        # def preprocess_function(examples):
        #     # Construct input texts by concatenating prompt and answer
        #     # 'X' is the prompt ending with "Answer: "
        #     # 'y' is the answer character (e.g., "A")
        #     inputs = [
        #         prompt + str(answer)
        #         for prompt, answer in zip(examples["X"], examples["y"])
        #     ]

        #     # Tokenize the combined texts
        #     # Ensure max_length and truncation are set to prevent overly long sequences
        #     model_inputs = fine_tuner.tokenizer(
        #         inputs, padding="max_length", max_length=3500, truncation=True
        #     )

        #     # For Causal LM, labels are typically the input_ids themselves
        #     model_inputs["labels"] = model_inputs["input_ids"].copy()
        #     return model_inputs

        # # Apply preprocessing
        # # Remove original columns to avoid conflicts
        # tokenized_train_dataset = raw_train_dataset.map(
        #     preprocess_function,
        #     batched=True,
        #     remove_columns=raw_train_dataset.column_names,
        # )
        # tokenized_eval_dataset = raw_eval_dataset.map(
        #     preprocess_function,
        #     batched=True,
        #     remove_columns=raw_eval_dataset.column_names,
        # )

        def _convert_to_prompt_response(example):
            return {
                "prompt": example["X"],
                "completion": example["y"],
            }

        train_dataset = raw_train_dataset.map(_convert_to_prompt_response, batched=True, remove_columns=raw_train_dataset.column_names)
        eval_dataset = raw_eval_dataset.map(_convert_to_prompt_response, batched=True, remove_columns=raw_eval_dataset.column_names)

        fine_tuner.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=f"./output/{model_id.split('/')[-1]}/medqa_5options",
            **sota_1b_model_kwargs[model_id],
        )
        fine_tuner.model.save_pretrained(
            f"./fine_tuned/{model_id.split('/')[-1]}_{option}/medqa_5options"
        )
        fine_tuner.tokenizer.save_pretrained(
            f"./fine_tuned/{model_id.split('/')[-1]}_{option}/medqa_5options"
        )
