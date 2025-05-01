from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.utils.quantization_config import BitsAndBytesConfig

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

model_id = sota_1b_model_id_list[0]

model = AutoModelForCausalLM.from_pretrained(
    model_id,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)