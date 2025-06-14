import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig


def fetch_model():
    # 요부분은 그냥 하드코딩 했습니다.
    adapter_id = "JoonseoHyeon/medgemma-27b-text-it"
    base_model = "google/medgemma-27b-text-it"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter_id, trust_remote_code=True)

    model = model.merge_and_unload()

    model.save_pretrained("medqa_agent")
    tokenizer.save_pretrained("medqa_agent")


if __name__ == "__main__":
    fetch_model()
    print("Model and tokenizer saved to 'medqa_agent' directory.")
