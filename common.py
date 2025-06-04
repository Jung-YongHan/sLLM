from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig


def get_quantization_config(torch_dtype="bfloat16") -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
    )


def load_model_and_tokenizer(
    model_id: str,
    quantization: bool = False,
    torch_dtype="bfloat16",
    trust_remote_code: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if quantization:
        quant_config = get_quantization_config(torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        # device_map="auto",  # device_map은 tokenizer에는 필요하지 않다고 함
        add_eos_token=True,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # TODO : 기존 EvaluationPipeline에서는 없는 로직인데, 필요할까?
    # 넣는 게 안전하다고는 함
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
