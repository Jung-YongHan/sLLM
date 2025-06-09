from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

class ModelHandler:
    def __init__(self, model_source: str, quantization: bool = False, torch_dtype="bfloat16", attn_implementation="flash_attention_2"):
        self.model_source = model_source
        self.is_quantization = quantization
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _get_quantization_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.torch_dtype,
        )

    def _load_model_and_tokenizer(self):
        if self.is_quantization:
            quant_config = self._get_quantization_config()
            model = AutoModelForCausalLM.from_pretrained(
                self.model_source,
                quantization_config=quant_config,
                attn_implementation=self.attn_implementation,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_source,
                attn_implementation=self.attn_implementation,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
    
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            # device_map="auto",  # device_map은 tokenizer에는 필요하지 않다고 함
            add_eos_token=True,
            trust_remote_code=True,
        )
    
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
        # TODO : 기존 EvaluationPipeline에서는 없는 로직인데, 필요할까?
        # 넣는 게 안전하다고는 함
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
    
        return model, tokenizer
