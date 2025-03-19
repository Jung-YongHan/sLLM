from datasets import load_dataset
from peft import get_peft_model
from peft.config import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Gemma3ForCausalLM, Trainer,
                          TrainingArguments)
from trl import GKDTrainer
from

'''
1. Base model
2. Knowledge Distillation, Teacher Model: LLM, student Model: sLLM(our model)
3. Supervised Fine-tuning for evaluation with PEFT(LoRA, Pruning, Quantization, Adapter)
4. RAG with Vector DB, LLM cache, vLLM
5. multi-query attention or grouped-query attention for KV cache
6. Make the class with AI Agent and English-Korean translation methods. The model will be argument for class.

PubMed DB must be used.
'''

# 0. Load dataset
pubmed_dataset = load_dataset("MedRAG/pubmed")
distillation_dataset = load_dataset()

# 1. Base model
# Qwen/QwQ-32B & google/gemma-3-12b-it
# Qwen/Qwen2.5-72B-Instruct & gemma-3-12b-it
teacher_model_name = "Qwen/QwQ-32B" # "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct-"
student_model_name = "google/gemma-3-12b-it" # "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
output_dir = "output"

# 1.1. Teacher model
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

# 1.2. Student model
# pip install flash-attn --no-build-isolation
student_model = Gemma3ForCausalLM.from_pretrained(student_model_name, attn_implementation="flash_attention_2")
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

# 2. Knowledge Distillation, Teacher Model: LLM, student Model: sLLM(our model)
distillation_trainer = GKDTrainer(
    model = student_model,
    teacher = teacher_model,
    processing_class=student_tokenizer,
    train_dataset=distillation_dataset,
    output_dir=output_dir,
)
distillation_trainer.train()
distillation_trainer.save_model(output_dir)
student_tokenizer.save_pretrained(output_dir)

# 3. Supervised Fine-tuning for evaluation
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    
)
model = AutoModelForCausalLM.from_pretrained(output_dir, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

train_dataset, valid_dataset = load_dataset("squad_v2", split="train"), load_dataset("squad_v2", split="validation")

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="best",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    bf16=True,
    fsdp="full_shard",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
)
trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)