# README

![The Result of Experiments](result.png)

## 참고 문헌 및 자료

## 옵션별 실험환경

- 소모되는 VRAM은 result_csv/vram.csv 참조
- 전체 DeepSpeed ZeRO 사용
- attn_implementation은 flash_attention_2 사용
- pre-trained 모델이 입력을 그대로 출력하는 경향 때문에 do_sample을 False로 설정하고 repeatition_penalty를 1.5로 줌

### baseline(no finetuning/no quantization)

- 모델 로딩 조건 그대로 inference만 진행

### baseline(full finetuning/no quantization)

- finetuning 하이퍼 파라미터는 다음과 같이 사용

#### Gemma3 1B

```python
training_args = TrainingArguments(
    output_dir=output_dir,  
    per_device_train_batch_size=kwargs.get("train_batch_size", 8),
    per_device_eval_batch_size=kwargs.get("eval_batch_size", 8),
    num_train_epochs=kwargs.get("num_train_epochs", 3),
    logging_dir=f"{output_dir}/logs",
    logging_steps=kwargs.get("logging_steps", 10),
    save_steps=kwargs.get("save_steps", 500),
    eval_strategy="steps",
    eval_steps=kwargs.get("eval_steps", 500),
    save_total_limit=kwargs.get("save_total_limit", 2),
    deepspeed="deepspeed_config.json"
)
```

### baseline(no finetuning/quantization)

- 모델 로딩 후 BitsAndBytes 인스턴스를 통해 하이퍼 파라미터를 다음과 같이 설정

|hyperparameter|value|
|:--|:--|
|load_in_4bit|True|
|bnb_4bit_use_double_quant|True|
|bnb_4bit_quant_type|nf4|
|bnb_4bit_compute_dtype|bloat16|

### baseline(full finetuning/quantization)

- 양자화 하이퍼 파라미터는 위와 동일
- finetuning 하이퍼 파라미터는 다음과 같이 사용

### lora(r=32/a=64)

### qlora(r=32/a=64)
