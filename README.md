# README

## 참고 문헌 및 자료

## 옵션별 실험환경

### baseline(no-finetuning, no quantization)

- 모델 로딩 조건 그대로 inference만 진행

### baseline(full finetuning, no quantization)

### baseline(no-finetuning, quantization)

- 모델 로딩 후 BitsAndBytes 인스턴스를 통해 하이퍼 파라미터를 다음과 같이 설정

|hyperparameter|value|
|:--|:--|
|load_in_4bit|True|
|bnb_4bit_use_double_quant|True|
|bnb_4bit_quant_type|nf4|
|bnb_4bit_compute_dtype|bloat16|

### baseline(full finetuning, quantization)

### lora(r=64,a=64)

### qlora(r=64,a=64)
