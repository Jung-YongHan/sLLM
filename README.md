# README

## 참고 문헌 및 자료

## 옵션별 실험환경

- 소모되는 VRAM은 result_csv/vram.csv 참조

### baseline(no-finetuning, no quantization)

- 모델 로딩 조건 그대로 inference만 진행

### baseline(full finetuning, no quantization)

- finetuning 하이퍼 파라미터는 다음과 같이 사용

|model|hyperparameter|value|
|:-:|:-|:-:|
|gemma3 1B||

### baseline(no-finetuning, quantization)

- 모델 로딩 후 BitsAndBytes 인스턴스를 통해 하이퍼 파라미터를 다음과 같이 설정

|hyperparameter|value|
|:--|:--|
|load_in_4bit|True|
|bnb_4bit_use_double_quant|True|
|bnb_4bit_quant_type|nf4|
|bnb_4bit_compute_dtype|bloat16|

### baseline(full finetuning, quantization)

- 양자화 하이퍼 파라미터는 위와 동일
- finetuning 하이퍼 파라미터는 다음과 같이 사용

### lora(r=64,a=64)

### qlora(r=64,a=64)
