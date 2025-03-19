import os
import subprocess

from datasets import load_dataset, load_metric
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, pipeline)
    

def train_model(model_name: str, dataset_name: str, output_dir: str):
    """
    Fine-tune a model on a specific dataset.
    :param model_name: Hugging Face model name or path.
    :param dataset_name: Dataset name from Hugging Face datasets.
    :param output_dir: Directory to save the fine-tuned model.
    """
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    dataset = load_dataset(dataset_name)
    tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()
    
    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def evaluate_on_squad2(model_name: str):
    """
    Evaluate the model on the SQuAD 2.0 dataset.
    :param model_name: Hugging Face model name or path.
    """

    # SQuAD 2.0 검증 데이터 로드
    squad_dataset = load_dataset("squad_v2", split="validation")
    
    # QA 파이프라인 생성 (모델과 토크나이저는 model_name을 사용)
    qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

    predictions = []
    references = []
    
    # 각 예제에 대해 예측 수행
    for example in squad_dataset:
        question = example["question"]
        context = example["context"]
        result = qa_pipeline(question=question, context=context)
        predictions.append({
            "id": example["id"],
            "prediction_text": result["answer"],
            "no_answer_probability": 1 - result["score"]  # 스코어를 보정한 no-answer 확률 (예시)
        })
        references.append({
            "id": example["id"],
            "answers": example["answers"]
        })
    
    # 메트릭 계산
    squad_metric = load_metric("squad_v2")
    results = squad_metric.compute(predictions=predictions, references=references)
    print("SQuAD 2.0 Evaluation:", results)

def evaluate_on_komt_bench(model_name: str, model_id: str, input_file: str):
    """
    Evaluate the model using the KoMT-Bench dataset by executing external scripts.
    
    Parameters:
      model_name: Model path or name for generating model answers.
      model_id:   Identifier used in KoMT-Bench evaluation.
      input_file: Path to input file for showing results.
    """
    # KoMT-Bench 기본 폴더 (현재 파일 기준 상대 경로)
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'KoMT-Bench')
    
    # 1. Generating Model Answer
    llm_judge_dir = os.path.join(base_dir, "fastchat", "llm_judge")
    gen_model_answer_cmd = (
        f"CUDA_VISIBLE_DEVICES=0 python gen_model_answer.py "
        f"--model-path {model_name} --model-id {model_id} --dtype bfloat16"
    )
    print("Generating Model Answer ...")
    subprocess.run(gen_model_answer_cmd, shell=True, cwd=llm_judge_dir)

    # 2. Evaluating Model Answer using gen_judgment.py
    gen_judgment_cmd = f"python gen_judgment.py --model-list {model_id}"
    print("Evaluating Model Answer ...")
    subprocess.run(gen_judgment_cmd, shell=True, cwd=llm_judge_dir)

    # 2.1 Penalizing non-Korean responses using detector.py
    detector_dir = os.path.join(base_dir, "data", "mt_bench", "model_judgment")
    detector_cmd = f"python detector.py --model_id {model_id}"
    print("Applying penalty for non-Korean responses ...")
    subprocess.run(detector_cmd, shell=True, cwd=detector_dir)

    # 3. Show Result
    show_result_cmd = f"python show_result.py --mode single --input-file {input_file}"
    print("Showing Results ...")
    subprocess.run(show_result_cmd, shell=True, cwd=llm_judge_dir)