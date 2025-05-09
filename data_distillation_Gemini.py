import json
import os
import time

from google.genai import Client, types
from tqdm import tqdm

from datasets_class import CustomDataset


def generate(text: str):
    client = Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-pro-exp-03-25"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    return client.models.generate_content(model=model, contents=contents, config=generate_content_config).text

if __name__ == "__main__":
    datasets = CustomDataset()
    
    for data_key in ["dentist", "doctor", "nurse", "pharm"]:
        distillation_jsonl_list = list()
        for X, y in tqdm(zip(datasets.kormedmcqa_datasets[data_key]["train"]["X"], datasets.kormedmcqa_datasets[data_key]["train"]["y"]),
                         desc=f"Distillation {data_key}", total=len(datasets.kormedmcqa_datasets[data_key]["train"]["X"])):
            distillation_jsonl_list.append(
                {
                    "question": X,
                    "label": y,
                    "answer": generate(X),
                }
            )
            time.sleep(5)
        with open(f"distillation_gemini/distillation_{data_key}.jsonl", "a", encoding="utf-8") as f:
            for row in distillation_jsonl_list:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        
    distillation_jsonl_list = list()        
    for X, y in tqdm(zip(datasets.medqa_5options_datasets["X"], datasets.medqa_5options_datasets["y"]),
                     desc="Distillation MedQA 5 options", total=len(datasets.medqa_5options_datasets["X"])):
        distillation_jsonl_list.append(
            {
                "question": X,
                "label": y,
                "answer": generate(X),
            }
        )
        time.sleep(5)
        with open(f"distillation_gemini/distillation_medqa_5options.jsonl", "a", encoding="utf-8") as f:
            for row in distillation_jsonl_list:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                
    distillation_jsonl_list = list()
    for X, y in tqdm(zip(datasets.medqa_4options_datasets["X"], datasets.medqa_4options_datasets["y"]),
                     desc="Distillation MedQA 4 options", total=len(datasets.medqa_4options_datasets["X"])):
        distillation_jsonl_list.append(
            {
                "question": X,
                "label": y,
                "answer": generate(X),
            }
        )
        time.sleep(1)
        with open(f"distillation_gemini/distillation_medqa_4options.jsonl", "a", encoding="utf-8") as f:
            for row in distillation_jsonl_list:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")