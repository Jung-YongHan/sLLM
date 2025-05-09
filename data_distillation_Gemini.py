import json
import os
import time

from google.genai import Client, types

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

    response = [chunk.text for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    )]
    
    return "".join(response)

if __name__ == "__main__":
    datasets = CustomDataset()
    
    for data_key in ["dentist", "doctor", "nurse", "pharm"]:
        distillation_jsonl_list = list()
        for X, y in zip(datasets.kormedmcqa_datasets[data_key]["train"], datasets.kormedmcqa_datasets[data_key]["y"]):
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
    for X, y in zip(datasets.medqa_5options_datasets["train"], datasets.medqa_5options_datasets["y"]):
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
    for X, y in zip(datasets.medqa_4options_datasets["train"], datasets.medqa_4options_datasets["y"]):
        distillation_jsonl_list.append(
            {
                "question": X,
                "label": y,
                "answer": generate(X),
            }
        )
        time.sleep(5)
        with open(f"distillation_gemini/distillation_medqa_4options.jsonl", "a", encoding="utf-8") as f:
            for row in distillation_jsonl_list:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")