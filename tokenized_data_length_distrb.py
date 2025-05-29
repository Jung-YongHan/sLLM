import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

# Load Korean datasets
kormedmcqa_dentist = load_dataset("json", data_dir="data/KorMedMCQA/dentist/", split=None)
kormedmcqa_doctor = load_dataset("json", data_dir="data/KorMedMCQA/doctor/", split=None)
kormedmcqa_nurse = load_dataset("json", data_dir="data/KorMedMCQA/nurse/", split=None)
kormedmcqa_pharm = load_dataset("json", data_dir="data/KorMedMCQA/pharm/", split=None)

# Load English datasets
medqa_4_options = load_dataset("json", data_dir="data/MedQA/4_options/", split=None)
medqa_5_options = load_dataset("json", data_dir="data/MedQA/5_options/", split=None)

def tokenize_and_count_length(dataset) -> list[int]:
    lengths = []
    for item in dataset:
        question = item["question"]
        tokenized_text = tokenizer(question, return_tensors="pt")
        token_length = tokenized_text.input_ids.shape[1]
        lengths.append(token_length)
    return lengths

tokenized_data_lengths = {
    "kormedmcqa_dentist": tokenize_and_count_length(kormedmcqa_dentist["train"]),
    "kormedmcqa_doctor": tokenize_and_count_length(kormedmcqa_doctor["train"]),
    "kormedmcqa_nurse": tokenize_and_count_length(kormedmcqa_nurse["train"]),
    "kormedmcqa_pharm": tokenize_and_count_length(kormedmcqa_pharm["train"]),
    "medqa_4_options": tokenize_and_count_length(medqa_4_options["train"]),
    "medqa_5_options": tokenize_and_count_length(medqa_5_options["train"]),
}

df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in tokenized_data_lengths.items()]))
print(df.describe())
#       kormedmcqa_dentist  kormedmcqa_doctor  kormedmcqa_nurse  kormedmcqa_pharm  medqa_4_options  medqa_5_options
#count          297.000000        1890.000000        582.000000        632.000000     10178.000000     10178.000000
#mean           138.750842         220.611111        132.522337        158.642405       229.422382       239.912556
#std             36.998976          82.938008         27.092282         69.006088        77.862043        78.484155
#min             80.000000          80.000000         78.000000         74.000000        65.000000        72.000000
#25%            113.000000         159.000000        112.000000        111.000000       174.000000       184.000000
#50%            134.000000         199.000000        130.000000        144.000000       218.000000       228.000000
#75%            158.000000         272.000000        148.000000        182.000000       271.000000       282.000000
#max            366.000000         555.000000        226.000000        539.000000       949.000000       960.000000