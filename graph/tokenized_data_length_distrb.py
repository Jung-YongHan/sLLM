import pandas as pd
from datasets import load_dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

# Load Korean datasets
kormedmcqa_dentist = load_dataset("json", data_dir="fine_tuning/data/KorMedMCQA/dentist/", split=None)
kormedmcqa_doctor = load_dataset("json", data_dir="fine_tuning/data/KorMedMCQA/doctor/", split=None)
kormedmcqa_nurse = load_dataset("json", data_dir="fine_tuning/data/KorMedMCQA/nurse/", split=None)
kormedmcqa_pharm = load_dataset("json", data_dir="fine_tuning/data/KorMedMCQA/pharm/", split=None)

# Load English datasets
medqa_4_options = load_dataset("json", data_dir="fine_tuning/data/MedQA/4_options/", split=None)
medqa_5_options = load_dataset("json", data_dir="fine_tuning/data/MedQA/5_options/", split=None)

# Load additional datasets to finetune
asan_healthinfo_data = load_dataset("json", data_files="fine_tuning/data/Asan-AMC-Healthinfo.jsonl", split=None)
gen_gpt_data = load_dataset("json", data_files="fine_tuning/data/GenMedGPT_5k_ko.jsonl", split=None)

# Load distillation datasets
distillation_data = load_dataset("json", data_dir="fine_tuning/data/distillation_gemini/", split=None)

def tokenize_and_count_length(dataset) -> list[int]:
    lengths = []
    for item in dataset:
        text = item["text"]
        tokenized_text = tokenizer.apply_chat_template(text, tokenize=True, add_special_tokens=True)
        lengths.append(len(tokenized_text))
    return lengths

tokenized_data_lengths = {
    "kormedmcqa_dentist": tokenize_and_count_length(kormedmcqa_dentist["train"]),
    "kormedmcqa_doctor": tokenize_and_count_length(kormedmcqa_doctor["train"]),
    "kormedmcqa_nurse": tokenize_and_count_length(kormedmcqa_nurse["train"]),
    "kormedmcqa_pharm": tokenize_and_count_length(kormedmcqa_pharm["train"]),
    "medqa_4_options": tokenize_and_count_length(medqa_4_options["train"]),
    "medqa_5_options": tokenize_and_count_length(medqa_5_options["train"]),
    "asan_healthinfo_data": tokenize_and_count_length(asan_healthinfo_data["train"]),
    "gen_gpt_data": tokenize_and_count_length(gen_gpt_data["train"]),
    "distillation_data": tokenize_and_count_length(distillation_data["train"]),
}

df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in tokenized_data_lengths.items()]))
print(df.describe())
#       kormedmcqa_dentist  kormedmcqa_doctor  kormedmcqa_nurse  kormedmcqa_pharm  medqa_4_options  medqa_5_options  asan_healthinfo_data  gen_gpt_data  distillation_data
#count          297.000000        1890.000000        582.000000        632.000000     10178.000000     10178.000000          19156.000000   5451.000000        3401.000000
#mean           188.898990         267.586772        180.723368        206.844937       274.046866       283.536942            167.570735    130.793983         791.352837
#std             42.707007          83.851403         30.370587         72.291526        78.643231        79.415921            138.987324     44.553746         303.240432
#min            122.000000         122.000000        123.000000        114.000000       107.000000       113.000000             53.000000     54.000000         119.000000
#25%            158.000000         206.000000        157.000000        155.000000       218.000000       227.000000             82.000000     97.000000         576.000000
#50%            183.000000         247.000000        178.000000        193.000000       263.000000       272.000000            128.000000    121.000000         745.000000
#75%            211.000000         320.000000        199.000000        237.000000       316.000000       326.000000            203.000000    158.000000         945.000000
#max            458.000000         607.000000        281.000000        587.000000       992.000000      1009.000000           2933.000000    358.000000        2747.000000