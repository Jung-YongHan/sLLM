from datasets import Dataset

train_dataset = Dataset.from_dict({
    "messages":[
        [
            {"role": "user", "content": "I want to know about the symptoms of COVID-19."},
            {"role": "assistant", "content": "COVID-19 symptoms include fever, cough, and shortness of breath."},
        ]
    ]
})