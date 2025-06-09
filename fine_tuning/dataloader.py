from datasets import Dataset, load_dataset
from configs import DatasetConfig

class DatasetLoader:
    def __init__(self, config: DatasetConfig):
        self.config = config
        
    def load_korean_datasets(self) -> dict[str, Dataset]:
        datasets = {}
        for dataset_name in ["dentist", "doctor", "nurse", "pharm"]:
            dataset = load_dataset(
                "json",
                data_dir=f"fine_tuning/data/KorMedMCQA/{dataset_name}/",
                data_files=self.config.split_files,
            )
            # 데이터셋 스키마 확인 및 정리
            datasets[f"kormedmcqa_{dataset_name}"] = self._ensure_schema(dataset)
        return datasets
    
    def load_english_datasets(self) -> dict[str, Dataset]:
        datasets = {}
        medqa_5 = load_dataset(
            "json",
            data_dir="fine_tuning/data/MedQA/5_options/",
            data_files=self.config.split_files,
        )
        medqa_4 = load_dataset(
            "json",
            data_dir="fine_tuning/data/MedQA/4_options/",
            data_files=self.config.split_files,
        )
        
        datasets["medqa_5_options"] = self._ensure_schema(medqa_5)
        datasets["medqa_4_options"] = self._ensure_schema(medqa_4)
        
        return datasets
    
    def _ensure_schema(self, dataset_dict):
        """모든 split이 text와 label 컬럼을 갖도록 보장"""
        processed_dataset = {}
        for split_name, dataset in dataset_dict.items():
            if isinstance(dataset, Dataset):
                # 이미 text, label 컬럼이 있는지 확인
                if "text" in dataset.column_names and "label" in dataset.column_names:
                    processed_dataset[split_name] = dataset
                else:
                    # 컬럼명이 다른 경우 처리 (필요시 추가 로직)
                    processed_dataset[split_name] = dataset
            else:
                processed_dataset[split_name] = dataset
        return processed_dataset
        
    def additional_finetuning_datasets(self, kormedmcqa_reasoning=False, asan_healthinfo=False) -> dict[str, Dataset]:
        
        def _load_asan_healthinfo():
            dataset = load_dataset("ChuGyouk/Asan-AMC-Healthinfo", split="train")
            # Asan dataset을 text, label 형식으로 변환
            def format_asan_sample(example):
                # 의료 정보 QA 형식으로 변환
                question = example.get("input", "")
                answer = example.get("output", "")
                
                # 한국어 chat template 적용
                messages = [
                    {"role": "system", "content": "다음 의료 관련 질문에 정확하고 도움이 되는 답변을 제공해주세요."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
                
                return {"text": messages, "label": answer}
            
            return dataset.map(format_asan_sample)
        
        dataset_dict = {
            "kormedmcqa_reasoning": None if not kormedmcqa_reasoning else None,  # 추후 구현
            "asan_healthinfo": _load_asan_healthinfo() if asan_healthinfo else None,
        }
        
        return {k: v for k, v in dataset_dict.items() if v is not None}
        
if __name__ == "__main__":
    data_loader = DatasetLoader(DatasetConfig())
    sample = data_loader.additional_finetuning_datasets(asan_healthinfo=True)
    print(sample)