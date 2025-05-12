import json
import re
from sklearn.metrics import accuracy_score, f1_score

def extract_prediction(answer_text):
    if answer_text is None:
        # print(f"Warning: answer_text is None. Cannot extract prediction.") # 디버깅용 메시지 (선택 사항)
        return None

    # 정답: **X** 또는 정답: X 또는 정답:X 형태의 예측을 추출
    # 패턴 순서 중요: 더 구체적인 패턴을 먼저 시도
    
    # 패턴 1: 정답: **[A-E]** (해설: 또는 이유: 또는 줄바꿈 또는 문자열 끝)
    match = re.search(r"정답: \*\*([A-E])\*\*(?:\n\n(?:해설:|이유:)|$)", answer_text)
    if match:
        return match.group(1)

    # 패턴 2: 정답: [A-E] (해설: 또는 이유: 또는 줄바꿈 또는 문자열 끝)
    match = re.search(r"정답: ([A-E])(?:\n\n(?:해설:|이유:)|$)", answer_text)
    if match:
        return match.group(1)

    # 패턴 3: ## 정답: [A-E] (해설: 또는 이유: 또는 줄바꿈 또는 문자열 끝)
    match = re.search(r"## 정답: ([A-E])(?:\n\n(?:해설:|이유:)|$)", answer_text)
    if match:
        return match.group(1)
        
    # 패턴 4: 정답: **[A-E]** (단순)
    match = re.search(r"정답: \*\*([A-E])\*\*", answer_text)
    if match:
        return match.group(1)

    # 패턴 5: 정답: [A-E] (단순)
    match = re.search(r"정답: ([A-E])", answer_text)
    if match:
        return match.group(1)
    
    # 패턴 6: 정답:[A-E] (공백 없음)
    match = re.search(r"정답:([A-E])", answer_text)
    if match:
        return match.group(1)
    
    # 패턴 8: 정답: **[A-E]
    match = re.search(r"정답: \*\*([A-E])", answer_text)
    if match:
        return match.group(1)

    # 패턴 9: 단순하게 첫 줄에서 A, B, C, D, E 중 하나로 시작하고 그 뒤에 콜론이나 줄바꿈이 오는 경우
    first_line = answer_text.split('\n')[0].strip()
    simple_match = re.match(r"^([A-E])(?:[:\s]|$)", first_line) # A:, A , A\n
    if simple_match and len(first_line) <= 5 : # "Apple" 같은 단어가 아닌 "A" 또는 "A:" 같은 짧은 문자열
        if "정답" in first_line or len(first_line) < 3: # 추가적인 휴리스틱
            return simple_match.group(1)

    # 패턴 10: 답변이 단일 문자 A-E인 경우
    if len(answer_text.strip()) == 1 and answer_text.strip() in ['A', 'B', 'C', 'D', 'E']:
        return answer_text.strip()

    print(f"Could not extract prediction from: {answer_text[:100]}") # 디버깅용
    return None

def evaluate_file(file_path):
    true_labels = []
    predicted_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1): # 줄 번호 추가
            if not line.strip(): # 빈 줄 스킵
                continue
            try:
                data = json.loads(line)
                if "label" not in data: # Check for "label"
                    print(f"Warning: Skipping line {line_number} in {file_path} due to missing 'label' field. Line: {line.strip()[:100]}")
                    continue
                        
                true_labels.append(data["label"])
                
                # Safely get the answer and initialize prediction
                answer_value = data.get("answer") 
                prediction = None # Initialize prediction

                if answer_value is None:
                    # If 'answer' is null (None), prediction remains None.
                    # extract_prediction will not be called.
                    pass
                elif isinstance(answer_value, str):
                    # If 'answer' is a string, attempt to extract the prediction.
                    prediction = extract_prediction(answer_value)
                else:
                    # Handles other non-string, non-None types.
                    print(f"Warning: 'answer' field at line {line_number} in {file_path} has unexpected type: {type(answer_value)}. Value: {str(answer_value)[:50]}. Treating as unparsable.")
                    # prediction remains None
                
                if prediction:
                    predicted_labels.append(prediction)
                else:
                    predicted_labels.append(None)
                    answer_preview = str(answer_value)[:50] if answer_value is not None else "None"
                    print(f"Warning: Could not parse prediction for line {line_number} in {file_path}.")
            except json.JSONDecodeError as e:
                print(f"Skipping line {line_number} due to JSONDecodeError: {e} in file {file_path} for line: {line.strip()[:100]}")
            except KeyError as e: 
                print(f"Skipping line {line_number} due to KeyError: {e} in file {file_path} for line: {line.strip()[:100]}")

    if not true_labels:
        print(f"No data found or processed in {file_path}. Cannot calculate metrics.")
        return None, None, 0, 0

    # 유효한 예측만 필터링
    filtered_true = []
    filtered_predicted = []
    valid_prediction_count = 0

    for i in range(len(true_labels)):
        if predicted_labels[i] is not None:
            filtered_true.append(true_labels[i])
            filtered_predicted.append(predicted_labels[i])
            valid_prediction_count +=1
        
    total_items = len(true_labels)
    
    if not filtered_true: 
        print(f"No valid predictions to evaluate in {file_path}. Total items: {total_items}")
        return 0, 0, total_items, 0

    # sklearn.metrics는 문자열 레이블을 직접 지원
    accuracy = accuracy_score(filtered_true, filtered_predicted)
    # labels 파라미터를 사용하여 모든 가능한 클래스 (A-E)에 대해 F1 점수를 계산하도록 명시
    f1 = f1_score(filtered_true, filtered_predicted, average='macro', zero_division=0, labels=['A', 'B', 'C', 'D', 'E']) 
    
    return accuracy, f1, total_items, valid_prediction_count

def main():
    file_paths = [
        'h:\\학과\\석사 1학기\\자연어처리이론과언어모델\\sLLM\\distillation_gemini\\distillation_dentist.jsonl',
        'h:\\학과\\석사 1학기\\자연어처리이론과언어모델\\sLLM\\distillation_gemini\\distillation_doctor.jsonl',
        'h:\\학과\\석사 1학기\\자연어처리이론과언어모델\\sLLM\\distillation_gemini\\distillation_nurse.jsonl',
        'h:\\학과\\석사 1학기\\자연어처리이론과언어모델\\sLLM\\distillation_gemini\\distillation_pharm.jsonl'
    ]

    total_processed_items_overall = 0
    total_valid_predictions_overall = 0

    for fp in file_paths:
        print(f"\nProcessing {fp}...")
        accuracy, f1, num_items, num_valid = evaluate_file(fp)
        
        if accuracy is not None:
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Macro: {f1:.4f}")
            print(f"  Total items processed: {num_items}")
            print(f"  Valid predictions evaluated: {num_valid}")
            if num_items != num_valid:
                print(f"  Items with failed prediction extraction: {num_items - num_valid}")
            
            total_processed_items_overall += num_items
            total_valid_predictions_overall += num_valid

        print("-" * 30)
    
    # 전체 파일에 대한 요약
    print("\nOverall Summary:")
    print(f"  Total items processed across all files: {total_processed_items_overall}")
    print(f"  Total valid predictions evaluated across all files: {total_valid_predictions_overall}")
    print("-" * 30)


if __name__ == "__main__":
    main()