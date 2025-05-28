import json
import os
import time

from datasets import load_dataset
from google.genai import Client, errors, types
from tqdm import tqdm


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
    generation_config_params = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    return client.models.generate_content(
        model=model,
        contents=contents,
        config=generation_config_params  # Corrected parameter name
    ).text


def reprocess_null_answers(filename, sleep_time):
    """
    지정된 JSONL 파일에서 'answer'가 null인 항목을 찾아 재처리합니다.
    재처리된 내용은 임시 파일에 저장 후 원본 파일을 대체합니다.
    """
    print(f"\n--- '{filename}' 파일의 null 답변 재처리 시작 ---")
    temp_filename = filename + ".tmp"
    reprocessed_count = 0
    null_answers_found = 0
    processed_lines = 0

    try:
        with open(filename, 'r', encoding='utf-8') as infile, \
             open(temp_filename, 'w', encoding='utf-8') as outfile:
            
            lines = infile.readlines()  # tqdm을 위해 전체 라인을 미리 읽음
            progress_bar = tqdm(lines, desc=f"재처리 중 {os.path.basename(filename)}", unit="line")

            for line_num, line_content in enumerate(progress_bar):
                processed_lines += 1
                try:
                    record = json.loads(line_content)
                    if record.get("answer") is None:
                        null_answers_found += 1
                        progress_bar.set_postfix_str(f"Null 발견 (항목 {line_num + 1}), API 요청 중...")
                        
                        new_answer = None
                        max_retries = 3
                        retry_count = 0
                        
                        while retry_count < max_retries:
                            try:
                                new_answer = generate(record["question"])
                                if new_answer is not None:  # 성공적으로 답변을 받으면 루프 탈출
                                    break 
                                print(f"\n항목 {line_num + 1} ('{record['question'][:30]}...') API 응답 null, 재시도 {retry_count + 1}/{max_retries}")
                                time.sleep(sleep_time * (retry_count + 1))  # 재시도 간격 증가
                            except errors.ServerError as e_api:
                                print(f"\n항목 {line_num + 1} 재처리 중 API 오류: {e_api}. {retry_count + 1}/{max_retries}번째 시도 실패.")
                                if retry_count == max_retries - 1:  # 마지막 재시도 실패
                                    print(f"항목 {line_num + 1} API 오류로 재처리 실패. 기존 null 값 유지.")
                                    break
                                time.sleep(sleep_time * (retry_count + 1))
                            except Exception as e_gen:
                                print(f"\n항목 {line_num + 1} 재처리 중 예기치 않은 오류: {e_gen}. {retry_count + 1}/{max_retries}번째 시도 실패.")
                                if retry_count == max_retries - 1:
                                    print(f"항목 {line_num + 1} 예기치 않은 오류로 재처리 실패. 기존 null 값 유지.")
                                    break
                                time.sleep(sleep_time * (retry_count + 1))
                            retry_count += 1
                            
                        if new_answer is not None:
                            record["answer"] = new_answer
                            reprocessed_count += 1
                            progress_bar.set_postfix_str(f"항목 {line_num + 1} 재처리 완료.")
                        else:
                            print(f"\n경고: 항목 {line_num + 1} ('{record['question'][:30]}...') 최대 재시도 후에도 null 응답 또는 오류 발생. 기존 null 값 유지.")

                        if line_num < len(lines) - 1:  # 마지막 라인이 아니면 sleep
                             time.sleep(sleep_time)

                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                    outfile.flush()

                except json.JSONDecodeError:
                    print(f"경고: '{filename}'의 {line_num + 1}번째 줄이 유효한 JSON이 아닙니다. 그대로 복사합니다.")
                    outfile.write(line_content)
                except Exception as e_outer:
                    print(f"\n'{filename}'의 {line_num + 1}번째 줄 처리 중 예기치 않은 오류: {e_outer}. 해당 줄은 건너뜁니다.")

        os.remove(filename)
        os.rename(temp_filename, filename)
        print(f"--- '{filename}' 파일 null 답변 재처리 완료 ---")
        print(f"  총 라인 수: {processed_lines}")
        print(f"  발견된 Null 답변 수: {null_answers_found}")
        print(f"  성공적으로 재처리된 항목 수: {reprocessed_count}")

    except FileNotFoundError:
        print(f"오류: '{filename}' 파일을 찾을 수 없습니다. 재처리를 건너뜁니다.")
    except IOError as e:
        print(f"파일 처리 중 오류 발생 ('{filename}' 또는 '{temp_filename}'): {e}")
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
                print(f"임시 파일 '{temp_filename}' 삭제 완료.")
            except OSError as e_del:
                print(f"임시 파일 '{temp_filename}' 삭제 실패: {e_del}")
    except Exception as e_global:
        print(f"재처리 함수 실행 중 예기치 않은 오류 발생: {e_global}")
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except OSError:
                pass


def process_and_save_dataset(dataset_X, dataset_y, output_filename, data_name: str, sleep_time: int | float):
    """
    데이터셋을 처리하고 결과를 JSONL 파일에 저장합니다.
    오류 발생 시 진행 상황을 저장하고 다음 시작 인덱스를 안내합니다.
    """
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 이어쓰기 전, 기존 파일에 있는 null 답변들을 먼저 재처리 시도
    if os.path.exists(output_filename):
        print(f"\n--- '{output_filename}'의 기존 null 답변에 대한 선행 재처리 시도 ---")
        reprocess_null_answers(output_filename, sleep_time) # sleep_time은 process_and_save_dataset의 인자를 사용
        print(f"--- '{output_filename}' 선행 재처리 시도 완료 ---")

    start_index = 0
    if os.path.exists(output_filename):
        valid_lines = 0
        try:
            with open(output_filename, 'r', encoding='utf-8') as f_check:
                for line_num, line_content in enumerate(f_check):
                    try:
                        json.loads(line_content)
                        valid_lines += 1
                    except json.JSONDecodeError:
                        print(f"경고: '{output_filename}'의 {line_num + 1}번째 줄이 유효한 JSON이 아닙니다. {valid_lines}번째 항목까지 처리된 것으로 간주합니다.")
                        break
            start_index = valid_lines
            if start_index > 0:
                print(f"'{output_filename}' 파일에서 {start_index}개의 유효한 항목을 감지했습니다. 이어서 처리합니다.")
        except Exception as e:
            print(f"'{output_filename}' 파일 읽기 중 오류 발생 (이어쓰기 정보 로드 실패): {e}. 처음부터 시작합니다.")
            start_index = 0
    
    if start_index >= len(dataset_X):
        print(f"'{output_filename}'의 모든 항목({start_index}/{len(dataset_X)})이 이미 처리된 것 같습니다.")
        return len(dataset_X)

    try:
        with open(output_filename, 'a', encoding='utf-8') as f:
            progress_bar = tqdm(range(start_index, len(dataset_X)),
                                  desc=data_name,
                                  total=len(dataset_X),
                                  initial=start_index,
                                  unit="item")

            for i in progress_bar:
                X = dataset_X[i]
                y = dataset_y[i]
                
                try:
                    answer = generate(X)
                    record = {"question": X, "label": y, "answer": answer}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

                    if i < len(dataset_X) - 1:
                        time.sleep(sleep_time)

                except errors.ServerError as e:
                    progress_bar.close()
                    print(f"\nAPI 오류 발생 (항목 인덱스 {i}, 첫 50자 질문: '{X[:50]}...'): {e}.")
                    print(f"현재까지의 진행 상황이 '{output_filename}'에 저장되었습니다.")
                    print(f"다음 실행 시 '{data_name}'에 대해 {i}번째 인덱스부터 재개해야 합니다.")
                    return i
                except Exception as e:
                    progress_bar.close()
                    print(f"\n항목 인덱스 {i} 처리 중 예기치 않은 오류 발생 (첫 50자 질문: '{X[:50]}...'): {e}")
                    print(f"현재까지의 진행 상황이 '{output_filename}'에 저장되었습니다.")
                    print(f"다음 실행 시 '{data_name}'에 대해 {i}번째 인덱스부터 재개해야 합니다.")
                    return i
            
            progress_bar.close()

    except IOError as e:
        print(f"\n파일 처리 중 오류 발생 ('{output_filename}'): {e}")
        return start_index

    print(f"\n{data_name}: 모든 {len(dataset_X)}개 항목 처리 완료. 결과가 '{output_filename}'에 저장되었습니다.")
    return len(dataset_X)

if __name__ == "__main__":
    
    # Load Korean datasets
    kormedmcqa_dentist = load_dataset("json", data_dir="data/KorMedMCQA/dentist/", split="train")
    kormedmcqa_doctor = load_dataset("json", data_dir="data/KorMedMCQA/doctor/", split="train")
    kormedmcqa_nurse = load_dataset("json", data_dir="data/KorMedMCQA/nurse/", split="train")
    kormedmcqa_pharm = load_dataset("json", data_dir="data/KorMedMCQA/pharm/", split="train")
    
    # Load English datasets
    medqa_4_options = load_dataset("json", data_dir="data/MedQA/4_options/", split="train")
    medqa_5_options = load_dataset("json", data_dir="data/MedQA/5_options/", split="train")
    
    processed_files = []
    current_sleep_time = 2.5

    for data_name, dataset in zip(["kormedmcqa_dentist", "kormedmcqa_doctor", "kormedmcqa_nurse", "kormedmcqa_pharm", "medqa_4_options", "medqa_5_options"],
                                 [kormedmcqa_dentist, kormedmcqa_doctor, kormedmcqa_nurse, kormedmcqa_pharm, medqa_4_options, medqa_5_options]):

        print(f"\n--- {data_name} 처리 시작 ---")
        processed_count = process_and_save_dataset(dataset["question"], dataset["answer"], output_filename, data_name, current_sleep_time)
        
        if os.path.exists(output_filename):
            processed_files.append(output_filename)

        if processed_count < len(dataset_X):
            print(f"--- {data_name} 처리가 중단되었습니다. 다음 실행 시 인덱스 {processed_count}부터 재개됩니다. ---")
        else:
            print(f"--- {data_name} 처리 완료 ---")
    
    print("\n모든 데이터셋에 대한 초기 처리 시도 완료.")

    if processed_files:
        print("\n--- 모든 생성된 파일에 대해 Null 답변 재처리 시작 ---")
        for filename_to_reprocess in processed_files:
            reprocess_null_answers(filename_to_reprocess, current_sleep_time)
        print("\n--- 모든 파일에 대한 Null 답변 재처리 시도 완료 ---")
    else:
        print("\n재처리할 파일이 없습니다 (초기 처리에서 파일이 생성되지 않았거나, 설정된 파일이 없음).")