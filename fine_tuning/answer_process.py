import re

class AnswerProcessor:
    def __init__(self):
        self.answer_pattern = {
            "A": [r"[Aa](?:\W|$)", r"[1](?:\W|$)"],
            "B": [r"[Bb](?:\W|$)", r"[2](?:\W|$)"],
            "C": [r"[Cc](?:\W|$)", r"[3](?:\W|$)"],
            "D": [r"[Dd](?:\W|$)", r"[4](?:\W|$)"],
            "E": [r"[Ee](?:\W|$)", r"[5](?:\W|$)"],
        }
        
    def preprocess_answer(self, answer: str) -> str:
        answer_lines = [
            line.split("Answer")[-1] if "Answer" in line else line.split("정답")[-1]
            for line in answer.strip().split("\n")
            if "정답" in line or "Answer" in line
        ]
        
        for line in answer_lines:
            for key, patterns in self.answer_pattern.items():
                if any(re.search(pattern, line) for pattern in patterns):
                    return key
                
        return "Z"