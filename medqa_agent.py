from typing import Literal

from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

from rag.stateful_assistant_agent import StatefulAssistantAgent

model_client = OpenAIChatCompletionClient(
    model="medqa-agent",  # 서빙 시 사용되는 모델 이름
    base_url="http://localhost:8000/v1",
    model_info=ModelInfo(
        vision=False,
        function_calling=False,
        json_output=True,
        family="unknown",
        structured_output=True,
    ),
    api_key="anything",
)


class MedQAResponse(BaseModel):
    thought: str
    response: Literal["A", "B", "C", "D", "E"]


class MedQAAgent(StatefulAssistantAgent):

    def __init__(self):
        super().__init__(
            name="MedQAAgent",
            model_client=model_client,
            system_message="You are a medical question answering agent.",
            output_content_type=MedQAResponse,
        )

    async def completion(self, query: str | list[str]) -> str:
        if isinstance(query, str):
            query = [query]

        results = []
        for q in query:
            resp = await self.run(task=q)
            print(f"질의: {q}")
            print(f"응답: {resp.messages[-1].content.response}")
            results.append(resp.messages[-1].content.response)

        # query와 key-value 쌍으로 합쳐서 반환
        return dict(zip(query, results))


if __name__ == "__main__":
    import asyncio

    agent = MedQAAgent()
    query = """
A 21-year-old sexually active male complains of fever.
pain during urination, and inflammation and pain in the right knee.
A culture of the joint fluid shows a bacteria that does not ferment maltose and has no polysaccharide capsule.
The physician orders antibiotic therapy for the patient.
The mechanism of action of action of the medication given blocks cell wall synthesis, which of the following was given?

"options"
    "A": "Chloramphenicol"
    "B": "Gentamicin"
    "C": "Ciprofloxacin"
    "D": "Ceftriaxone"
    "E": "Trimethoprim"
"""

    result = asyncio.run(agent.completion(query=query))
    print("Final Result:", result)
