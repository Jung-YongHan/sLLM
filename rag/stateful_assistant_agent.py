from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from rag.vectordb_factory import create_preferences_memory


class StatefulAssistantAgent(AssistantAgent):
    """
    memory를 갖는 AssistantAgent.
    해당 Agent를 상속받아 사용하기
    """

    def __init__(self, name: str, model_client: OpenAIChatCompletionClient, **kwargs):
        super().__init__(
            name=name,
            model_client=model_client,
            memory=[create_preferences_memory()],
            system_message=kwargs.get("system_message", None),
            output_content_type=kwargs.get("output_content_type", None),
        )
