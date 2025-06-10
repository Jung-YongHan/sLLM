from typing import List

from autogen_core.memory import MemoryContent
from autogen_core.memory._base_memory import ContentType
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
)


class VectorDB(ChromaDBVectorMemory):
    def __init__(self, config: PersistentChromaDBVectorMemoryConfig):
        """Initialize the VectorDB with the given configuration.

        Args:
            config (PersistentChromaDBVectorMemoryConfig): The configuration for the vector database.

        Example:
            PersistentChromaDBVectorMemoryConfig(
                collection_name="preferences",
                persistence_path=os.path.join(str(Path.home()), "chroma_db"),
                k=2,
                score_threshold=0.4,
            )
        """
        super().__init__(config)
        self.config = config

    async def add_content(self, content: MemoryContent) -> None:
        """
        content 삽입 함수. 예시의 형태를 따라야 함.
        metadata는 dict 형태로, 알아서 넣으면 됨.

        Args:
            content (MemoryContent): The content to be added to the vector database.

        Example:
            MemoryContent(
                content="The weather should be in metric units",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "preferences", "type": "units"},
            )
        """
        await super().add(content)

    async def retrieve_relevant(self, query: str) -> List[ContentType]:
        """
        query와 연관있는 결과 반환.
        이때, 임계값은 config에서 설정된 score_threshold 사용.

        Args:
            query (str): The content to query the vector database with.
        Example:
            query_content = MemoryContent(
                content="What is the weather in metric units?",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "preferences", "type": "units"},
            )
        """
        results = (await super().query(query=query)).results
        return [r.content for r in results if isinstance(r, MemoryContent)]

    async def clear_memory(self) -> None:
        """DB의 구조는 유지한채, 데이터만 정리"""
        await super().clear()

    async def reset_memory(self) -> None:
        """DB의 구조와 데이터 모두 정리"""
        # allow_reset이 True이어야 함
        self.config.allow_reset = True
        await super().reset()
