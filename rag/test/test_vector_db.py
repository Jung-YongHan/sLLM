import shutil
import tempfile
import unittest

from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import PersistentChromaDBVectorMemoryConfig

from rag.vector_db import VectorDB


class TestChromaDBVectorMemory(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.config = PersistentChromaDBVectorMemoryConfig(
            collection_name="test_collection",
            persistence_path=self.tempdir,
            k=2,
            score_threshold=0.4,
        )
        self.memory = VectorDB(config=self.config)

        # 기존에 남아있는 데이터 초기화
        await self.memory.clear_memory()

    async def asyncTearDown(self):
        # 테스트 종료 후 메모리와 디렉터리 정리
        await self.memory.clear_memory()
        shutil.rmtree(self.tempdir)

    async def test_add_and_retrieve(self):
        docs = [
            "서울의 날씨는 맑습니다.",
            "서울의 현재 기온은 20도입니다.",
            "울산의 기온은 15도입니다.",
            "부산의 날씨는 흐립니다.",
        ]
        for text in docs:
            content = MemoryContent(
                content=text,
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "weather", "location": "대한민국"},
            )
            await self.memory.add_content(content)

        query = "현재 서울의 날씨를 알려주세요."
        results = await self.memory.retrieve_relevant(query)

        self.assertTrue(any(r == docs[0] for r in results))

        self.assertGreaterEqual(len(results), 0)

    async def test_clear_memory(self):
        # 메모리에 데이터 추가
        await self.test_add_and_retrieve()

        # 메모리 비우기
        await self.memory.clear_memory()

        # 정상 처리되었는지 확인
        results = await self.memory.retrieve_relevant("서울의 날씨는 어떤가요?")
        self.assertEqual(len(results), 0, "Memory should be empty after clearing.")


if __name__ == "__main__":
    unittest.main()
