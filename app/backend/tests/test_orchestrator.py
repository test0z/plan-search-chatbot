import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.orchestrator import Orchestrator
from model.models import ChatMessage
from utils.enum import SearchEngine


class DummySettings:
    AZURE_OPENAI_API_KEY = "dummy"
    AZURE_OPENAI_API_VERSION = "dummy"
    AZURE_OPENAI_ENDPOINT = "dummy"
    AZURE_OPENAI_DEPLOYMENT_NAME = "dummy"
    AZURE_OPENAI_QUERY_DEPLOYMENT_NAME = "dummy"
    MAX_TOKENS = 128
    DEFAULT_TEMPERATURE = 0.5
    TIME_ZONE = "Asia/Seoul"


@pytest.fixture
def orchestrator():
    settings = DummySettings()
    orch = Orchestrator(settings)
    orch.client = MagicMock()
    orch.query_rewriter = MagicMock()
    orch.search_crawler = MagicMock()
    orch.bing_grounding_search = MagicMock()
    return orch


@pytest.mark.asyncio
async def test_generate_response_no_query_rewrite(orchestrator):
    # Mock the OpenAI response properly
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello!"

    orchestrator.client.chat.completions.create = AsyncMock(return_value=mock_response)
    orchestrator.search_crawler.search = MagicMock(return_value=[])
    orchestrator.search_crawler.extract_contexts_async = AsyncMock(return_value=[])

    messages = [ChatMessage(role="user", content="Hi")]
    gen = orchestrator.generate_response(
        messages,
        query_rewrite=False,
        search_engine=SearchEngine.GOOGLE_SEARCH_CRAWLING,
        search_crawler=orchestrator.search_crawler,
    )
    result = [
        x
        async for x in gen
        if x is not None
        and not x.startswith("data:")
        and not x.startswith("Response generated")
    ]
    assert len(result) > 0
    assert result[0] == "Hello!"


@pytest.mark.asyncio
async def test_generate_response_with_query_rewrite_and_search(orchestrator):
    orchestrator.query_rewriter.rewrite_query = AsyncMock(
        return_value={
            "search_query": "Microsoft product info",
            "llm_query": "Tell me about Microsoft products.",
        }
    )
    orchestrator.search_crawler.search = MagicMock(
        return_value=[
            {"link": "http://a", "snippet": "info1"},
            {"link": "http://b", "snippet": "info2"},
        ]
    )
    orchestrator.search_crawler.extract_contexts_async = AsyncMock(
        return_value=["context1", "context2"]
    )

    # Mock the OpenAI response properly
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Microsoft info response"

    orchestrator.client.chat.completions.create = AsyncMock(return_value=mock_response)

    messages = [ChatMessage(role="user", content="What about Microsoft?")]
    gen = orchestrator.generate_response(
        messages,
        query_rewrite=True,
        search_engine=SearchEngine.GOOGLE_SEARCH_CRAWLING,
        search_crawler=orchestrator.search_crawler,
    )
    result = [x async for x in gen if x is not None]
    assert len(result) > 0
    print("result:", result)
    if result:
        assert "Microsoft info response" in result


@pytest.mark.asyncio
async def test_generate_response_streaming(orchestrator):
    class DummyChunk:
        def __init__(self, content):
            self.choices = [MagicMock(delta=MagicMock(content=content))]

    async def dummy_stream(*args, **kwargs):
        for c in ["A", "B", "C"]:
            yield DummyChunk(c)

    orchestrator.client.chat.completions.create = AsyncMock(side_effect=dummy_stream)
    orchestrator.search_crawler.search = MagicMock(return_value=[])
    orchestrator.search_crawler.extract_contexts_async = AsyncMock(return_value=[])

    messages = [ChatMessage(role="user", content="Stream test")]
    gen = orchestrator.generate_response(
        messages,
        stream=True,
        query_rewrite=False,
        search_engine=SearchEngine.GOOGLE_SEARCH_CRAWLING,
        search_crawler=orchestrator.search_crawler,
    )
    result = [
        x
        async for x in gen
        if x is not None
        and not x.startswith("data:")
        and not x.startswith("Response generated")
        and x.strip()
    ]
    assert len(result) == 3
    assert "".join(result) == "ABC"


@pytest.mark.asyncio
async def test_generate_response_exception(orchestrator):
    orchestrator.client.chat.completions.create = AsyncMock(
        side_effect=Exception("fail")
    )
    orchestrator.search_crawler.search = MagicMock(return_value=[])
    orchestrator.search_crawler.extract_contexts_async = AsyncMock(return_value=[])

    messages = [ChatMessage(role="user", content="Hi")]
    gen = orchestrator.generate_response(
        messages,
        query_rewrite=False,
        search_engine=SearchEngine.GOOGLE_SEARCH_CRAWLING,
        search_crawler=orchestrator.search_crawler,
    )
    result = [x async for x in gen]
    assert len(result) > 0
    assert "Error: fail" in result[0]
