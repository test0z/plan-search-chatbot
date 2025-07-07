import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from services.plan_executor import PlanExecutor
from model.models import ChatMessage
from utils.enum import SearchEngine
from config.config import Settings


@pytest.fixture
def mock_settings():
    return Settings(
        AZURE_OPENAI_API_KEY="test_key",
        AZURE_OPENAI_API_VERSION="test_version",
        AZURE_OPENAI_ENDPOINT="https://test.openai.azure.com",
        AZURE_OPENAI_DEPLOYMENT_NAME="test_deployment",
        AZURE_OPENAI_QUERY_DEPLOYMENT_NAME="test_query_deployment",
        MAX_TOKENS=1000,
        DEFAULT_TEMPERATURE=0.7,
        
    )


@pytest.fixture
def plan_executor(mock_settings):
    executor = PlanExecutor(settings=mock_settings)
    executor.client = AsyncMock()
    executor.query_rewriter = AsyncMock()
    return executor


@pytest.mark.asyncio
async def test_generate_response_no_question(plan_executor):
    """Test response when no question is provided."""
    messages = []
    
    response = [msg async for msg in plan_executor.generate_response(messages)]
    
    assert response == ["질문을 입력해주세요."]
    plan_executor.client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_with_query_rewrite(plan_executor):
    """Test response generation with query rewriting enabled."""
    messages = [ChatMessage(role="user", content="What is Galaxy s25?")]
    
    # Mock query rewriter
    plan_executor.query_rewriter.rewrite_query.return_value = {
        "llm_query": "Microsoft Galaxy s25 features specifications release date"
    }
    
    # Mock planner response with proper JSON format
    planner_completion = AsyncMock()
    planner_json = json.dumps({
        "search_queries": [
            "Microsoft Galaxy s25 specifications",
            "Galaxy s25 features",
            "Galaxy s25 release date"
        ]
    })
    planner_completion.choices = [
        MagicMock(message=MagicMock(content=planner_json))
    ]
    
    # Mock search results
    plan_executor.search_crawler.search = MagicMock(return_value = [
        {"link": "https://Microsoft.com/galaxy-s25", "snippet": "Galaxy s25 info"}
    ])
    plan_executor.search_crawler.extract_contexts_async = AsyncMock(return_value=[
        "The Galaxy s25 features a 6.1-inch display with 120Hz refresh rate."
    ])
    
    # Mock final answer response
    answer_completion = AsyncMock()
    answer_completion.choices = [
        MagicMock(message=MagicMock(content="✨ The Microsoft Galaxy s25 is a flagship phone with...\n\n[More information](https://Microsoft.com/galaxy-s25)"))
    ]
    
    plan_executor.client.chat.completions.create.side_effect = [
        planner_completion,
        answer_completion
    ]
    
    response = [msg async for msg in plan_executor.generate_response(
        messages,
        query_rewrite=True,
        search_engine=SearchEngine.GOOGLE_SEARCH_CRAWLING  # Fixed search engine value
    )]
    
    # Filter out the streaming progress messages and find the actual content
    actual_content = [msg for msg in response if not msg.startswith("data:")]
    assert len(actual_content) > 0
    assert "✨" in actual_content[0] or "검색" in actual_content[0]


@pytest.mark.asyncio
async def test_generate_response_streaming(plan_executor):
    """Test streaming response generation."""
    messages = [ChatMessage(role="user", content="What is Galaxy s25?")]
    
    # Mock planner response with proper JSON format
    planner_completion = AsyncMock()
    planner_json = json.dumps({
        "search_queries": [
            "Microsoft Galaxy s25 specifications",
            "Galaxy s25 features"
        ]
    })
    planner_completion.choices = [
        MagicMock(message=MagicMock(content=planner_json))
    ]
    
    plan_executor.search_crawler.search = MagicMock(return_value=[
        {"link": "https://Microsoft.com/galaxy-s25", "snippet": "Galaxy s25 info"}
    ])
    plan_executor.search_crawler.extract_contexts_async = AsyncMock(return_value=[
        "The Galaxy s25 features a 6.1-inch display with 120Hz refresh rate."
    ])
    
    # Mock streaming chunks
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="The Galaxy"))]
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content=" s25 is"))]
    chunk3 = MagicMock()
    chunk3.choices = [MagicMock(delta=MagicMock(content=" a flagship phone"))]
    
    # Set up side effects for completions.create
    def side_effect(*args, **kwargs):
        if kwargs.get("stream", False):
            async def async_generator():
                for chunk in [chunk1, chunk2, chunk3]:
                    yield chunk
            return async_generator()
        return planner_completion
    
    plan_executor.client.chat.completions.create.side_effect = side_effect
    
    response = [msg async for msg in plan_executor.generate_response(
        messages,
        query_rewrite=False,
        search_engine=SearchEngine.GOOGLE_SEARCH_CRAWLING,
        search_crawler=plan_executor.search_crawler,
        stream=True
    )]
    
    # Since the exact order and content of streaming messages can vary,
    # check for the key components instead of exact equality
    for expected in ["질문 분석", "검색 계획", "검색 중", "답변 생성"]:
        assert any(expected in msg for msg in response)
    
    # Check that the final content pieces are present
    final_pieces = ["The Galaxy", " s25 is", " a flagship phone"]
    for piece in final_pieces:
        assert piece in response


@pytest.mark.asyncio
async def test_generate_response_no_contexts(plan_executor):
    """Test response when no search contexts are found."""
    messages = [ChatMessage(role="user", content="What is Galaxy s25?")]
    
    # Mock planner response with proper JSON format
    planner_completion = AsyncMock()
    planner_json = json.dumps({
        "search_queries": [
            "Microsoft Galaxy s25 specifications",
            "Galaxy s25 features"
        ]
    })
    planner_completion.choices = [
        MagicMock(message=MagicMock(content=planner_json))
    ]
    
    # Mock empty search results and empty contexts
    plan_executor.search_crawler.search = MagicMock(return_value=[])
    plan_executor.search_crawler.extract_contexts_async = AsyncMock(return_value=[])
    
    # Set up the completion side effects for the planner and the final message
    final_completion = AsyncMock()
    final_completion.choices = [
        MagicMock(message=MagicMock(content="검색결과가 부족하여 답변을 할 수 없습니다."))
    ]
    
    plan_executor.client.chat.completions.create.side_effect = [
        planner_completion,
        final_completion
    ]
    
    response = [msg async for msg in plan_executor.generate_response(
        messages,
        query_rewrite=False,
        search_engine=SearchEngine.GOOGLE_SEARCH_CRAWLING
    )]
    
    assert "검색결과가 부족하여 답변을 할 수 없습니다." in response

    @pytest.mark.asyncio
    async def test_generate_response_error_on_planner(plan_executor):
        """Test error handling when planner API call fails."""
        messages = [ChatMessage(role="user", content="Tell me about the moon.")]

        # Simulate planner API failure
        plan_executor.client.chat.completions.create.side_effect = Exception("Planner API Failure")

        response = [msg async for msg in plan_executor.generate_response(messages, query_rewrite=False)]
        assert len(response) == 1
        assert "Error:" in response[0]
        assert "Planner API Failure" in response[0]


    @pytest.mark.asyncio
    async def test_generate_response_error_on_answer_generation(plan_executor):
        """Test error handling when answer generation API call fails after planner succeeds."""
        messages = [ChatMessage(role="user", content="Tell me about the sun.")]

        # Mock planner response
        planner_completion = MagicMock()
        planner_json = json.dumps({"search_queries": ["sun facts"]})
        planner_completion.choices = [MagicMock(message=MagicMock(content=planner_json))]

        # Set up completions.create to succeed for planner, fail for answer
        def side_effect(*args, **kwargs):
            if not hasattr(side_effect, "called"):
                side_effect.called = True
                return planner_completion
            raise Exception("Answer API Failure")
        plan_executor.client.chat.completions.create.side_effect = side_effect

        plan_executor.search_crawler.search = MagicMock(return_value=[
            {"link": "https://nasa.gov/sun", "snippet": "The Sun is a star."}
        ])
        plan_executor.search_crawler.extract_contexts_async = AsyncMock(return_value=[
            "The Sun is a star at the center of the Solar System."
        ])

        response = [msg async for msg in plan_executor.generate_response(messages, query_rewrite=False, stream=False)]
        assert any("Error:" in r for r in response)
        assert any("Answer API Failure" in r for r in response)


    @pytest.mark.asyncio
    async def test_generate_response_streaming_error(plan_executor):
        """Test error handling in streaming mode."""
        messages = [ChatMessage(role="user", content="Tell me about Mars.")]

        # Mock planner response
        planner_completion = MagicMock()
        planner_json = json.dumps({"search_queries": ["Mars facts"]})
        planner_completion.choices = [MagicMock(message=MagicMock(content=planner_json))]

        # completions.create: planner ok, answer raises in streaming
        def side_effect(*args, **kwargs):
            if kwargs.get("stream", False):
                async def error_stream():
                    raise Exception("Streaming API Error")
                    yield  # for async generator compliance
                return error_stream()
            return planner_completion
        plan_executor.client.chat.completions.create.side_effect = side_effect

        plan_executor.search_crawler.search = MagicMock(return_value=[
            {"link": "https://nasa.gov/mars", "snippet": "Mars is the fourth planet."}
        ])
        plan_executor.search_crawler.extract_contexts_async = AsyncMock(return_value=[
            "Mars is the fourth planet from the Sun."
        ])

        response = [msg async for msg in plan_executor.generate_response(messages, query_rewrite=False, stream=True)]
        assert any("Error:" in r for r in response)
        assert any("Streaming API Error" in r for r in response)
