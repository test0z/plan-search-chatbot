import asyncio
import pytest
import logging
from typing import List, Dict, Any
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Settings
from model.models import ChatMessage
from services.plan_executor import PlanExecutor
from services.search_crawler import GoogleSearchCrawler
from services.query_rewriter import QueryRewriter
from utils.enum import SearchEngine

# This plugin is needed for pytest.mark.asyncio
pytest_plugins = ["pytest_asyncio"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_CASES = [
    {
        "question": "마이크로소프트 서피스 노트북의 카메라 성능은 어떤가요?",
    }
]

@pytest.fixture
def settings():
    """Load settings for tests."""
    return Settings()

@pytest.fixture
def plan_executor(settings):
    """Create and configure a PlanExecutor instance."""
    executor = PlanExecutor(settings)
    
    redis_config = {
        "host": settings.REDIS_HOST,
        "port": settings.REDIS_PORT,
        "password": settings.REDIS_PASSWORD,
        "db": settings.REDIS_DB,
        "decode_responses": True
    }
    
    executor.query_rewriter = QueryRewriter(client=executor.client, settings=settings)
    
    return executor

async def collect_streaming_response(response_generator):
    """Collect streaming response into a single string."""
    response_chunks = []
    async for chunk in response_generator:
        if chunk.startswith("data: "):
            chunk = chunk[6:]
        response_chunks.append(chunk)
    return "".join(response_chunks)

@pytest.mark.asyncio
async def test_plan_executor_with_real_questions(plan_executor):
    """Test PlanExecutor with real-world questions."""
    for test_case in TEST_CASES:
        question = test_case["question"]
        
        logger.info(f"Testing question: {question}")
        
        messages = [ChatMessage(role="user", content=question)]
        
        response_generator = plan_executor.generate_response(
            messages=messages,
            query_rewrite=True,
            search_engine=SearchEngine.GOOGLE_SEARCH_CRAWLING,
            stream=True  
        )
        
        response = await collect_streaming_response(response_generator)
        
        logger.info(f"Response: {response}")
        
        # Simply verify that we got a non-empty response
        assert response, f"Response should not be empty for question: {question}"
        assert len(response) > 0, f"Response should have content for question: {question}"
        
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(pytest.main(["-xvs", __file__]))
