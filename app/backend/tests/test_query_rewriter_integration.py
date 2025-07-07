import pytest
import json
import sys
import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.query_rewriter import QueryRewriter
from openai import AsyncAzureOpenAI
from config.config import Settings

pytest_plugins = ["pytest_asyncio"]

# Configure logging for pytest
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)

# Also add a handler specifically for pytest
pytest_logger = logging.getLogger()
pytest_logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
pytest_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)


# Test cases for integration testing
INTEGRATION_TEST_CASES = [
    {
        "query": "Surface Pro는 어떤 특징이 있나요?",
        "locale": "ko-KR",
        "expected_contains": ["마이크로소프트", "서피스 프로", "특징"],
        "expected_not_contains": ["단점", "가격"]
    },
    {
        "query": "m365 제품군의 장점은 무엇인가요?",
        "locale": "ko-KR",
        "expected_contains": ["마이크로소프트", "m365"],
        "expected_not_contains": ["단점", "가격"]
    },
]

@pytest.fixture
def settings():
    """Load settings for tests."""
    return Settings()

@pytest.fixture
async def real_query_rewriter(settings):
    """Create QueryRewriter with real OpenAI client for integration tests."""
    client = AsyncAzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
    )
    
    query_rewriter = QueryRewriter(client=client, settings=settings)
    
    yield query_rewriter
    
    # Properly close the client after tests
    await client.close()

@pytest.mark.parametrize("test_case", INTEGRATION_TEST_CASES)
@pytest.mark.skip(reason="Skipping this for rewrite and plan tests")
@pytest.mark.asyncio
async def test_parametrized_real_query_rewriter(test_case, real_query_rewriter):
    """Parametrized integration test for various queries with real LLM"""
    query = test_case["query"]
    locale = test_case["locale"]
    expected_contains = test_case["expected_contains"]
    
    logger.info(f"Running parametrized integration test for query: {query}")
    
    result = await real_query_rewriter.rewrite_query(query, locale=locale)
    
    logger.info(f"##### Rewrited Search query: {result['search_query']}")
    logger.info(f"##### Rewrited LLM query: {result['llm_query']}")
    
    # Verify response structure
    assert "search_query" in result
    assert "llm_query" in result
    
    # Verify expected content is present (case insensitive)
    search_query_lower = result["search_query"].lower()
    llm_query_lower = result["llm_query"].lower()
    
    for expected in expected_contains:
        expected_lower = expected.lower()
        assert expected_lower in search_query_lower or expected_lower in llm_query_lower, \
            f"Expected '{expected}' not found in either search_query or llm_query"

    for unexpected in test_case.get("expected_not_contains", []):
        unexpected_lower = unexpected.lower()
        assert unexpected_lower not in search_query_lower and unexpected_lower not in llm_query_lower, \
            f"Unexpected '{unexpected}' found in either search_query or llm_query"

    # Small delay between tests
    await asyncio.sleep(1)

@pytest.mark.parametrize("test_case", INTEGRATION_TEST_CASES)
#@pytest.mark.repeat(2)
@pytest.mark.asyncio
async def test_parametrized_real_rewrite_plan(test_case, real_query_rewriter):
    """Parametrized integration test for various queries with real LLM"""
    query = test_case["query"]
    locale = test_case["locale"]
    expected_contains = test_case["expected_contains"]
    
    logger.info(f"Running parametrized integration test for query: {query}")
    
    result = await real_query_rewriter.rewrite_plan_query(query, locale=locale)
    
    logger.info(f"##### Rewrited and planed Search query: {result['search_queries']}")
    logger.info(f"##### Rewrited and planed LLM query: {result['expanded_query']}")
    
    # Verify response structure
    assert "search_queries" in result
    assert "expanded_query" in result

    # Verify expected content is present (case insensitive)
    search_query_list = result["search_queries"]
    llm_query_lower = result["expanded_query"].lower()

    
    for expected in expected_contains:
        expected_lower = expected.lower()
        # Check if expected text is found in any search query element OR in llm_query
        found_in_search = any(expected_lower in query.lower() for query in search_query_list)
        found_in_llm = expected_lower in llm_query_lower
        
        assert found_in_search or found_in_llm, \
            f"Expected '{expected}' not found in either search_query or llm_query"

    for unexpected in test_case.get("expected_not_contains", []):
        unexpected_lower = unexpected.lower()
        # Check that unexpected text is NOT found in any search query element AND NOT in llm_query
        found_in_search = any(unexpected_lower in query.lower() for query in search_query_list)
        found_in_llm = unexpected_lower in llm_query_lower
        
        assert not found_in_search and not found_in_llm, \
            f"Unexpected '{unexpected}' found in either search_query or llm_query"

    # Small delay between tests
    await asyncio.sleep(1)