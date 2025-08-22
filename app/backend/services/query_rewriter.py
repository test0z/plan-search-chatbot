import json
import logging
import os
import pytz
from typing import Dict, Optional
from datetime import datetime
from config.config import Settings
from langchain.prompts import load_prompt
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

QUERY_REWRITE_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "query_rewrite_prompt.yaml"),
    encoding="utf-8",
)
REWRITE_PLAN_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "rewrite_plan_prompt_ko.yaml"),
    encoding="utf-8",
)


class QueryRewriter:
    """
    Class for rewriting user queries for search optimization and LLM processing.
    """

    def __init__(
        self, client: Optional[AzureOpenAI] = None, settings: Optional[Settings] = None
    ):
        """
        Initialize the QueryRewriter with an OpenAI client.

        Args:
            client: AzureOpenAI client instance from Orchestrator
            settings: Application settings
        """
        self.client = client
        self.settings = settings

        # Make sure we have deployment name from settings
        if settings:
            self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME
            self.query_deployment_name = settings.AZURE_OPENAI_QUERY_DEPLOYMENT_NAME
            self.planner_max_plans = settings.PLANNER_MAX_PLANS
            if isinstance(settings.TIME_ZONE, str):
                self.timezone = pytz.timezone(settings.TIME_ZONE)
            else:
                self.timezone = pytz.UTC
        else:
            logger.warning(
                "Settings not provided to QueryRewriter, deployment name may be missing"
            )
            self.query_deployment_name = None

    async def rewrite_query(
        self,
        original_query: str,
        temperature: float = 0.9,
        locale: Optional[str] = "ko-KR",
    ) -> Dict[str, str]:
        """
        Rewrite a query for both search and LLM processing.

        Args:
            original_query: The original user query
            temperature: Temperature parameter for the LLM (0.0 to 1.0)

        Returns:
            Dictionary with 'search_query' and 'llm_query' keys
        """
        if not self.client:
            logger.error("No OpenAI client available for query rewriting")
            if locale == "ko-KR":
                search_query = f"마이크로소프트 {original_query}"
                llm_query = f"마이크로소프트 {original_query}"
            else:
                search_query = f"Microsoft {original_query}"
                llm_query = f"Microsoft {original_query}"
            return {"search_query": search_query, "llm_query": llm_query}

        try:

            formatted_prompt = QUERY_REWRITE_PROMPT.format(
                user_query=original_query, locale=locale
            )
            logger.info(
                f"==============Rewriting query: {original_query} with locale: {locale}"
            )
            # Make the API call
            response = await self.client.chat.completions.create(
                model=self.query_deployment_name,
                messages=[
                    {"role": "system", "content": formatted_prompt},
                    {"role": "user", "content": original_query},
                ],
                temperature=temperature,
                max_tokens=300,
                response_format={"type": "json_object"},
            )

            # Parse and validate the response
            result = json.loads(response.choices[0].message.content.strip())

            # Ensure both required keys are present
            if "search_query" not in result or "llm_query" not in result:
                raise ValueError("API response missing required keys")

            return result

        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            # Fallback to a simple rewrite
            if locale == "ko-KR":
                search_query = f"마이크로소프트 {original_query}"
                llm_query = f"마이크로소프트 {original_query}"
            else:
                search_query = f"Microsoft {original_query}"
                llm_query = f"Microsoft {original_query}"
            return {"search_query": search_query, "llm_query": llm_query}

    async def rewrite_plan_query(
        self,
        original_query: str,
        temperature: float = 0.9,
        locale: Optional[str] = "ko-KR",
    ) -> Dict[str, str | list]:
        """
        Rewrite and plan a query for both search and LLM processing.

        Args:
            original_query: The original user query
            temperature: Temperature parameter for the LLM (0.0 to 1.0)

        Returns:
            Dictionary with 'expanded_query' and 'search_queries' keys
        """

        try:
            current_date = datetime.now(tz=self.timezone).strftime("%Y-%m-%d")

            formatted_prompt = REWRITE_PLAN_PROMPT.format(
                user_query=original_query,
                planner_max_plans=self.planner_max_plans,
                date=current_date,
                locale=locale,
            )
            print(formatted_prompt)
            logger.info(
                f"==============Rewrite and Plan query: {original_query} with locale: {locale}"
            )
            # Make the API call
            response = await self.client.chat.completions.create(
                model=self.query_deployment_name,
                messages=[
                    {"role": "system", "content": formatted_prompt},
                    {"role": "user", "content": original_query},
                ],
                temperature=temperature,
                max_tokens=300,
                response_format={"type": "json_object"},
            )

            # Parse and validate the response
            result = json.loads(response.choices[0].message.content.strip())

            # Ensure both required keys are present
            if "expanded_query" not in result or "search_queries" not in result:
                raise ValueError("API response missing required keys")

            return result

        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            # Fallback to a simple rewrite
            expanded_query = f"Microsoft {original_query}"
            search_queries = [f"Microsoft {original_query}"]
            return {"expanded_query": expanded_query, "search_queries": search_queries}
