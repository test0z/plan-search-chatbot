import os
import requests
import asyncio
import httpx
import redis
import scrapy
from urllib.parse import urljoin
from typing import List, Tuple, Optional, Dict, Any
import logging
from datetime import datetime
from langchain.prompts import PromptTemplate
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.ai.agents.models import (
    BingGroundingTool,
    MessageRole,
    RunStep,
    ThreadMessage,
    ThreadRun,
    MessageDeltaTextUrlCitationAnnotation,
    MessageDeltaTextContent,
    AgentEventHandler,
    MessageDeltaChunk
)
import pytz 
from services.search_crawler import SearchCrawler
from config.config import Settings
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import AgentStreamEvent, RunStepDeltaChunk, RunStatus
from azure.ai.projects import AIProjectClient
from langchain_core.prompts import load_prompt, PromptTemplate

logger = logging.getLogger(__name__)


SEARCH_GENERATE_PROMPT_TEMPLATE = """
You are an intelligent chatbot that provides guidance on Microsoft products in **Markdown** format based on real-time web search results.

🎯 Objective:
- Provide users with accurate and reliable answers based on the latest web information.
- Actively utilize web search results to generate rich and specific answers.
- Respond in Markdown format, including 1-2 emojis to increase readability and friendliness.

📌 Guidelines:  
1. don't response with any greeting messages, just response with the answer to the user's question.
2. Always generate answers based on search results and avoid making unfounded assumptions.  
3. Always include reference links and format them using the Markdown `[text](URL)` format.  
4. When providing product price information, base it on the official website's prices and links.
"""

SEARCH_PROMPT_TEMPLATE = """
You are an intelligent chatbot that can perform real-time web searches for Microsoft products.
"""


# for BingGroundingSearch.search_and_generate_by_bing_grounding_ai_agent method
SEARCH_GEN_PROMPT = PromptTemplate(
    template=SEARCH_GENERATE_PROMPT_TEMPLATE,
)
# for BingGroundingCrawler.search method
SEARCH_PROMPT = PromptTemplate(
    template=SEARCH_PROMPT_TEMPLATE,
)

class BingGroundingSearch():
    """
    Class for handling web search and content extraction from search results.
    Set these environment variables with your own values:
    1) PROJECT_ENDPOINT - The Azure AI Project endpoint, as found in the Overview 
                          page of your Azure AI Foundry portal.
    2) MODEL_DEPLOYMENT_NAME - The deployment name of the AI model, as found under the "Name" column in 
       the "Models + endpoints" tab in your Azure AI Foundry project.
    3) AZURE_BING_CONNECTION_ID - The ID of the Bing connection, in the format of:
       /subscriptions/{subscription-id}/resourceGroups/{resource-group-name}/providers/Microsoft.CognitiveServices/accounts/{ai-service-account-name}/projects/{project-name}/connections/{connection-name}
       
    """
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        """
        Initialize the Bing Grounding Search with the given configuration.

        Args:
            redis_config: Configuration dictionary for Redis connection
        """
        self.settings = Settings()
        if isinstance(self.settings.TIME_ZONE, str):
            self.timezone = pytz.timezone(self.settings.TIME_ZONE)
        else:
            self.timezone = pytz.UTC
        self.project_endpoint = os.getenv("BING_GROUNDING_PROJECT_ENDPOINT")
        self.connection_id = os.getenv("BING_GROUNDING_CONNECTION_ID")
        self.agent_model_deployment_name = os.getenv("BING_GROUNDING_AGENT_MODEL_DEPLOYMENT_NAME")
        self.max_results = int(os.getenv("BING_GROUNDING_MAX_RESULTS", 10))
        self.market = os.getenv("BING_GROUNDING_MARKET", "ko-KR")
        self.set_lang = os.getenv("BING_GROUNDING_SET_LANG", "ko")
        self.search_gen_agent_id_env = os.getenv("SEARCH_GEN_AGENT_ID")
        
        # Initialize credentials based on environment
        self.creds = self._get_azure_credential()
        
        self.search_gen_agent = None
        
        self.agents_client = AgentsClient(
            endpoint=self.project_endpoint,
            credential=self.creds,
        )

        bing = BingGroundingTool(connection_id=self.connection_id, market=self.market, set_lang=self.set_lang, count=int(self.max_results))

        if self.search_gen_agent_id_env:
            self.search_gen_agent = self.agents_client.get_agent(self.search_gen_agent_id_env)
            logger.info(f"Using existing search-gen-agent, ID: {self.search_gen_agent.id}")
            if(self.search_gen_agent is None):
                logger.error("search_gen_agent is None, please check the environment variable SEARCH_GEN_AGENT_ID")
                self.search_gen_agent = self.agents_client.create_agent(
                    model=self.agent_model_deployment_name,
                    name="temp-search-gen-agent",
                    instructions=SEARCH_GEN_PROMPT.format(),
                    tools=bing.definitions,
                )
            else:
                self.agents_client.update_agent(
                    self.search_gen_agent.id,
                    model=self.agent_model_deployment_name,
                    name="temp-search-gen-agent",
                    instructions=SEARCH_GEN_PROMPT.format(),
                    tools=bing.definitions
                )
            
        else:
            self.search_gen_agent = self.agents_client.create_agent(
                model=self.agent_model_deployment_name,
                name="temp-search-gen-agent",
                instructions=SEARCH_GEN_PROMPT.format(),
                tools=bing.definitions
            )
            logger.info(f"Created new search-gen-agent, ID: {self.search_gen_agent.id}")
            
        # Check if Redis should be used (default to False if not specified)
        self.use_redis = os.getenv("REDIS_USE", "False").lower() == "true"
        
        if not self.use_redis:
            logger.info("Redis usage is disabled by configuration")
            self.redis_client = None
            return
            
        # Default Redis configuration
        default_redis_config = {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", 6379)),
            "password": os.getenv("REDIS_PASSWORD", ""),
            "db": int(os.getenv("REDIS_DB", 0)),
            "decode_responses": True
        }
        
        # Use provided config or default
        redis_config = redis_config or default_redis_config
        
        # Initialize Redis client if credentials are available
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(**redis_config)
            self.redis_client.ping()  # Test connection
            logger.info("Redis cache connection established successfully")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {str(e)}. Caching disabled.")
            self.redis_client = None
        except Exception as e:
            logger.warning(f"Redis initialization error: {str(e)}. Caching disabled.")
            self.redis_client = None
            
        # Cache expiration in seconds (default: 7 days)
        self.cache_expiration = int(os.getenv("REDIS_CACHE_EXPIRED_SECOND", 604800))

    def deleteAgent(self):
        self.agents_client.delete_agent(self.search_gen_agent.id)
        logger.info(f"Deleted search-gen-agent, ID: {self.search_gen_agent.id}")


    def _get_azure_credential(self):
        """
        Get appropriate Azure credential based on the environment.
        
        Returns:
            Azure credential instance
        """
        try:            
            if  os.getenv("APP_USER_ASSIGNED_MANAGED_IDENTITY_CLIENT_ID"):                
                logger.info("Detected Azure environment, using ManagedIdentityCredential")
                client_id = os.getenv("APP_USER_ASSIGNED_MANAGED_IDENTITY_CLIENT_ID") 
                if client_id:
                    logger.info(f"Using user-assigned managed identity with client ID: {client_id}")
                    return ManagedIdentityCredential(client_id=client_id)
                else:
                    logger.info("Using system-assigned managed identity")
                    return ManagedIdentityCredential()
            else:
                logger.info("Using DefaultAzureCredential for local development")
                return DefaultAzureCredential()
                
        except Exception as e:
            logger.warning(f"Error initializing Azure credential: {str(e)}")
            logger.info("Falling back to DefaultAzureCredential")
            return DefaultAzureCredential()

    async def search_and_generate_by_bing_grounding_ai_agent(self, 
            queries: Dict[str, str],
            max_tokens: int = 1024,
            temperature: float = 0.7,
            stream: bool = False,
            locale: str = "ko-KR",
            ):
        
        if not self.creds or not self.project_endpoint:
            logger.error("Bing Grounding credentials are missing")
            return 
        
        thread = self.agents_client.threads.create()
        logger.info(f"Created thread, ID: {thread.id}")
        # Define the system prompt template using LangChain's PromptTemplate
        SEARCH_GEN_USER_PROMPT_TEMPLATE = """
            don't say hello or any greetings, directly respond with the answer.
            please provide as rich and specific an answer and reference links as possible for `{llm_query}` of Microsoft.
            Today is {current_date}. Results should be based on the recent information available. 
        """

        # Create a LangChain PromptTemplate
        SEARCH_GEN_USER_PROMPT = PromptTemplate(
            template=SEARCH_GEN_USER_PROMPT_TEMPLATE,
            input_variables=["search_keyword","llm_query", "current_date", "max_results"]
        )
        current_date = datetime.now(tz=self.timezone).strftime("%Y-%m-%d")
        
        search_gen_instruction = SEARCH_GEN_USER_PROMPT.format(
            search_keyword=queries["search_query"],
            current_date=current_date,
            llm_query=queries["llm_query"],
            max_results=self.max_results
        )

        # Create message to thread
        logger.info(f"final user_instruction on Azure AI Agent: {search_gen_instruction}")
        message = self.agents_client.messages.create(
            thread_id=thread.id,
            role=MessageRole.USER,
            content=search_gen_instruction,
        )
        logger.info(f"Created message, ID: {message.id}")
        
        async def get_agent_message_with_retry(thread_id, role, max_retries=5, delay=1.0):
            """
            Poll for the agent message, retrying if not immediately available.
            """
            for attempt in range(max_retries):
                response_message = self.agents_client.messages.get_last_message_by_role(
                    thread_id=thread_id, 
                    role=role,
                )
                if response_message is not None and getattr(response_message, "content", None):
                    logger.info(f"Agent message found on attempt {attempt+1}")
                    return response_message
                logger.info(f"Agent message not found (attempt {attempt+1}/{max_retries}), retrying after {delay}s...")
                await asyncio.sleep(delay)
            return None

        if stream:
            with self.agents_client.runs.stream(
                    thread_id=thread.id, agent_id=self.search_gen_agent.id, 
                    max_prompt_tokens=max_tokens, 
                    temperature=temperature
                    ) as stream:
                for event_type, event_data, _ in stream:
                    if isinstance(event_data, MessageDeltaChunk):
                        yield f"{event_data.text}"
                    elif isinstance(event_data, ThreadMessage):
                        logger.info(f"ThreadMessage created. ID: {event_data.id}, Status: {event_data.status}")
                    elif isinstance(event_data, ThreadRun):
                        logger.info(f"ThreadRun status: {event_data.status}")
                        if event_data.status == RunStatus.FAILED:
                            logger.info(f"Run failed. Data: {event_data}")
                    elif isinstance(event_data, RunStep):
                        logger.info(f"RunStep type: {event_data.type}, Status: {event_data.status}")
                    elif event_type == AgentStreamEvent.ERROR:
                        logger.info(f"An error occurred. Data: {event_data}")
                    elif event_type == AgentStreamEvent.DONE:
                        logger.info("Stream completed.")
                        break
                    else:
                        logger.info(f"Unhandled Event Type: {event_type}, Data: {event_data}")

            response_message = await get_agent_message_with_retry(thread_id=thread.id, role=MessageRole.AGENT)
            if not response_message or not getattr(response_message, "content", None):
                logger.error("Agent response message not found after streaming (NoneType).")
                yield "Error: Agent is still processing. Please try again in a few seconds."
                return

            logger.info(f"Agent response message: {response_message.content}")
            yield(f"  \n\n")
            # for annotation in getattr(response_message, "url_citation_annotations", []):
            #     yield(f"### Reference: [{annotation.url_citation.title}]({annotation.url_citation.url}) \n\n")

        else:
            logger.info("Creating and processing agent run in thread with tools")
            run = self.agents_client.runs.create_and_process(thread_id=thread.id, agent_id=self.search_gen_agent.id)
            logger.info(f"Run finished with status: {run.status}")

            if run.status == "failed":
                logger.error(f"Run failed: {run.last_error}")
                yield f"Error: Run failed - {run.last_error}"
                return

            # Fetch run steps to get the details of the agent run
            run_steps = self.agents_client.run_steps.list(thread_id=thread.id, run_id=run.id)
            for step in run_steps:
                logger.info(f"Step {step['id']} status: {step['status']}")
                step_details = step.get("step_details", {})
                tool_calls = step_details.get("tool_calls", [])

                if tool_calls:
                    logger.info("  Tool calls:")
                    for call in tool_calls:
                        logger.info(f"    Tool Call ID: {call.get('id')}")
                        logger.info(f"    Type: {call.get('type')}")

                        bing_grounding_details = call.get("bing_grounding", {})
                        if bing_grounding_details:
                            logger.info(f"  Bing Grounding ID: {bing_grounding_details.get('requesturl')}")


            # Print the Agent's response message with optional citation
            try:
                # 메시지 polling/retry 사용
                response_message = await get_agent_message_with_retry(
                    thread_id=thread.id, 
                    role=MessageRole.AGENT,
                    max_retries=8,
                    delay=0.5
                )
                if response_message is None or not getattr(response_message, "content", None):
                    logger.error("Agent response message not found after retries.")
                    yield "Error: Agent response message not found. Please try again."
                    return
                logger.info(f"Agent response message: {response_message.content}")
                if response_message.content:
                    for content_item in response_message["content"]:
                        if content_item["type"] == "text":
                            text_content = content_item["text"]["value"]
                            yield text_content
            except Exception as e:
                logger.error(f"Error retrieving agent response: {str(e)}")
                yield f"Error retrieving response: {str(e)}"


        return

   
        
class BingGroundingCrawler(SearchCrawler):
    """
    Class for handling web search and content extraction from search results.
    Set these environment variables with your own values:
    1) PROJECT_ENDPOINT - The Azure AI Project endpoint, as found in the Overview 
                          page of your Azure AI Foundry portal.
    2) MODEL_DEPLOYMENT_NAME - The deployment name of the AI model, as found under the "Name" column in 
       the "Models + endpoints" tab in your Azure AI Foundry project.
    3) AZURE_BING_CONNECTION_ID - The ID of the Bing connection, in the format of:
       /subscriptions/{subscription-id}/resourceGroups/{resource-group-name}/providers/Microsoft.CognitiveServices/accounts/{ai-service-account-name}/projects/{project-name}/connections/{connection-name}
       
    """
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        """
        Initialize the Bing Grounding Search with the given configuration.

        Args:
            redis_config: Configuration dictionary for Redis connection
        """
        self.settings = Settings()
        if isinstance(self.settings.TIME_ZONE, str):
            self.timezone = pytz.timezone(self.settings.TIME_ZONE)
        else:
            self.timezone = pytz.UTC
        self.project_endpoint = os.getenv("BING_GROUNDING_PROJECT_ENDPOINT")
        self.connection_id = os.getenv("BING_GROUNDING_CONNECTION_ID")
        self.agent_model_deployment_name = os.getenv("BING_GROUNDING_AGENT_MODEL_DEPLOYMENT_NAME")
        self.max_results = int(os.getenv("BING_GROUNDING_MAX_RESULTS", 10))
        self.market = os.getenv("BING_GROUNDING_MARKET", "ko-KR")
        self.set_lang = os.getenv("BING_GROUNDING_SET_LANG", "ko-KR")
        self.search_agent_id_env = os.getenv("SEARCH_AGENT_ID")
        self.search_agent = None
        
        # Initialize credentials based on environment
        self.creds = self._get_azure_credential()
        
        self.agents_client = AgentsClient(
            endpoint=self.project_endpoint,
            credential=self.creds,
        )

        bing = BingGroundingTool(connection_id=self.connection_id, market=self.market, set_lang=self.set_lang, count=int(self.max_results))

        if self.search_agent_id_env:
            self.search_agent = self.agents_client.get_agent(self.search_agent_id_env)
            logger.info(f"Using existing search agent, ID: {self.search_agent.id}")
            
            if(self.search_agent is None):
                logger.error("search_agent is None, please check the environment variable SEARCH_AGENT_ID")
                self.search_agent = self.agents_client.create_agent(
                    model=self.agent_model_deployment_name,
                    name="temp-search-agent",
                    instructions=SEARCH_PROMPT.format(),
                    tools=bing.definitions,
                )
            else:
                logger.info(f"Updating existing search-agent, ID: {self.search_agent.id}")
                self.agents_client.update_agent(
                    self.search_agent.id,
                    model=self.agent_model_deployment_name,
                    name="temp-search-agent",
                    instructions=SEARCH_PROMPT.format(),
                    tools=bing.definitions
                )
            
        else:
            self.search_agent = self.agents_client.create_agent(
                model=self.agent_model_deployment_name,
                name="temp-search-agent",
                instructions=SEARCH_PROMPT.format(),
                tools=bing.definitions,
            )
            logger.info(f"Created new search-agent, ID: {self.search_agent.id}")

        # Check if Redis should be used (default to False if not specified)
        self.use_redis = os.getenv("REDIS_USE", "False").lower() == "true"
        
        if not self.use_redis:
            logger.info("Redis usage is disabled by configuration")
            self.redis_client = None
            return
            
        # Default Redis configuration
        default_redis_config = {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", 6379)),
            "password": os.getenv("REDIS_PASSWORD", ""),
            "db": int(os.getenv("REDIS_DB", 0)),
            "decode_responses": True
        }
        
        # Use provided config or default
        redis_config = redis_config or default_redis_config
        
        # Initialize Redis client if credentials are available
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(**redis_config)
            self.redis_client.ping()  # Test connection
            logger.info("Redis cache connection established successfully")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {str(e)}. Caching disabled.")
            self.redis_client = None
        except Exception as e:
            logger.warning(f"Redis initialization error: {str(e)}. Caching disabled.")
            self.redis_client = None
            
        # Cache expiration in seconds (default: 7 days)
        self.cache_expiration = int(os.getenv("REDIS_CACHE_EXPIRED_SECOND", 604800))
        
    def _get_azure_credential(self):
        """
        Get appropriate Azure credential based on the environment.
        
        Returns:
            Azure credential instance
        """
        try:
            if os.getenv("APP_USER_ASSIGNED_MANAGED_IDENTITY_CLIENT_ID"):                
                logger.info("Detected Azure environment, using ManagedIdentityCredential") 
                client_id = os.getenv("APP_USER_ASSIGNED_MANAGED_IDENTITY_CLIENT_ID")
                if client_id:
                    logger.info(f"Using user-assigned managed identity with client ID: {client_id}")
                    return ManagedIdentityCredential(client_id=client_id)
                else:
                    logger.info("Using system-assigned managed identity")
                    return ManagedIdentityCredential()
            else:
                logger.info("Using DefaultAzureCredential for local development")
                return DefaultAzureCredential()
                
        except Exception as e:
            logger.warning(f"Error initializing Azure credential: {str(e)}")
            logger.info("Falling back to DefaultAzureCredential")
            return DefaultAzureCredential()

    def search(self, query: str, locale: Optional[str] = "en-US") -> List[Dict[str, str]]:
        """
        Perform a Grounding with Bing search using Azure AI Agent.
        
        Args:
            query: The search query
            
        Returns:
            A list of search result items
        """
        if not self.creds or not self.project_endpoint:
            logger.error("Bing Grounding credentials are missing")
            return []

        thread = self.search_agent.threads.create()
        logger.info(f"Created thread, ID: {thread.id}")
        
        # Define the system prompt template using LangChain's PromptTemplate
        USER_PROMPT_TEMPLATE = """
            Search the web for: {query}. Return only the top {max_results} most relevant results as a list.
            Today is {current_date}. Results should be based on the recent information available.
        """

        # Create a LangChain PromptTemplate
        USER_PROMPT = PromptTemplate(
            template=USER_PROMPT_TEMPLATE,
            input_variables=["query","current_date", "max_results"]
        )
        current_date = datetime.now(tz=self.timezone).strftime("%Y-%m-%d")
        
        search_instruction = USER_PROMPT.format(
            current_date=current_date,
            query=query,
            max_results=self.max_results
        )
        
        logger.info(f"final user_instruction on Azure AI Agent: {search_instruction}")
        message = self.agents_client.messages.create(
            thread_id=thread.id,
            role=MessageRole.USER,
            content=search_instruction,
        )

        logger.info(f"Created message, ID: {message.id}")
        
        run = self.agents_client.runs.create_and_process(thread_id=thread.id, agent_id=self.search_agent.id)

        if run.status == "failed":
            logger.error(f"Bing Grounding execution failed: {run.last_error}")
            return []
        
        results = []
        response_message = self.agents_client.messages.get_last_message_by_role(
            thread_id=thread.id,
            role=MessageRole.AGENT,
        )
        logger.info(f"response_message: {response_message}")
            
        if response_message.url_citation_annotations:
            for annotation in response_message.url_citation_annotations:
                url = annotation.url_citation.url
                title = annotation.url_citation.title
                results.append({"link": url, "snippet": title})

        logger.info(f"Search results: {results}")
        if not results:
            logger.warning("No search results found.")
            return []
        return results



