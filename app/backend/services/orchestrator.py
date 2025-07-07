import logging
from datetime import datetime
from typing import AsyncGenerator, List, Optional
import pytz 
from config.config import Settings
from langchain_core.prompts import PromptTemplate
from model.models import ChatMessage
from openai import AsyncAzureOpenAI
from utils.enum import SearchEngine
import json
from services.bing_grounding_search import BingGroundingSearch, BingGroundingCrawler
from services.search_crawler import SearchCrawler, GoogleSearchCrawler, BingSearchCrawler
from services.query_rewriter import QueryRewriter
from i18n.locale_msg import LOCALE_MESSAGES

logger = logging.getLogger(__name__)

# Define the system prompt template using LangChain's PromptTemplate
PROMPT_TEMPLATE = """

        너는 마이크로소프트 제품 관련 정보를 제공하는 챗봇이야. 다음의 규칙을 따라 답변해줘

        * 아래 제공하는 검색엔진에서 검색한 결과를 컨텍스트로 활용하여 질문에 대한 답변을 제공해야 해. 
        * 컨텍스트를 최대한 활용하여 풍부하고 상세히 답변을 해야해. 
        * 경쟁사 제품들은 언급하지 말고 마이크로소프트 제품에 대한 정보만 제공해.
        * 마이크로소프트 제품이나 서비스, 회사에 대한 부정적인 내용은 절대 포함하지 말아야 해. 
        * 답변은 마크다운으로 이모지를 1~2개 포함해서 작성해줘.
        * 컨텍스트가 부족하면 대답을 하지 말고 "검색결과가 부족하여 답변을 할 수 없습니다."라고 대답해.
        * 현재는 {date} 이므로 최신의 데이터를 기반으로 답변을 해줘.
        * 검색된 참조링크를 바탕으로 사용자가 클릭할 수 있도록 참조 링크를 제공해줘.
        * 출력언어 설정에 따라 답변해줘. 

        검색엔진에서 검색된 컨텍스트: 
        {contexts}
        검색된 참조링크:
        {url_citation}
        사용자질문: 
        {question}
        출력언어:
        {locale}

    """

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["date", "contexts", "question"],
    optional_variables=["url_citation", "locale"]
)

if not all(var in PROMPT.input_variables for var in ["date", "contexts", "question"]):
    raise ValueError("Loaded prompt does not contain required input variables")        

class Orchestrator:
    """Service for orchestrating the chatbot pipeline, including Azure OpenAI interactions"""
    
    def __init__(
        self, 
        settings: Settings
    ):
        self.settings = settings
        if isinstance(settings.TIME_ZONE, str):
            self.timezone = pytz.timezone(settings.TIME_ZONE)
        else:
            self.timezone = pytz.UTC
            
        #
        self.client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )
        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.query_deployment_name = settings.AZURE_OPENAI_QUERY_DEPLOYMENT_NAME
        # TODO : change the test code to use the settings
        self.query_rewriter: QueryRewriter = None
        self.bing_grounding_search: BingGroundingSearch = None
        self.search_crawler: SearchCrawler = None
        logger.debug(f"Orchestrator initialized with Azure OpenAI deployment: {self.deployment_name}")
        
    async def generate_response(
        self, 
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        query_rewrite: bool = True,
        search_engine: SearchEngine = SearchEngine.GOOGLE_SEARCH_CRAWLING,
        search_crawler: SearchCrawler = GoogleSearchCrawler(),    
        stream: bool = False,
        elapsed_time: bool = True,
        locale: Optional[str] = "ko-KR"
    ) -> AsyncGenerator[str, None]:
        """
        Generate a chat response using Azure OpenAI, with optional streaming support.
        
        Args:
            messages: List of chat messages in the conversation
            max_tokens: Maximum tokens to generate
            temperature: Controls randomness (0.0 to 1.0)
            query_rewrite: Whether to rewrite the query for better results
            search_engine: Search engine to use for RAG
            stream: Whether to stream the response or return it all at once
            
        Yields:
            Response text. For non-streaming requests, yields exactly once.
            For streaming requests, yields multiple chunks.
        """
        try:
            start_time = datetime.now(tz=self.timezone)
            if elapsed_time:
                logger.info(f"Starting response generation at {start_time}")
                ttft_time = None
            
            messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]
            current_date = datetime.now(tz=self.timezone).strftime("%Y-%m-%d")
            last_user_message = next(
                (msg["content"] for msg in reversed(messages_dict) if msg["role"] == "user"), 
                "No question provided"
            )

            LOCALE_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])

            if last_user_message == "No question provided":
                yield LOCALE_MSG["input_needed"]
                return
            
            contexts_text = "No search results available"
            
            system_content = PROMPT.format(
                date=current_date,
                contexts=contexts_text,
                question=last_user_message,
                url_citation=[],
                locale=locale
            )

            messages_dict.insert(0, {"role": "system", "content": system_content})

            if max_tokens is None:
                max_tokens = self.settings.MAX_TOKENS
                
            if temperature is None:
                temperature = self.settings.DEFAULT_TEMPERATURE
            final_messages = messages_dict
            
            if stream:
                yield f"data: ### {LOCALE_MSG['analyzing']}\n\n"

            if query_rewrite and self.query_rewriter:
                if last_user_message != "No question provided":
                    queries = await self.query_rewriter.rewrite_query(last_user_message, locale=locale)
                    if stream:
                        yield f"data: ### {LOCALE_MSG['rewrite_complete']}: {queries['llm_query']}\n\n"
            else:
                queries = {
                    "llm_query": last_user_message,
                    "search_query": last_user_message
                }

            if stream:
                yield f"data: ### {LOCALE_MSG['searching']}\n\n"

            if search_engine == SearchEngine.GOOGLE_SEARCH_CRAWLING or search_engine == SearchEngine.BING_SEARCH_CRAWLING or search_engine == SearchEngine.BING_GROUNDING_CRAWLING:
                logger.info(f"##### Using External Search Engine ##### (queries={queries}) ")
                search_results = search_crawler.search(
                    queries["search_query"],
                )
                
                logger.info(f"Search results: {search_results}")
                
                if search_results:
                    url_snippet_tuples = [(r["link"], r["snippet"]) for r in search_results]

                    if stream:
                        yield f"data: ### {LOCALE_MSG['searching']}\n\n"

                    contexts = await search_crawler.extract_contexts_async(url_snippet_tuples)

                    if stream:
                        yield f"data: ### {LOCALE_MSG['preparing_response']}\n\n"
                    logger.debug(f"url_snippet_tuples: {url_snippet_tuples}")
                    contexts_text = "\n\n".join(contexts)
                    
                    system_content = PROMPT.format(
                        date=current_date,
                        contexts=contexts_text,
                        question=queries["llm_query"],
                        url_citation=json.dumps(url_snippet_tuples, ensure_ascii=False),
                        locale=locale
                    )
                    
                    final_messages[0]["content"] = system_content

                    for i, msg in enumerate(final_messages):
                        if msg["role"] == "user" and msg["content"] == last_user_message:
                            final_messages[i]["content"] = queries["llm_query"]
                            break
                
                if stream:
                    yield f"data: ### {LOCALE_MSG['answering']}\n\n"

                log_message = (
                    f"Sending {'streaming ' if stream else ''}request to Azure OpenAI: "
                    f"deployment={self.deployment_name}, "
                    f"max_tokens={max_tokens}, "
                    f"temperature={temperature}, "
                    f"message_count={len(final_messages)}"
                )
                logger.debug(log_message)

                response = await self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=final_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream
                )

                if stream:
                    ttft_time = datetime.now(tz=self.timezone) - start_time
                    async for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield f"{chunk.choices[0].delta.content}"
                else:
                    ttft_time = datetime.now(tz=self.timezone) - start_time
                    message_content = response.choices[0].message.content
                    yield message_content
            
            elif search_engine == SearchEngine.BING_GROUNDING:
                logger.info(f"##### Using Bing Grounding for search ##### (queries={queries}) ")
                if stream:
                    yield f"data: ### {LOCALE_MSG['search_and_answer']}\n\n"

                search_generator = self.bing_grounding_search.search_and_generate_by_bing_grounding_ai_agent(
                    queries=queries,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                    locale=locale
                )
                
                if stream:
                    first_chunk = True
                    async for chunk in search_generator:
                        if first_chunk:
                            ttft_time = datetime.now(tz=self.timezone) - start_time
                            first_chunk = False
                        yield chunk
                else:
                    ttft_time = datetime.now(tz=self.timezone) - start_time
                    results = []
                    async for result in search_generator:
                        if result:
                            results.append(result)
                    yield "\n".join(results)
            else:
                error_msg = f"Unsupported search engine: {search_engine}"
                logger.error(error_msg)
                yield f"Error: {error_msg}"

            if elapsed_time and ttft_time is not None:
                logger.info(f"Response generated successfully in {ttft_time.total_seconds()} seconds")
                yield " \n\n"
                yield " \n\n"
                yield f"Response generated successfully in {ttft_time.total_seconds()} seconds \n\n"
            return
                        
        except Exception as e:
            error_msg = f"Azure OpenAI API {'streaming ' if stream else ''}error: {str(e)}"
            logger.error(error_msg)

            yield f"Error: {str(e)}"


