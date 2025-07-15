# locale_msg_front.py

UI_TEXT = {
    "en-US": {
        "title": "# Microsoft Plan and Search Chat",
        "select_agent_mode": "### Select Agentic Mode",
        "query_rewrite_title": "#### Query Rewrite",
        "query_rewrite_desc": "GPT will respond rewriting your query",
        "plan_execute_title": "#### Plan & Execute", 
        "plan_execute_desc": "GPT will plan & execute when the queries are complex",
        "search_engine_title": "#### Search Engine",
        "search_engine_desc": "Search engines use crawling except for Grounding Gen",
        "enable_label": "Enable",
        "send_button": "Send",
        "clear_chat_button": "Clear Chat",
        "try_prompts": "### Try following prompts",
        "connecting_api": "⟳ Connecting to API...",
        "searching_response": "⟳ Searching response...",
        "processing_message": "Processing message...",
        "language_toggle": "Language / 언어"
    },
    "ko-KR": {
        "title": "# Microsoft Plan and Search Chat",
        "select_agent_mode": "### 에이전트 모드 선택!!!!!",
        "query_rewrite_title": "#### 쿼리 재작성",
        "query_rewrite_desc": "GPT가 쿼리를 재작성하여 응답합니다",
        "plan_execute_title": "#### 계획 및 실행",
        "plan_execute_desc": "복잡한 쿼리에 대해 GPT가 계획하고 실행합니다",
        "search_engine_title": "#### 검색 엔진",
        "search_engine_desc": "Grounding Gen을 제외한 검색 엔진은 크롤링을 사용합니다",
        "enable_label": "Enable",
        "send_button": "전송",
        "clear_chat_button": "채팅 지우기",
        "try_prompts": "### 다음 프롬프트를 시도해보세요",
        "connecting_api": "⟳ API 연결 중...",
        "searching_response": "⟳ 응답 검색 중...",
        "processing_message": "메시지 처리 중...",
        "language_toggle": "Language / 언어"
    }
}

EXAMPLE_PROMPTS = {
    "en-US": {
        "question_Microsoft": {
            "title": "Microsoft General Questions",
            "description": "Have a general question about Microsoft",
            "prompt": "How is the Microsoft stock mood these days?"
        },
        "product_info": {
            "title": "Questions about Product", 
            "description": "Ask questions about various Microsoft products",
            "prompt": "Can you recommend a Surface model other than 12inches from Microsoft?"
        },
        "recommendation": {
            "title": "Recommendation",
            "description": "Ask recommendation for a product line", 
            "prompt": "Can you recommend a good office software for my parents?"
        },
        "comparison": {
            "title": "Comparison",
            "description": "Ask for a comparison between two different models",
            "prompt": "Can you compare Samsung Galaxy Book and Microsoft Surface Pro?"
        },
        "support_questions": {
            "title": "Support Questions",
            "description": "Ask for support-related inquiries",
            "prompt": "What is the warranty period for Microsoft products?"
        },
        "tools": {
            "title": "Tool (Time & Weather)",
            "description": "Ask time and weather information from external APIs",
            "prompt": "What time is it in New York?"
        }
    },
    "ko-KR": {
        "question_Microsoft": {
            "title": "마이크로소프트 일반 질문",
            "description": "마이크로소프트에 대한 일반적인 질문하기",
            "prompt": "마이크로소프트 주식 분위기가 요즘 어때?"
        },
        "product_info": {
            "title": "제품 질문",
            "description": "다양한 마이크로소프트 제품에 대한 질문하기",
            "prompt": "마이크로소프트 제품 중 12인치가 아닌 서피스 모델 추천해줘"
        },
        "recommendation": {
            "title": "추천",
            "description": "제품 라인에 대한 추천 요청하기",
            "prompt": "부모님께 좋은 오피스 소프트웨어를 추천해줄 수 있어?"
        },
        "comparison": {
            "title": "비교",
            "description": "두 가지 다른 모델 간의 비교 요청하기",
            "prompt": "삼성노트북과 마이크로소프트 서피스를 비교해줄 수 있어?"
        },
        "support_questions": {
            "title": "지원 질문",
            "description": "지원 관련 문의하기",
            "prompt": "마이크로소프트 제품의 보증 기간은 어떻게 되나요?"
        },
        "tools": {
            "title": "도구 (시간 및 날씨)",
            "description": "외부 API에서 시간 및 날씨 정보 요청하기",
            "prompt": "뉴욕은 지금 몇 시야?"
        }
    }
}