import aiohttp
import os
import asyncio
import warnings
from jinja2 import Template
from typing import Annotated, Optional
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from dotenv import load_dotenv
from datetime import datetime
from datetime import timedelta
import pytz

load_dotenv(override=True)
DART_API_KEY = os.getenv("DART_API_KEY")
DART_API_URL = os.getenv("DART_API_URL", "https://opendart.fss.or.kr/api/")
TIME_ZONE = os.getenv("TIME_ZONE", "Asia/Seoul")

REPORT_CODES = {          # DART "reprt_code"
    "Q1"     : "11013",
    "HALF"   : "11012",
    "Q3"     : "11014",
    "ANNUAL" : "11011",
}

# statutory filing deadlines (month, day) — used to decide what *should* exist
DEADLINES = [
    ("Q3"  , (11, 14)),   # 3Q report
    ("HALF", ( 8, 14)),   # half-year report
    ("Q1"  , ( 5, 15)),   # 1Q report
    ("ANNUAL", ( 3, 31)), # business report
]

class DartPlugin:
    BASE_URL = DART_API_URL

    def __init__(self, load_corp_code_list=False):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 타임아웃 설정 추가
        self.timeout = aiohttp.ClientTimeout(
            total=60,        # 전체 요청 타임아웃 60초
            connect=10,      # 연결 타임아웃 10초
            sock_read=30,    # 소켓 읽기 타임아웃 30초
            sock_connect=10  # 소켓 연결 타임아웃 10초
        )
        
        self.connector_kwargs = {
            'limit': 100,           # 전체 연결 풀 크기
            'limit_per_host': 30,   # 호스트당 연결 제한
            'ttl_dns_cache': 300,   # DNS 캐시 TTL
            'use_dns_cache': True,  # DNS 캐시 사용
            'ssl': False if DART_API_URL.startswith('http://') else None,  # SSL 설정
            'force_close': True,    # 연결 강제 종료
        }
        
        if load_corp_code_list:
            self.corp_list = None
        else:
            self.corp_list = None

    async def _create_session(self):
        """새로운 세션 생성"""
        connector = aiohttp.TCPConnector(**self.connector_kwargs)
        return aiohttp.ClientSession(
            headers=self.headers,
            timeout=self.timeout,
            connector=connector
        )

    async def get_company_profile(self, corp_code: str):
        url = f"{self.BASE_URL}/company.json?crtfc_key={DART_API_KEY}&corp_code={corp_code}"
        async with await self._create_session() as session:
            async with session.get(url) as resp:
                return await resp.json()
        
    async def get_recent_filings(self, corp_code: str, bgn_de: str, end_de: str):
        url = f"{self.BASE_URL}/list.json?crtfc_key={DART_API_KEY}&corp_code={corp_code}&bgn_de={bgn_de}&end_de={end_de}"
        async with await self._create_session() as session:
            async with session.get(url) as resp:
                return await resp.json()

    async def get_financial_stats(self, corp_code: str, bsns_year: str, reprt_code: str):    
        url = f"{self.BASE_URL}/fnlttSinglAcnt.json?crtfc_key={DART_API_KEY}&corp_code={corp_code}&bsns_year={bsns_year}&reprt_code={reprt_code}"
        async with await self._create_session() as session:
            async with session.get(url) as resp:
                return await resp.json()

    async def get_financial_stats_with_fallback(self, corp_code: str, max_attempts: int = 5) -> dict:
        """
        Try to fetch financial data with fallback logic for the latest report.
        If the latest report is not available, it will try previous reports in a specific order.
        The order is: Q3 → HALF → Q1 → ANNUAL.
        
        Args:
            corp_code (str): DART 기업 코드
            max_attempts (int): 최대 시도 횟수
        
        Returns:
            dict: 재무 데이터가 포함된 JSON 응답
        """
        timezone = pytz.timezone(TIME_ZONE)
        today = datetime.now(tz=timezone)
        year, reprt_code = await self._guess_latest_period(today)
        
        print(f"Starting financial data fetch from: {year}, {reprt_code}")
        
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}: Trying year={year}, reprt_code={reprt_code}")
            
            try:
                result = await self.get_financial_stats(corp_code, year, reprt_code)
                
                # Check if the result is valid
                if result.get('status') == '000' and result.get('list'):
                    print(f"### Successfully fetched financial data for {year}, {reprt_code}")
                    return result
                else:
                    print(f"❌ No data for {year}, {reprt_code}: {result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ Error fetching {year}, {reprt_code}: {str(e)}")
            
            # fallback to get the next report
            year, reprt_code = await self._fallback(year, reprt_code)
            print(f"Falling back to: {year}, {reprt_code}")
        
        # 모든 시도가 실패한 경우
        print(f"❌ Failed to fetch financial data after {max_attempts} attempts")
        return {'status': '999', 'message': 'No financial data available', 'list': []}
    
    @staticmethod
    async def _guess_latest_period(today: datetime) -> tuple[str, str]:
        """
        Return (year, reprt_code) that *should* be available today.
        Falls back to the previous–year 3Q report if nothing for the current year
        should have been disclosed yet (Jan~Mar period).
        """
        y = today.year
        for label, (m, d) in DEADLINES:
            if today >= datetime(y, m, d, tzinfo=today.tzinfo):
                return str(y), REPORT_CODES[label]
        # before Mar 31 → last year's 3Q
        return str(y - 1), REPORT_CODES["Q3"]
    
    @staticmethod
    async def _fallback(year: int, code: str) -> tuple[int, str]:
        """
        Earlier report within the same year, or previous-year ANNUAL as last resort.
        Order: Q3 → HALF → Q1 → ANNUAL.
        """
        order = ["Q3", "HALF", "Q1", "ANNUAL"]
        inv   = {v: k for k, v in REPORT_CODES.items()}
        idx   = order.index(inv[code])

        if idx < len(order) - 1:                  
            return year, REPORT_CODES[order[idx + 1]]
        return year - 1, REPORT_CODES["ANNUAL"]   

    @kernel_function(
        name="fetch_dart_data",
        description="기업 개요, 공시, 재무정보를 병렬 조회 후 요약 텍스트를 생성합니다."
    )
    async def fetch_dart_data(self, corp_code: Annotated[str, "DART 기업 코드"] = None) -> str:
        try:
            timezone = pytz.timezone(TIME_ZONE)
            today = datetime.now(tz=timezone)
            
            print(f"Starting fetch_dart_data for corp_code: {corp_code}")
            
            bgn_de = (today - timedelta(days=90)).strftime("%Y%m%d")
            end_de = today.strftime("%Y%m%d")

            print(f"bgn_de: {bgn_de}, end_de: {end_de}")
            
            # 하나의 세션을 사용하여 모든 요청을 처리
            async with await self._create_session() as session:
                # 세션을 공유하는 내부 함수들
                async def get_profile_with_session():
                    url = f"{self.BASE_URL}/company.json?crtfc_key={DART_API_KEY}&corp_code={corp_code}"
                    async with session.get(url) as resp:
                        return await resp.json()

                async def get_filings_with_session():
                    url = f"{self.BASE_URL}/list.json?crtfc_key={DART_API_KEY}&corp_code={corp_code}&bgn_de={bgn_de}&end_de={end_de}"
                    async with session.get(url) as resp:
                        return await resp.json()

                async def get_financials_with_session():
                    # 기존 fallback 로직을 세션과 함께 사용
                    timezone = pytz.timezone(TIME_ZONE)
                    today = datetime.now(tz=timezone)
                    year, reprt_code = await self._guess_latest_period(today)
                    
                    for attempt in range(5):
                        try:
                            url = f"{self.BASE_URL}/fnlttSinglAcnt.json?crtfc_key={DART_API_KEY}&corp_code={corp_code}&bsns_year={year}&reprt_code={reprt_code}"
                            async with session.get(url) as resp:
                                result = await resp.json()
                                
                            if result.get('status') == '000' and result.get('list'):
                                return result
                            else:
                                year, reprt_code = await self._fallback(year, reprt_code)
                                
                        except Exception as e:
                            print(f"❌ Error fetching {year}, {reprt_code}: {str(e)}")
                            year, reprt_code = await self._fallback(year, reprt_code)
                    
                    return {'status': '999', 'message': 'No financial data available', 'list': []}

                # 병렬로 데이터 가져오기
                profile, filings, financials = await asyncio.gather(
                    get_profile_with_session(),
                    get_filings_with_session(),
                    get_financials_with_session()
                )
            
            context_template = Template("""
                📌 기업 개요
                
                회사명: {{ profile.get('corp_name', 'N/A') }}
                
                사업자등록번호: {{ profile.get('bizr_no', 'N/A') }}
                
                업종코드: {{ profile.get('induty_code', 'N/A') }} 
                
                대표자명: {{ profile.get('ceo_nm', 'N/A') }}
                
                주소: {{ profile.get('adres', 'N/A') }}
                
                
                📄 최근 공시 목록
                {% if filings.get('list') %}
                {% for item in filings.get('list', [])[:3] %}
                - [{{ item.get('rcept_dt', 'N/A') }}] {{ item.get('report_nm', 'N/A') }}
                {% endfor %}
                {% else %}
                - 공시 정보: {{ filings.get('message', '데이터가 없습니다.') }}
                {% endif %}
                
                📊 재무 정보 
                {% if financials.get('list') %}
                {% for item in financials.get('list', [])[:5] %}
                - 기준년도 ({{ item.get('bsns_year') }}년, {{ item.get('thstrm_dt') }})
                - 계정: {{ item.get('account_nm', 'N/A') }} / 금액: {{ item.get('thstrm_amount', 'N/A') }}
                {% endfor %}
                {% else %}
                - 재무 정보를 불러올 수 없습니다.
                {% endif %}
                """)
            
            summary = context_template.render(profile=profile, filings=filings, financials=financials)
            return summary
            
        except Exception as e:
            return f"Error fetching DART data: {str(e)}"

    async def close(self):
        """더 이상 필요하지 않음 - 각 요청마다 새 세션 사용"""
        pass
