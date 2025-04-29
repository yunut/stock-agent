from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
import json
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 뉴스 분석 에이전트
class NewsAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.client_id = os.getenv('NAVER_CLIENT_ID')
        self.client_secret = os.getenv('NAVER_CLIENT_SECRET')
        if not self.client_id or not self.client_secret:
            raise ValueError("NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET이 환경 변수에 설정되어 있지 않습니다.")
        self._setup_prompts()
        
    def _setup_prompts(self):
        """프롬프트 템플릿을 설정합니다."""
        self.news_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 주식 투자 전문가입니다. 주어진 뉴스들을 분석하여 투자 의견을 제시해주세요.
            다음 사항들을 고려하여 분석해주세요:
            1. 뉴스의 긍정/부정적 영향
            2. 회사의 미래 전망
            3. 투자 위험 요소
            4. 투자 기회 요소
            
            분석 결과는 다음 형식으로 작성해주세요:
            - 긍정적 요소: [항목들]
            - 부정적 요소: [항목들]
            - 투자 의견: [의견]
            - 투자 위험도: [낮음/중간/높음]
            """),
            ("user", "다음은 {company}에 대한 최근 뉴스들입니다:\n{news}")
        ])
        
        self.news_chain = self.news_analysis_prompt | self.llm
    
    def search_news(self, company: str) -> List[Dict[str, str]]:
        """네이버 검색 API를 사용하여 뉴스를 검색합니다.
        
        Args:
            company (str): 종목 코드 또는 회사명
            
        Returns:
            List[Dict[str, str]]: 뉴스 정보 리스트
        """
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        params = {
            "query": company,
            "display": 5,  # 검색 결과 개수
            "sort": "date"  # 최신순 정렬
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            news_data = response.json()
            news_list = []
            
            for item in news_data.get('items', []):
                # HTML 태그 제거
                title = BeautifulSoup(item['title'], 'html.parser').get_text()
                description = BeautifulSoup(item['description'], 'html.parser').get_text()
                
                # 날짜 형식 변환
                pub_date = datetime.strptime(item['pubDate'], '%a, %d %b %Y %H:%M:%S %z')
                formatted_date = pub_date.strftime('%Y.%m.%d %H:%M')
                
                news_list.append({
                    'title': title,
                    'date': formatted_date,
                    'source': item.get('source', '네이버 뉴스'),
                    'link': item['link'],
                    'body': description
                })
            
            return news_list
            
        except Exception as e:
            print(f"뉴스 검색 중 오류 발생: {str(e)}")
            return []
    
    def _compress_news(self, news_list: List[Dict]) -> str:
        """뉴스 데이터를 압축하여 중요 정보만 추출합니다."""
        compressed_news = []
        for news in news_list:
            compressed_news.append(
                f"제목: {news['title']}\n"
                f"날짜: {news['date']}\n"
                f"내용 요약: {news.get('body', '내용 없음')}"
            )
        return "\n\n".join(compressed_news)
    
    def analyze_news(self, company: str, news_list: List[Dict]) -> Dict:
        """뉴스 목록을 분석하여 투자 의견을 생성합니다."""
        # 뉴스 데이터 압축
        compressed_news = self._compress_news(news_list)
        
        # LLM을 사용하여 뉴스 분석
        analysis = self.news_chain.invoke({
            "company": company,
            "news": compressed_news
        })
        
        return {
            "news_list": news_list,
            "analysis": analysis.content  # AIMessage 객체에서 content만 추출
        }
    
    def get_news_analysis(self, company: str) -> Dict:
        """기업 이름을 받아 뉴스를 검색하고 분석합니다."""
        news_list = self.search_news(company)
        return self.analyze_news(company, news_list) 