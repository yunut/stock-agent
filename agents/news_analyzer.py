from typing import List, Dict
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# 뉴스 분석 에이전트
class NewsAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.ddgs = DDGS()  # DDGS 객체 초기화
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
    
    def search_news(self, company: str) -> List[Dict]:
        """뉴스 검색"""
        query = f"{company} stock news"
        try:
            results = list(self.ddgs.news(
                query,
                max_results=5  # 최대 5개의 뉴스만 검색
            ))
            return results
        except Exception as e:
            print(f"뉴스 검색 중 오류 발생: {str(e)}")
            return []
    
    def _compress_news(self, news_list: List[Dict]) -> str:
        """뉴스 데이터를 압축하여 중요 정보만 추출합니다."""
        compressed_news = []
        for news in news_list:
            # 제목과 날짜만 포함
            compressed_news.append(f"제목: {news['title']}\n날짜: {news['date']}")
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
            "analysis": analysis
        }
    
    def get_news_analysis(self, company: str) -> Dict:
        """기업 이름을 받아 뉴스를 검색하고 분석합니다."""
        news_list = self.search_news(company)
        return self.analyze_news(company, news_list) 