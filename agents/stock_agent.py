from typing import Dict, List, TypedDict
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os
from .news_analyzer import NewsAnalyzer
from .financial_analyzer import FinancialAnalyzer
from .chart_analyzer import ChartAnalyzer
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore', category=UserWarning)

# 환경 변수 로드
load_dotenv()

class AgentState(TypedDict):
    """에이전트의 상태를 정의합니다."""
    messages: List[Dict]
    intermediate_steps: List

class StockAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.news_analyzer = NewsAnalyzer()
        self.financial_analyzer = FinancialAnalyzer()
        self.chart_analyzer = ChartAnalyzer()
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
        
    def _setup_tools(self) -> List[Tool]:
        """주식 분석에 필요한 도구들을 설정합니다."""
        tools = [
            Tool(
                name="chart_analysis",
                func=self._analyze_chart,
                description="주식 차트를 분석합니다. 입력: 티커 심볼"
            ),
            Tool(
                name="news_analysis",
                func=self._analyze_news,
                description="주식 관련 뉴스를 분석합니다. 입력: 티커 심볼"
            ),
            Tool(
                name="financial_analysis",
                func=self._analyze_financials,
                description="재무제표를 분석합니다. 입력: 티커 심볼"
            )
        ]
        return tools
    
    def _create_agent(self):
        """에이전트를 생성합니다."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """주식 투자 전문가로서 주어진 데이터를 분석하여 투자 의견을 제시해주세요.
            
            분석 결과는 다음 형식으로 작성해주세요:
            - 기술적 분석: [차트 분석 결과]
            - 뉴스 분석: [뉴스 분석 결과]
            - 재무 분석: [재무 분석 결과]
            - 종합 투자 의견: [최종 의견]
            - 투자 위험도: [낮음/중간/높음]
            - 매매 시그널: [매수/매도/홀딩]
            """),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        return create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
    
    def _analyze_chart(self, ticker: str) -> str:
        """주식 차트를 분석합니다."""
        result = self.chart_analyzer.get_chart_analysis(ticker)
        return result["analysis"].content if hasattr(result["analysis"], "content") else result["analysis"]
    
    def _analyze_news(self, ticker: str) -> str:
        """뉴스를 검색하고 분석합니다."""
        result = self.news_analyzer.get_news_analysis(ticker)
        return result["analysis"].content if hasattr(result["analysis"], "content") else result["analysis"]
    
    def _analyze_financials(self, ticker: str) -> str:
        """재무제표를 분석합니다."""
        result = self.financial_analyzer.get_financial_analysis(ticker)
        return result["analysis"].content if hasattr(result["analysis"], "content") else result["analysis"]
    
    def analyze_stock(self, ticker: str, thread_id: str = None) -> Dict:
        """주식 종목을 분석하고 투자 의견을 반환합니다."""
        try:
            # 에이전트 실행기 생성
            agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                max_iterations=3,  # 최대 반복 횟수 제한
                handle_parsing_errors=True  # 파싱 오류 처리
            )
            
            # 에이전트 실행
            result = agent_executor.invoke(
                {"input": f"주식 심볼 {ticker}에 대해 분석해주세요."}
            )
            
            return {
                "messages": [{"role": "assistant", "content": result["output"]}]
            }
        except Exception as e:
            print(f"분석 중 오류 발생: {str(e)}")
            return {
                "messages": [{"role": "assistant", "content": "분석 중 오류가 발생했습니다."}]
            }
    
    def _calculate_financial_ratios(self, financial_data: Dict) -> Dict:
        """재무비율 계산"""
        try:
            ratios = {}
            
            # 영업이익률
            if financial_data.get('totalRevenue', 0) != 0:
                ratios['operating_margin'] = (financial_data.get('operatingIncome', 0) / 
                                            financial_data.get('totalRevenue', 0)) * 100
            else:
                ratios['operating_margin'] = 0
                
            # 순이익률
            if financial_data.get('totalRevenue', 0) != 0:
                ratios['net_margin'] = (financial_data.get('netIncome', 0) / 
                                      financial_data.get('totalRevenue', 0)) * 100
            else:
                ratios['net_margin'] = 0
                
            # 부채비율
            if financial_data.get('totalStockholderEquity', 0) != 0:
                ratios['debt_ratio'] = (financial_data.get('totalDebt', 0) / 
                                      financial_data.get('totalStockholderEquity', 0)) * 100
            else:
                ratios['debt_ratio'] = 0
                
            # ROE
            if financial_data.get('totalStockholderEquity', 0) != 0:
                ratios['roe'] = (financial_data.get('netIncome', 0) / 
                               financial_data.get('totalStockholderEquity', 0)) * 100
            else:
                ratios['roe'] = 0
                
            return ratios
        except Exception as e:
            print(f"재무비율 계산 중 오류 발생: {str(e)}")
            return {
                'operating_margin': 0,
                'net_margin': 0,
                'debt_ratio': 0,
                'roe': 0
            }
    
    def plot_chart(self, ticker: str, save_path: str = None):
        """주식 차트를 시각화합니다."""
        self.chart_analyzer.plot_chart(ticker, save_path) 