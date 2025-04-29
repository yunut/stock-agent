from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
import yfinance as yf
import warnings
import seaborn as sns

# 경고 메시지 숨기기
warnings.filterwarnings('ignore', category=UserWarning)

# 차트 분석 에이전트
class ChartAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self._setup_prompts()
        
    def _setup_prompts(self):
        """차트 분석을 위한 프롬프트를 설정합니다."""
        self.chart_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 주식 차트 분석 전문가입니다. 주어진 차트 데이터를 분석하여 투자 의견을 제시해주세요.
            
            다음 지표들을 중심으로 분석해주세요:
            1. 추세 분석
               - 주가 추세
               - 이동평균선
               - 추세선
            2. 기술적 지표
               - RSI (과매수/과매도)
               - MACD
               - 볼린저 밴드
            3. 거래량 분석
               - 거래량 추이
               - 거래량 패턴
            
            분석 결과는 다음 형식으로 작성해주세요:
            - 추세 분석: [분석내용]
            - 기술적 지표 분석: [분석내용]
            - 거래량 분석: [분석내용]
            - 종합 의견: [의견]
            - 투자 위험도: [낮음/중간/높음]
            - 매매 시그널: [매수/매도/홀딩]
            """),
            ("user", "다음은 {company}의 차트 데이터입니다:\n\n{chart_data}")
        ])
        
        self.chart_chain = self.chart_analysis_prompt | self.llm
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표를 계산합니다."""
        # 이동평균선
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df
    
    def _compress_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터를 압축하여 중요 정보만 추출합니다."""
        # 최근 30일 데이터만 사용
        recent_data = df.tail(30)
        
        # 일간 데이터로 리샘플링
        daily_data = recent_data.resample('D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'MA20': 'last',
            'MA60': 'last',
            'RSI': 'last',
            'MACD': 'last',
            'Signal': 'last'
        })
        
        return daily_data
    
    def _generate_chart_summary(self, df: pd.DataFrame) -> str:
        """차트 데이터를 요약하여 문자열로 변환합니다."""
        # 기술적 지표 계산
        df = self._calculate_technical_indicators(df)
        
        # 데이터 압축
        compressed_data = self._compress_data(df)
        
        # 중요 지점 추출
        current_price = compressed_data['Close'].iloc[-1]
        ma20 = compressed_data['MA20'].iloc[-1]
        ma60 = compressed_data['MA60'].iloc[-1]
        rsi = compressed_data['RSI'].iloc[-1]
        macd = compressed_data['MACD'].iloc[-1]
        signal = compressed_data['Signal'].iloc[-1]
        
        # 차트 데이터 요약
        chart_summary = f"""
        현재가: {current_price:.2f}
        20일 이동평균: {ma20:.2f}
        60일 이동평균: {ma60:.2f}
        RSI: {rsi:.2f}
        MACD: {macd:.2f}
        Signal: {signal:.2f}
        가격 추세: {'상승' if current_price > ma20 else '하락'}
        이동평균선 정렬: {'상승' if ma20 > ma60 else '하락'}
        RSI 상태: {'과매수' if rsi > 70 else '과매도' if rsi < 30 else '중립'}
        MACD 상태: {'상승' if macd > signal else '하락'}
        """
        
        return chart_summary
    
    def get_chart_analysis(self, ticker: str) -> Dict:
        """주식 차트를 분석합니다."""
        # 야후 파이낸스에서 데이터 가져오기
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")  # 3개월 데이터만 사용
        
        # 차트 데이터 요약 생성
        chart_summary = self._generate_chart_summary(hist)
        
        # LLM을 사용하여 차트 분석
        analysis = self.chart_chain.invoke({
            "company": ticker,
            "chart_data": chart_summary
        })
        
        return {
            "data": hist.to_dict(),
            "analysis": analysis
        }
    
    def plot_chart(self, ticker: str, save_path: str = None):
        """차트를 시각화합니다."""
        # 데이터 가져오기
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")  # 3개월 데이터만 사용
        hist = self._calculate_technical_indicators(hist)
        
        # 차트 생성
        plt.figure(figsize=(15, 10))
        
        # 주가 차트
        plt.subplot(2, 1, 1)
        plt.plot(hist.index, hist['Close'], label='Close')
        plt.plot(hist.index, hist['MA20'], label='MA20')
        plt.plot(hist.index, hist['MA60'], label='MA60')
        plt.title(f'{ticker} Stock Price')
        plt.legend()
        
        # 거래량 차트
        plt.subplot(2, 1, 2)
        plt.bar(hist.index, hist['Volume'], label='Volume')
        plt.title('Volume')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show() 