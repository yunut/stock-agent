from typing import Dict, List
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
import yfinance as yf

# 재무재표 분석 에이전트
class FinancialAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self._setup_prompts()
        
    def _setup_prompts(self):
        """재무제표 분석을 위한 프롬프트를 설정합니다."""
        self.financial_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 재무제표 분석 전문가입니다. 주어진 재무제표를 분석하여 기업의 재무 건전성과 투자 가치를 평가해주세요.
            
            다음 지표들을 중심으로 분석해주세요:
            1. 수익성 지표
               - ROE(자기자본이익률)
               - 영업이익률
               - 순이익률
            2. 안정성 지표
               - 부채비율
               - 유동비율
               - 당좌비율
            3. 성장성 지표
               - 매출액 증가율
               - 영업이익 증가율
               - 순이익 증가율
            4. 밸류에이션 지표
               - PER(주가수익비율)
               - PBR(주가순자산비율)
               - EPS(주당순이익)
            
            분석 결과는 다음 형식으로 작성해주세요:
            - 수익성 평가: [평가내용]
            - 안정성 평가: [평가내용]
            - 성장성 평가: [평가내용]
            - 밸류에이션 평가: [평가내용]
            - 종합 투자 의견: [의견]
            - 투자 위험도: [낮음/중간/높음]
            """),
            ("user", "다음은 {company}의 재무제표 데이터입니다:\n\n{financial_data}")
        ])
        
        self.financial_chain = self.financial_analysis_prompt | self.llm
    
    def _calculate_financial_ratios(self, financials: pd.DataFrame, info: Dict) -> Dict:
        """재무비율을 계산합니다."""
        try:
            # 최근 4분기 데이터만 사용
            recent_data = financials.iloc[:, :4]
            
            ratios = {}
            
            # 수익성 지표
            if 'Net Income' in recent_data.index and 'Total Stockholder Equity' in recent_data.index:
                equity = recent_data.loc['Total Stockholder Equity'].mean()
                if equity != 0:
                    ratios['ROE'] = (recent_data.loc['Net Income'].sum() / equity)
                else:
                    ratios['ROE'] = 0
            
            if 'Operating Income' in recent_data.index and 'Total Revenue' in recent_data.index:
                revenue = recent_data.loc['Total Revenue'].sum()
                if revenue != 0:
                    ratios['영업이익률'] = (recent_data.loc['Operating Income'].sum() / revenue)
                else:
                    ratios['영업이익률'] = 0
            
            if 'Net Income' in recent_data.index and 'Total Revenue' in recent_data.index:
                revenue = recent_data.loc['Total Revenue'].sum()
                if revenue != 0:
                    ratios['순이익률'] = (recent_data.loc['Net Income'].sum() / revenue)
                else:
                    ratios['순이익률'] = 0
            
            # 성장성 지표 (전년 대비)
            if 'Total Revenue' in recent_data.index and len(financials.columns) >= 8:
                current_revenue = recent_data.loc['Total Revenue'].sum()
                prev_revenue = financials.loc['Total Revenue'].iloc[4:8].sum()
                if prev_revenue != 0:
                    ratios['매출액증가율'] = ((current_revenue - prev_revenue) / prev_revenue)
                else:
                    ratios['매출액증가율'] = 0
            else:
                ratios['매출액증가율'] = 0
            
            # 밸류에이션 지표
            if info and 'marketCap' in info and 'sharesOutstanding' in info and info['sharesOutstanding'] != 0:
                ratios['시가총액'] = info['marketCap']
                net_income = recent_data.loc['Net Income'].sum() if 'Net Income' in recent_data.index else 0
                ratios['EPS'] = net_income / info['sharesOutstanding']
                if ratios['EPS'] != 0:
                    ratios['PER'] = info['marketCap'] / (ratios['EPS'] * info['sharesOutstanding'])
                else:
                    ratios['PER'] = 0
            else:
                ratios['시가총액'] = 0
                ratios['EPS'] = 0
                ratios['PER'] = 0
            
            return ratios
        except Exception as e:
            print(f"재무비율 계산 중 오류 발생: {str(e)}")
            return {
                'ROE': 0,
                '영업이익률': 0,
                '순이익률': 0,
                '매출액증가율': 0,
                'EPS': 0,
                'PER': 0,
                '시가총액': 0
            }
    
    def _compress_financial_data(self, financials: pd.DataFrame) -> pd.DataFrame:
        """재무제표 데이터를 압축하여 중요 지표만 추출합니다."""
        # 중요 지표만 선택
        important_metrics = [
            'Total Revenue',
            'Operating Income',
            'Net Income',
            'Total Stockholder Equity',
            'Total Debt',
            'Total Current Assets',
            'Total Current Liabilities',
            'Cash And Cash Equivalents'
        ]
        
        # 중요 지표만 필터링
        compressed_data = financials[financials.index.isin(important_metrics)]
        
        # 최근 4분기 데이터만 사용
        return compressed_data.iloc[:, :4]
    
    def get_financial_analysis(self, ticker: str) -> Dict:
        """기업의 재무제표를 분석합니다."""
        # 야후 파이낸스에서 데이터 가져오기
        stock = yf.Ticker(ticker)
        financials = stock.financials
        info = stock.info
        
        # 재무제표 데이터 압축
        compressed_financials = self._compress_financial_data(financials)
        
        # 재무비율 계산
        ratios = self._calculate_financial_ratios(compressed_financials, info)
        
        # 주요 재무 지표만 선택
        key_metrics = {
            'ROE': ratios.get('ROE', 0),
            '영업이익률': ratios.get('영업이익률', 0),
            '순이익률': ratios.get('순이익률', 0),
            '매출액증가율': ratios.get('매출액증가율', 0),
            'EPS': ratios.get('EPS', 0),
            'PER': ratios.get('PER', 0)
        }
        
        # 재무제표와 비율을 문자열로 변환
        financial_text = f"""주요 재무 지표:
        ROE: {key_metrics['ROE']:.2%}
        영업이익률: {key_metrics['영업이익률']:.2%}
        순이익률: {key_metrics['순이익률']:.2%}
        매출액증가율: {key_metrics['매출액증가율']:.2%}
        EPS: {key_metrics['EPS']:.2f}
        PER: {key_metrics['PER']:.2f}"""
        
        # LLM을 사용하여 재무제표 분석
        analysis = self.financial_chain.invoke({
            "company": ticker,
            "financial_data": financial_text
        })
        
        return {
            "ratios": key_metrics,
            "analysis": analysis
        } 