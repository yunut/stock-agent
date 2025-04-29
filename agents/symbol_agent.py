from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

# 환경 변수 로드
load_dotenv()

class StockSymbol(BaseModel):
    """주식 심볼 정보"""
    company_name: str = Field(description="회사명")
    symbol: str = Field(description="주식 심볼")
    market: str = Field(description="시장 (KOSPI, KOSDAQ, NASDAQ, NYSE 등)")
    confidence: float = Field(description="매칭 신뢰도 (0-1)")

class SymbolAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo"
        )
        self.parser = PydanticOutputParser(pydantic_object=StockSymbol)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 주식 심볼 매핑 전문가입니다.
            사용자가 입력한 회사명이나 주식 심볼을 분석하여 정확한 주식 정보를 제공합니다.
            
            한국 주식의 경우:
            - KOSPI: .KS 접미사 사용 (예: 005930.KS)
            - KOSDAQ: .KQ 접미사 사용 (예: 035720.KQ)
            
            미국 주식의 경우:
            - NASDAQ: 심볼 사용 (예: NVDA)
            - NYSE: 심볼 사용 (예: AAPL)
            
            주의사항:
            1. 정확한 심볼을 사용해야 합니다
            2. 한국 주식은 반드시 접미사를 포함해야 합니다
            3. 미국 주식은 접미사 없이 심볼만 사용합니다
            4. 모호한 경우 신뢰도(confidence)를 낮게 설정합니다
            5. 확실하지 않은 경우 None을 반환합니다
            
            잘 알려진 주식의 예시:
            - 삼성전자: 005930.KS
            - 현대자동차: 005380.KS
            - NVIDIA: NVDA
            - Apple: AAPL
            - Microsoft: MSFT
            - Tesla: TSLA
            
            응답 형식:
            {format_instructions}
            """),
            ("user", "{input}")
        ])
        
        self.prompt = self.prompt.partial(
            format_instructions=self.parser.get_format_instructions()
        )

    def get_symbol(self, input_text: str) -> Optional[StockSymbol]:
        """회사명이나 심볼을 입력받아 주식 정보를 반환"""
        try:
            # 입력 텍스트 정제
            input_text = input_text.strip()
            
            # LLM을 사용하여 심볼 매핑
            chain = self.prompt | self.llm | self.parser
            result = chain.invoke({"input": input_text})
            
            # 신뢰도가 낮은 경우 None 반환
            if result and result.confidence < 0.7:
                return None
                
            return result
        except Exception as e:
            print(f"심볼 매핑 중 오류 발생: {str(e)}")
            return None

    def validate_symbol(self, symbol: str) -> bool:
        """심볼의 유효성을 검증"""
        try:
            import yfinance as yf
            stock = yf.Ticker(symbol)
            # 간단한 데이터 조회로 심볼 유효성 검증
            info = stock.info
            return True
        except:
            return False

    def get_symbol_with_validation(self, input_text: str) -> Optional[StockSymbol]:
        """심볼을 검증하여 반환"""
        result = self.get_symbol(input_text)
        if result and self.validate_symbol(result.symbol):
            return result
        return None 