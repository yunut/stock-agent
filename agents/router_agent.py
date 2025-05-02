from typing import Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import json

class RouterAgent:
    def __init__(self):
        """라우터 에이전트를 초기화합니다."""
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7
        )
        self._setup_prompts()
        
    def _setup_prompts(self):
        """프롬프트 템플릿을 설정합니다."""
        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", """사용자의 질문을 분석하여 적절한 에이전트로 라우팅해주세요.

다음 에이전트 중 하나를 선택해야 합니다:
1. general_agent: 일반적인 질문에 답변
2. etf_agent: ETF 관련 질문에 답변
3. stock_agent: 주식 투자 관련 질문에 답변

다음 형식의 JSON으로 응답해주세요:
{
    "agent": "general_agent" | "etf_agent" | "stock_agent",
    "reason": "선택한 에이전트를 선택한 이유"
}"""),
            ("user", "{query}")
        ])
        
        self.general_prompt = ChatPromptTemplate.from_messages([
            ("system", """사용자의 질문에 친절하고 전문적으로 답변해주세요.
답변은 한국어로 작성해주세요."""),
            ("user", "{query}")
        ])
        
    def route_query(self, query: str) -> Dict:
        """질문을 분석하여 적절한 에이전트를 선택합니다."""
        try:
            # 질문 분석
            chain = self.router_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"query": query})
            
            # JSON 파싱
            try:
                routing = json.loads(result)
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 기본 에이전트로 라우팅
                routing = {
                    "agent": "general_agent",
                    "reason": "질문 분석 실패"
                }
            
            return routing
            
        except Exception as e:
            print(f"라우팅 중 오류 발생: {str(e)}")
            return {
                "agent": "general_agent",
                "reason": "오류 발생"
            }
            
    def answer_general_query(self, query: str) -> str:
        """일반적인 질문에 답변합니다."""
        try:
            chain = self.general_prompt | self.llm | StrOutputParser()
            return chain.invoke({"query": query})
            
        except Exception as e:
            print(f"일반 질문 답변 중 오류 발생: {str(e)}")
            return "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다." 