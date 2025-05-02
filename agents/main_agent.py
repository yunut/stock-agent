from typing import Dict, Optional
from .router_agent import RouterAgent
from .etf_analyzer import ETFAnalyzer
from .stock_agent import StockAgent

class MainAgent:
    def __init__(self):
        """메인 에이전트를 초기화합니다."""
        self.router = RouterAgent()
        self.etf_analyzer = ETFAnalyzer()
        self.stock_agent = StockAgent()
        
    def process_query(self, query: str) -> Dict:
        """사용자의 질문을 처리합니다."""
        try:
            # 1. 질문 라우팅
            routing = self.router.route_query(query)
            
            # 2. 적절한 에이전트로 라우팅
            if routing["agent"] == "general_agent":
                response = self.router.answer_general_query(query)
                return {
                    "agent": "general_agent",
                    "response": response,
                    "reason": routing["reason"]
                }
                
            elif routing["agent"] == "etf_agent":
                response = self.etf_analyzer.analyze_query(query)
                return {
                    "agent": "etf_agent",
                    "response": response,
                    "reason": routing["reason"]
                }
                
            elif routing["agent"] == "stock_agent":
                response = self.stock_agent.analyze_stock(query)
                return {
                    "agent": "stock_agent",
                    "response": response,
                    "reason": routing["reason"]
                }
                
            else:
                # 예상치 못한 에이전트 타입
                response = self.router.answer_general_query(query)
                return {
                    "agent": "general_agent",
                    "response": response,
                    "reason": "알 수 없는 에이전트 타입"
                }
                
        except Exception as e:
            print(f"질문 처리 중 오류 발생: {str(e)}")
            return {
                "agent": "general_agent",
                "response": "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다.",
                "reason": "오류 발생"
            } 