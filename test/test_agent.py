from agents.main_agent import MainAgent
import json

def main():
    """메인 함수"""
    agent = MainAgent()
    
    # 테스트 쿼리
    test_queries = [
        "안녕하세요",
        "KODEX 200 ETF에 대해 알려주세요",
        "삼성전자 주식 분석해주세요",
        "ETF와 주식의 차이점이 무엇인가요?",
        "최근 인기 있는 ETF는 무엇인가요?",
        "주식 투자 초보자를 위한 조언을 해주세요"
    ]
    
    # 각 쿼리 테스트
    for query in test_queries:
        print(f"\n질문: {query}")
        result = agent.process_query(query)
        
        print(f"선택된 에이전트: {result['agent']}")
        print(f"선택 이유: {result['reason']}")
        print("응답:")
        if isinstance(result['response'], dict):
            print(json.dumps(result['response'], ensure_ascii=False, indent=2))
        else:
            print(result['response'])
            
if __name__ == "__main__":
    main() 