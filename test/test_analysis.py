from agents.stock_agent import StockAgent
from agents.symbol_agent import SymbolAgent
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore', category=UserWarning)

def main():
    # 에이전트 초기화
    stock_agent = StockAgent()
    symbol_agent = SymbolAgent()
    
    # 사용자 입력 받기
    stock_name = input("분석할 주식 이름을 입력하세요 (예: 엔비디아, 애플): ").strip()
    
    # 주식 심볼 찾기
    symbol_info = symbol_agent.get_symbol_with_validation(stock_name)
    if not symbol_info:
        print(f"'{stock_name}'에 대한 주식 심볼을 찾을 수 없습니다.")
        return
    
    print(f"\n{symbol_info.company_name}({symbol_info.symbol}) 분석을 시작합니다...")
    
    # 주식 분석 실행
    thread_id = f"analysis_{symbol_info.symbol}"
    result = stock_agent.analyze_stock(symbol_info.symbol, thread_id)
    
    # 분석 결과 출력
    if result and "messages" in result:
        print("\n분석 결과:")
        if isinstance(result["messages"], list):
            print(result["messages"][-1]["content"])
        else:
            print(result["messages"].content)
    else:
        print("\n분석 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main() 