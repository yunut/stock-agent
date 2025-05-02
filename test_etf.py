from agents.etf_analyzer import ETFAnalyzer
import json

def main():
    analyzer = ETFAnalyzer()
    
    # 사용자 질문 예시
    questions = [
        "1년 수익률이 좋고 성장성이 있는 섹터의 ETF를 추천해줘",
        "수수료가 낮고 안정적인 ETF를 추천해줘",
        "반도체와 2차전지 관련 ETF 중에서 추천해줘"
    ]
    
    for question in questions:
        print(f"\n질문: {question}")
        result = analyzer.analyze_query(question)
        
        if "error" in result:
            print(f"오류 발생: {result['error']}")
            continue
            
        print("\n추천 ETF:")
        for i, etf in enumerate(result["recommendations"][:3], 1):
            print(f"{i}. {etf['name']} ({etf['code']})")
            print(f"   - 운용사: {etf['company']}")
            print(f"   - 점수: {etf['score']:.2f}")
            print(f"   - 운용보수: {etf['expense_ratio']}%")
            print(f"   - 추적오차: {etf['tracking_error']}%")
            print(f"   - 순자산: {etf['total_assets']:,}원")
            print(f"   - 가입자수: {etf['subscribers']:,}명")
        
        print("\n상세 분석:")
        print(result["detailed_analysis"])
        
        print("\n추천 기준:")
        print(json.dumps(result["criteria"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main() 