import os
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def update_etf_data():
    """ETF 데이터를 업데이트합니다."""
    try:
        # KRX API를 통해 ETF 목록 가져오기
        url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020101"
        }
        data = {
            "bld": "dbms/MDC/STAT/standard/MDCSTAT01501",
            "locale": "ko_KR",
            "mktId": "STK",
            "share": "1",
            "csvxls_isNo": "false",
            "name": "fileDown",
            "url": "dbms/MDC/STAT/standard/MDCSTAT01501"
        }
        
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        
        # 응답을 DataFrame으로 변환
        etf_data = response.json()
        if 'output' not in etf_data:
            print("API 응답 형식이 예상과 다릅니다. 응답 내용:", etf_data)
            return
            
        df = pd.DataFrame(etf_data['output'])
        
        # 필요한 컬럼만 선택
        df = df[['ISU_CD', 'ISU_NM', 'MKT_NM', 'SECUGRP_NM', 'LIST_DD']]
        df.columns = ['code', 'name', 'market', 'category', 'listing_date']
        
        # 추가 정보 수집
        df['expense_ratio'] = 0.0
        df['tracking_error'] = 0.0
        df['total_assets'] = 0
        df['subscribers'] = 0
        
        # 데이터 저장
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # 현재 날짜로 백업 파일 생성
        backup_file = f"{data_dir}/etf_list_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(backup_file, index=False)
        
        # 최신 파일 업데이트
        latest_file = f"{data_dir}/etf_list.csv"
        df.to_csv(latest_file, index=False)
        
        print(f"ETF 데이터가 성공적으로 업데이트되었습니다. ({len(df)}개 ETF)")
        
    except Exception as e:
        print(f"ETF 데이터 업데이트 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    update_etf_data() 