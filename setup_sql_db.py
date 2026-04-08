import sqlite3
import pandas as pd

def init_db():
    conn = sqlite3.connect('financial_data.db')
    
    # 建立結構化財報資料表 (模擬 ARM 需要看的硬數據)
    data = {
        'company_code': ['2330', '2330', '2330', '2454', '2454'],
        'company_name': ['台積電', '台積電', '台積電', '聯發科', '聯發科'],
        'quarter': ['2023Q2', '2023Q3', '2023Q4', '2023Q3', '2023Q4'],
        'revenue_bn': [480.8, 546.7, 625.5, 110.0, 129.5],
        'gross_margin': [54.1, 54.3, 53.0, 47.4, 48.3],
        'eps': [7.01, 8.14, 9.21, 11.64, 16.15]
    }
    df = pd.DataFrame(data)
    df.to_sql('financials', conn, if_exists='replace', index=False)
    print("✅ SQLite 'financials' table created successfully with mock data.")
    conn.close()

if __name__ == "__main__":
    init_db()
