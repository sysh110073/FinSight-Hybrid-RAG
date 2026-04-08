import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

def test_langsmith_connection():
    try:
        client = Client()
        # 建立一個測試用的專案
        project_name = os.environ.get("LANGCHAIN_PROJECT", "FinSight-Hybrid-RAG")
        print(f"嘗試連線至 LangSmith...")
        print(f"專案名稱: {project_name}")
        
        # 簡單呼叫以確認 API Key 正確
        projects = list(client.list_projects())
        print("✅ 連線成功！你的 LangSmith 已經可以開始監控了！")
    except Exception as e:
        print("❌ 連線失敗，請確認是否已正確設定 LANGCHAIN_API_KEY。")
        print(f"錯誤訊息: {e}")

if __name__ == "__main__":
    test_langsmith_connection()
