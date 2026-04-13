import argparse
from hybrid_rag_agent import hybrid_agent_executor
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="FinSight Hybrid-RAG CLI 測試工具")
    parser.add_argument("--query", "-q", type=str, required=True, help="請輸入您的問題 (例如: -q '請比較旺宏法說會與投顧對 2025Q4 毛利的看法')")
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print(f"👤 [使用者提問]: {args.query}")
    print("="*50)
    print("🤖 AI 大腦 (Agentic RAG) 正在思考並檢索多來源資料庫...\n")
    
    try:
        response = hybrid_agent_executor.invoke({"messages": [("user", args.query)]})
        print("\n" + "🎯"*10 + " [Hybrid RAG 最終報告] " + "🎯"*10)
        print(f"\n{response['messages'][-1].content}\n")
        print("="*50 + "\n")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}\n")

if __name__ == "__main__":
    main()
