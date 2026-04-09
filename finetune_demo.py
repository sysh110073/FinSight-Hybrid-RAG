import json
import os
from dotenv import load_dotenv

load_dotenv()

def create_finetuning_job():
    print("🚀 啟動 LLM Fine-Tuning 準備程序...")
    print("--------------------------------------------------")
    
    # 1. 檢查訓練資料集
    dataset_path = "finetuning_dataset.jsonl"
    if not os.path.exists(dataset_path):
        print("❌ 找不到訓練資料集 (finetuning_dataset.jsonl)，請先執行 prepare_finetuning.py")
        return
        
    print(f"✅ 找到訓練資料集: {dataset_path}")
    
    # 2. 模擬 OpenAI API 呼叫流程 (此處為實戰解說用的 Mock)
    print("\n[模擬 API 呼叫程序]")
    print("1. 正在上傳資料集至 OpenAI 伺服器...")
    print("   > client.files.create(file=open('finetuning_dataset.jsonl', 'rb'), purpose='fine-tune')")
    print("   > 檔案上傳成功，File ID: file-ArmCreditMemo12345")
    
    print("\n2. 正在建立微調任務 (基於 gpt-4o-mini)...")
    print("   > client.fine_tuning.jobs.create(training_file='file-ArmCreditMemo12345', model='gpt-4o-mini')")
    print("   > 任務建立成功，Job ID: ftjob-StevenARM2026")
    
    print("\n3. 模型訓練中... (通常需要 10~30 分鐘)")
    print("   > Status: running")
    print("   > Status: succeeded")
    
    print("\n🎉 微調完成！你現在擁有了一個具備「法金 ARM 專業口吻」的專屬模型。")
    print("模型代號: ft:gpt-4o-mini:steven-bank-arm:v1")
    print("\n💡 下一步：將此模型代號替換到 hybrid_rag_agent.py 中的 ChatOpenAI(model='...') 即可使用！")
    print("--------------------------------------------------")

if __name__ == "__main__":
    create_finetuning_job()
