import time
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 定義要監控的資料夾
RAG_INBOX = "./rag_inbox"
FINETUNE_INBOX = "./finetune_reports"

class AutoProcessHandler(FileSystemEventHandler):
    def on_created(self, event):
        # 忽略資料夾的建立事件
        if event.is_directory:
            return
        
        filepath = event.src_path
        filename = os.path.basename(filepath)
        
        # 處理 RAG 新法說會/財報檔案 (.pdf, .txt)
        if filepath.startswith(os.path.abspath(RAG_INBOX)) or filepath.startswith(RAG_INBOX):
            if filename.endswith('.pdf') or filename.endswith('.txt'):
                print(f"\n[Watcher👀] 偵測到新的知識庫檔案: {filename}，正在啟動 Data Pipeline...")
                time.sleep(2) # 等待檔案完全寫入磁碟
                subprocess.run(["python", "data_pipeline.py"])
                print(f"[Watcher👀] {filename} 已成功匯入 Vector DB！等待下一個檔案...\n")
        
        # 處理 Fine-tune 新徵信報告 (.docx)
        elif filepath.startswith(os.path.abspath(FINETUNE_INBOX)) or filepath.startswith(FINETUNE_INBOX):
            if filename.endswith('.docx') and not filename.startswith('~'):
                print(f"\n[Watcher👀] 偵測到新的徵信報告: {filename}，正在啟動微調資料萃取器...")
                time.sleep(2) # 等待檔案完全寫入磁碟
                subprocess.run(["python", "auto_finetune_generator.py"])
                print(f"[Watcher👀] {filename} 已成功萃取成 JSONL 訓練集！等待下一個檔案...\n")

def start_watcher():
    # 確保資料夾存在
    os.makedirs(RAG_INBOX, exist_ok=True)
    os.makedirs(FINETUNE_INBOX, exist_ok=True)
    
    event_handler = AutoProcessHandler()
    observer = Observer()
    
    # 監控兩個資料夾
    observer.schedule(event_handler, path=RAG_INBOX, recursive=False)
    observer.schedule(event_handler, path=FINETUNE_INBOX, recursive=False)
    
    print("==================================================")
    print("📡 [FinSight Watcher] 自動化監控程式已啟動！")
    print(f"正在監控 RAG 知識庫收件匣: {RAG_INBOX}")
    print(f"正在監控 Fine-tune 報告收件匣: {FINETUNE_INBOX}")
    print("請隨時將 PDF 或 Word 檔案拖曳進資料夾，系統將自動處理...")
    print("==================================================\n")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 [Watcher] 監控已停止。")
    observer.join()

if __name__ == "__main__":
    start_watcher()
