import os
import shutil
import time
from pdf_vision_ingest import process_pdf, ingest_to_db

INBOX_DIR = "./rag_inbox"
PROCESSED_DIR = "./rag_processed"

def setup_directories():
    os.makedirs(INBOX_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def run_pipeline():
    setup_directories()
    print("🔄 開始執行自動化資料注入 Pipeline (Data Ingestion Pipeline)...")
    
    files = [f for f in os.listdir(INBOX_DIR) if f.endswith('.pdf') or f.endswith('.txt')]
    
    if not files:
        print("📭 目前收件匣 (data_inbox) 沒有新檔案需要處理。")
        return

    for file_name in files:
        file_path = os.path.join(INBOX_DIR, file_name)
        print(f"\n📂 偵測到新檔案: {file_name}，準備處理...")
        
        try:
            if file_name.endswith('.pdf'):
                # 處理 PDF (包含多模態視覺 OCR)
                text = process_pdf(file_path)
                ingest_to_db(text)
            elif file_name.endswith('.txt'):
                # 處理純文字
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                ingest_to_db(text)
                
            # 處理成功，移動到已處理區
            shutil.move(file_path, os.path.join(PROCESSED_DIR, file_name))
            print(f"✅ 檔案 {file_name} 處理完畢，已移至 {PROCESSED_DIR}")
            
        except Exception as e:
            print(f"❌ 檔案 {file_name} 處理失敗: {e}")

if __name__ == "__main__":
    run_pipeline()
