import os
import json
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import fitz
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def extract_metadata(pdf_path):
    filename = os.path.basename(pdf_path)
    try:
        doc = fitz.open(pdf_path)
        first_page = doc.load_page(0).get_text()
        
        prompt = f"""你是一個專業的金融資料工程師。
請根據這個法說會簡報的檔名與首頁文字，精準提取以下 Metadata：
1. "company_name" (公司名稱，例如 '台積電')
2. "year_quarter" (年份與季別，例如 '2023Q4'，若無請填 'Unknown')
3. "industry" (產業別，例如 '半導體', '金融業', '電子零組件')

檔名：{filename}
首頁文字內容：
{first_page[:1000]}
"""
        messages = [
            SystemMessage(content="你只能輸出純 JSON 格式，不要有 ```json 標記。"),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        content = response.content.replace("```json", "").replace("```", "").strip()
        meta = json.loads(content)
        meta["source"] = filename
        return meta
    except Exception as e:
        print(f"⚠️ Metadata 提取失敗，將使用預設檔名。錯誤: {e}")
        return {"source": filename}

def update_existing_db():
    print("啟動 Vector DB 更新程序...")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
    
    pdf_files = [f for f in os.listdir("rag_processed") if f.endswith(".pdf")]
    
    for pdf_file in pdf_files:
        print(f"處理檔案: {pdf_file}")
        source_name = pdf_file
        pdf_path = os.path.join("rag_processed", pdf_file)
        
        # 從檔名與首頁提取 metadata
        meta = extract_metadata(pdf_path)
        print(f"萃取到的 Metadata: {meta}")
        
        # 從 Chroma 找出這個 source 的所有 doc IDs
        try:
            existing_data = vectorstore.get()
            if existing_data and existing_data.get('metadatas'):
                ids_to_update = []
                for idx, metadata in zip(existing_data['ids'], existing_data['metadatas']):
                    if metadata and metadata.get("source") == source_name:
                        ids_to_update.append(idx)
                
                if ids_to_update:
                    print(f"更新 {len(ids_to_update)} 個區塊的 Metadata...")
                    # Update each document's metadata
                    for idx in ids_to_update:
                        vectorstore._collection.update(ids=[idx], metadatas=[meta])
                    print("✅ 更新成功！")
                else:
                    print("📭 Vector DB 中找不到該檔案的內容。")
        except Exception as e:
            print(f"更新失敗: {e}")
            
    print("所有檔案處理完畢！")

if __name__ == "__main__":
    update_existing_db()
