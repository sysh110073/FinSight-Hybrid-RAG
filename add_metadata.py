import fitz
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def extract_metadata(pdf_path):
    filename = os.path.basename(pdf_path)
    try:
        doc = fitz.open(pdf_path)
        first_page = doc.load_page(0).get_text()
        
        prompt = f"""你是一個專業的金融資料工程師。
請根據這個法說會簡報的檔名與第一頁文字，提取以下 Metadata：
1. "company_name" (公司名稱，例如 '台積電')
2. "year_quarter" (年份與季別，例如 '2023Q4'，若無請填 'Unknown')
3. "industry" (產業別，例如 '半導體', '金融業', '航運業')

檔名：{filename}
首頁文字內容：
{first_page[:1000]}

請直接輸出 JSON 格式，包含上述三個欄位，不要有 ```json 等 markdown 標記。"""

        messages = [
            SystemMessage(content="你只能輸出純 JSON 格式。"),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        content = response.content.replace("```json", "").replace("```", "").strip()
        meta = json.loads(content)
        meta["source"] = filename
        return meta
    except Exception as e:
        print(f"Metadata 提取失敗: {e}")
        return {"source": filename}

print(extract_metadata("rag_processed/1815_富喬_2025-12-31.pdf"))
