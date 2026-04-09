import os
import glob
import json
import shutil
import docx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# 初始化用於生成訓練資料的模型 (這裡需要高智商的 GPT-4o 來逆向推導)
llm_extractor = ChatOpenAI(model="gpt-4o", temperature=0.2)

def extract_text_from_docx(file_path):
    """從 Word 檔案萃取純文字"""
    try:
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
    except Exception as e:
        print(f"讀取 {file_path} 失敗: {e}")
        return ""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_qa_pairs(report_text):
    """讓 GPT-4o 閱讀 ARM 報告，逆向生成 JSONL 格式的微調問答對"""
    prompt = f"""你是一位資深的金融資料工程師。
    我將提供一段「銀行法金 ARM 撰寫的真實徵信報告段落」。
    請你閱讀後，逆向推導出：
    1. **user_query (使用者提問)**: 假設這是一句簡短、隨意、白話的提問（例如：「這家公司 Q3 營收怎樣？毛利為什麼掉？」）。
    2. **assistant_response (ARM 專業回覆)**: 報告中對應的精準、專業、客觀的段落原文。

    請將結果輸出為嚴格的 JSON 陣列格式，包含至少 2-3 個問答對。
    每個物件必須有 "user_query" 和 "assistant_response" 兩個欄位。
    
    【徵信報告原文】：
    {report_text[:3000]}  # 限制長度避免爆 Token
    """
    
    messages = [
        SystemMessage(content="你只能輸出純 JSON 陣列，不要有任何 markdown 標記 (如 ```json) 或其他廢話。"),
        HumanMessage(content=prompt)
    ]
    
    response = llm_extractor.invoke(messages)
    try:
        # 清理可能存在的 markdown 標籤
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失敗: {e}")
        print(f"GPT 回傳原始內容: {response.content}")
        return []

def process_reports(input_dir="./finetune_reports", output_file="finetuning_dataset.jsonl", processed_dir="./finetune_processed"):
    """掃描資料夾內所有 Word 檔，自動生成微調資料集並歸檔"""
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    docx_files = glob.glob(os.path.join(input_dir, "*.docx"))
    
    if not docx_files:
        print(f"📭 找不到任何 .docx 檔案在 {input_dir} 資料夾中。")
        return

    print(f"🚀 開始將 {len(docx_files)} 份 Word 徵信報告轉換為 Fine-tuning 訓練集...")
    all_qa_pairs = []

    for file_path in docx_files:
        print(f"📄 處理報告: {os.path.basename(file_path)}")
        text = extract_text_from_docx(file_path)
        if not text:
            continue
            
        qa_pairs = generate_qa_pairs(text)
        for qa in qa_pairs:
            # 轉換成 OpenAI Fine-tuning 的標準格式
            formatted_item = {
                "messages": [
                    {"role": "system", "content": "你是專業的法金 ARM (法人金融客戶關係經理) 助理。"},
                    {"role": "user", "content": qa.get("user_query", "")},
                    {"role": "assistant", "content": qa.get("assistant_response", "")}
                ]
            }
            all_qa_pairs.append(formatted_item)

        # 處理成功後，將檔案移至 processed 資料夾
        shutil.move(file_path, os.path.join(processed_dir, os.path.basename(file_path)))
        print(f"✅ 檔案已歸檔至: {processed_dir}")

    # 如果有新資料，採用「追加寫入 (Append)」模式或讀取舊資料合併
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        for item in all_qa_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"🎉 成功生成！共萃取出 {len(all_qa_pairs)} 筆微調訓練資料。")
    print(f"📁 檔案已存檔至: {os.path.abspath(output_file)}")
    print("💡 面試亮點: 這套自動化資料產生器 (Synthetic Data Generator) 大幅降低了人工標註微調資料的成本！")

if __name__ == "__main__":
    process_reports()
