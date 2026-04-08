import fitz  # PyMuPDF
import base64
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 載入環境變數
load_dotenv()

# 初始化具備視覺能力的模型 (GPT-4o)
llm_vision = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1000)

def encode_image(image_bytes):
    """將圖片 bytes 轉換為 base64 編碼"""
    return base64.b64encode(image_bytes).decode('utf-8')

def extract_text_from_image(image_bytes):
    """呼叫 GPT-4o Vision 模型解析圖片內容"""
    base64_image = encode_image(image_bytes)
    message = HumanMessage(
        content=[
            {
                "type": "text", 
                "text": "你是一個專業的金融數據分析師。請將這張圖片中的財務數據、圖表或文字，精確轉換為 Markdown 格式的純文字與表格。如果圖片只是裝飾性背景或沒有意義，請直接回覆 'IGNORE'。"
            },
            {
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            }
        ]
    )
    response = llm_vision.invoke([message])
    return response.content

def process_pdf(pdf_path):
    """讀取 PDF，抽取純文字與圖片，並將圖片送去 OCR 轉換"""
    doc = fitz.open(pdf_path)
    full_text = ""
    print(f"📄 開始解析 PDF: {pdf_path}")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # 1. 抽取純文字
        page_text = page.get_text()
        full_text += f"\n\n--- 第 {page_num+1} 頁文字 ---\n" + page_text

        # 2. 抽取圖片並用 Vision 模型轉換
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            print(f"🔍 發現圖片 (頁碼 {page_num+1}, 圖片 {img_index+1})，正在呼叫 GPT-4o 進行視覺解析...")
            vision_text = extract_text_from_image(image_bytes)
            
            # 如果 AI 判斷這不是裝飾圖片，就將解析出來的 Markdown 加入總文本
            if "IGNORE" not in vision_text.upper():
                print(f"✅ 圖片解析成功！內容已轉換為 Markdown。")
                full_text += f"\n\n--- 第 {page_num+1} 頁圖片解析內容 ---\n" + vision_text
            else:
                print(f"⏭️ 忽略裝飾性圖片。")

    return full_text

def ingest_to_db(text):
    """將解析後的完整文本切塊並存入 Vector DB"""
    print("💾 開始將解析後的內容存入 Vector DB (Chroma)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    
    # 存入 ChromaDB，持久化儲存
    vectorstore = Chroma.from_texts(
        texts=texts, 
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )
    print(f"🎉 成功存入 Vector DB，共 {len(texts)} 個文本區塊！")

if __name__ == "__main__":
    import sys
    
    # 測試用的檔案路徑，可透過 command line 傳入
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "sample_report.pdf"
    
    if os.path.exists(pdf_file):
        final_text = process_pdf(pdf_file)
        ingest_to_db(final_text)
    else:
        print(f"❌ 找不到檔案 {pdf_file}！")
        print("💡 請在終端機輸入: python pdf_vision_ingest.py <你的PDF檔名.pdf>")
