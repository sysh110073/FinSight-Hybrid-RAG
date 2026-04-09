#  FinSight Hybrid-RAG (企業級法金授信 AI 助理)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green)
![LangSmith](https://img.shields.io/badge/LLMOps-LangSmith-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

本專案旨在建立一個企業級的「混合檢索增強生成 (Hybrid RAG)」系統，專為法人金融 (ARM/RM) 撰寫徵信報告與風險評估所設計。系統能夠同時從「結構化資料庫 (SQL)」獲取精確財務數據，並從「非結構化資料庫 (Vector DB)」檢索法說會逐字稿與新聞輿情，並具備處理財報圖表的多模態解析能力。

##  核心亮點 (Core Features)

1. **混合檢索大腦 (Hybrid RAG Router)**
   * 使用 `LangGraph ReAct Agent`。AI 能夠自動判斷使用者意圖，精準路由至 SQL Database 撈取財報數字，或去 Chroma Vector DB 撈取質化報告，最後融合為一篇無幻覺的專業回覆。
2. **多模態圖表解析 (Multimodal PDF Parser)**
   * 針對金融業痛點開發：運用 `PyMuPDF` 結合 `GPT-4o-mini Vision` 模型，自動將法說會簡報中的「圖片與表格」進行 OCR 並轉為 Markdown 儲存至向量庫。
   * **成本控管機制 (Cost Optimization)**：內建圖片大小過濾 (Size Filtering)、MD5 雜湊去重 (Deduplication)，以及基於 `Tenacity` 的指數退避重試 (Exponential Backoff)，有效解決 API 429 限制並節省 80% Token 成本。
3. **企業級 LLMOps 監控 (LangSmith Integration)**
   * 完美整合 LangSmith，對每一次 LLM 呼叫進行 Token 成本、延遲 (Latency) 以及檢索路徑 (Trace) 的全方位監控。
4. **視覺化互動介面 (Streamlit UI)**
   * 拋棄終端機黑畫面，提供友善的對話式 Web 介面。

##  技術棧 (Tech Stack)
* **LLM & Agent**: OpenAI (GPT-4o / GPT-4o-mini), LangChain, LangGraph
* **Databases**: SQLite (Structured), ChromaDB (Vector)
* **LLMOps**: LangSmith
* **Frontend**: Streamlit
* **Utils**: PyMuPDF, Tenacity, Pandas

##  快速啟動 (Quick Start)

### 1. 系統環境要求
確保已安裝 Python 3.10+。

### 2. 安裝套件
```bash
git clone https://github.com/sysh110073/FinSight-Hybrid-RAG.git
cd FinSight-Hybrid-RAG
python -m venv .venv
source .venv/bin/activate  # Windows 請使用 .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 環境變數設定
請在專案根目錄建立 `.env` 檔案（此檔案已加入 `.gitignore`，請勿推送到 GitHub）：
```env
OPENAI_API_KEY="sk-proj-你的OpenAI金鑰"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="你的LangSmith金鑰"
LANGCHAIN_PROJECT="FinSight-Hybrid-RAG"
```

### 4. 初始化資料庫與啟動
```bash
# 建立 SQL 測試資料庫
python setup_sql_db.py

# 啟動 Streamlit 網頁對話介面
streamlit run app.py
```

##  專案結構 (Project Structure)
```text
FinSight-RAG/
├── app.py                   # Streamlit 網頁主程式
├── hybrid_rag_agent.py      # LangGraph 混合檢索路由大腦
├── pdf_vision_ingest.py     # 多模態 PDF 解析與成本控管管線
├── setup_sql_db.py          # SQLite 資料庫初始化腳本
├── test_langsmith.py        # LLMOps 監控測試腳本
├── test_sql_agent.py        # SQL 檢索測試
├── test_vector_rag.py       # Vector DB 檢索測試
├── spec.md                  # 系統架構規格書
└── requirements.txt         # 依賴套件清單
```

---
*Developed by Steven Huang | Data Scientist & AI Application Developer*
