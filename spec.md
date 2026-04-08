# FinSight Hybrid-RAG (企業級法金授信 AI 助理)

## 1. 專案概述 (Project Overview)
本專案旨在建立一個企業級的「混合檢索增強生成 (Hybrid RAG)」系統，專為法人金融 (ARM/RM) 撰寫徵信報告與風險評估所設計。
系統能夠同時從「結構化資料庫 (SQL)」獲取精確財務數據，並從「非結構化資料庫 (Vector DB)」檢索法說會逐字稿與新聞輿情，最終由 LLM 統整出符合法金專業口吻的分析報告。

## 2. 核心技術棧 (Tech Stack)
* **程式語言**: Python 3.10+
* **LLM 框架**: LangChain, OpenAI API (GPT-4o)
* **結構化資料庫**: SQLite / PostgreSQL
* **向量資料庫 (Vector DB)**: ChromaDB / FAISS
* **LLMOps 與監控**: LangSmith
* **部署與基礎架構**: Docker, Google Cloud Platform (Cloud Run)
* **版本控制**: Git (遵循 Conventional Commits 規範)

## 3. 系統架構設計 (Architecture)

### 3.1 混合檢索模組 (Hybrid Retrieval Engine)
* **SQL Agent**: 負責將使用者的自然語言問題（如「台積電 2023 Q3 的 EPS 是多少？」）轉換為 SQL Query，並從關聯式資料庫中提取精確數字。
* **Vector Retriever**: 負責將自然語言問題向量化，從 Vector DB 中進行相似度搜尋，提取質化文本（如「台積電對於未來資本支出的看法？」）。
* **Router / Orchestrator**: LangChain 核心路由機制，根據問題意圖判斷該呼叫 SQL 工具、Vector 工具，或是兩者皆呼叫並合併上下文。

### 3.2 模型微調與生成 (Fine-tuning & Generation)
* 收集過往優質的法金徵信報告作為訓練集，對 OpenAI 模型進行 Fine-tuning，確保生成的文字段落符合「銀行授信報告的專業、客觀語氣」。

### 3.3 驗證與監控 (LLMOps)
* 導入 LangSmith，監控每一筆 LLM 呼叫的 Trace。
* 追蹤 Token 使用量、Latency（延遲），以及透過自定義評估函數監控「幻覺率 (Groundedness)」。
* 建立自動化腳本，當新財報發布時自動清洗數據並更新 Vector DB 與 SQL。

## 4. 開發階段規劃 (Phases)

### Phase 1: 基礎混合檢索引擎開發 (MVP)
- [ ] 建立 Python 虛擬環境與安裝依賴套件。
- [ ] 準備 Mock Data（上市櫃公司財報 CSV 與法說會逐字稿 TXT）。
- [ ] 實作 SQL Database 建置與 LangChain SQL Agent。
- [ ] 實作 Vector DB 建置與 Document Loader / Text Splitter。
- [ ] 實作 Hybrid RAG Chain，結合兩者進行問答。

### Phase 2: LLMOps 監控與評估
- [ ] 註冊並整合 LangSmith API。
- [ ] 實作效能監控儀表板連線。
- [ ] 設計評估腳本 (Evaluator) 測試 RAG 準確度。

### Phase 3: 自動化 Pipeline 與 Fine-tuning
- [ ] 撰寫資料更新排程腳本 (Data Ingestion Pipeline)。
- [ ] 準備 JSONL 訓練資料並進行 OpenAI Fine-tuning。

### Phase 4: 容器化與雲端部署
- [ ] 撰寫 Dockerfile 進行容器化。
- [ ] 部署至 GCP Cloud Run，並透過 API 形式提供服務。

## 5. Git 開發規範 (Workflow)
* `main`: 穩定可部署版本。
* `dev`: 開發中版本。
* 開發新功能時，從 `dev` 切出 `feature/xxx` 分支，完成後 Merge 回 `dev`。
* Commit 訊息規範：`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`。