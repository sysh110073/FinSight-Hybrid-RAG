# 使用輕量級的 Python 3.10 映像檔
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴 (針對 PyMuPDF 等套件需要的底層 C 庫)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝 Python 套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有專案原始碼到容器內
COPY . .

# 曝露 Streamlit 預設的 8501 Port
EXPOSE 8501

# 設定環境變數確保 Python 輸出不被緩衝 (即時顯示 Log)
ENV PYTHONUNBUFFERED=1

# 啟動 Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
