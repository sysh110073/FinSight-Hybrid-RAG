import streamlit as st
from hybrid_rag_agent import hybrid_agent_executor
import csv
import datetime
import os
import uuid

# 設定網頁標題與圖示
st.set_page_config(page_title="FinSight Hybrid-RAG", page_icon="🏦", layout="centered")

st.title("🏦 FinSight Hybrid-RAG")
st.subheader("法人金融專屬 AI 助理 (SQL + Vector 混合檢索)")
st.markdown("本系統整合了 **SQL 結構化財務數據** 與 **ChromaDB 法說會逐字稿**。您可直接詢問財報數字與管理層展望。")

# --- 🚀 新增的日誌寫入函數 ---
def log_feedback(query, response, feedback):
    log_file = "feedback_log.csv"
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "query", "response", "feedback"]) # 寫入標頭
        
        writer.writerow([datetime.datetime.now().isoformat(), query, response, feedback])


# 初始化對話紀錄
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好！我是 FinSight 助理。您可以問我：「台積電 2023Q4 的營收與 EPS 是多少？管理層對於 AI 伺服器的看法為何？」", "id": "init"}
    ]

# 顯示過去的對話
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 在 AI 的歷史訊息旁也加上按鈕 (但只顯示，不觸發功能)
        if msg["role"] == "assistant" and msg["id"] != "init":
            feedback_cols = st.columns(10)
            with feedback_cols[0]:
                st.button("👍", key=f"like_{msg['id']}_history", disabled=True)
            with feedback_cols[1]:
                st.button("👎", key=f"dislike_{msg['id']}_history", disabled=True)

# 處理使用者輸入
if prompt := st.chat_input("請輸入您的提問..."):
    # 1. 顯示使用者輸入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 呼叫 Hybrid RAG Agent
    with st.chat_message("assistant"):
        with st.spinner("🧠 AI 正在分析意圖，並檢索多來源資料庫..."):
            try:
                # 取得 AI 回覆
                response = hybrid_agent_executor.invoke({"messages": [("user", prompt)]})
                ai_reply = response["messages"][-1].content
                st.markdown(ai_reply)
                
                # 為這次對話產生獨一無二的 ID
                msg_id = str(uuid.uuid4())
                
                # 將 AI 的回答與 ID 存入 session_state
                st.session_state.messages.append({"role": "assistant", "content": ai_reply, "id": msg_id})
                
                # --- 🚀 新增的回饋按鈕與日誌功能 ---
                feedback_cols = st.columns(10)
                with feedback_cols[0]:
                    if st.button("👍", key=f"like_{msg_id}"):
                        log_feedback(prompt, ai_reply, "like")
                        st.toast("感謝您的正面回饋！", icon="🎉")
                
                with feedback_cols[1]:
                    if st.button("👎", key=f"dislike_{msg_id}"):
                        log_feedback(prompt, ai_reply, "dislike")
                        st.toast("感謝回饋，我會將此案例加入優化清單！", icon="📝")

            except Exception as e:
                st.error(f"系統發生錯誤: {e}")
