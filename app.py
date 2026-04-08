import streamlit as st
from hybrid_rag_agent import hybrid_agent_executor

# 設定網頁標題與圖示
st.set_page_config(page_title="FinSight Hybrid-RAG", page_icon="🏦", layout="centered")

st.title("🏦 FinSight Hybrid-RAG")
st.subheader("法人金融專屬 AI 助理 (SQL + Vector 混合檢索)")
st.markdown("本系統整合了 **SQL 結構化財務數據** 與 **ChromaDB 法說會逐字稿**。您可直接詢問財報數字與管理層展望。")

# 初始化對話紀錄
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好！我是 FinSight 助理。您可以問我：「台積電 2023Q4 的營收與 EPS 是多少？管理層對於 AI 伺服器的看法為何？」"}
    ]

# 顯示過去的對話
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 處理使用者輸入
if prompt := st.chat_input("請輸入您的提問..."):
    # 1. 顯示使用者輸入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 呼叫 Hybrid RAG Agent
    with st.chat_message("assistant"):
        with st.spinner("🧠 AI 正在分析意圖，並檢索 SQL 與 Vector DB..."):
            try:
                # 傳入對話紀錄讓 Agent 執行
                response = hybrid_agent_executor.invoke({"messages": [("user", prompt)]})
                ai_reply = response["messages"][-1].content
                st.markdown(ai_reply)
                
                # 將 AI 的回答存入 session_state
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            except Exception as e:
                st.error(f"系統發生錯誤，請聯絡工程師: {e}")
