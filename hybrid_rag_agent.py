import os
import warnings
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

# 載入環境變數
load_dotenv()

# ==========================================
# 模組 1: SQL Agent (處理結構化財報數據)
# ==========================================
db = SQLDatabase.from_uri("sqlite:///financial_data.db")
sql_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
sql_agent_executor = create_sql_agent(sql_llm, db=db, agent_type="openai-tools", verbose=False)

@tool
def query_financial_data(query: str) -> str:
    """用來查詢公司的財報硬數據，例如營收、毛利率、EPS。"""
    return sql_agent_executor.invoke({"input": query})["output"]


# ==========================================
# 模組 2: Vector RAG (處理非結構化法說會/新聞)
# ==========================================
loader = TextLoader("mock_tsmc_earnings_call.txt", encoding="utf-8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
system_prompt = (
    "請根據以下檢索到的法說會內容來回答問題。若內容中沒有提到，請回答「資料中未提及」。\n\n檢索資料：\n{context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
question_answer_chain = create_stuff_documents_chain(rag_llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@tool
def query_earnings_call(query: str) -> str:
    """用來查詢公司法說會的質化資訊，例如未來展望、資本支出、地緣政治風險、AI 發展。"""
    return rag_chain.invoke({"input": query})["answer"]


# ==========================================
# 模組 3: Hybrid Orchestrator (混合檢索大腦)
# ==========================================
tools = [query_financial_data, query_earnings_call]
main_llm = ChatOpenAI(model="gpt-4o", temperature=0) # 使用較強的模型做判斷與統整
main_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是專業的法金 ARM 助理。請根據使用者的提問，呼叫合適的工具來回答。\n"
               "如果問題同時包含數字與質化資訊，請**分別呼叫兩個工具**，並將結果整合成一篇專業的徵信報告段落。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

hybrid_agent_executor = create_react_agent(main_llm, tools)

if __name__ == "__main__":
    query = "請幫我總結台積電 2023Q4 的 EPS 與營收表現，並說明管理層對 AI 伺服器營收貢獻的未來展望。"
    print(f"\n[主管提問]: {query}\n")
    
    response = hybrid_agent_executor.invoke({"messages": [("user", query)]})
    print(f"\n====================================")
    print(f"🎯 [Hybrid RAG 最終報告]:\n{response['messages'][-1].content}")
    print(f"====================================\n")
