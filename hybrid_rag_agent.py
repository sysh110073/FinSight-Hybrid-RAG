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
# 模組 2: Vector RAG (處理非結構化法說會/投顧報告/新聞)
# ==========================================
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

rag_llm = ChatOpenAI(model="ft:gpt-4.1-2025-04-14:personal::DSmsiQ2n", temperature=0)
system_prompt = (
    "請根據以下檢索到的資料來回答問題。若內容中沒有提到，請回答「資料中未提及」。\n\n檢索資料：\n{context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
question_answer_chain = create_stuff_documents_chain(rag_llm, prompt)

def create_filtered_retriever(doc_type: str):
    # 使用 Chroma 的 metadata filtering 功能，增加檢索數量以涵蓋圖片 OCR 產生的表格
    return vectorstore.as_retriever(search_kwargs={"k": 10, "filter": {"doc_type": doc_type}})

official_rag_chain = create_retrieval_chain(create_filtered_retriever("法說會"), question_answer_chain)
financial_statement_chain = create_retrieval_chain(create_filtered_retriever("財報"), question_answer_chain)
analyst_rag_chain = create_retrieval_chain(create_filtered_retriever("投顧報告"), question_answer_chain)

@tool
def query_company_official_view(query: str) -> str:
    """查詢公司自己發布的法說會或經營層展望資訊（例如：公司的毛利率預估、擴產計畫）。"""
    return official_rag_chain.invoke({"input": query})["answer"]

@tool
def query_financial_statement(query: str) -> str:
    """用來查詢公司發布的官方財務報告、四大報表數據（例如：資產負債表的資產/負債金額、現金流量表數據）。"""
    return financial_statement_chain.invoke({"input": query})["answer"]

@tool
def query_analyst_reports(query: str) -> str:
    """查詢外部投顧公司、券商分析師對該公司的評價、預估與看法（例如：外資降評、分析師對盈餘的預測）。"""
    return analyst_rag_chain.invoke({"input": query})["answer"]

# ==========================================
# 模組 3: Hybrid Orchestrator (混合檢索大腦)
# ==========================================
tools = [query_financial_data, query_company_official_view, query_financial_statement, query_analyst_reports]
main_llm = ChatOpenAI(model="gpt-4o", temperature=0) # 使用較強的模型做判斷與統整
main_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是專業的法金 ARM 助理。請根據使用者的提問，呼叫合適的工具來回答。\n"
               "1. 若詢問「EPS、營收、毛利率等精確硬數據」，優先呼叫 query_financial_data。\n"
               "2. 若詢問「資產負債表、資產總額、負債總額等報表科目數據」，必須呼叫 query_financial_statement。\n"
               "3. 當遇到對公司未來展望的評估時，如果問題提及『交叉比對』或需要『客觀分析』，"
               "請分別使用『query_company_official_view (公司官方說法)』與『query_analyst_reports (外部投顧看法)』進行比對，"
               "並將兩者的觀點融合成一篇專業的徵信報告段落。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

hybrid_agent_executor = create_react_agent(main_llm, tools)

if __name__ == "__main__":
    query = "請比較旺宏公司派（法說會）與外部投顧分析師對於 2025Q4 以後毛利率或產能利用率的看法差異？"
    print(f"\n[主管提問]: {query}\n")
    
    response = hybrid_agent_executor.invoke({"messages": [("user", query)]})
    print(f"\n====================================")
    print(f"🎯 [Hybrid RAG 最終報告]:\n{response['messages'][-1].content}")
    print(f"====================================\n")
