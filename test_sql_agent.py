import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

def run_sql_agent():
    load_dotenv()
    
    # 連結 SQLite 資料庫
    db = SQLDatabase.from_uri("sqlite:///financial_data.db")
    
    # 初始化 LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 建立 LangChain SQL Agent
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
    
    query = "請問台積電 2023Q4 的 EPS 和營收分別是多少？"
    print(f"\n[使用者提問]: {query}")
    
    response = agent_executor.invoke({"input": query})
    print(f"\n[AI 報告]: {response['output']}\n")

if __name__ == "__main__":
    run_sql_agent()
