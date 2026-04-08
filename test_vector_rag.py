import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def run_vector_rag():
    load_dotenv()
    
    print("Loading documents...")
    loader = TextLoader("mock_tsmc_earnings_call.txt", encoding="utf-8")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    print("Creating Vector DB (Chroma)...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_prompt = (
        "你是專業的法金 ARM (法人金融客戶關係經理) 助理。"
        "請根據以下檢索到的法說會內容來回答問題。若內容中沒有提到，請回答「資料中未提及」。"
        "回答請保持專業、客觀，並符合銀行徵信報告的語氣。\n\n"
        "檢索資料：\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    query = "請問管理層如何看待地緣政治風險？以及他們在美國與日本的佈局進度為何？"
    print(f"\n[使用者提問]: {query}")
    
    response = rag_chain.invoke({"input": query})
    print(f"\n[AI 報告]: {response['answer']}\n")

if __name__ == "__main__":
    run_vector_rag()
