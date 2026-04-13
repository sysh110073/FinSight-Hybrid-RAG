import json
from hybrid_rag_agent import create_filtered_retriever, question_answer_chain, rag_llm
from langchain_core.prompts import ChatPromptTemplate

retriever = create_filtered_retriever("財報")
print("1. 測試檢索 (Retriever):")
docs = retriever.invoke("根據旺宏官方財報,查詢114年12月31日的總資產金額")
for i, d in enumerate(docs):
    print(f"--- Doc {i} (Source: {d.metadata.get('source')}) ---")
    print(d.page_content[:150] + "...")

print("\n2. 測試 RAG 鏈直接呼叫:")
from langchain.chains import create_retrieval_chain
chain = create_retrieval_chain(retriever, question_answer_chain)
res = chain.invoke({"input": "根據旺宏官方財報,查詢114年12月31日的總資產金額"})
print("RAG Answer:", res["answer"])
