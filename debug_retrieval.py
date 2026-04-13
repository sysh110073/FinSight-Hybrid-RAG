from hybrid_rag_agent import create_filtered_retriever, question_answer_chain, rag_llm
from langchain_classic.chains import create_retrieval_chain

retriever = create_filtered_retriever("財報")
chain = create_retrieval_chain(retriever, question_answer_chain)
res = chain.invoke({"input": "請在旺宏財報(資產負債表)中尋找『總資產』或『資產總計』在114年12月31日的具體金額數字，請不要回報無法查詢"})
print("\n[強制檢索 RAG Answer]:", res["answer"])
