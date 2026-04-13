from hybrid_rag_agent import create_filtered_retriever
retriever = create_filtered_retriever("財報")
docs = retriever.invoke("旺宏 114年12月31日 總資產")
for i, d in enumerate(docs):
    print(f"--- Doc {i} ---")
    print(d.page_content)
