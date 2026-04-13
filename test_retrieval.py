from hybrid_rag_agent import create_filtered_retriever
retriever = create_filtered_retriever("財報")
docs = retriever.invoke("資產負債表 存貨 現金流量")
for i, d in enumerate(docs):
    print(f"--- Doc {i} ---")
    print(d.page_content)
    print("----------------")
