from hybrid_rag_agent import create_filtered_retriever
r = create_filtered_retriever("財報")
# Increase k to 15 to find the target document
r.search_kwargs["k"] = 15
docs = r.invoke("旺宏 114年12月31日 資產負債表 總資產金額")
for i, d in enumerate(docs):
    if "78,500,385" in d.page_content:
        print(f"--- Doc {i} (FOUND!) ---")
        print(d.page_content)
