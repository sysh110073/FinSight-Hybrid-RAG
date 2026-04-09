from hybrid_rag_agent import hybrid_agent_executor

query = "請幫我總結聯發科 2023Q4 的 EPS 與營收表現，並說明蔡力行對於 Edge AI 和天璣 9300 的看法。"
print(f"\n[主管提問]: {query}\n")

response = hybrid_agent_executor.invoke({"messages": [("user", query)]})
print(f"\n====================================")
print(f"🎯 [Hybrid RAG 最終報告]:\n{response['messages'][-1].content}")
print(f"====================================\n")
