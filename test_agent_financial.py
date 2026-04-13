from hybrid_rag_agent import hybrid_agent_executor

query = "請根據旺宏官方財報，查詢其 114年12月31日 的『總資產』金額為多少？"
print(f"\n[主管提問]: {query}\n")

response = hybrid_agent_executor.invoke({"messages": [("user", query)]})
print(f"\n====================================")
print(f"🎯 [Hybrid RAG 最終報告]:\n{response['messages'][-1].content}")
print(f"====================================\n")
