from hybrid_rag_agent import hybrid_agent_executor

print(">>> 測試代理大腦路由:")
query = "根據旺宏官方財報,查詢114年12月31日的總資產金額為多少?"
for step in hybrid_agent_executor.stream({"messages": [("user", query)]}):
    print(step)
