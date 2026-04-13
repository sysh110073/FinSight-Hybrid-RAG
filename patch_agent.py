import re

with open('hybrid_rag_agent.py', 'r') as f:
    content = f.read()

# Update embedding
content = content.replace(
    'vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())',
    'vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))'
)

# Restore k to 5
content = content.replace(
    'search_kwargs={"k": 10',
    'search_kwargs={"k": 5'
)

with open('hybrid_rag_agent.py', 'w') as f:
    f.write(content)
