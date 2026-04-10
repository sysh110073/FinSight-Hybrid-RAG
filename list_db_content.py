from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
data = vectorstore.get()

print("Vector DB 筆數:", len(data['ids']))
if len(data['ids']) > 0:
    for meta in data['metadatas'][:5]:
        print("Metadata:", meta)
