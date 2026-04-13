import re

with open('pdf_vision_ingest.py', 'r') as f:
    content = f.read()

# Add MarkdownHeaderTextSplitter import
if 'MarkdownHeaderTextSplitter' not in content:
    content = content.replace(
        'from langchain_text_splitters import RecursiveCharacterTextSplitter',
        'from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter'
    )

# Replace ingest_to_db function
new_ingest = """def ingest_to_db(text, metadata=None):
    \"\"\"將解析後的完整文本切塊並附加 Metadata 存入 Vector DB\"\"\"
    print("💾 開始將解析後的內容存入 Vector DB (Chroma)...")
    
    # 方案 B: 語義感知切塊 (MarkdownHeaderTextSplitter)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_splits = markdown_splitter.split_text(text)
    
    # 方案 A: 加大 Chunk Size (針對 Markdown 切不開的超長表格或段落)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    final_splits = text_splitter.split_documents(md_splits)
    
    # 確保 Metadata 附加
    if metadata:
        for doc in final_splits:
            doc.metadata.update(metadata)
    
    # 方案 A: 升級為 text-embedding-3-large
    vectorstore = Chroma.from_documents(
        documents=final_splits, 
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory="./chroma_db"
    )
    print(f"🎉 成功存入 Vector DB，共 {len(final_splits)} 個文本區塊！附帶 Metadata: {metadata['source'] if metadata else 'None'}")
"""

# Regex replace the old ingest_to_db
content = re.sub(r'def ingest_to_db\(text, metadata=None\):.*?(?=if __name__ == "__main__":)', new_ingest + '\n', content, flags=re.DOTALL)

with open('pdf_vision_ingest.py', 'w') as f:
    f.write(content)

