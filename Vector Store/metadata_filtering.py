from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# 2. Connect to the existing ChromaDB
vector_store = Chroma(
    collection_name="os_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# 3. Perform Similarity Search with Metadata Filter
query = "How does memory work?"

results = vector_store.similarity_search(
    query=query,
    k=2,
    filter={
        "category": "Memory Management"
    }
)


print("\n========== METADATA FILTERING ==========")

for document in results:

    print("\nContent:", document.page_content)
    print("Metadata:", document.metadata)