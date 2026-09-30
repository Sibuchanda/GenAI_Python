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


# 3. Get an existing document ID
data = vector_store.get(
    include=["documents", "metadatas"]
)

document_id = data["ids"][0]

print("Document ID:", document_id)


# 4. Delete the document
vector_store.delete(
    ids=[document_id]
)

print("Document deleted successfully!")


# 5. Check remaining documents
data = vector_store.get(
    include=["documents", "metadatas"]
)

print("\nRemaining documents:", len(data["ids"]))

for i in range(len(data["ids"])):
    print("\nID:", data["ids"][i])
    print("Content:", data["documents"][i])
    print("Metadata:", data["metadatas"][i])