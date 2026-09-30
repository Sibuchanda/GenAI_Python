from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


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


# 4. Create the updated document
updated_doc = Document(
    page_content=(
        "A process is a program in execution. "
        "It has its own address space and system resources. "
        "The operating system manages processes through different states "
        "and uses scheduling to allocate CPU time."
    ),
    metadata={
        "topic": "Process",
        "category": "Process Management"
    }
)


# 5. Update the document
vector_store.update_document(
    document_id=document_id,
    document=updated_doc
)

print("Document updated successfully!")


# 6. View the updated document
data = vector_store.get(
    include=["documents", "metadatas"]
)

for i in range(len(data["ids"])):
    if data["ids"][i] == document_id:

        print("\nUpdated Document:")
        print("Content:", data["documents"][i])
        print("Metadata:", data["metadatas"][i])