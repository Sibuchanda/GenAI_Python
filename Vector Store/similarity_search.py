from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()


# 1. Create the same embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# 2. Create Chroma Vector Store
vector_store = Chroma(
    collection_name="os_knowledge",
    embedding_function=embeddings, # Here we link openAI embedding model
    persist_directory="./chroma_db" # create a persistant storage locally , although we have to generate the embedding everytime
)


# 3. Perform Similarity Search
query = "What happens when processes wait for each other's resources?"

results = vector_store.similarity_search(
    query=query,
    k=2 # How many most relevant documents do we want back from the vector store
)



"""
results will return a list of langchain documents -->

results = [
    Document(
        page_content="Deadlock occurs when...",
        metadata={
            "topic": "Deadlock",
            "category": "Process Management"
        }
    ),

    Document(
        page_content="A process is a program...",
        metadata={
            "topic": "Process",
            "category": "Process Management"
        }
    )
]

"""


# 4. Display Results
print("\n========== SIMILARITY SEARCH ==========")

for document in results:
    print("\nContent:", document.page_content)
    print("Metadata:", document.metadata)

