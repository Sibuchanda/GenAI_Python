from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# 2. Connect to the existing ChromaDB
vector_store=Chroma(
    collection_name="os_knowledge",
    embedding_function=embeddings,
    persist_directory='./chroma_db'
)


# 3. Perform Similarity Search With Score
query = "How does the operating system manage memory larger than RAM?"

results = vector_store.similarity_search_with_score(
    query=query,
    k=2
)

"""
Here 'results' returns list of touple ( because we used --> similarity_search_with_score) -->

results = [
    (Document(...), score),
    (Document(...), score)
]


"""


# 4. Display Results
print("\n========== SIMILARITY SEARCH WITH SCORE ==========")

for document, score in results:

    print("\nContent:", document.page_content)
    print("Metadata:", document.metadata)
    print("Score:", score)