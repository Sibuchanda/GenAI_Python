from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store=Chroma(
    collection_name="tech",
    embedding_function=embedding,
    persist_directory='./vector_db'
)

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

# vector_store.add_documents(docs)

retriver = vector_store.as_retriever(
  search_type="mmr", # this enables mmr searching
  search_kwargs={"k": 3, "lambda_mult": 0.5}  # k = top results, lambda_mult = relevance-diversity balance
)

query="what is langchain"


# Performing Standard Similarity search VS MMR
similarity_retriver = vector_store.as_retriever(
    search_kwargs={"k":3}
)

res = similarity_retriver.invoke(query)

for i,doc in enumerate(res):
    print(f"Result : {i+1}")
    print(doc.page_content)


print(f"\n\n=============MMR==============\n\n")
#======================================================
# Performing MMR
mmr_retriver = vector_store.as_retriever(
    search_type="mmr", # This enables MMR
    search_kwargs={"k": 3, "fetch_k": 4, "lambda_mult": 0.5}
)
mmr_res = mmr_retriver.invoke(query)

for i,doc in enumerate(mmr_res):
    print(f"Result : {i+1}")
    print(doc.page_content)




"""
NOTE : 

lambda_mult: Controls the trade-off between relevance and diversity:
   --> 1.0: Identical to standard cosine similarity search (100% relevance).
   --> 0.0: Maximizes diversity among results (100% diversity).
   --> 0.5 - 0.7: Recommended balanced range.

   


NOTE : 

fetch_k Reference
  --> What it does
       --> fetch_k sets the size of the initial candidate pool retrieved from the vector database based purely on similarity scores, before MMR filtering occurs.

  --> Why is fetch_k Important?
        --> Without fetch_k, MMR wouldn't have a broad enough pool of options to select distinct topics.

        --> If fetch_k is too small (e.g., fetch_k = 2 when k = 2): MMR has only 2 candidates to choose from, so it can't add any diversity. It acts exactly like regular similarity search.

        --> If fetch_k is sufficiently larger than k (e.g., fetch_k = 10, k = 3): MMR gets a rich pool of 10 relevant documents to pick 3 distinct, non-repetitive answers from.



Rule of Thumb :  Set fetch_k to 2x to 4x the value of k.

"""