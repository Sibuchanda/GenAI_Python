from dotenv import load_dotenv
from langchain_community.retrievers import WikipediaRetriever


load_dotenv()


# Initialize the retriever (optional: set language and top_k)
retriever = WikipediaRetriever(top_k_results=2, lang="en")

query="Give me the nutrients presents in Soya chunks as per WHO"

# Get relevant Wikipedia documents
docs = retriever.invoke(query)


# Print retrieved content
for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")  # truncate for display