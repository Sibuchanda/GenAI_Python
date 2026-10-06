from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()


# 1. Define models
cheap_model = ChatOpenAI(
    model="gpt-4o-mini", temperature=0
)  # Fast & cheap for compression
smart_model = ChatOpenAI(
    model="gpt-4o", temperature=0
)  # Powerful for final reasoning


embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store=Chroma(
    collection_name="Random",
    embedding_function=embedding,
    persist_directory='./vector_db'
)

# Relevant health & wellness documents
docs = [
    Document(
        page_content=(
            "Type 1 and Type 2 diabetes are chronic metabolic conditions. "
            "Common symptoms of diabetes include frequent urination, excessive thirst, "
            "unexplained weight loss, and extreme fatigue. "
            "The global prevalence of diabetes has risen significantly over the past three decades. "
            "Treatments range from daily insulin injections to dietary management and physical exercise."
        ),
        metadata={"source": "medical_journal_v1"}
    ),
    Document(
        page_content=(
            "Cardiovascular health depends on maintaining low cholesterol levels, regular aerobic exercise, "
            "and avoiding tobacco products. High blood pressure is a primary risk factor for heart attacks. "
            "Medical centers recommend annual check-ups to monitor systolic and diastolic pressure levels."
        ),
        metadata={"source": "heart_health_guide"}
    )
]

# vector_store.add_documents(docs)

base_retriver = vector_store.as_retriever(search_kwargs={"k":2})

# 4. Create the Compressor and Contextual Compression Retriever
compressor = LLMChainExtractor.from_llm(cheap_model)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriver
)

# 3. Setup final RAG Chain using SMART model
prompt = ChatPromptTemplate.from_template("""
Answer the question based strictly on the context below:
{context}

Question: {question}
""")

rag_chain = (
    {"context": compression_retriever, "question": RunnablePassthrough()}
    | prompt
    | smart_model  # High-tier model gets the compressed text
    | StrOutputParser()
)

# 4. Invoke
response = rag_chain.invoke("What are the symptoms of diabetes?")
print(response)





"""

[User Query] 
     │
     ▼
[Vector DB] ──(Retrieves 5 full documents)──► [Cheap LLM / Extractor]
                                                       │
                                            (Trims 90% of filler text)
                                                       │
                                                       ▼
[Smart LLM] ◄──(Receives only 2 clean sentences)──────┘
     │
     ▼
[Final Answer]


"""