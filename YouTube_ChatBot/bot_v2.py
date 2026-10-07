from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
parser = StrOutputParser()

yt_api = YouTubeTranscriptApi()
video_id="KVI8MKzDnZI"

# Step : 1  Loading docements
try:
    fetched_transcript = yt_api.fetch(video_id,languages=["hi"])
    transcript_text = ""
    for snippet in fetched_transcript:
        transcript_text+=snippet.text + " "
except TranscriptsDisabled:
    print("No captions available for this video")
    transcript_text = ""

if len(transcript_text)==0:
    print("No transcripted Text found")
    exit()

# Step : 2  Indexing( Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(transcript_text)


# Step 3 : Indexing (Embedding Generation and Storing in Vector Store)
vector_store= Chroma(
    embedding_function=embedding,
    collection_name="YouTube_Content",
    persist_directory='./vector_DB'
)

vector_store.add_texts(chunks)
retriver = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})


# 6. Format Retrieved Documents
def format_docs(docs):
    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"
    return context


prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say I don't know based on the provided context.

      {context}
      Question: {question}
    """,
    input_variables=['context','question']
)

parallel_chain = RunnableParallel({
   'context' : retriver | RunnableLambda(format_docs),
   'question' : RunnablePassthrough()
})

main_chain = parallel_chain | prompt | llm | parser

question = input("Ask anything ? \n")
ans = main_chain.invoke(question)

print(f"\n Answer : {ans}")