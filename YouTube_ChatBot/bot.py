from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate



load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

yt_api = YouTubeTranscriptApi()
video_id="J5_-l7WIO_w"

# Step : 1  Loading docements
fetched_transcript = yt_api.fetch(video_id,languages=['hi'])
transcript_text=""

try:
    for snipped in fetched_transcript:
        transcript_text=transcript_text+snipped.text
except TranscriptsDisabled:
    print("No caption availabale for this video")


# Step : 2  Indexing( Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_text(transcript_text)
# print(len(chunks))
# print(chunks[1])


# Step 3 : Indexing (Embedding Generation and Storing in Vector Store)
vector_store= Chroma(
    embedding_function=embedding,
    collection_name="YouTube_Content",
    persist_directory='./vector_DB'
)

# vector_store.add_texts(chunks)
retriver = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=['context','question']
)

question =input("Ask Question?")
context = retriver.invoke(question)

final_prompt = prompt.invoke({"context": context, "question": question})

answer = llm.invoke(final_prompt)
print(answer.content)