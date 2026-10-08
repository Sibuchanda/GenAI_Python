1. Get the transcripts of any video using api.
2. Merge the transcript text.
3. Splits the text.
4. Embedded the split text and then store in vector DB.
5. Get user query and convert in vector.
6. Retrive the vector DB based on user query.
7. Prepare a standard Template(To prevent Hallucination) and pass context and query to llm.
8. Return result to user.