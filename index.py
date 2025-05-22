import requests
from pprint import pformat
from pprint import pprint
import html2text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models, QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
import os
from mistralai import Mistral


encoder = SentenceTransformer("all-MiniLM-L6-v2")
api_key = "" 
model = "open-mistral-7b"
mistral_client = Mistral(api_key=api_key)


url = input('enter your webpage link : ')
response = requests.get(url)
text = html2text.html2text(response.text)


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=250,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
text_chunks = text_splitter.create_documents([text])

qdrant_client = QdrantClient(":memory:")

# embedding chunks
embeddings = []
for text_chunk in text_chunks:
    embeddings.append(encoder.encode(str(text_chunk)))

# creating qdrant colllection
qdrant_client.create_collection(
    collection_name="Scrap_collection",
    vectors_config=VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE,
    )
)

# creating points

points = []

for i in range(len(text_chunks)):
    point = models.PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={
            'chunk': str(text_chunks[i]),
            'url': url
        }
    )
    points.append(point)


# points = [
#     models.PointStruct(
#         id=idx,
#         vector=embedding,
#         payload={
#             'chunk': chunk,
#             'source': url
#         }
#     )
#     for idx, (chunk, embedding) in enumerate(zip(text_chunks, embeddings))
# ]

# uploading points
qdrant_client.upsert(
    collection_name="Scrap_collection",
    points=points
)

while True:
    userInput = input('Ask me anything about the webpage now ! : ')
    # embedding it
    embedded_user_input = encoder.encode(userInput)

    # searching in Qdrant
    search_result = qdrant_client.query_points(
        collection_name="Scrap_collection",
        query=embedded_user_input,
        limit=3
    ).points

    # filtering based on confidence score

    bad_points = []

    for search in search_result:
        if search.score <= 0.13:
            bad_points.append(search)
    for bad in bad_points:
        search_result.remove(bad)

    chat_response = mistral_client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": userInput
            },
            {
                "role": "system",
                "content": f"""
                            You are an AI assistant that represents the information entity below. Your ONLY source of truth is the content provided between the delimiters << >>.

                            <<
                            {search_result}
                            >>

                            Strict Rules:
                            - Only answer using the information inside the << >>.
                            - If the information does NOT answer the user's question, reply exactly: "I don't know."
                            - DO NOT guess. DO NOT create polite explanations.
                            - DO NOT ever say: "The data provided does not mention..." or similar.
                            - DO NOT refer to the delimiters << >> or mention "provided information."
                            - Use pronouns "we," "us," or "our" when answering, because you represent the entity.
                            - If any information is missing, vague, incomplete, or irrelevant to the question, reply exactly: "I don't know."
                            - Never invent new information.
                            - Reply directly with the final answer only. No extra thoughts, no commentary.

                            WARNING: If you break any of these instructions, you will fail the task.
                            """



            }
        ]
    )

    pprint(chat_response.choices[0].message.content)
