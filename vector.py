from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

import os
import pandas as pd

db_location = "FAISS_langchain_db"
index_file = os.path.join(db_location, "index.faiss")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
df = pd.read_csv("realistic_restaurant_reviews.csv")

# Prepare documents always — needed if FAISS is missing
documents = []
ids = []

for i, row in df.iterrows():
    doc = Document(
        page_content=row["Title"] + "\n" + row["Review"],
        metadata={
            "rating": row["Rating"],
            "date": row["Date"],
            "id": i,
        },
        id=str(i)
    )
    documents.append(doc)
    ids.append(str(i))

# Load or create FAISS index
if os.path.exists(index_file):
    vector_store = FAISS.load_local(
        db_location,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("⚠️ FAISS index not found, creating a new one.")
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(db_location)

    # Add extra safeguard: if index was created empty
    if not vector_store.index.ntotal:
        vector_store.add_documents(documents=documents, ids=ids)

# Export retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
