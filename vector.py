from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=row['Title'] + "\n" + row['Review'],
            metadata={
                "rating": row['Rating'],
                "date": row['Date'],
                "id": i
            },
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = FAISS.load_local(
    db_location,           # same folder you used for save_local
    embeddings,
    allow_dangerous_deserialization=True     # required if running on PyPI wheels
)


if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5
    }    
)
    