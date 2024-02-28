from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import StorageContext, load_index_from_storage
from IPython.display import Markdown, display
import os
import openai
import requests
import json
import time


openai.api_key=os.getenv("OPENAI_API_KEY")
API_KEY=os.getenv("OPENAI_API_KEY")


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from elasticsearch import Elasticsearch, helpers, exceptions
from urllib.request import urlopen


# Create the client instance
client = Elasticsearch(
    # For local development
     hosts=["http://localhost:9999"]
)
print("---------------------------------------------------------------------------------------------------")
print(client.info())
print("---------------------------------------------------------------------------------------------------")

dada = client.search(
    index="openai-movie-embeddings",
    size=3,
    knn={
        "field": "plot_embedding",
        "query_vector_builder": {
            "text_embedding": {
                "model_id": "my_openai_embedding_model",
                "model_text": "Fighting movie",
            }
        },
        "k": 10,
        "num_candidates": 100,
    },
)

for hit in dada["hits"]["hits"]:
    doc_id = hit["_id"]
    score = hit["_score"]
    title = hit["_source"]["title"]
    plot = hit["_source"]["plot"]
    print(f"Score: {score}\nTitle: {title}\nPlot: {plot}\n")
