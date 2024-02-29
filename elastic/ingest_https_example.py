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
    ["https://localhost:9200"],
    verify_certs=True,
    ca_certs='/Users/philippebogaerts/elastic/elasticsearch-1/config/certs/http_ca.crt',
     basic_auth=("USERID", "PWD")
    # any other configuration options
)

print("---------------------------------------------------------------------------------------------------")
print(client.info())
print("---------------------------------------------------------------------------------------------------")


client.inference.put_model(
    task_type="text_embedding",
    model_id="my_openai_embedding_model",
    body={
        "service": "openai",
        "service_settings": {"api_key": API_KEY},
        "task_settings": {"model": "text-embedding-3-small"},
    },
)
print("---------------------------------------------------------------------------------------------------")


client.ingest.put_pipeline(
    id="openai_embeddings_pipeline",
    description="Ingest pipeline for OpenAI inference.",
    processors=[
        {
            "inference": {
                "model_id": "my_openai_embedding_model",
                "input_output": {
                    "input_field": "plot",
                    "output_field": "embedding",
                },
            }
        }
    ],
)

print("---------------------------------------------------------------------------------------------------")

client.indices.delete(index="openai-movie-embeddings", ignore_unavailable=True)
client.indices.create(
    index="openai-movie-embeddings",
    settings={"index": {"default_pipeline": "openai_embeddings_pipeline"}},
    mappings={
        "properties": {
            "embedding": {
                "type": "dense_vector",
                "dims": 1536,
                "similarity": "dot_product",
            },
            "plot": {"type": "text"},
        }
    },
)

print("---------------------------------------------------------------------------------------------------")


url = "https://raw.githubusercontent.com/elastic/elasticsearch-labs/main/notebooks/search/movies.json"
response = requests.get(url, verify=False)
print("Load data Finished from URL")
print(response.content)

print("---------------------------------------------------------------------------------------------------")

# Load the response data into a JSON object
data_json = json.loads(response.content)
print(data_json)
print("---------------------------------------------------------------------------------------------------")

# Prepare the documents to be indexed
# Load the response data into a JSON object
data_json = json.loads(response.content)

# Prepare the documents to be indexed
documents = []
for doc in data_json:
    documents.append(
        {
            "_index": "openai-movie-embeddings",
            "_source": doc,
        }
    )
print("---------------------------------------------------------------------------------------------------")

# Use helpers.bulk to index
response = helpers.bulk(client, documents, index='openai-movie-embeddings')
print(response)
print("stuff added")
print("---------------------------------------------------------------------------------------------------")

print("Done indexing documents into `openai-movie-embeddings` index!")
time.sleep(10)

print("---------------------------------------------------------------------------------------------------")

dada = client.search(
    index="openai-movie-embeddings",
    size=3,
    knn={
        "field": "embedding",
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
