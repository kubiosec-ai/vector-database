from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

from IPython.display import Markdown, display
import os
import openai
import requests
import json
import time


openai.api_key=os.getenv("OPENAI_API_KEY")
API_KEY=os.getenv("OPENAI_API_KEY")

es = ElasticsearchStore(
    index_name="my_index",
    es_url="http://127.0.0.1:9999",
)

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

from llama_index.core import StorageContext

vector_store = ElasticsearchStore(
    es_url="http://localhost:9999",
    index_name="paul_graham",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine()
response = query_engine.query("what were his investments in Y Combinator?")
print(response)


