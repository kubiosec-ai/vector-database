from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
import openai
import logging
import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

documents = SimpleDirectoryReader("data").load_data()

openai.api_key = "xxxxxxxxxxxxxxx"

index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist()

query_engine = index.as_query_engine()
response = query_engine.query("what about robotics")
print(response)
