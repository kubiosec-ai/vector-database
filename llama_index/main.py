from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
import openai
import logging
import sys
import os

# For detailed logging, uncomment the following 2 lines
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Set your OS env variable like export OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxx
openai.api_key=os.getenv("OPENAI_API_KEY")

# Load the data files in the ./data directory
documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist()

query_engine = index.as_query_engine()
response = query_engine.query("what about robotics")
print(response)
