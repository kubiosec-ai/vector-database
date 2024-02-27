from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
import openai
import logging
import sys
import os
import time

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

documents = SimpleDirectoryReader("data").load_data()

openai.api_key=os.getenv("OPENAI_API_KEY")

PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    print(time.time())
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(time.time())
    print("--- Saving persistent model ----")

else:
    # load the existing index
    print(time.time())
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    print(time.time())
    print("--- From persistent model ----")

query_engine = index.as_query_engine(similarity_top_k=2)
response = query_engine.query("What was the impact of COVID? Show statements in bullet form and show page reference after each statement.")
print(response)

for node in response.source_nodes:
    print("-----")
    text_fmt = node.node.get_content().strip().replace("\n", " ")[:1000]
    print(f"Text:\t {text_fmt} ...")
    print(f"Metadata:\t {node.node.metadata}")
    print(f"Score:\t {node.score:.3f}")
