import nest_asyncio
import os

nest_asyncio.apply()

# https://cloud.llamaindex.ai/api-key
# wget https://arxiv.org/pdf/1706.03762.pdf -O attention.pdf


from llama_parse import LlamaParse

# https://cloud.llamaindex.ai/api-key

parser = LlamaParse(
    api_key="llx-xxxx",  
    result_type="markdown",  # "markdown" and "text" are available
    num_workers=4, 
    verbose=False,
    language="en" 

)

file_extractor = {".pdf": parser}

documents = parser.load_data("./attention.pdf")

print(documents[0].text)

