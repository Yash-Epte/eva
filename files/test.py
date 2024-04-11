import argparse
import params
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain_openai import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import warnings
from vectorize import docs
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFacePipeline
import ssl

ssl._https_verify_certificates(enable=False)

warnings.filterwarnings("ignore", category=UserWarning, module="langchain.chains.llm")

# Initialize MongoDB python client
client = MongoClient(params.mongodb_conn_string, ssl_cert_reqs=ssl.CERT_NONE)
collection = client[params.db_name][params.collection_name]

# Initialize vector store
embeddings = OpenAIEmbeddings(openai_api_key=params.openai_api_key)
vectorStore = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, collection=collection, index_name=params.index_name
)

# Initialize OpenAI LLM
openai_llm = OpenAI(openai_api_key=params.openai_api_key, temperature=0.7)

# Initialize Mistral LLM
mistral_model_path = "/Desktop/mongo/mistral-7b-v0.1.Q3_K_L.gguf"
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_path)
mistral_model = AutoModelForCausalLM.from_pretrained(mistral_model_path)
mistral_llm = HuggingFacePipeline(model=mistral_model, tokenizer=mistral_tokenizer)

# Contextual Compression
compressor = LLMChainExtractor.from_llm(openai_llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=vectorStore.as_retriever()
)

print("Welcome to the Chat Bot! Type 'exit' to quit.")

# Initialize a cache dictionary
cache = {}

while True:
    query = input("\nYour question: ")
    if query.lower() == "exit":
        break
    print("---------------")
    relevant_docs = vectorStore.similarity_search(query)
    if relevant_docs:
        print(relevant_docs[0].page_content)
    else:
        print("No matching documents found.")
    print("\nAI Response:")
    print("-----------")
    compressed_docs = compression_retriever.get_relevant_documents(query)
    if compressed_docs:
        combined_content = "\n".join([doc.page_content for doc in compressed_docs])
        if combined_content in cache:
            # Retrieve the cached response
            response = cache[combined_content]
            print(response)
        else:
            # Generate a new response using OpenAI LLM
            openai_response = openai_llm(combined_content + "\nHuman: " + query + "\nAssistant:")
            print("OpenAI Response:")
            print(openai_response)

            # Generate a new response using Mistral LLM
            mistral_response = mistral_llm(combined_content + "\nHuman: " + query + "\nAssistant:")
            print("\nMistral Response:")
            print(mistral_response)

            # Store the responses in the cache
            # cache[combined_content] = openai_response
            # cache[combined_content] = mistral_response
    else:
        print("No matching documents found.")