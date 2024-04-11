# https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/mongodb_atlas
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
import params

# Step 1: Load
loaders = [
    CSVLoader("/Users/yashepte/Desktop/mongo/Students.csv")
]

data = []
for loader in loaders:
    data.extend(loader.load())

# Step 2: Transform (Split)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[
    "\n\n", "\n", "(?<=\\. )", " "], length_function=len)  # Escape the period with double backslashes
docs = text_splitter.split_documents(data)
print('Split into ' + str(len(docs)) + ' docs')

# Step 3: Embed
# https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.openai.OpenAIEmbeddings.html
embeddings = OpenAIEmbeddings(openai_api_key=params.openai_api_key)

# Step 4: Store
# Initialize MongoDB python client
client = MongoClient(params.mongodb_conn_string)
collection = client[params.db_name][params.collection_name]

# Reset w/out deleting the Search Index
collection.delete_many({})

# Insert the documents in MongoDB Atlas with their embedding
# https://github.com/hwchase17/langchain/blob/master/langchain/vectorstores/mongodb_atlas.py
docsearch = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, collection=collection, index_name=params.index_name
)





"""
df = pd.read_csv('/Users/yashepte/Desktop/mongo/Students.csv')
loaders = [
    CSVLoader("/Users/yashepte/Desktop/mongo/Students.csv")
]

data = []
for loader in loaders:
    data.extend(loader.load())

# Step 2: Transform (Split)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[
    "\n\n", "\n", "(?<=\\. )", " "], length_function=len)  # Escape the period with double backslashes
docs = text_splitter.split_documents(data)
print('Split into ' + str(len(docs)) + ' docs')





# Create a list of Document objects
docs = [Document(page_content=doc) for doc in df.apply(lambda row: ', '.join([f'{key}: {value}' for key, value in row.items()]), axis=1)]

print(docs)
# Step 3: Embed
# https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.openai.OpenAIEmbeddings.html
embeddings = OpenAIEmbeddings(openai_api_key=params.openai_api_key)

# Step 4: Store
# Initialize MongoDB python client
client = MongoClient(params.mongodb_conn_string)
collection = client[params.db_name][params.collection_name]

# Reset w/out deleting the Search Index
collection.delete_many({})

# Insert the documents in MongoDB Atlas with their embedding
# https://github.com/hwchase17/langchain/blob/master/langchain/vectorstores/mongodb_atlas.py
docsearch = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, collection=collection, index_name=params.index_name
)
"""