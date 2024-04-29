import streamlit as st
import pandas as pd
from io import StringIO
import json
#from langchain.document_loaders import PyMuckedUnstructuredLoader
from langchain.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pymongo import MongoClient
from langchain.mongodb import MongoDBAtlasVectorSearch
import params
import requests
from io import StringIO
# Additional code
import argparse
from langchain_openai import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.chains.llm")
import ast
import ast
import plotly.graph_objects as go
from PIL import Image
import params
import requests

import ast  # Import the ast module

def get_response(query_input, vectorStore, compression_retriever, cache, llm, model_choice):
    st.write("---------------")
    relevant_docs = vectorStore.similarity_search(query_input)
    if relevant_docs:
        data = relevant_docs[0].page_content
        print("data is : ", data)
        print(type(data))
        compressed_docs = compression_retriever.get_relevant_documents(query_input)
        if compressed_docs:
            combined_content = "\n".join([doc.page_content for doc in compressed_docs])
            if combined_content in cache:
                # Retrieve the cached response
                response = cache[combined_content]
                st.write(response)
            else:
                # Generate a new response using the chosen LLM
                if model_choice == "OpenAI":
                    response_llm = llm(combined_content + "\nHuman: " + query_input + "\nAssistant:")
                    st.write(f"LLM Response: {response_llm}")
                elif model_choice == "Mistral":
                    MISTRAL_API_KEY = {params.MISTRAL_API_KEY}
                # Define the input messages
                    context_message = {
                        "role": "user",
                        "content": str(ast.literal_eval(data))  # Convert the context data to a string
                    }
                    query_message = {
                        "role": "user",
                        "content": "Human: " + query_input
                    }
                    input_messages = [context_message, query_message]
                    print("Input Messages:", input_messages)
                    # Send the request to the Mistral API
                    headers = {
                        'Authorization': f'Bearer {MISTRAL_API_KEY}',
                        'Content-Type': 'application/json'
                    }
                    payload = {
                        'query': {
                            'inputs': input_messages,
                            'task': 'text-generation',
                            'model': 'open-mistral-7b',
                            'mode': 'default'
                        }
                    }
                    print("Payload:", json.dumps(payload, indent=2))
                    response_mistral = requests.post('https://api.mistral.ai/query', headers=headers, json=payload)
                    # Check if the request was successful
                    if response_mistral.status_code == 200:
                        response_data = response_mistral.json()
                        response_text = response_data['results'][0]['generated_text']
                        st.write(f"Mistral Response: {response_text}")
                    else:
                        print(f'Error: {response_mistral.status_code} - {response_mistral.text}')
                                    # Store the response in the cache
                if model_choice == "OpenAI":
                    cache[combined_content] = response_llm
        else:
            st.write("No matching documents found.")
    else:
        st.info('Retry Genrating Again', icon="🔃")
    if relevant_docs:
        try:
            # Convert the string to a Python dictionary
            data_dict = ast.literal_eval(data)

            # Create a DataFrame from the dictionary
            df = pd.DataFrame.from_dict(data_dict, orient='index').transpose()

            # Get the student name and store it in a variable
            student_name = df.loc[0, 'Name '].strip()  # Remove extra whitespace

            # Exclude 'Seat Number', 'Name', and 'Result' columns
            df = df.drop(['Seat Number', 'Name ', 'Result '], axis=1)

            # Melt the DataFrame to create a long format
            df_melted = pd.melt(df.reset_index(), id_vars=['index'], value_vars=df.columns, var_name='Subject', value_name='Marks')

            # Plot the bar chart
            fig_bar = go.Figure(data=[go.Bar(x=df_melted['Subject'], y=df_melted['Marks'])])
            fig_bar.update_layout(title_text=f'{student_name} - Overall Performance')
            st.plotly_chart(fig_bar)

            # Create a pie chart
            fig_pie = go.Figure(data=[go.Pie(labels=df.columns, values=df.iloc[0], hole=0.4)])
            fig_pie.update_layout(title_text=f'{student_name} - Subject-wise Marks Distribution')
            st.plotly_chart(fig_pie)

        except Exception as e:
            st.info('We are currently Unable To Perfrom Visualization On this dataset', icon="ℹ️")
    else:
         st.info('Retry Genrating Again', icon="🔃")

def main():
    st.set_page_config(page_title="CSV-EVA", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(to right, rgb(29, 34, 55), rgb(29, 34, 55));
        }
        header .css-1595audm {
            display: none;
        }
        section[data-testid="stSidebar"] > div {
            background-color: rgb(35, 41, 64);
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: left;'>
        <h1 style='border-bottom: 3px solid;
                    border-image: linear-gradient(to right, violet, indigo, blue, green, yellow, orange, red) 1;'>
            💬 CHAT WITH CSV 📊 
        </h1>
    </div>
""", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        container = st.container()
        image = Image.open("female-robot-ai,-futuristic.png")
        with container:
            st.image(image, width=200)
        st.markdown("<h1 style='text-align: center'>EVA</h1>", unsafe_allow_html=True)
        st.divider()
        st.subheader("Upload Your Marksheet")

        # Create a file uploader without a visible label
        uploaded_file = st.file_uploader("", type=["csv"], label_visibility="hidden")

        if uploaded_file is not None:
            # Read the contents of the uploaded file
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            # Load the CSV data from StringIO using pandas
            df = pd.read_csv(stringio)

            # Create a preview button
            if st.button("Preview File"):
                # Open a container on the main screen
                with st.container():
                    st.write("### Preview of the uploaded file:")
                    st.write(df)
                    st.markdown(
                        "<a href='#' onclick='window.history.back();return false;'>&#9746; Cancel</a>",
                        unsafe_allow_html=True,
                    )

        # Add a selectbox to choose the LLM model
        model_choice = st.selectbox("Choose LLM Model", ["OpenAI", "Mistral{Disable for now}"])

    # Main content
    with st.container():
        if uploaded_file is not None:
            try:
                # Read the contents of the uploaded file
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                # Load the CSV data from StringIO using pandas
                df = pd.read_csv(stringio)
                data = df.to_dict('records')

                # Step 2: Transform (Split)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=["\n\n", "\n", r"(?<=\.)", " "], length_function=len)
                docs = [Document(page_content=str(doc)) for doc in data]
                docs = text_splitter.split_documents(docs)

                # Step 3: Embed
                embeddings = OpenAIEmbeddings(openai_api_key=params.openai_api_key)

                # Step 4: Store
                client = MongoClient(params.mongodb_conn_string)
                collection = client[params.db_name][params.collection_name]
                collection.delete_many({})  # Reset without deleting the Search Index

                # Insert documents in MongoDB Atlas with their embeddings
                docsearch = MongoDBAtlasVectorSearch.from_documents(docs, embeddings, collection=collection, index_name=params.index_name)

                st.success("CSV data successfully vectorized and stored in MongoDB Atlas!")

                # Additional code
                # Initialize MongoDB python client
                client = MongoClient(params.mongodb_conn_string)
                collection = client[params.db_name][params.collection_name]

                # Initialize vector store
                embeddings = OpenAIEmbeddings(openai_api_key=params.openai_api_key)
                vectorStore = MongoDBAtlasVectorSearch.from_documents(docs, embeddings, collection=collection, index_name=params.index_name)

                # Initialize LLM
                llm = OpenAI(openai_api_key=params.openai_api_key, temperature=0.7)

                # Contextual Compression
                compressor = LLMChainExtractor.from_llm(llm)
                compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vectorStore.as_retriever())

                # Initialize a cache dictionary
                cache = {}

                # User input and response
                query_input = st.text_input("Your question:", key="query_input")
                if st.button("Get Response"):
                    get_response(query_input, vectorStore, compression_retriever, cache, llm, model_choice)

            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()