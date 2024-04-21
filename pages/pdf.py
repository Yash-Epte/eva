import os
import PyPDF2
import random
import itertools
from io import StringIO
from PIL import Image

import streamlit as st
from dotenv import load_dotenv
import params
# NEED TO - pip install faiss-cpu to use FAISS
from langchain.vectorstores import FAISS
# NEED TO - pip install openai to use OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever

from langchain.chains import RetrievalQA, QAGenerationChain

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


if "shared" not in st.session_state:
   st.session_state["shared"] = True

@st.cache_resource
def load_docs(files):
    st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text


@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "Similarity Search":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error creating vectorstore: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)
    elif retriever_type == "Support Vector Machines(not completed yet)":
        retriever = SVMRetriever.from_texts(splits, _embeddings)

    return retriever


@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    # Split texts
    # IN: text, chunk size, overlap, split_method
    # OUT: list of str splits

    st.info("`Splitting doc ...`")

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits


@st.cache_resource
def generate_eval(text, N, chunk):

    # Generate N questions from context of chunk chars
    # IN: text, N questions, chunk size to draw question from in the doc
    # OUT: eval set as JSON list

    st.info("`Generating sample questions ...`")
    n = len(text)
    starting_indices = [random.randint(0, n-chunk) for _ in range(N)]
    sub_sequences = [text[i:i+chunk] for i in starting_indices]
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
            st.write("Creating Question:", i+1)
        except:
            st.warning('Error generating question %s.' % str(i+1), icon="⚠️")
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full


def main():

    load_dotenv()
    st.set_page_config(page_title="PDF-EVA", page_icon=":📑", layout="wide", initial_sidebar_state="collapsed")
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
            💬 CHAT WITH PDF 📑 
        </h1>
    </div>
""", unsafe_allow_html=True)
        # uploading file


    with st.sidebar:
        
        container = st.container()
        image = Image.open("/Users/yashepte/Desktop/mongo/female-robot-ai,-futuristic.png")
        with container:
            st.image(image, width=200)
        st.markdown("<h1 style='text-align: center'>EVA</h1>", unsafe_allow_html=True)
        st.divider()
        uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
            "pdf", "txt"], accept_multiple_files=True)

    embedding_option = st.sidebar.radio(
        "Choose Embeddings", ["OpenAI Embeddings", "MistralAI"])
    
    retriever_type = "Similarity Search"

   
    # Use RecursiveCharacterTextSplitter as the default and only text splitter
    splitter_type = "RecursiveCharacterTextSplitter"

    if uploaded_files:
        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
            if 'eval_set' in st.session_state:
                del st.session_state['eval_set']

        # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(uploaded_files)
        st.write("Documents uploaded and processed.")

        # Split the document into chunks
        splits = split_texts(loaded_text, chunk_size=1000,
                             overlap=0, split_method=splitter_type)

        # Display the number of text chunks
        num_chunks = len(splits)
        st.write(f"Number of text chunks: {num_chunks}")

        # Embed using OpenAI embeddings or HuggingFace embeddings
        # Embed using OpenAI embeddings or MistralAI embeddings
        if embedding_option == "OpenAI Embeddings":
            embeddings = OpenAIEmbeddings()
        elif embedding_option == "MistralAI":
            # Initialize Mistral client
            mistral_client = MistralClient(api_key=params.MISTRAL_API_KEY)
            # Initialize Mistral sentence client
            sentence_client = MistralSentenceClient(mistral_client)

            # Generate embeddings using Mistral
            embeddings = []
            for chunk in splits:
                embedding = sentence_client.embed(chunk)
                embeddings.append(embedding)

        retriever = create_retriever(embeddings, splits, retriever_type)

# Initialize the language model (LLM) based on the chosen option
        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = CallbackManager([callback_handler])
        if embedding_option == "OpenAI Embeddings":
            llm = ChatOpenAI(
                streaming=True,
                callback_manager=callback_manager,
                verbose=True,
                temperature=0,
            )
        elif embedding_option == "MistralAI":
            llm = MistralAI(
                streaming=True,
                callback_manager=callback_manager,
                verbose=True,
                temperature=0,
                mistral_client=mistral_client,
            )

        qa = RetrievalQA.from_chain_type(
            llm=llm, retriever=retriever, chain_type="stuff", verbose=True
        )

        # Check if there are no generated question-answer pairs in the session state
        if 'eval_set' not in st.session_state:
            # Use the generate_eval function to generate question-answer pairs
            num_eval_questions = 3  # Number of question-answer pairs to generate
            st.session_state.eval_set = generate_eval(
                loaded_text, num_eval_questions, 3000)

        # Display the question-answer pairs in the sidebar with smaller text
        for i, qa_pair in enumerate(st.session_state.eval_set):
            st.sidebar.markdown(
                f"""
                <div class="css-card">
                <span class="card-tag">Question {i + 1}</span>
                    <p style="font-size: 12px;">{qa_pair['question']}</p>
                    <p style="font-size: 12px;">{qa_pair['answer']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # <h4 style="font-size: 14px;">Question {i + 1}:</h4>
            # <h4 style="font-size: 14px;">Answer {i + 1}:</h4>
        st.write("Ready to answer questions.")

        # Question and answering
        user_question = st.text_input("Enter your question:")
        if user_question:
            answer = qa.run(user_question)
            st.write("Answer:", answer)


if __name__ == '__main__':
    main()
