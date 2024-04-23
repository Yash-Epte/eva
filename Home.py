import streamlit as st
from PIL import Image


def main():
    im = Image.open("female-robot-ai,-futuristic.png")
    st.set_page_config(page_title="Home-EVA", page_icon=im, layout="wide", initial_sidebar_state="collapsed")

    st.markdown(
        """
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

            section[data-testid="stSidebar"] > div:nth-child(1) {
                display: none;
            }

            section[data-testid="stSidebar"] > div:nth-child(2) {
                position: absolute;
                bottom: 0;
                left: 0;
                width: 100%;
            }

            .glassmorphism-box {
                background: rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                font-size: 16px;
                position: relative;
                height: 300px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .glassmorphism-box::before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                border-radius: 20px;
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            }

            .box-content {
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
            }

            .title {
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 10px;
            }

            .content {
                margin-bottom: 20px;
            }

            .page-link-button {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-radius: 5px;
                padding: 10px 20px;
                color: #fff;
                text-decoration: none;
                transition: background-color 0.3s ease;
            }

            .page-link-button:hover {
                background-color: rgba(255, 255, 255, 0.3);
            }

            .page-link-button i {
                margin-right: 5px;
            }
        </style>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        """,
        unsafe_allow_html=True,
    )

    # Display the avatar image
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image("female-robot-ai,-futuristic.png", width=140)

    with col2:
        st.markdown(
            """
            <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative;'>
                <span style='font-size: 24px; background-color:; color: white; padding: 10px 20px;'>EVA</span>
                <hr style='width: 100%; border: 0; height: 2px; background-image: linear-gradient(to right, #ff0000, #ffa500, #ffff00, #008000, #0000ff, #4b0082, #ee82ee);'>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # Create two columns
    col1, col2 = st.columns([10,8])

    # Create the first glassmorphism box with a page link and image
    with col1:
        with st.container():
            #st.markdown('<div class="glassmorphism-box">', unsafe_allow_html=True)
            st.markdown('<div class="box-content">', unsafe_allow_html=True)
            st.markdown('<div class="title">Instructions for Chat With PDF</div>', unsafe_allow_html=True)
            st.image("data/pdf.png", width=550)
            st.markdown('<div class="content"> 1.This Projects Runs On Pdf Input and Genrate text as well as questions from that input given.<br>2.The project uses OpenAI LLM Model</div>', unsafe_allow_html=True)
            st.markdown('<div class="page-link">', unsafe_allow_html=True)
            st.page_link("pdf.png", label="Chat With PDF", icon="📄")
            st.markdown('</div></div></div>', unsafe_allow_html=True)

    # Create the second glassmorphism box with a page link and image
    with col2:
        with st.container():
           # st.markdown('<div class="glassmorphism-box">', unsafe_allow_html=True)
            st.markdown('<div class="box-content">', unsafe_allow_html=True)
            st.markdown('<div class="title">Instructions for Chat With CSV</div>', unsafe_allow_html=True)
            st.image("data/csv.png", width=550)
            st.markdown('<div class="content">1.This Projects Runs On CSV Input and Genrate text as well as visualize the input.<br>2.The project uses OpenAI LLM Model</div>', unsafe_allow_html=True)
            st.markdown('<div class="page-link">', unsafe_allow_html=True)
            st.page_link("csv.png", label="Chat With CSV", icon="📊")
            st.markdown('</div></div></div>', unsafe_allow_html=True)

if __name__ == "__main__": 
    main()