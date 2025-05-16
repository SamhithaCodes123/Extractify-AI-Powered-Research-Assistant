import os
import streamlit as st
import requests
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQAWithSourcesChain
from qdrant_client import QdrantClient
from googlesearch import search
from streamlit_lottie import st_lottie

# Qdrant API Keys
QDRANT_API_KEY = "Not mentioned here for privacy purpose"
QDRANT_URL = "https://9be38dbf-070f-4053-9fc3-645458aa72dd.us-west-2-0.aws.cloud.qdrant.io"
COLLECTION_NAME = "qdrant_cloud_documents"
GOOGLE_COLLECTION_NAME = "google_search_documents"

# OpenAI API Key
OPENAI_API_KEY = "Not mentioned here for privacy purpose"
# Function to Load Lottie Animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottieurl("https://lottie.host/17b0300c-5a09-4802-93dd-9686c8595ef7/2WsD31YVrA.json")

if lottie_animation:
    st_lottie(lottie_animation, speed = 1, height = 200, key = "animation")
else:
    st.error("⚠️ Lottie animation could not be loaded.")

# Apply CSS for Background Image
st.markdown(
    """
    <style>
        /* Background Image */
        .stApp {
            background: url("https://images.unsplash.com/photo-1617896376265-28bcb35febf7?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8YmxhY2slMjBzcGFjZXxlbnwwfHwwfHx8MA%3D%3D");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        /* Centered Title */
        .title {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            color: white;
            text-shadow: 3px 3px 10px rgba(0, 0, 0, 0.8);
        }

        /* Styled Input Fields */
        .stTextInput>div>div>input {
            border-radius: 10px;
            padding: 12px;
            font-size: 18px;
            border: 2px solid #FF4B4B;
            background: rgba(255, 255, 255, 0.7);
            color: black;
        }

        /* Button Styling */
        .stButton>button {
            border-radius: 10px;
            background: linear-gradient(135deg, #FF4B4B, #FF7878);
            color: white;
            font-weight: bold;
            padding: 12px 24px;
            font-size: 18px;
            transition: all 0.3s ease-in-out;
            box-shadow: 2px 2px 15px rgba(255, 75, 75, 0.5);
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #FF7878, #FF4B4B);
            transform: scale(1.07);
        }

        /* Sidebar Customization */
        .css-1d391kg {
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 12px;
            padding: 15px;
            color: white;
        }

        /* Response Box */
        .response-box {
            background: rgba(0, 0, 0, 0.7);
            padding: 15px;
            border-radius: 12px;
            color: white;
            font-weight: bold;
            font-size: 18px;
            text-shadow: 1px 1px 5px rgba(0, 0, 0, 0.5);
        }

        /* Adjust size of Sources heading */
        .stSubheader {
            font-size: 18px;
            font-weight: bold;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Webpage Header
st.markdown("<h1 class='title'>📰 News Research Tool 🚀</h1>", unsafe_allow_html=True)

# Sidebar - URL Input
st.sidebar.title("🔗 News Article URLs")
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("⚡ Process URLs")

# Google Search Option
use_google_search = st.sidebar.checkbox("🔎 Use Google Search")
google_query = st.sidebar.text_input("Enter Google Search Query")
search_google_clicked = st.sidebar.button("🔍 Search & Process")

# Query Input
query = st.text_input("🔍 Ask a Question:")
get_answer_clicked = st.button("💡 Get Answer")

# Initialize Qdrant Client
qdrant_client = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)

def store_documents_in_qdrant(urls, collection_name):
    """Processes and stores content from URLs into Qdrant."""
    try:
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        Qdrant.from_documents(docs, embeddings, url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name=collection_name)
        st.sidebar.success(f"✅ Documents stored in {collection_name}!")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")

def google_search(query, num_results=5):
    """Fetches top Google search results."""
    try:
        return list(search(query, num_results=num_results))
    except Exception as e:
        st.sidebar.error(f"❌ Google Search Error: {e}")
        return []

if process_url_clicked:
    urls = [url.strip() for url in urls if url.strip()]
    if urls:
        store_documents_in_qdrant(urls, COLLECTION_NAME)
    else:
        st.sidebar.error("⚠️ Please provide at least one valid URL.")

if search_google_clicked:
    if google_query:
        search_urls = google_search(google_query)
        if search_urls:
            store_documents_in_qdrant(search_urls, GOOGLE_COLLECTION_NAME)
    else:
        st.sidebar.error("⚠️ Please enter a Google search query.")

if get_answer_clicked and query:
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        collection_to_use = GOOGLE_COLLECTION_NAME if use_google_search else COLLECTION_NAME
        qdrant_cloud = Qdrant(client=qdrant_client, collection_name=collection_to_use, embeddings=embeddings)
        retriever = qdrant_cloud.as_retriever()
        llm = OpenAI(temperature=0.7, max_tokens=1000, openai_api_key=OPENAI_API_KEY)
        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)
        result = chain({"question": query}, return_only_outputs=True)
        st.markdown("### 📝 Response")
        st.write(result["answer"])
        sources = result.get("sources", "")
        if sources:
            st.markdown("### 📌 Sources")
            for source in sources.split("\n"):
                st.write(source)
    except Exception as e:
        st.error(f"❌ Error: {e}")
