import os
import streamlit as st
import html2text
from dotenv import load_dotenv

# LangChain v2 imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores import Qdrant

from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ---------------------------------------------------
# WEBSITE LOADER
# ---------------------------------------------------
def load_website(url: str, max_depth: int = 2, max_pages: int = 50):
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=max_depth,
        extractor=lambda html: html2text.html2text(html)
    )
    docs = loader.load()
    return docs[:max_pages]

# ---------------------------------------------------
# BUILD RAG (Crawl → Chunk → Embed → Qdrant)
# ---------------------------------------------------
def build_rag_from_url(url: str):

    st.info("🔍 Crawling website...")
    documents = load_website(url)

    st.info("✂️ Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    st.info("🧠 Generating embeddings...")
    embed_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    st.info("📦 Storing in Qdrant...")
    client = QdrantClient(url=QDRANT_URL, api_key=None)

    collection_name = (
        url.replace("https://", "")
        .replace("http://", "")
        .replace("/", "_")
    )

    vectorstore = Qdrant.from_documents(
    chunks,
    embedding=embed_model,
    url=QDRANT_URL,
    api_key=None,
    collection_name=collection_name,
    force_recreate=True
)


    return vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------------------------------------------------
# MANUAL QA CHAIN (No deprecated modules)
# ---------------------------------------------------
def build_qa_chain(retriever, model_name="gpt-4o-mini", temperature=0):

    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=OPENAI_API_KEY,
        temperature=temperature,
    )

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Use the following context to answer.\n\n"
            "CONTEXT:\n{context}\n\n"
            "QUESTION:\n{question}\n\n"
            "Answer in clear English.\n"
        ),
        input_variables=["context", "question"]
    )

    def chain_fn(question):
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        final_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(final_prompt)
        return response.content

    return chain_fn

# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------
st.set_page_config(page_title="Website RAG Chatbot", layout="wide")
st.title("🔎 RAG Project")

with st.sidebar:
    st.header("Settings")
    url = st.text_input("Website URL", placeholder="https://example.com")
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
    start_btn = st.button("Load Website")

# State
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Build RAG
if start_btn:
    if not url:
        st.error("Enter a URL first.")
    else:
        with st.spinner("Building RAG pipeline..."):
            try:
                retriever = build_rag_from_url(url)
                chain = build_qa_chain(retriever, model_name, temperature)

                st.session_state.retriever = retriever
                st.session_state.qa_chain = chain

                st.success("Website indexed successfully!")
            except Exception as e:
                st.exception(e)

# Chat
query = st.chat_input("Ask something about the site...")

if query:
    if st.session_state.qa_chain is None:
        st.error("Load a website first.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.qa_chain(query)
            except Exception as e:
                st.exception(e)
                answer = ""

        st.chat_message("user").markdown(query)
        st.chat_message("assistant").markdown(answer)
