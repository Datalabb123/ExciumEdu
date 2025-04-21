import streamlit as st
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import HuggingFaceHub
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# ---------------------- App Configuration ----------------------
st.set_page_config(page_title="EduMind Chat", layout="wide")

# ---------------------- Sidebar Features ----------------------
st.sidebar.title("🔧 Settings & Info")

# Theme selector
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
            body, .stApp {
                background-color: #1e1e1e;
                color: white;
            }
            .stTextInput > div > input {
                background-color: #333;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

elif theme == "Light":
    st.markdown("""
        <style>
            /* App background */
            .stApp {
                background-color: #ffffff;
                color: #000000;
            }

            /* Sidebar background & text */
            section[data-testid="stSidebar"] {
                background-color: #f7f7f7;
                color: #000000 !important;
            }

            /* Sidebar input and text color */
            section[data-testid="stSidebar"] * {
                color: #000000 !important;
            }

            /* Placeholder text */
            ::placeholder {
                color: #999 !important;
                opacity: 1 !important;
            }

            /* Input fields */
            input, textarea {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ccc;
            }

            /* Headers */
            h1, h2, h3, h4, h5, h6 {
                color: #000000 !important;
            }

            /* Markdown text */
            .markdown-text-container, .stMarkdown {
                color: #000000 !important;
            }

            /* Buttons */
            button {
                color: #000000 !important;
                background-color: #e0e0e0;
            }
        </style>
    """, unsafe_allow_html=True)

elif theme == "Dark":
    st.markdown("""
        <style>
            .stApp {
                background-color: #ffffff;
                color: #000000;
            }

            section[data-testid="stSidebar"] {
                background-color: #f7f7f7;
                color: #000000 !important;
            }

            section[data-testid="stSidebar"] * {
                color: #000000 !important;
            }

            ::placeholder {
                color: #999 !important;
                opacity: 1 !important;
            }

            input, textarea {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ccc;
            }

            h1, h2, h3, h4, h5, h6 {
                color: #000000 !important;
            }

            .markdown-text-container, .stMarkdown {
                color: #000000 !important;
            }

            /* Button styling */
            button {
                color: #000000 !important;
                background-color: #eeeeee !important;
                border: none !important;
            }

            /* Button in sidebar */
            section[data-testid="stSidebar"] button {
                color: #000000 !important;
                background-color: #eeeeee !important;
                border: none !important;
            }
        </style>
    """, unsafe_allow_html=True)

# Show history toggle
show_history = st.sidebar.checkbox("🕘 Show Full Chat History")

# Clear history button
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state["input"] = ""
    st.success("Chat history cleared!")

# Feedback form
st.sidebar.markdown("### 💬 Feedback")
feedback = st.sidebar.text_area("What do you think about EduMind?")
if st.sidebar.button("Submit Feedback"):
    st.sidebar.success("Thank you for your feedback!")
    # Optionally save feedback to a file or database

# About section
st.sidebar.markdown("### ℹ️ About EduMind")
st.sidebar.write(
    "EduMind Chatbot is a sophisticated AI-powered assistant designed to help "
    "you with your educational queries using state-of-the-art LLMs and document retrieval."
)

# Resource links
st.sidebar.markdown("### 🔗 Helpful Links")
st.sidebar.markdown("[Documentation](https://link-to-docs.com)")
st.sidebar.markdown("[FAQ](https://link-to-faq.com)")

# ---------------------- Initial Setup ----------------------
@st.cache_resource(show_spinner="Loading vector store and model...")
def setup_chatbot():
    # Load documents
    loader = DirectoryLoader(
        path=r"data",
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )

    # Vector store
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # LLM setup
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="gemma2-9b-it"
    )

    # Prompts
    system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n{context}"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, "
                   "formulate a standalone question. Do NOT answer the question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Session-based memory
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_rag_chain

# Initialize chatbot
chat_chain = setup_chatbot()

# ---------------------- Chat UI ----------------------
st.markdown("""
    <h1 style="color:#4e8cff; font-size: 36px;">EduMind Chat 🤖📚</h1>
    <p style="font-size: 18px;">An intelligent educational assistant that remembers your questions.</p>
""", unsafe_allow_html=True)

session_id = st.session_state.get("session_id", "default_session")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.markdown("### Ask your educational questions:")
user_input = st.text_input("Type your question here...", key="input")
submit = st.button("Ask")

if submit and user_input:
    with st.spinner("Thinking..."):
        response = chat_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        st.session_state.chat_history.append((user_input, response["answer"]))

# Show the last message
if st.session_state.chat_history:
    last_q, last_a = st.session_state.chat_history[-1]
    st.markdown(f"**You:** {last_q}")
    st.markdown(f"**EduMind:** {last_a}")
    st.markdown("---")

# Show full chat history if enabled
if show_history and len(st.session_state.chat_history) > 1:
    with st.expander("🕘 Full Chat History", expanded=True):
        for q, a in reversed(st.session_state.chat_history[:-1]):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**EduMind:** {a}")
            st.markdown("---")
