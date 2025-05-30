import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

st.set_page_config(page_title="Floating RAG Chatbot", layout="wide", initial_sidebar_state="collapsed")

# Inject CSS for floating chat widget fixed bottom-right and toggle button
st.markdown(
    """
    <style>
    /* Hide Streamlit default menu and footer */
    #MainMenu, footer, header {visibility: hidden;}
    /* Floating chatbot container */
    #chatbot-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 350px;
        max-height: 480px;
        background: white;
        border-radius: 10px 10px 0 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        display: flex;
        flex-direction: column;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 14px;
        user-select: none;
        z-index: 9999;
    }
    #chatbot-header {
        background: linear-gradient(90deg, #0d8bff, #005bcc);
        color: white;
        padding: 14px 16px;
        font-weight: 700;
        font-size: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        cursor: default;
    }
    #chatbot-header span {
        user-select: text;
    }
    #chatbot-close-btn {
        background: transparent;
        border: none;
        color: white;
        font-size: 20px;
        line-height: 20px;
        cursor: pointer;
        font-weight: 900;
    }
    #chatbot-content {
        flex: 1;
        padding: 12px;
        overflow-y: auto;
        background: #f9f9f9;
        display: flex;
        flex-direction: column;
        scrollbar-width: thin;
        scrollbar-color: #0d8bff #e0e0e0;
    }
    #chatbot-content::-webkit-scrollbar {
        width: 8px;
    }
    #chatbot-content::-webkit-scrollbar-track {
        background: #e0e0e0;
        border-radius: 10px;
    }
    #chatbot-content::-webkit-scrollbar-thumb {
        background-color: #0d8bff;
        border-radius: 10px;
    }
    .message {
        max-width: 75%;
        margin-bottom: 10px;
        padding: 10px 14px;
        border-radius: 20px;
        font-size: 14px;
        line-height: 1.3;
        word-wrap: break-word;
    }
    .message.user {
        background: #0d8bff;
        color: white;
        align-self: flex-end;
        border-radius: 20px 20px 0 20px;
    }
    .message.bot {
        background: #e3e6e8;
        color: #202124;
        align-self: flex-start;
        border-radius: 20px 20px 20px 0;
    }
    #chatbot-input-area {
        display: flex;
        border-top: 1px solid #ddd;
        padding: 8px 12px;
        background: white;
    }
    #chat-input {
        flex: 1;
        font-size: 14px;
        border: 1px solid #ccc;
        border-radius: 20px;
        padding: 8px 12px;
        outline: none;
        transition: border-color 0.2s;
    }
    #chat-input:focus {
        border-color: #0d8bff;
    }
    #send-btn {
        background: #0d8bff;
        border: none;
        color: white;
        font-weight: 600;
        padding: 8px 16px;
        border-radius: 20px;
        margin-left: 8px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    #send-btn:hover {
        background: #005bcc;
    }
    #chatbot-toggle-btn {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: #0d8bff;
        color: white;
        border: none;
        border-radius: 50%;
        width: 56px;
        height: 56px;
        cursor: pointer;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
        font-size: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    }
    #chatbot-toggle-btn:hover {
        background: #005bcc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Toggle chatbot open/close state stored in session_state
if "chatbot_open" not in st.session_state:
    st.session_state.chatbot_open = False

def open_chat():
    st.session_state.chatbot_open = True

def close_chat():
    st.session_state.chatbot_open = False

def main():
    st.title(" ")  # Blank title to reduce main area

    # Load embeddings, vector store, and LLM with caching
    @st.cache_resource(show_spinner=True)
    def load_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Note the underscore here to avoid hashing error in Streamlit cache
    @st.cache_resource(show_spinner=True)
    def load_vectorstore(_embeddings):
        from langchain.docstore.document import Document

        kb_docs = [
            Document(page_content="Our support hours are Monday to Friday, 9am to 5pm."),
            Document(page_content="You can reach us by emailing support@example.com."),
            Document(page_content="Our product pricing depends on the plan you select."),
            Document(page_content="We offer a 14-day free trial with all subscriptions."),
            Document(page_content="To reset your password, click 'Forgot Password' on the login screen."),
        ]
        return FAISS.from_documents(kb_docs, _embeddings)

    @st.cache_resource(show_spinner=True)
    def get_llm():
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("Google API key not found. Set the GOOGLE_API_KEY environment variable.")
            st.stop()
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key, temperature=0.3, convert_system_message_to_human=True)

    embeddings = load_embeddings()
    vector_store = load_vectorstore(embeddings)
    llm = get_llm()

    if "history" not in st.session_state:
        st.session_state.history = []

    # Render toggle button if chatbot closed
    if not st.session_state.chatbot_open:
        if st.button("💬", key="open_chat_btn"):
            open_chat()
        return

    # Chat header and close button
    col1, col2 = st.columns([8,1])
    with col1:
        st.markdown("<div style='padding:6px; font-weight: 700; font-size: 16px; color: white; background: linear-gradient(90deg,#0d8bff,#005bcc); border-radius: 10px 10px 0 0;'>Support Chat</div>", unsafe_allow_html=True)
    with col2:
        if st.button("✕", key="close_chat_btn", help="Close Chat"):
            close_chat()
            st.experimental_rerun()

    # Chat content area
    chat_container = st.empty()

    # Display previous chat messages
    with chat_container.container():
        for msg in st.session_state.history:
            if msg["role"]=="user":
                st.markdown(f"<div style='background:#0d8bff; color:white; border-radius: 20px 20px 0 20px; padding:10px; max-width:75%; margin-left:auto; margin-bottom:10px; word-wrap:break-word;'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background:#e3e6e8; color:#202124; border-radius: 20px 20px 20px 0; padding:10px; max-width:75%; margin-bottom:10px; word-wrap:break-word;'>{msg['content']}</div>", unsafe_allow_html=True)

    # Input and send button in one row
    input_col, btn_col = st.columns([8,1])
    with input_col:
        user_input = st.text_input(" ", key="chat_input", placeholder="Type your message here...", label_visibility="collapsed")
    with btn_col:
        send_clicked = st.button("Send", key="send_btn")

    if send_clicked and user_input:
        # Add user message
        st.session_state.history.append({"role":"user", "content":user_input})

        # Retrieve context from KB
        relevant_docs = vector_store.similarity_search(user_input, k=3)
        context_text = "\n".join([doc.page_content for doc in relevant_docs])

        # Build messages for LLM
        system_prompt = (
            "You are a helpful customer support assistant. Use the following context to answer user questions. "
            "If the answer is not found in the context, politely say you don't know."
        )
        messages = [SystemMessage(content=system_prompt)]

        for entry in st.session_state.history[:-1]:
            if entry["role"] == "user":
                messages.append(HumanMessage(content=entry["content"]))
            else:
                messages.append(AIMessage(content=entry["content"]))

        user_text = f"Context:\n{context_text}\n\nQuestion:\n{user_input}"
        messages.append(HumanMessage(content=user_text))

        response = llm(messages)
        answer = response.content.strip()

        # Add bot response
        st.session_state.history.append({"role":"bot", "content":answer})

        # Clear input
        st.session_state.chat_input = ""
        # Rerun to show updated chat
        st.experimental_rerun()

if __name__ == "__main__":
    main()
