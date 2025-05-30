import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
# from sentence_transformers import SentenceTransformer # Not directly used, LangChain wrapper is
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.docstore.document import Document
import json
import logging

# --- LOGGER SETUP ---
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "chatbot.log")

def setup_logger(name='chatbot_logger', log_file=LOG_FILE, level=logging.DEBUG):
    """Function to set up a configured logger."""
    os.makedirs(LOG_DIR, exist_ok=True) # Ensure logs directory exists

    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(level)

    # Prevent adding multiple handlers if logger already exists and configured
    if not logger_instance.handlers:
        # File Handler
        fh = logging.FileHandler(log_file, mode='a') # Append mode
        fh.setLevel(level)

        # Console Handler (optional, good for development to see logs in terminal too)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO) # Log INFO and above to console

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger_instance.addHandler(fh)
        logger_instance.addHandler(ch)
    
    return logger_instance

# Get a logger instance for this module
logger = setup_logger()

# --- CONFIGURATION & INITIALIZATION ---
def load_env_vars():
    """Loads environment variables from .env file."""
    load_dotenv()
    logger.debug("Environment variables loaded.")

def get_gemini_llm():
    """Initializes and returns the Gemini LLM."""
    load_env_vars() # Ensures API key is loaded
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.critical("GOOGLE_API_KEY not found in environment variables.")
            raise ValueError("GOOGLE_API_KEY not set.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logger.info("Gemini LLM initialized successfully ('gemini-1.5-flash-latest').")
        return model
    except Exception as e:
        logger.error(f"Error configuring Gemini: {e}", exc_info=True)
        raise

def get_embedding_model(model_name='all-MiniLM-L6-v2'):
    """Initializes and returns the Sentence Transformer embedding model via LangChain wrapper."""
    logger.info(f"Initializing embedding model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)

# --- PROMPT TEMPLATES ---
INITIAL_ANALYSIS_PROMPT_TEMPLATE = """
User Query: "{user_query}"

Analyze the user query for an IT support chatbot. Determine the most appropriate primary knowledge source and extract key entities.
Possible knowledge sources are:
- "Internal_Docs": For questions related to IT procedures, troubleshooting specific hardware/software, how-to guides, company policies, FAQs, or any information typically found in internal documentation, user manuals, SOPs, or knowledge bases.
- "Web_Search": For very general queries, current events, or if the query is clearly outside the scope of typical IT support or internal documentation.

Also, provide a concise version of the query suitable for semantic search. Focus on key entities and concepts that would help retrieve relevant document sections, even if the exact phrasing isn't present.

Output your decision strictly in JSON format like this (do not add any other text before or after the JSON block):
{{
  "best_source": "Internal_Docs" | "Web_Search",
  "simplified_query_for_search": "concise version of the query"
}}

Example 1:
User Query: "How do I reset my Windows password?"
JSON Output:
{{
  "best_source": "Internal_Docs",
  "simplified_query_for_search": "reset windows password"
}}

Example 2:
User Query: "My Dell laptop screen is flickering and showing strange artifacts after the recent update. I've already tried restarting."
JSON Output:
{{
  "best_source": "Internal_Docs",
  "simplified_query_for_search": "Dell laptop screen flickering after update"
}}

Example 3:
User Query: "How to import sharepoint to new account?"
JSON Output:
{{
  "best_source": "Internal_Docs",
  "simplified_query_for_search": "import sharepoint to new account"
}}

Example 4:
User Query: "What's the weather like today?"
JSON Output:
{{
  "best_source": "Web_Search",
  "simplified_query_for_search": "current weather"
}}

Example 5:
User Query: "Notice for built-in rechargeable battery"
JSON Output:
{{
  "best_source": "Internal_Docs",
  "simplified_query_for_search": "rechargeable battery information notice"
}}

Now, analyze the User Query at the top of this prompt.
"""

RESPONSE_GENERATION_PROMPT_TEMPLATE = """
You are a helpful IT support assistant. Your goal is to provide clear, concise, and actionable answers.
Answer the user's query: "{user_query}"
Based *only* on the following provided context.

**Instructions for Answering:**
1.  **Natural Language:** Formulate your answer in a natural, conversational way. Avoid overly technical jargon unless the query implies a technical user.
2.  **Conciseness:** Get straight to the point. Provide the information the user needs without unnecessary fluff.
3.  **Markdown Formatting:** Use Markdown formatting where appropriate to enhance readability and clarity. This includes:
    *   **Lists:** For sequences of steps or instructions, use bullet points (e.g., `- Item` or `* Item`) or numbered lists (e.g., `1. Item`).
    *   **Emphasis:** Use bold (`**text**`) or italics (`*text*`) for emphasis where it aids understanding.
    *   **Code Blocks:** If providing code snippets or commands, use Markdown code blocks (e.g., ```python\ncode\n``` or `inline code`).
    *   **Headings:** For longer, structured answers, consider using Markdown headings (`## Heading`) if it improves organization, but use sparingly.
4.  **Relevance:** If the context contains information relevant to the user's query, synthesize it into a helpful answer.
5.  **Address Specificity (If Applicable):**
    *   If the user asks for a *specific section, notice, or type of document* (e.g., "Liquid crystal display (LCD) notice", "Notice for built-in rechargeable battery") and the context *does not explicitly contain that exact section title or document type*, clearly state that the specific "notice" or "section" was not found.
    *   However, *crucially*, even if the specific "notice" or "section" is not found, if the context *does* contain general information related to the query (e.g., general LCD troubleshooting, battery care tips), you MUST still provide that available relevant information to the user. Do not simply say "not found" if related information exists.
6.  **Handling Insufficient Context:** If the context is genuinely insufficient or contains no relevant information at all, then politely state that you couldn't find specific information for that query.
7.  **No Fabrication:** Do not make up information.
8.  **Source Attribution (Subtle):**
    *   Do NOT explicitly state "This information is from FAQ file X" or "According to SOP Y."
    *   If the context is from a web search, you can subtly indicate this if it adds value (e.g., "According to some online sources..." or "A web search suggests...").
    *   For internal documents, integrate the information seamlessly. If different pieces of context point to slightly different aspects of a solution (e.g., one for general backup, another for Outlook-specific backup), try to weave them together coherently. If a reference link is available in the metadata and highly relevant, you *may* include it if it seems genuinely helpful for the user to explore further, but don't do this routinely.

Context (Source: {source_type_used}):
"{context}"
---
Answer:
"""


# --- DATA LOADING & PROCESSING ---
def load_faqs(file_path="data/faqs/faq_data.xlsx"):
    """Loads FAQs from an Excel file and adds 'doc_type' metadata."""
    docs = []
    logger.info(f"Attempting to load FAQs from: {file_path}")
    try:
        df = pd.read_excel(file_path)
        if 'Question' not in df.columns or 'Answer' not in df.columns:
            logger.error("FAQ Excel must contain 'Question' and 'Answer' columns (case-sensitive).")
            return []

        for _, row in df.iterrows():
            content = f"Question: {row['Question']}\nAnswer: {row['Answer']}"
            ref_link = row.get('ref link')
            # Use the filename as the primary source identifier for consistency
            metadata = {"doc_type": "faq", "source": os.path.basename(file_path)}
            if pd.notna(ref_link) and ref_link:
                content += f"\nReference Link: {str(ref_link)}"
                metadata["reference_link"] = str(ref_link)
            docs.append(Document(page_content=content, metadata=metadata))
        logger.info(f"Successfully loaded {len(docs)} FAQs with 'doc_type' metadata.")
    except FileNotFoundError:
        logger.error(f"FAQ file not found at {file_path}. Please create it.")
    except Exception as e:
        logger.error(f"Error loading FAQs: {e}", exc_info=True)
    return docs

def load_sops(sops_dir="data/sops/"):
    """Loads SOP documents, adds 'doc_type' metadata, and splits them into chunks."""
    raw_docs = []
    logger.info(f"Attempting to load SOPs from directory: {sops_dir}")
    if not os.path.exists(sops_dir) or not os.listdir(sops_dir):
        logger.warning(f"SOPs directory '{sops_dir}' is empty or does not exist. No SOPs loaded.")
        return []

    for filename in os.listdir(sops_dir):
        file_path = os.path.join(sops_dir, filename)
        loader = None
        logger.debug(f"Processing SOP file: {filename}")
        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.lower().endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)

            if loader:
                loaded_docs_for_file = loader.load()
                for doc in loaded_docs_for_file:
                    doc.metadata["doc_type"] = "sop" # Add doc_type
                    doc.metadata["source"] = filename # Keep original filename as source
                raw_docs.extend(loaded_docs_for_file)
                logger.debug(f"Loaded {len(loaded_docs_for_file)} pages/parts from {filename} with 'doc_type' metadata.")
        except Exception as e:
            logger.error(f"Error loading SOP file {filename}: {e}", exc_info=True)

    if not raw_docs:
        logger.warning("No SOP documents were successfully loaded or processed.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(raw_docs)
    logger.info(f"Loaded and split {len(raw_docs)} raw SOP document pages/parts into {len(split_docs)} chunks.")
    return split_docs

def load_all_documents():
    """Loads all documents (FAQs and SOPs) and returns a combined list."""
    logger.info("Loading all documents (FAQs and SOPs).")
    all_docs = []
    
    faq_docs = load_faqs()
    if faq_docs:
        all_docs.extend(faq_docs)
        logger.info(f"Added {len(faq_docs)} FAQs to the combined list.")
        
    sop_docs = load_sops()
    if sop_docs:
        all_docs.extend(sop_docs)
        logger.info(f"Added {len(sop_docs)} SOPs to the combined list.")
        
    if not all_docs:
        logger.warning("No documents (FAQs or SOPs) were loaded.")
    else:
        logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs

# --- VECTOR STORE & RETRIEVAL ---
def create_or_load_faiss_index(index_name, docs_loader_func, embedding_model, force_recreate=False):
    index_base_path = "data/vector_store"
    os.makedirs(index_base_path, exist_ok=True)
    index_path = os.path.join(index_base_path, index_name)

    if os.path.exists(index_path) and not force_recreate:
        logger.info(f"Loading existing FAISS index: {index_name} from {index_path}")
        try:
            return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.warning(f"Error loading FAISS index {index_name}: {e}. Will try to recreate.", exc_info=True)
            # Fall through to recreate

    logger.info(f"Attempting to create FAISS index: {index_name}")
    docs = docs_loader_func()
    if not docs:
        logger.warning(f"No documents found by docs_loader_func for {index_name}. Index not created.")
        return None
    try:
        logger.debug(f"Creating FAISS index '{index_name}' from {len(docs)} documents...")
        vector_store = FAISS.from_documents(docs, embedding_model)
        vector_store.save_local(index_path)
        logger.info(f"FAISS index {index_name} created and saved to {index_path}.")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating FAISS index {index_name}: {e}", exc_info=True)
        return None

# def get_faq_retriever(embedding_model, force_recreate=False, k_results=2):
#     logger.debug(f"Getting FAQ retriever (force_recreate={force_recreate}, k={k_results})")
#     vector_store = create_or_load_faiss_index("faiss_faq_index", load_faqs, embedding_model, force_recreate)
#     if vector_store:
#         return vector_store.as_retriever(search_kwargs={"k": k_results})
#     logger.warning("FAQ vector store not available, retriever is None.")
#     return None

# def get_sop_retriever(embedding_model, force_recreate=False, k_results=5):
#     logger.debug(f"Getting SOP retriever (force_recreate={force_recreate}, k={k_results})")
#     vector_store = create_or_load_faiss_index("faiss_sop_index", load_sops, embedding_model, force_recreate)
#     if vector_store:
#         return vector_store.as_retriever(search_kwargs={"k": k_results})
#     logger.warning("SOP vector store not available, retriever is None.")
#     return None

def get_combined_retriever(embedding_model, force_recreate=False, k_results=5):
    """Gets a retriever for the combined FAQ and SOP documents."""
    index_name = "faiss_combined_index"
    logger.debug(f"Getting combined retriever for '{index_name}' (force_recreate={force_recreate}, k={k_results})")
    vector_store = create_or_load_faiss_index(index_name, load_all_documents, embedding_model, force_recreate)
    if vector_store:
        return vector_store.as_retriever(search_kwargs={"k": k_results})
    logger.warning(f"{index_name} vector store not available, retriever is None.")
    return None

# --- SEARCH TOOL ---
def perform_duckduckgo_search(query_text):
    logger.info(f"Performing DuckDuckGo search for: '{query_text}'")
    search = DuckDuckGoSearchRun()
    try:
        results = search.run(query_text)
        if not results or "No good DuckDuckGo Search Result was found" in results:
            logger.warning(f"DuckDuckGo search for '{query_text}' yielded no specific results.")
            return "Web search did not yield specific results for this query."
        logger.info(f"DuckDuckGo search for '{query_text}' successful.")
        return results
    except Exception as e:
        logger.error(f"DuckDuckGo search error for '{query_text}': {e}", exc_info=True)
        return "Web search failed or encountered an error."

# --- LLM UTILITY ---
def clean_json_response(llm_response_text):
    logger.debug(f"Attempting to clean JSON from LLM response: '{llm_response_text[:200]}...'")
    try:
        json_start = llm_response_text.index("{")
        json_end = llm_response_text.rindex("}") + 1
        json_str = llm_response_text[json_start:json_end]
        parsed_json = json.loads(json_str)
        logger.debug(f"Successfully parsed JSON: {parsed_json}")
        return parsed_json
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing JSON from LLM response: {e}\nRaw response: {llm_response_text}", exc_info=True)
        return None
