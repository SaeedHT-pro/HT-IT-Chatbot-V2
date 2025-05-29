import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer # We'll use LangChain's wrapper
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.docstore.document import Document
import json

# --- CONFIGURATION & INITIALIZATION ---
def load_env_vars():
    """Loads environment variables from .env file."""
    load_dotenv()

def get_gemini_llm():
    """Initializes and returns the Gemini LLM."""
    load_env_vars()
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash-latest') # Using a specific fast model
        return model
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        raise

def get_embedding_model(model_name='all-MiniLM-L6-v2'):
    """Initializes and returns the Sentence Transformer embedding model via LangChain wrapper."""
    return HuggingFaceEmbeddings(model_name=model_name)

# --- PROMPT TEMPLATES ---
INITIAL_ANALYSIS_PROMPT_TEMPLATE = """
User Query: "{user_query}"

Analyze the user query for an IT support chatbot. Determine the most appropriate primary knowledge source and extract key entities.
Possible knowledge sources are: "FAQ_Store" (for common, direct questions), "SOP_Store" (for detailed procedures, troubleshooting specific product issues), or "Web_Search" (for very general queries or if other sources fail).

Also, provide a concise version of the query suitable for semantic search.

Output your decision strictly in JSON format like this (do not add any other text before or after the JSON block):
{{
  "best_source": "FAQ_Store" | "SOP_Store" | "Web_Search",
  "simplified_query_for_search": "concise version of the query"
}}

Example 1:
User Query: "How do I reset my Windows password?"
JSON Output:
{{
  "best_source": "FAQ_Store",
  "simplified_query_for_search": "reset windows password"
}}

Example 2:
User Query: "My Dell laptop screen is flickering and showing strange artifacts after the recent update. I've already tried restarting."
JSON Output:
{{
  "best_source": "SOP_Store",
  "simplified_query_for_search": "Dell laptop screen flickering after update"
}}

Example 3:
User Query: "What's the weather like today?"
JSON Output:
{{
  "best_source": "Web_Search",
  "simplified_query_for_search": "current weather"
}}

Now, analyze the User Query at the top of this prompt.
"""

RESPONSE_GENERATION_PROMPT_TEMPLATE = """
You are a helpful IT support assistant.
Answer the user's query: "{user_query}"
Based *only* on the following provided context. If the context is insufficient or doesn't directly answer, politely state that you couldn't find specific information for that query.
Do not make up information. If the context is from a web search, you can optionally mention that.

Context from {source_type_used}:
"{context}"
---
Answer:
"""

# --- DATA LOADING & PROCESSING ---
def load_faqs(file_path="data/faqs/faq_data.xlsx"):
    """Loads FAQs from an Excel file."""
    docs = []
    try:
        df = pd.read_excel(file_path)
        if 'question' not in df.columns or 'answer' not in df.columns:
            print("FAQ Excel must contain 'question' and 'answer' columns.")
            return []

        for _, row in df.iterrows():
            content = f"Question: {row['question']}\nAnswer: {row['answer']}"
            ref_link = row.get('ref link') # Optional column
            metadata = {"source": "faq"}
            if pd.notna(ref_link) and ref_link:
                content += f"\nReference Link: {ref_link}"
                metadata["reference_link"] = str(ref_link)
            docs.append(Document(page_content=content, metadata=metadata))
        print(f"Loaded {len(docs)} FAQs.")
    except FileNotFoundError:
        print(f"FAQ file not found at {file_path}. Please create it.")
    except Exception as e:
        print(f"Error loading FAQs: {e}")
    return docs

def load_sops(sops_dir="data/sops/"):
    """Loads SOP documents from a directory and splits them into chunks."""
    raw_docs = []
    if not os.path.exists(sops_dir) or not os.listdir(sops_dir):
        print(f"SOPs directory '{sops_dir}' is empty or does not exist. No SOPs loaded.")
        return []
        
    for filename in os.listdir(sops_dir):
        file_path = os.path.join(sops_dir, filename)
        loader = None
        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.lower().endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
            # Add more loaders for other file types if needed

            if loader:
                loaded_docs_for_file = loader.load()
                for doc in loaded_docs_for_file: # Add source filename to metadata
                    doc.metadata["source"] = filename
                raw_docs.extend(loaded_docs_for_file)
        except Exception as e:
            print(f"Error loading SOP file {filename}: {e}")

    if not raw_docs:
        print("No SOP documents were successfully loaded.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(raw_docs)
    print(f"Loaded and split {len(raw_docs)} SOP documents into {len(split_docs)} chunks.")
    return split_docs

# --- VECTOR STORE & RETRIEVAL ---
def create_or_load_faiss_index(index_name, docs_loader_func, embedding_model, force_recreate=False):
    """Creates or loads a FAISS index. `docs_loader_func` is a function that returns a list of LangChain Documents."""
    index_base_path = "data/vector_store"
    os.makedirs(index_base_path, exist_ok=True) # Ensure base directory exists
    index_path = os.path.join(index_base_path, index_name)

    if os.path.exists(index_path) and not force_recreate:
        print(f"Loading existing FAISS index: {index_name} from {index_path}")
        try:
            # FAISS.load_local requires allow_dangerous_deserialization for custom embeddings like HuggingFaceEmbeddings
            return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading FAISS index {index_name}: {e}. Will try to recreate.")
            # Fall through to recreate if loading fails

    print(f"Creating FAISS index: {index_name}")
    docs = docs_loader_func() # Call the provided function (e.g., load_faqs or load_sops)
    if not docs:
        print(f"No documents found for {index_name}. Index not created.")
        return None
    try:
        vector_store = FAISS.from_documents(docs, embedding_model)
        vector_store.save_local(index_path)
        print(f"FAISS index {index_name} created and saved to {index_path}.")
        return vector_store
    except Exception as e:
        print(f"Error creating FAISS index {index_name}: {e}")
        return None

def get_faq_retriever(embedding_model, force_recreate=False, k_results=2):
    """Gets a FAISS retriever for FAQs."""
    vector_store = create_or_load_faiss_index("faiss_faq_index", load_faqs, embedding_model, force_recreate)
    if vector_store:
        return vector_store.as_retriever(search_kwargs={"k": k_results})
    return None

def get_sop_retriever(embedding_model, force_recreate=False, k_results=3):
    """Gets a FAISS retriever for SOPs."""
    vector_store = create_or_load_faiss_index("faiss_sop_index", load_sops, embedding_model, force_recreate)
    if vector_store:
        return vector_store.as_retriever(search_kwargs={"k": k_results})
    return None

# --- SEARCH TOOL ---
def perform_duckduckgo_search(query_text):
    """Performs a DuckDuckGo search and returns a summary of results."""
    search = DuckDuckGoSearchRun()
    try:
        results = search.run(query_text)
        # Basic check for empty or "no good result" type responses
        if not results or "No good DuckDuckGo Search Result was found" in results:
            return "Web search did not yield specific results for this query."
        return results
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return "Web search failed or encountered an error."

# --- LLM UTILITY ---
def clean_json_response(llm_response_text):
    """Attempts to extract a valid JSON string from the LLM's potentially messy output."""
    # Find the start and end of the JSON block
    try:
        json_start = llm_response_text.index("{")
        json_end = llm_response_text.rindex("}") + 1
        json_str = llm_response_text[json_start:json_end]
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error parsing JSON from LLM response: {e}\nRaw response: {llm_response_text}")
        return None # Or a default dict