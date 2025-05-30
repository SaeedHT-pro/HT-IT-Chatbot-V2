import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
# from sentence_transformers import SentenceTransformer # Not directly used, LangChain wrapper is
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.tools import DuckDuckGoSearchRun # Will be replaced
from duckduckgo_search import DDGS # Import for direct use
from langchain.docstore.document import Document
import json
import logging
import requests
from bs4 import BeautifulSoup

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
RELEVANCE_CHECK_PROMPT_TEMPLATE = """
Original User Query: "{user_query}"
Simplified Search Query Used: "{simplified_query}"

Retrieved Context Snippet(s) from Internal Documents:
---
{retrieved_context}
---
Based on the "Retrieved Context Snippet(s)", is it highly likely to contain a direct and useful answer to the "Original User Query"?
The context is only relevant if it directly addresses the main subject of the user's query.
Consider if the context is specific enough to be helpful or just vaguely related.
Answer strictly with only "YES" or "NO".
"""

# --- PROMPT TEMPLATES (Ensure INITIAL_ANALYSIS_PROMPT_TEMPLATE is clear about Internal_Docs vs Web_Search) ---
INITIAL_ANALYSIS_PROMPT_TEMPLATE = """
User Query: "{user_query}"

Analyze the user query for an IT support chatbot. Your primary goal is to determine if the query can likely be answered by our internal documentation (FAQs, SOPs, manuals) or if it requires a general web search.

Consider the following:
- If the query contain random typos like "adbajfb","kdnaiof", "asdasd", or similar gibberish, it is likely not a valid query and should given response as it should given by chatbot.
- If the query asks for specific internal procedures, troubleshooting for company-supported hardware/software, "how-to" for internal tools, or seems like a common IT question that would be documented internally, choose "Internal_Docs".
- If the query is very general, about consumer products not typically managed by corporate IT, news, current events, or clearly outside the scope of internal IT documentation, choose "Web_Search".

Also, provide a concise version of the query suitable for semantic search against our internal documents.

Output your decision strictly in JSON format like this (do not add any other text before or after the JSON block):
{{
  "best_source": "Internal_Docs" | "Web_Search",
  "simplified_query_for_search": "concise version of the query"
}}

Example 1 (Internal):
User Query: "How do I reset my Windows password on my work laptop?"
JSON Output:
{{
  "best_source": "Internal_Docs",
  "simplified_query_for_search": "reset windows work laptop password"
}}

Example 2 (Internal - Specific SOP type):
User Query: "My Dell Latitude 7400 is not connecting to the VPN after the new software update."
JSON Output:
{{
  "best_source": "Internal_Docs",
  "simplified_query_for_search": "Dell Latitude 7400 VPN connection issue after software update"
}}

Example 3 (Web - Too general or consumer):
User Query: "What's the best free antivirus software for a home PC?"
JSON Output:
{{
  "best_source": "Web_Search",
  "simplified_query_for_search": "best free antivirus home pc"
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
    *   **Link Previews (IMPORTANT!):** If you include an external URL that the user would benefit from visiting (e.g., a support article, documentation), YOU MUST format it for a preview like this: `[PREVIEW](https://example.com/some-article)`. The system will then attempt to fetch the page title to make the link more informative. For example, if you want to link to `https://support.example.com/article/123`, write it as `[PREVIEW](https://support.example.com/article/123)`. For any other links where a title preview is not necessary or for internal references, use standard Markdown: `[Visible Text](https://example.com)`.
4.  **Relevance:** If the context contains information relevant to the user's query, synthesize it into a helpful answer.
5.  **Address Specificity (If Applicable):**
    *   If the user asks for a *specific section, notice, or type of document* (e.g., "Liquid crystal display (LCD) notice", "Notice for built-in rechargeable battery") and the context *does not explicitly contain that exact section title or document type*, clearly state that the specific "notice" or "section" was not found.
    *   However, *crucially*, even if the specific "notice" or "section" is not found, if the context *does* contain general information related to the query (e.g., general LCD troubleshooting, battery care tips), you MUST still provide that available relevant information to the user. Do not simply say "not found" if related information exists.
6.  **Handling Insufficient Context:** If the context is genuinely insufficient or contains no relevant information at all, then politely state that you couldn't find specific information for that query.
7.  **No Fabrication:** Do not make up information.
8.  **Source Attribution (Subtle):**
    *   Do NOT explicitly state "This information is from FAQ file X" or "According to SOP Y."
    *   **Citing Web Sources**: If `Context (Source: ...)` indicates "Web Search" and your answer directly uses information from a specific article or support page URL found within that web search context, you **MUST** cite that URL. Use the `[PREVIEW](URL_HERE)` format for this citation. For example: "According to [PREVIEW](https://source.example.com/article-name), the steps are..." or "You can find more details at [PREVIEW](https://another.example.com/support-page)."
    *   For internal documents, integrate the information seamlessly.
    *   Remember to use `[PREVIEW](URL_HERE)` for any other external helpful links as described in instruction 3, even if not explicitly citing a web search source.

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
def perform_duckduckgo_search(query_text: str, max_results: int = 3) -> str:
    """
    Performs a DuckDuckGo search and returns a formatted string of results
    including titles, URLs, and snippets.
    """
    logger.info(f"Performing DuckDuckGo search for: '{query_text}' with max_results={max_results}")
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query_text, max_results=max_results))

        if not search_results:
            logger.warning(f"DuckDuckGo search for '{query_text}' yielded no results.")
            return "Web search did not yield specific results for this query."

        formatted_results = "Web Search Results:\n\n"
        for i, result in enumerate(search_results):
            title = result.get('title', 'N/A')
            url = result.get('href', 'N/A')
            snippet = result.get('body', 'N/A')
            
            formatted_results += f"Result {i+1}:\n"
            formatted_results += f"Title: {title}\n"
            formatted_results += f"URL: {url}\n"
            formatted_results += f"Snippet: {snippet}\n---\n"
            
            # Log individual result details for clarity
            logger.debug(f"Search Result {i+1} for '{query_text}': Title='{title}', URL='{url}', Snippet='{snippet[:100]}...'")


        logger.info(f"DuckDuckGo search for '{query_text}' successful, found {len(search_results)} results.")
        return formatted_results.strip()

    except Exception as e:
        logger.error(f"DuckDuckGo search error for '{query_text}': {e}", exc_info=True)
        return "Web search failed or encountered an error."

# --- LLM UTILITY ---
def clean_json_response(llm_response_text):
    logger.debug(f"Attempting to clean JSON from LLM response: '{llm_response_text[:200]}...'")
    try:
        # Handle potential markdown code block ```json ... ```
        if llm_response_text.strip().startswith("```json"):
            llm_response_text = llm_response_text.split("```json", 1)[1].rsplit("```", 1)[0]
        elif llm_response_text.strip().startswith("```"):
             llm_response_text = llm_response_text.split("```", 1)[1].rsplit("```", 1)[0]

        json_start = llm_response_text.index("{")
        json_end = llm_response_text.rindex("}") + 1
        json_str = llm_response_text[json_start:json_end]
        parsed_json = json.loads(json_str)
        logger.debug(f"Successfully parsed JSON: {parsed_json}")
        return parsed_json
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing JSON from LLM response: {e}\nRaw response: {llm_response_text}", exc_info=True)
        return None

# --- URL TITLE FETCHER ---
def fetch_url_title(url: str) -> str:
    """
    Fetches the title of a given URL.
    Returns the title string or a default message if fetching fails or title is not found.
    """
    logger.info(f"Attempting to fetch title for URL: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        soup = BeautifulSoup(response.content, 'html.parser')

        # Try to get title tag
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            title = title_tag.string.strip()
            logger.info(f"Found title for {url}: '{title}'")
            return title

        # Fallback: Try to get the first <h1> tag
        h1_tag = soup.find('h1')
        if h1_tag and h1_tag.string:
            title = h1_tag.string.strip()
            logger.info(f"Found h1 for {url} as fallback title: '{title}'")
            return title
        
        # Fallback: Try to get OpenGraph title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
            logger.info(f"Found OpenGraph title for {url}: '{title}'")
            return title

        logger.warning(f"No <title>, <h1>, or og:title found for URL: {url}")
        return url # Return the URL itself if no title found
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}", exc_info=True)
        return url # Return URL on error
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching title for {url}: {e}", exc_info=True)
        return url # Return URL on unexpected error
