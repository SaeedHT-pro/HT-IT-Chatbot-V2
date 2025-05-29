import streamlit as st
from chatbot_utils import (
    get_gemini_llm,
    get_embedding_model,
    get_faq_retriever,
    get_sop_retriever,
    perform_duckduckgo_search,
    INITIAL_ANALYSIS_PROMPT_TEMPLATE,
    RESPONSE_GENERATION_PROMPT_TEMPLATE,
    clean_json_response
)
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="IT Support Chatbot", layout="wide")
st.title("IT Support Chatbot Assistant 🤖")
st.caption("Powered by Gemini, FAISS, LangChain & Streamlit")

# --- INITIALIZATION & CACHING (runs once per session or on script rerun) ---
@st.cache_resource # Cache LLM and embedding model for the session
def load_models_and_retrievers():
    print("Loading models and retrievers...")
    llm = get_gemini_llm()
    embedding_model = get_embedding_model()
    # Consider adding a button or flag to force_recreate for debugging
    faq_retriever = get_faq_retriever(embedding_model, force_recreate=False)
    sop_retriever = get_sop_retriever(embedding_model, force_recreate=False)
    print("Models and retrievers loaded.")
    return llm, faq_retriever, sop_retriever

llm, faq_retriever, sop_retriever = load_models_and_retrievers()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your IT Support Assistant. How can I help you today?"}]

# --- CHAT UI ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask your IT question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        thinking_message = "🤔 Thinking..."
        response_placeholder.markdown(thinking_message)
        
        current_response_content = "" # To build up the assistant's verbose response

        # 1. Initial Analysis with LLM to determine best source
        analysis_prompt_filled = INITIAL_ANALYSIS_PROMPT_TEMPLATE.format(user_query=user_query)
        best_source = "SOP_Store" # Default
        simplified_query = user_query # Default

        try:
            analysis_response = llm.generate_content(analysis_prompt_filled)
            analysis_result_json = clean_json_response(analysis_response.text)

            if analysis_result_json:
                best_source = analysis_result_json.get("best_source", "SOP_Store")
                simplified_query = analysis_result_json.get("simplified_query_for_search", user_query)
                current_response_content += f"*(Decided to check: {best_source.replace('_', ' ')} first)*\n\n"
            else:
                current_response_content += "*(Could not reliably determine the best source, will try internal documents.)*\n\n"
        except Exception as e:
            st.error(f"Error in initial analysis: {e}")
            current_response_content += f"*(Error in analysis, defaulting to SOPs: {e})*\n\n"
        
        response_placeholder.markdown(thinking_message + "\n" + current_response_content) # Update UI

        # 2. Retrieve Context based on `best_source`
        retrieved_context_str = ""
        source_type_used = ""
        docs_found = False

        # Attempt 1: FAQ Store (if chosen or as a quick check)
        if best_source == "FAQ_Store" and faq_retriever:
            source_type_used = "FAQs"
            try:
                relevant_docs = faq_retriever.get_relevant_documents(simplified_query)
                if relevant_docs:
                    retrieved_context_str = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
                    docs_found = True
                    current_response_content += f"Found some info in {source_type_used}.\n"
            except Exception as e:
                st.warning(f"Error retrieving from FAQs: {e}")
                current_response_content += f"Error accessing FAQs: {e}\n"
        
        response_placeholder.markdown(thinking_message + "\n" + current_response_content)

        # Attempt 2: SOP Store (if chosen, or if FAQ failed)
        if not docs_found and (best_source == "SOP_Store" or (best_source == "FAQ_Store" and not docs_found)) and sop_retriever:
            if source_type_used == "FAQs": # Meaning FAQ was tried but found nothing
                 current_response_content += f"No specific match in FAQs, now checking SOPs...\n"
            source_type_used = "SOPs"
            try:
                relevant_docs = sop_retriever.get_relevant_documents(simplified_query)
                if relevant_docs:
                    retrieved_context_str = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
                    docs_found = True
                    current_response_content += f"Found some info in {source_type_used}.\n"
            except Exception as e:
                st.warning(f"Error retrieving from SOPs: {e}")
                current_response_content += f"Error accessing SOPs: {e}\n"
        
        response_placeholder.markdown(thinking_message + "\n" + current_response_content)

        # Attempt 3: Web Search (if chosen, or if internal KBs failed)
        if not docs_found or best_source == "Web_Search":
            if docs_found: # This means best_source was Web_Search but we still checked internal
                 current_response_content += f"Also checking the web as requested or for broader context...\n"
            elif source_type_used: # Internal KBs were tried but found nothing
                 current_response_content += f"No specific match in {source_type_used}, now searching the web...\n"
            else: # No internal KBs were even available or best_source was web from start
                 current_response_content += f"No internal documents available or applicable, searching the web...\n"
            
            source_type_used = "Web Search"
            response_placeholder.markdown(thinking_message + "\n" + current_response_content + "🌐 Searching online...")
            time.sleep(0.5) # Brief pause for UX
            
            web_search_results = perform_duckduckgo_search(simplified_query)
            if web_search_results and "did not yield specific results" not in web_search_results and "failed" not in web_search_results :
                retrieved_context_str = web_search_results
                docs_found = True # Consider web search as finding docs
                current_response_content += f"Found some info via {source_type_used}.\n"
            else:
                current_response_content += f"Web search did not return specific results or failed.\n"
                if not retrieved_context_str: # Only if nothing else was found
                    retrieved_context_str = "No relevant information found in internal documents or through web search."
        
        response_placeholder.markdown(thinking_message + "\n" + current_response_content)

        # 3. Generate Final Response with LLM
        final_answer = "Sorry, I encountered an issue and couldn't generate a response."
        if not retrieved_context_str and not docs_found: # If truly nothing was found anywhere
            retrieved_context_str = "No specific information was found regarding your query in any available source."
            source_type_used = "any available source"
        
        final_prompt_filled = RESPONSE_GENERATION_PROMPT_TEMPLATE.format(
            user_query=user_query,
            source_type_used=source_type_used if source_type_used else "available knowledge",
            context=retrieved_context_str
        )
        
        current_response_content += "Generating final answer...\n"
        response_placeholder.markdown(thinking_message + "\n" + current_response_content)

        try:
            final_gemini_response = llm.generate_content(final_prompt_filled)
            final_answer = final_gemini_response.text
        except Exception as e:
            st.error(f"Error generating final response with Gemini: {e}")
            final_answer = f"Sorry, I encountered an error trying to generate a response from the LLM: {e}"

        # Display final answer (replace thinking message and accumulated steps)
        response_placeholder.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})