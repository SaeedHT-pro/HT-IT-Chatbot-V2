import streamlit as st
from chatbot_utils import (
    get_gemini_llm,
    get_embedding_model,
    get_combined_retriever, # Corrected import
    perform_duckduckgo_search,
    INITIAL_ANALYSIS_PROMPT_TEMPLATE,
    RESPONSE_GENERATION_PROMPT_TEMPLATE,
    clean_json_response,
    logger # Import the logger instance
)
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="IT Support Chatbot", layout="wide")
st.title("IT Support Chatbot Assistant 🤖")
st.caption("Powered by Gemini, FAISS, LangChain & Streamlit")

# --- INITIALIZATION & CACHING ---
@st.cache_resource
def load_models_and_retrievers():
    logger.info("Attempting to load models and retrievers for Streamlit session...")
    llm = get_gemini_llm()
    embedding_model = get_embedding_model()
    
    # You can add a checkbox in Streamlit UI to control this for easier debugging
    # force_recreate_indexes = st.sidebar.checkbox("Recreate Vector Indexes", False) 
    force_recreate_indexes = False # Default to False

    combined_retriever = get_combined_retriever(embedding_model, force_recreate=force_recreate_indexes) # Corrected
    logger.info("Models and retrievers loading complete for Streamlit session.")
    return llm, combined_retriever # Corrected

llm, combined_retriever = load_models_and_retrievers() # Corrected

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
        thinking_message_ui = "🤔 Thinking..." # For UI display
        response_placeholder.markdown(thinking_message_ui)
        
        # This string will accumulate the "reasoning" steps shown to the user in the UI.
        # Detailed debug logs go to the log file.
        assistant_reasoning_steps_for_ui = "" 

        logger.info(f"--- New User Query Processing Started ---")
        logger.info(f"User Query: {user_query}")

        # 1. Initial Analysis with LLM
        analysis_prompt_filled = INITIAL_ANALYSIS_PROMPT_TEMPLATE.format(user_query=user_query)
        best_source = "Internal_Docs" # Default changed
        simplified_query = user_query # Default

        try:
            logger.debug("Sending query to LLM for initial analysis...")
            analysis_response = llm.generate_content(analysis_prompt_filled)
            logger.debug(f"Raw LLM Analysis Response Text: {analysis_response.text}")
            analysis_result_json = clean_json_response(analysis_response.text)
            logger.info(f"Parsed LLM Analysis JSON: {analysis_result_json}")

            if analysis_result_json:
                best_source = analysis_result_json.get("best_source", "Internal_Docs") # Default changed
                simplified_query = analysis_result_json.get("simplified_query_for_search", user_query)
                ui_source_display = "Internal Documents" if best_source == "Internal_Docs" else best_source.replace('_', ' ')
                assistant_reasoning_steps_for_ui += f"*(Decided to check: {ui_source_display} first using query: '{simplified_query}')*\n\n"
            else:
                assistant_reasoning_steps_for_ui += "*(Could not reliably parse LLM analysis. Defaulting to Internal Documents with original query.)*\n\n" # Changed
                logger.warning("LLM analysis JSON parsing failed.")
        except Exception as e:
            st.error(f"Error during LLM initial analysis: {e}") # Show error in UI
            logger.error(f"Error in LLM initial analysis: {e}", exc_info=True)
            assistant_reasoning_steps_for_ui += f"*(Error during analysis phase: {e}. Defaulting to Internal Documents.)*\n\n" # Changed
        
        logger.info(f"After Analysis - best_source: {best_source}, simplified_query: '{simplified_query}'")
        response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)

        # 2. Retrieve Context
        retrieved_context_str = ""
        source_type_used_for_response_prompt = "" # For the final LLM prompt
        docs_found = False

        # Attempt 1: Internal Documents (Combined FAQs and SOPs)
        if best_source == "Internal_Docs":
            source_type_used_for_response_prompt = "Internal Documents"
            assistant_reasoning_steps_for_ui += f"Attempting to retrieve from {source_type_used_for_response_prompt}...\n"
            response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)
            if combined_retriever:
                try:
                    logger.debug(f"Querying Combined retriever with: '{simplified_query}'")
                    relevant_docs = combined_retriever.get_relevant_documents(simplified_query)
                    logger.info(f"Combined Retriever found {len(relevant_docs)} docs for '{simplified_query}'.")
                    if relevant_docs:
                        # Include metadata like source and doc_type in the context string for better LLM understanding
                        context_parts = []
                        for doc in relevant_docs:
                            doc_info = f"Source: {doc.metadata.get('source', 'N/A')}, Type: {doc.metadata.get('doc_type', 'N/A')}"
                            context_parts.append(f"{doc_info}\nContent: {doc.page_content}")
                        retrieved_context_str = "\n\n---\n\n".join(context_parts)
                        docs_found = True
                        assistant_reasoning_steps_for_ui += f"Found relevant info in {source_type_used_for_response_prompt}.\n"
                    else:
                        assistant_reasoning_steps_for_ui += f"No specific match found in {source_type_used_for_response_prompt}.\n"
                except Exception as e:
                    st.warning(f"Error retrieving from {source_type_used_for_response_prompt}: {e}") # UI warning
                    logger.warning(f"Error retrieving from {source_type_used_for_response_prompt}: {e}", exc_info=True)
                    assistant_reasoning_steps_for_ui += f"Error accessing {source_type_used_for_response_prompt}: {e}\n"
            else:
                assistant_reasoning_steps_for_ui += f"{source_type_used_for_response_prompt} retriever not available.\n"
                logger.warning("Combined retriever was None during search attempt.")
        
        response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)

        # Attempt 2: Web Search (if chosen, or if internal KBs failed)
        # Logic for web search remains largely the same, but the conditions to trigger it are simplified.
        if best_source == "Web_Search" or (best_source == "Internal_Docs" and not docs_found):
            if best_source == "Web_Search" and docs_found: # This case should ideally not happen if logic is correct
                 assistant_reasoning_steps_for_ui += f"Initial plan was web search, but found internal docs. Performing web search anyway as per plan...\n"
            elif not docs_found and source_type_used_for_response_prompt == "Internal Documents":
                 assistant_reasoning_steps_for_ui += f"No specific match in Internal Documents, now searching the web...\n"
            elif best_source == "Web_Search" and not docs_found : # Explicitly chosen web search and nothing found yet
                 assistant_reasoning_steps_for_ui += f"Searching the web as per initial plan...\n"
            
            source_type_used_for_response_prompt = "Web Search" # Set this regardless of previous state if we reach here
            assistant_reasoning_steps_for_ui += f"🌐 Searching online for '{simplified_query}'...\n"
            response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)
            time.sleep(0.1) 
            
            web_search_results = perform_duckduckgo_search(simplified_query)
            logger.info(f"Web Search for '{simplified_query}' returned: {web_search_results[:300]}...")
            if web_search_results and "did not yield specific results" not in web_search_results and "failed" not in web_search_results:
                retrieved_context_str = web_search_results
                docs_found = True 
                assistant_reasoning_steps_for_ui += f"Found info via {source_type_used_for_response_prompt}.\n"
            else:
                assistant_reasoning_steps_for_ui += f"Web search did not return specific usable results or failed.\n"
                if not retrieved_context_str: 
                    retrieved_context_str = "No relevant information found in internal documents or through web search."
        
        response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)
        logger.debug(f"Final retrieved_context_str before LLM (first 300 chars): {retrieved_context_str[:300]}...")
        logger.info(f"Final source_type_used_for_response_prompt: {source_type_used_for_response_prompt}")
        logger.info(f"Final docs_found status: {docs_found}")

        # 3. Generate Final Response with LLM
        final_answer_for_ui = "Sorry, I encountered an issue and couldn't generate a response."
        if not retrieved_context_str and not docs_found: 
            retrieved_context_str = "No specific information was found regarding your query in any available source."
            source_type_used_for_response_prompt = "any available source" # Update for prompt
        
        final_prompt_filled = RESPONSE_GENERATION_PROMPT_TEMPLATE.format(
            user_query=user_query,
            source_type_used=source_type_used_for_response_prompt if source_type_used_for_response_prompt else "available knowledge", # Ensure it's not empty
            context=retrieved_context_str
        )
        
        assistant_reasoning_steps_for_ui += "Generating final answer...\n"
        response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)
        logger.debug("Sending context and query to LLM for final response generation.")

        try:
            final_gemini_response = llm.generate_content(final_prompt_filled)
            final_answer_for_ui = final_gemini_response.text
            logger.info("Successfully generated final response from LLM.")
        except Exception as e:
            st.error(f"Error generating final response with Gemini: {e}") # UI error
            logger.error(f"Error generating final response with Gemini: {e}", exc_info=True)
            final_answer_for_ui = f"Sorry, I encountered an error trying to generate a response from the LLM: {e}"

        # Display final answer, replacing the "thinking" process
        response_placeholder.markdown(final_answer_for_ui)
        st.session_state.messages.append({"role": "assistant", "content": final_answer_for_ui})
        logger.info(f"--- User Query Processing Ended ---")
