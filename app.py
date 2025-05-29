import streamlit as st
from chatbot_utils import (
    get_gemini_llm,
    get_embedding_model,
    get_faq_retriever,
    get_sop_retriever,
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

    faq_retriever = get_faq_retriever(embedding_model, force_recreate=force_recreate_indexes)
    sop_retriever = get_sop_retriever(embedding_model, force_recreate=force_recreate_indexes)
    logger.info("Models and retrievers loading complete for Streamlit session.")
    return llm, faq_retriever, sop_retriever

llm, faq_retriever, sop_retriever = load_models_and_retrievers()

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
        best_source = "SOP_Store" # Default
        simplified_query = user_query # Default

        try:
            logger.debug("Sending query to LLM for initial analysis...")
            analysis_response = llm.generate_content(analysis_prompt_filled)
            logger.debug(f"Raw LLM Analysis Response Text: {analysis_response.text}")
            analysis_result_json = clean_json_response(analysis_response.text)
            logger.info(f"Parsed LLM Analysis JSON: {analysis_result_json}")

            if analysis_result_json:
                best_source = analysis_result_json.get("best_source", "SOP_Store")
                simplified_query = analysis_result_json.get("simplified_query_for_search", user_query)
                assistant_reasoning_steps_for_ui += f"*(Decided to check: {best_source.replace('_', ' ')} first using query: '{simplified_query}')*\n\n"
            else:
                assistant_reasoning_steps_for_ui += "*(Could not reliably parse LLM analysis. Defaulting to SOPs with original query.)*\n\n"
                logger.warning("LLM analysis JSON parsing failed.")
        except Exception as e:
            st.error(f"Error during LLM initial analysis: {e}") # Show error in UI
            logger.error(f"Error in LLM initial analysis: {e}", exc_info=True)
            assistant_reasoning_steps_for_ui += f"*(Error during analysis phase: {e}. Defaulting to SOPs.)*\n\n"
        
        logger.info(f"After Analysis - best_source: {best_source}, simplified_query: '{simplified_query}'")
        response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)

        # 2. Retrieve Context
        retrieved_context_str = ""
        source_type_used_for_response_prompt = "" # For the final LLM prompt
        docs_found = False

        # Attempt 1: FAQ Store
        if best_source == "FAQ_Store":
            source_type_used_for_response_prompt = "FAQs"
            assistant_reasoning_steps_for_ui += f"Attempting to retrieve from {source_type_used_for_response_prompt}...\n"
            response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)
            if faq_retriever:
                try:
                    logger.debug(f"Querying FAQ retriever with: '{simplified_query}'")
                    relevant_docs = faq_retriever.get_relevant_documents(simplified_query)
                    logger.info(f"FAQ Retriever found {len(relevant_docs)} docs for '{simplified_query}'.")
                    if relevant_docs:
                        retrieved_context_str = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
                        docs_found = True
                        assistant_reasoning_steps_for_ui += f"Found relevant info in {source_type_used_for_response_prompt}.\n"
                    else:
                        assistant_reasoning_steps_for_ui += f"No specific match found in {source_type_used_for_response_prompt}.\n"
                except Exception as e:
                    st.warning(f"Error retrieving from FAQs: {e}") # UI warning
                    logger.warning(f"Error retrieving from FAQs: {e}", exc_info=True)
                    assistant_reasoning_steps_for_ui += f"Error accessing {source_type_used_for_response_prompt}: {e}\n"
            else:
                assistant_reasoning_steps_for_ui += "FAQ retriever not available.\n"
                logger.warning("FAQ retriever was None during search attempt.")
        
        response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)

        # Attempt 2: SOP Store (if chosen, or if FAQ failed/wasn't primary)
        if not docs_found and (best_source == "SOP_Store" or (best_source == "FAQ_Store" and not docs_found)):
            if source_type_used_for_response_prompt == "FAQs": 
                 assistant_reasoning_steps_for_ui += f"Now checking SOPs...\n"
            source_type_used_for_response_prompt = "SOPs"
            assistant_reasoning_steps_for_ui += f"Attempting to retrieve from {source_type_used_for_response_prompt}...\n"
            response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)
            if sop_retriever:
                try:
                    logger.debug(f"Querying SOP retriever with: '{simplified_query}'")
                    relevant_docs = sop_retriever.get_relevant_documents(simplified_query)
                    logger.info(f"SOP Retriever found {len(relevant_docs)} docs for '{simplified_query}'.")
                    if relevant_docs:
                        retrieved_context_str = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
                        docs_found = True
                        assistant_reasoning_steps_for_ui += f"Found relevant info in {source_type_used_for_response_prompt}.\n"
                    else:
                        assistant_reasoning_steps_for_ui += f"No specific match found in {source_type_used_for_response_prompt}.\n"
                except Exception as e:
                    st.warning(f"Error retrieving from SOPs: {e}") # UI warning
                    logger.warning(f"Error retrieving from SOPs: {e}", exc_info=True)
                    assistant_reasoning_steps_for_ui += f"Error accessing {source_type_used_for_response_prompt}: {e}\n"
            else:
                assistant_reasoning_steps_for_ui += "SOP retriever not available.\n"
                logger.warning("SOP retriever was None during search attempt.")
        
        response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)

        # Attempt 3: Web Search (if chosen, or if internal KBs failed)
        if not docs_found or best_source == "Web_Search":
            if docs_found and best_source == "Web_Search": 
                 assistant_reason_steps_for_ui += f"Initial plan was web search. Performing web search...\n"
            elif not docs_found and source_type_used_for_response_prompt: 
                 assistant_reasoning_steps_for_ui += f"No specific match in {source_type_used_for_response_prompt}, now searching the web...\n"
            elif not docs_found and not source_type_used_for_response_prompt : 
                 assistant_reasoning_steps_for_ui += f"No internal documents found or applicable, searching the web...\n"
            
            source_type_used_for_response_prompt = "Web Search"
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