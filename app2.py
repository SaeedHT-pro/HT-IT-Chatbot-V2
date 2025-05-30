# app.py
import streamlit as st
from chatbot_utils import (
    get_gemini_llm,
    get_embedding_model,
    get_combined_retriever,
    perform_duckduckgo_search,
    INITIAL_ANALYSIS_PROMPT_TEMPLATE,
    RESPONSE_GENERATION_PROMPT_TEMPLATE,
    RELEVANCE_CHECK_PROMPT_TEMPLATE, # Make sure this is imported
    clean_json_response,
    fetch_url_title, # Import the new function
    logger
)
import time
import re # Import regex module

# --- HELPER FUNCTION FOR LINK PREVIEWS ---
def process_response_for_link_previews(text: str) -> str:
    """
    Finds [PREVIEW](URL) syntax in text, fetches URL titles, and replaces with Markdown links.
    """
    logger.debug(f"Starting link preview processing for text: {text[:100]}...")
    # Regex to find [PREVIEW](url)
    # It captures the full URL inside the parentheses
    preview_link_pattern = r'\[PREVIEW\]\(([^)]+)\)'

    def replace_link(match):
        url = match.group(1)
        logger.info(f"Found [PREVIEW] link for URL: {url}")
        
        # Basic validation for common image/media file extensions - skip fetching title for these
        media_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mp3', '.pdf', '.zip', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
        if any(url.lower().endswith(ext) for ext in media_extensions):
            logger.info(f"URL {url} appears to be a direct media link. Skipping title fetch, using URL as title.")
            return f"[{url.split('/')[-1]}]({url})" # Use filename as link text

        title = fetch_url_title(url)
        if title and title != url: # If title was fetched and is not just the URL itself
            # Sanitize title for Markdown (escape brackets)
            title = title.replace("[", "\\[").replace("]", "\\]")
            logger.info(f"Replacing [PREVIEW]({url}) with title: [{title}]({url})")
            return f"[{title}]({url})"
        else: # If title fetch failed or returned the URL
            logger.info(f"Failed to fetch a distinct title for {url}. Using URL as link text.")
            return f"[{url}]({url})"

    processed_text = re.sub(preview_link_pattern, replace_link, text)
    
    if processed_text != text:
        logger.debug(f"Link preview processing completed. Text was modified.")
    else:
        logger.debug(f"Link preview processing completed. No [PREVIEW] links found or no changes made.")
        
    return processed_text

# --- PAGE CONFIG --- (no change)
st.set_page_config(page_title="IT Support Chatbot", layout="wide")
st.title("IT Support Chatbot Assistant ")

# --- INITIALIZATION & CACHING --- (no change)
@st.cache_resource
def load_models_and_retrievers():
    logger.info("Attempting to load models and retrievers for Streamlit session...")
    llm = get_gemini_llm()
    embedding_model = get_embedding_model()
    force_recreate_indexes = False # Default
    combined_retriever = get_combined_retriever(embedding_model, force_recreate=force_recreate_indexes)
    logger.info("Models and retrievers loading complete for Streamlit session.")
    return llm, combined_retriever

llm, combined_retriever = load_models_and_retrievers()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your IT Support Assistant. How can I help you today?"}]

# --- CHAT UI --- (no change)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask your IT question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        thinking_message_ui = "🤔 Thinking..."
        response_placeholder.markdown(thinking_message_ui)
        
        assistant_reasoning_steps_for_ui = "" 

        logger.info(f"--- New User Query Processing Started ---")
        logger.info(f"User Query: {user_query}")

        # 1. Initial Analysis with LLM (Your existing code for this is good)
        analysis_prompt_filled = INITIAL_ANALYSIS_PROMPT_TEMPLATE.format(user_query=user_query)
        llm_decision_best_source = "Internal_Docs" # Default
        simplified_query = user_query

        try:
            logger.debug("Sending query to LLM for initial analysis...")
            analysis_response = llm.generate_content(analysis_prompt_filled)
            logger.debug(f"Raw LLM Analysis Response Text: {analysis_response.text}")
            analysis_result_json = clean_json_response(analysis_response.text)
            logger.info(f"Parsed LLM Analysis JSON: {analysis_result_json}")

            if analysis_result_json:
                llm_decision_best_source = analysis_result_json.get("best_source", "Internal_Docs")
                simplified_query = analysis_result_json.get("simplified_query_for_search", user_query)
                ui_source_display = "Internal Documents" if llm_decision_best_source == "Internal_Docs" else llm_decision_best_source.replace('_', ' ')
                assistant_reasoning_steps_for_ui += f"*(LLM suggests checking: {ui_source_display} first using query: '{simplified_query}')*\n\n"
            else:
                assistant_reasoning_steps_for_ui += "*(Could not reliably parse LLM analysis. Defaulting to Internal Documents.)*\n\n"
                logger.warning("LLM analysis JSON parsing failed.")
        except Exception as e:
            st.error(f"Error during LLM initial analysis: {e}")
            logger.error(f"Error in LLM initial analysis: {e}", exc_info=True)
            assistant_reasoning_steps_for_ui += f"*(Error during analysis phase: {e}. Defaulting to Internal Documents.)*\n\n"
        
        logger.info(f"After Analysis - LLM suggested best_source: {llm_decision_best_source}, simplified_query: '{simplified_query}'")
        response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)

        # 2. Retrieve Context
        retrieved_context_str = ""
        source_type_used_for_response_prompt = "" 
        relevant_internal_docs_found = False # Key flag

        # Function for LLM Relevance Check (copied from previous good suggestion)
        def check_context_relevance(query_for_relevance, simplified_search_query, context_to_check, llm_model):
            if not context_to_check:
                return False
            relevance_prompt_filled = RELEVANCE_CHECK_PROMPT_TEMPLATE.format(
                user_query=query_for_relevance,
                simplified_query=simplified_search_query,
                retrieved_context=context_to_check[:3000] # Limit context
            )
            logger.debug(f"Asking LLM to check relevance of context for query: '{query_for_relevance}'")
            try:
                relevance_response = llm_model.generate_content(relevance_prompt_filled)
                relevance_answer = relevance_response.text.strip().upper()
                logger.info(f"LLM relevance check answer: {relevance_answer}")
                return "YES" in relevance_answer
            except Exception as e_rel:
                logger.error(f"Error during LLM relevance check: {e_rel}", exc_info=True)
                return False

        # Attempt 1: Internal Documents (Combined FAQs and SOPs)
        if llm_decision_best_source == "Internal_Docs":
            source_type_used_for_response_prompt = "Internal Documents" # Tentative
            assistant_reasoning_steps_for_ui += f"Attempting to retrieve from {source_type_used_for_response_prompt}...\n"
            response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)

            if combined_retriever:
                try:
                    logger.debug(f"Querying Combined retriever with: '{simplified_query}'")
                    # Retrieve more documents initially to give relevance checker more to work with
                    retrieved_docs_list = combined_retriever.search_kwargs['k'] = 5 # Temporarily increase k
                    retrieved_docs_list = combined_retriever.get_relevant_documents(simplified_query)
                    combined_retriever.search_kwargs['k'] = 3 # Reset k if you changed it, or manage k better

                    logger.info(f"Combined Retriever initially found {len(retrieved_docs_list)} docs for '{simplified_query}'.")
                    
                    if retrieved_docs_list:
                        potential_context_parts = []
                        for doc in retrieved_docs_list:
                            doc_info = f"Source: {doc.metadata.get('source', 'N/A')}, Type: {doc.metadata.get('doc_type', 'N/A')}"
                            potential_context_parts.append(f"{doc_info}\nContent: {doc.page_content}")
                        potential_context = "\n\n---\n\n".join(potential_context_parts)

                        if check_context_relevance(user_query, simplified_query, potential_context, llm):
                            retrieved_context_str = potential_context # Use all retrieved if relevant
                            relevant_internal_docs_found = True
                            assistant_reasoning_steps_for_ui += f"Found relevant info in {source_type_used_for_response_prompt}.\n"
                        else:
                            assistant_reasoning_steps_for_ui += f"Found entries in {source_type_used_for_response_prompt}, but they don't seem directly relevant to your query.\n"
                            logger.info("Internal docs found but deemed not relevant by LLM.")
                    else:
                        assistant_reasoning_steps_for_ui += f"No specific match found in {source_type_used_for_response_prompt}.\n"
                        logger.info("No docs found by internal retriever.")
                except Exception as e:
                    st.warning(f"Error retrieving from {source_type_used_for_response_prompt}: {e}")
                    logger.warning(f"Error retrieving from {source_type_used_for_response_prompt}: {e}", exc_info=True)
                    assistant_reasoning_steps_for_ui += f"Error accessing {source_type_used_for_response_prompt}: {e}\n"
            else:
                assistant_reasoning_steps_for_ui += f"{source_type_used_for_response_prompt} retriever not available.\n"
                logger.warning("Combined retriever was None during search attempt.")
        
        response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)

        # Attempt 2: Web Search (if LLM chose Web_Search OR if Internal_Docs search failed to find RELEVANT info)
        if llm_decision_best_source == "Web_Search" or not relevant_internal_docs_found:
            if llm_decision_best_source == "Internal_Docs" and not relevant_internal_docs_found:
                 assistant_reasoning_steps_for_ui += f"No relevant information found in Internal Documents. Now searching the web...\n"
            elif llm_decision_best_source == "Web_Search":
                 assistant_reasoning_steps_for_ui += f"Searching the web as per initial plan...\n"
            
            source_type_used_for_response_prompt = "Web Search"
            assistant_reasoning_steps_for_ui += f"🌐 Searching online for '{simplified_query}'...\n"
            response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)
            time.sleep(0.1) 
            
            web_search_results = perform_duckduckgo_search(simplified_query)
            logger.info(f"Web Search for '{simplified_query}' returned (first 300 chars): {web_search_results[:300]}...")
            
            if web_search_results and "did not yield specific results" not in web_search_results and "failed" not in web_search_results:
                # Optional: Relevance check for web results too, but often web snippets are more direct if search is good.
                # For now, assume web results are used if found.
                retrieved_context_str = web_search_results # This will overwrite internal context if internal was not relevant
                relevant_internal_docs_found = True # Set to true because we now have *some* context for the LLM
                assistant_reasoning_steps_for_ui += f"Found info via {source_type_used_for_response_prompt}.\n"
            else:
                assistant_reasoning_steps_for_ui += f"Web search did not return specific usable results or failed.\n"
                if not retrieved_context_str: # Only if nothing else was found (internal context was also empty)
                    retrieved_context_str = "No relevant information found in internal documents or through web search."
                    source_type_used_for_response_prompt = "any available source" # Update if truly nothing
        
        response_placeholder.markdown(thinking_message_ui + "\n" + assistant_reasoning_steps_for_ui)
        logger.debug(f"Final retrieved_context_str before LLM (first 300 chars): {retrieved_context_str[:300]}...")
        logger.info(f"Final source_type_used_for_response_prompt: {source_type_used_for_response_prompt}")
        logger.info(f"Final relevant_internal_docs_found (or web found): {relevant_internal_docs_found}") # Name is a bit misleading now

        # 3. Generate Final Response with LLM
        final_answer_for_ui = "Sorry, I encountered an issue and couldn't generate a response."
        
        # If after all attempts, retrieved_context_str is still the "not found" message or empty
        if not retrieved_context_str or "No relevant information found" in retrieved_context_str:
            final_answer_for_ui = "I couldn't find specific information regarding your query in our knowledge base or on the web. Could you please try rephrasing or ask something else?"
            logger.info("No usable context found after all search attempts.")
        else:
            final_prompt_filled = RESPONSE_GENERATION_PROMPT_TEMPLATE.format(
                user_query=user_query,
                source_type_used=source_type_used_for_response_prompt if source_type_used_for_response_prompt else "available knowledge",
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
                st.error(f"Error generating final response with Gemini: {e}")
                logger.error(f"Error generating final response with Gemini: {e}", exc_info=True)
                final_answer_for_ui = f"Sorry, I encountered an error trying to generate a response from the LLM: {e}"

        # Process response for link previews
        processed_final_answer_for_ui = process_response_for_link_previews(final_answer_for_ui)
        
        response_placeholder.markdown(processed_final_answer_for_ui)
        st.session_state.messages.append({"role": "assistant", "content": processed_final_answer_for_ui})
        logger.info(f"--- User Query Processing Ended ---")
