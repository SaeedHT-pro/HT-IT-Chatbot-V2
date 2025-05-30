from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from chatbot_utils import (
    get_gemini_llm,
    get_embedding_model,
    get_combined_retriever,
    perform_duckduckgo_search,
    INITIAL_ANALYSIS_PROMPT_TEMPLATE,
    RESPONSE_GENERATION_PROMPT_TEMPLATE,
    RELEVANCE_CHECK_PROMPT_TEMPLATE,
    clean_json_response,
    fetch_url_title,
    logger
)
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

llm = get_gemini_llm()
embedding_model = get_embedding_model()
retriever = get_combined_retriever(embedding_model)

class QueryRequest(BaseModel):
    user_query: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=dict)
async def chat(request: QueryRequest):
    user_query = request.user_query
    logger.info(f"User asked: {user_query}")

    # === REUSE YOUR LOGIC ===
    from chatbot_utils import RELEVANCE_CHECK_PROMPT_TEMPLATE

    # Step 1: Decide Source
    analysis_prompt = INITIAL_ANALYSIS_PROMPT_TEMPLATE.format(user_query=user_query)
    source = "Internal_Docs"
    simplified_query = user_query

    try:
        analysis_response = llm.generate_content(analysis_prompt)
        parsed = clean_json_response(analysis_response.text)
        if parsed:
            source = parsed.get("best_source", "Internal_Docs")
            simplified_query = parsed.get("simplified_query_for_search", user_query)
    except Exception as e:
        logger.warning(f"Analysis failed: {e}")

    context = ""
    if source == "Internal_Docs" and retriever:
        try:
            docs = retriever.get_relevant_documents(simplified_query)
            context = "\n\n---\n\n".join([f"Source: {d.metadata.get('source')}\n{d.page_content}" for d in docs])

            relevance_prompt = RELEVANCE_CHECK_PROMPT_TEMPLATE.format(
                user_query=user_query,
                simplified_query=simplified_query,
                retrieved_context=context[:3000]
            )
            rel_check = llm.generate_content(relevance_prompt).text.strip().upper()
            if "NO" in rel_check:
                context = ""
        except Exception as e:
            logger.error(f"Retriever error: {e}")

    if not context:
        context = perform_duckduckgo_search(simplified_query)
        source = "Web Search"

    final_prompt = RESPONSE_GENERATION_PROMPT_TEMPLATE.format(
        user_query=user_query,
        source_type_used=source,
        context=context
    )

    try:
        final_response = llm.generate_content(final_prompt).text
        return {"response": final_response}
    except Exception as e:
        logger.error(f"Gemini final error: {e}")
        return {"response": "Sorry, I encountered an issue generating your response."}
