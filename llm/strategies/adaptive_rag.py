#!/usr/bin/env python
"""
Adaptive Retrieval using Classification of Query Complexity.
Stores the LLM-generated job description embedding into Redis Search.
Output is structured in the format:
{
  "user_input": { ... },
  "retrieved_results": {
    "query": { ... },
    "retrieved_results": [ ... ],
    "generated_job_description": { ... },
    "timings": { ... }
  }
}
"""

import os
import json
import time
import asyncio
import logging
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END
from functools import wraps
from langchain_google_vertexai import ChatVertexAI
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import PromptTemplate
from google.api_core.exceptions import GoogleAPICallError, DeadlineExceeded, InvalidArgument
from dotenv import load_dotenv # type: ignore
load_dotenv()

# Import utilities from utils.py
from llm.strategies.util import (
    redis_client,
    store_jd_in_redis,
    retrieve_cached_jd,
    classify_query,
    route_retrieval,
    parse_query_fields
)
# Import retriever modules and exceptions
from llm.retrievers.atlas_retriever import MongoDBAtlasRetriever
from llm.retrievers.simple_query import SimpleQueryRetriever, simple_query_retrieval
from llm.retrievers.complex_query import ComplexQueryRetriever
from llm.exceptions import QueryClassificationError, RetrievalError

# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Vertex AI & MongoDB Configuration
# ---------------------------
SERVICE_ACCOUNT_KEY = os.path.abspath("xooper.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-V2")

PROJECT_ID = "xooper-450012"
LOCATION = "us-central1"
MONGO_URI = (
    "mongodb+srv://xooper:lsBAmSmNcI0s7uUW@xoopercluster.alvrs.mongodb.net/?retryWrites=true&w=majority&appName=xoopercluster"
)
DB_NAME = "jobs"
COLLECTION_NAME = "job_postings"

vertex_ai_llm = ChatVertexAI(
    model="gemini-1.5-flash-001",
    project=PROJECT_ID,
    location=LOCATION,
    temperature=0.7
)

def performance_monitor(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️ {func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

# ---------------------------
# Adaptive RAG: Detailed Timing in LLM Call
# ---------------------------
@performance_monitor
async def process_with_vertex_ai(results, query_text_for_redis_search, user_query, timeout=15):
    if not results:
        logger.info("No results retrieved for processing.")
        return None

    total_start = time.time()

    # --- Prompt Construction ---
    prompt_start = time.time()
    output_parser = StructuredOutputParser.from_response_schemas([
        {"name": "job_title", "description": "Short job title derived from the user query or job postings.", "type": "string"},
        {"name": "experience", "description": "Minimum and maximum experience required, formatted as 'min-max' years (e.g., '2-5').", "type": "string"},
        {"name": "location", "description": "Geographic location mentioned by the user or found in job postings.", "type": "string"},
        {"name": "job_type", "description": "Nature of the job role (e.g., Full-time, Contract, Internship).", "type": "string"},
        {"name": "employment_type", "description": "Employment classification (e.g., Remote, Hybrid, On-site).", "type": "string"},
        {"name": "summary", "description": "Concise professional summary of the role, mentioning experience and key specializations without listing responsibilities.", "type": "string"},
        {"name": "responsibilities", "description": "Bullet-point list detailing each specialized requirement or skill individually.", "type": "array"},
        {"name": "required_skills", "description": "List of all essential skills or technologies required for the role.", "type": "array"},
        {"name": "department", "description": "The department or division relevant to this role.", "type": "string"},
        {"name": "qualifications", "description": "Education and experience level required, formatted as a list (e.g., ['Bachelor’s degree', '3 years experience']).", "type": "array"}
    ])

    prompt_template = PromptTemplate(
        template="""
        You are an AI-powered Job Description Generator. Generate a structured, **language-adaptive** job description with the right tone and complexity.

        ### **Key Guidelines**
        1. **Language & Tone Detection:**  
           - Identify the **language** from the user query (default: English) and generate entire description in that language.  
           - Identify the **tone** (Professional, General, Friendly) (default: Professional) and generate entire description in tone.  

        2. **Experience-Based Complexity:**  
           - **0-2 years** → Simple, guided learning, fundamental tasks.  
           - **3-5 years** → Balanced expertise, independent execution.  
           - **5+ years** → Advanced skills, leadership, and strategic contributions.  

        3. **Job Description Structure:**  
           - **Summary:** Concise, grouping specialized skills together without listing them.  
           - **Responsibilities:** Bullet points elaborating each key skill separately.  
           - **Qualifications & Experience:** Exactly as per user input.  
           - **Skills:** Only include those explicitly mentioned in the query or postings.  

        4. **Output Format:**  
           - **Strict JSON structure** with no extra text.  

        **User Query:**  
        {user_query}

        **Retrieved Job Postings:**  
        {results}

        {format_instructions}
        """,
        input_variables=["user_query", "results"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    prompt = prompt_template.format(
        user_query=json.dumps(user_query, indent=2),
        results=json.dumps(results, indent=2)
    )
    prompt_elapsed = time.time() - prompt_start
    # logger.info("Prompt construction time: %.4f seconds", prompt_elapsed)

    # --- LLM Call ---
    llm_start = time.time()
    try:
        response_future = asyncio.to_thread(vertex_ai_llm.invoke, prompt)
        response = await asyncio.wait_for(response_future, timeout=timeout)
    except Exception as e:
        logger.error("❌ LLM call error: %s", e)
        return None
    # llm_elapsed = time.time() - llm_start
    # logger.info("LLM call time: %.4f seconds", llm_elapsed)

    # --- Response Parsing ---
    parse_start = time.time()
    response_text = response.content.strip()
    structured_output = output_parser.parse(response_text)
    parse_elapsed = time.time() - parse_start
    # logger.info("Response parsing time: %.4f seconds", parse_elapsed)

    # --- Storing in Redis ---
    store_start = time.time()
    generated_jd = structured_output.get("generated_job_description", structured_output)
    global embedding_model
    store_jd_in_redis(generated_jd, user_query, embedding_model)
    store_elapsed = time.time() - store_start
    # logger.info("Storing in Redis time: %.4f seconds", store_elapsed)

    total_elapsed = time.time() - total_start
    # logger.info("Total process_with_vertex_ai time: %.4f seconds", total_elapsed)

    return structured_output

# ---------------------------
# Retrieval Node Implementations with Timing
# ---------------------------
@performance_monitor
async def structured_retrieval_node(s, atlas_retriever, embedding_model):
    start_time = time.time()
    try:
        combined_query = " ".join(filter(None, [
            s["query"].get("role", ""),
            " ".join(s["query"].get("skills", [])),
            s["query"].get("description", "")
        ]))
        query_vector = await asyncio.to_thread(embedding_model.encode, combined_query)
        s["results"] = await atlas_retriever.semantic_vector_search(query_vector.tolist(), top_k=1)
        elapsed = time.time() - start_time
        # logger.info("Time taken by structured_retrieval_node: %.4f seconds", elapsed)
        s.setdefault("timings", {})["structured_retrieval_time"] = elapsed
        return s
    except Exception as e:
        logger.error("❌ Error in structured retrieval: %s", e, exc_info=True)
        raise RetrievalError(f"Error in structured retrieval: {e}")

@performance_monitor
async def simple_query_retrieval_node(s, simple_retriever):
    start_time = time.time()
    try:
        logger.info("🔍 Executing Simple Query Retrieval with Query: %s", s["query"])
        s["results"] = await simple_query_retrieval(s["query"], simple_retriever)
        elapsed = time.time() - start_time
        # logger.info("Time taken by simple_query_retrieval_node: %.4f seconds", elapsed)
        s.setdefault("timings", {})["simple_query_retrieval_time"] = elapsed
        return s
    except Exception as e:
        logger.error("❌ Error in simple query retrieval: %s", e, exc_info=True)
        raise RetrievalError(f"Error in simple query retrieval: {e}")

@performance_monitor
async def complex_query_retrieval_node(s, complex_retriever, embedding_model):
    start_time = time.time()
    try:
        s["results"] = await complex_retriever.complex_query_retrieval(s["query"], embedding_model, top_k=1)
        elapsed = time.time() - start_time
        # logger.info("Time taken by complex_query_retrieval_node: %.4f seconds", elapsed)
        s.setdefault("timings", {})["complex_query_retrieval_time"] = elapsed
        return s
    except Exception as e:
        logger.error("❌ Error in complex query retrieval: %s", e, exc_info=True)
        raise RetrievalError(f"Error in complex query retrieval: {e}")

# ---------------------------
# Graph Node Wrappers
# ---------------------------

@performance_monitor
async def structured_retrieval_wrapper(s):
    return await structured_retrieval_node(s, atlas_retriever, embedding_model)


@performance_monitor
async def simple_query_retrieval_wrapper(s):
    return await simple_query_retrieval_node(s, simple_retriever)


@performance_monitor
async def complex_query_retrieval_wrapper(s):
    return await complex_query_retrieval_node(s, complex_retriever, embedding_model)


async def rag_operation_wrapper(s):
    return await rag_operation(s)

@performance_monitor
async def rag_operation(state):
    start_time = time.time()
    try:
        processed_results = await process_with_vertex_ai(state["results"], state['query_text_for_redis_search'], state["query"])
        state["processed_results"] = processed_results
        return state
    except Exception as e:
        logger.error("❌ Error in rag_operation: %s", e)
        return state

# ---------------------------
# Output Formatting Helper
# ---------------------------
def format_output(user_query, retrieved_results, generated_job_description):
    output = {
        "user_input": user_query,
        "retrieved_results": {
            "query": user_query,
            "retrieved_results": retrieved_results if isinstance(retrieved_results, list) else [retrieved_results],
            "generated_job_description": generated_job_description
        }
    }
    return output

@performance_monitor
async def process_the_query(query_text_for_redis_search, timeout=15):
    prompt = (
        f"Convert the following JSON data into a single, coherent sentence optimized for semantic search. "
        f"Every key and its corresponding value must be explicitly included in the sentence in the format 'key: value'. "
        f"In particular, ensure that the tone (e.g., 'tone: professional') is clearly visible. "
        f"The output should include all details such as role, min_experience, max_experience, skills, description, tone, language, industry, location, job_type, and employment_type.\n\n"
        f"JSON Data:\n{query_text_for_redis_search}\n\n"
        f"Optimized Query:"
    )

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(vertex_ai_llm.invoke, prompt),
            timeout=timeout
        )
        return response.content.strip()
    except Exception as e:
        logger.error("Vertex AI query error: %s", e)
        return None

# ---------------------------
# Main process_query Function for API Usage
# ---------------------------
@performance_monitor
async def process_query(query):
    total_start_time = time.time()
    try:
        global atlas_retriever, simple_retriever, complex_retriever, embedding_model

        # Timer: Sentence Transformer model loading
        start_time_model = time.time()
        logger.info("✅ Loaded Sentence Transformer model. (%.4f seconds)", time.time() - start_time_model)

        # Timer: Construct a single query text from the input
        start_time_construct_query = time.time()
        query_text = " ".join(filter(None, [
            query.get("role", ""),
            " ".join(query.get("skills", [])),
            query.get("description", "")
        ]))
        logger.info("Constructed query text: %s", query_text)
        logger.info("Query text construction took: %.4f seconds", time.time() - start_time_construct_query)

        # Timer: Construct query text for Redis search
        start_time_construct_redis = time.time()
        query_text_for_redis_search = " ".join(
            f"{key}: {value}"
            for key, value in (
                ("role", query.get("role", "")),
                ("skills", " ".join(query.get("skills", []))),
                ("description", query.get("description", "")),
                ("tone", query.get("tone", "")),
                ("language", query.get("language", "")),
                ("industry", query.get("industry", "")),
                ("location", query.get("location", "")),
                ("job_type", query.get("job_type", "")),
                ("employment_type", query.get("employment_type", ""))
            ) if value  # Only include non-empty values
        )
        logger.info("Constructed query text for Redis search: %s", query_text_for_redis_search)
        logger.info("Query text for Redis search construction took: %.4f seconds", time.time() - start_time_construct_redis)

        # Timer: Retrieve cached job description from Redis
        start_time_redis = time.time()
        parsed_fields = parse_query_fields(query_text_for_redis_search)
        cached = retrieve_cached_jd(
            query_fields=query,
            embed_model=embedding_model,
            base_similarity_threshold=0.8
        )
        logger.info("Redis cached retrieval took: %.4f seconds", time.time() - start_time_redis)

        if cached:
            logger.info("Returning cached job description.")
            logger.info("Total processing time: %.4f seconds", time.time() - total_start_time)
            return format_output(query, cached, cached)
        else:
            # Timer: Instantiate retrievers if not found in cache
            start_time_retrievers = time.time()
            atlas_retriever = MongoDBAtlasRetriever(MONGO_URI, DB_NAME, COLLECTION_NAME)
            simple_retriever = SimpleQueryRetriever(MONGO_URI, DB_NAME, COLLECTION_NAME)
            complex_retriever = ComplexQueryRetriever(MONGO_URI, DB_NAME, COLLECTION_NAME)
            logger.info("✅ Retrievers instantiated. (%.4f seconds)", time.time() - start_time_retrievers)

            # Timer: Query classification
            start_time_classification = time.time()
            strategy = await classify_query(query)
            logger.info("🔍 Query classified as: %s (%.4f seconds)", strategy, time.time() - start_time_classification)

            state = {
                "query": query,
                "query_text_for_redis_search": query_text_for_redis_search,
                "strategy": strategy,
                "results": []
            }

            # Timer: Graph execution (LLM process/RAG operation)
            start_time_graph = time.time()
            graph = StateGraph(dict)
            graph.add_node("structured_retrieval", structured_retrieval_wrapper)
            graph.add_node("simple_query_retrieval", simple_query_retrieval_wrapper)
            graph.add_node("complex_query_retrieval", complex_query_retrieval_wrapper)
            graph.add_node("rag_operation", rag_operation_wrapper)
            graph.add_conditional_edges(START, route_retrieval, {
                "structured_retrieval": "structured_retrieval",
                "simple_query_retrieval": "simple_query_retrieval",
                "complex_query_retrieval": "complex_query_retrieval"
            })
            graph.add_edge("structured_retrieval", "rag_operation")
            graph.add_edge("simple_query_retrieval", "rag_operation")
            graph.add_edge("complex_query_retrieval", "rag_operation")
            graph.add_edge("rag_operation", END)
            compiled_graph = graph.compile()
            final_state = await compiled_graph.ainvoke(state)
            logger.info("Graph execution took: %.4f seconds", time.time() - start_time_graph)

            if "processed_results" in final_state:
                logger.info("Total processing time: %.4f seconds", time.time() - total_start_time)
                return format_output(query, final_state["results"], final_state["processed_results"])
            else:
                logger.warning("⚠️ 'processed_results' missing in final state.")
                logger.info("Total processing time: %.4f seconds", time.time() - total_start_time)
                return "Error: Failed to generate job description."
    except Exception as e:
        logger.error("❌ Unexpected error in process_query: %s", e)
        return "Error: Failed to process query."

# ---------------------------
# Multiprocessing for CPU-Bound Tasks
# ---------------------------
def compute_embedding(query):
    global embedding_model
    return embedding_model.encode(query)

# with ProcessPoolExecutor() as executor:
#     query_vector = executor.submit(compute_embedding, combined_query).result()
