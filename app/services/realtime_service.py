from typing import List, Optional, Iterator, Tuple, Any
from tavily import TavilyClient
import os
import time
import logging
from app.services.groq_service import GroqService, escape_curly_braces, AllGroqApisFailedError
from app.services.vector_store import VectorStoreService
from app.utils.retry import with_retry
from config import REALTIME_CHAT_ADDENDUM, GROQ_API_KEYS, GROQ_MODEL, INTENT_CLASSIFY_MODEL
from langchain_groq import ChatGroq

logger = logging.getLogger("J.A.R.V.I.S")

GROQ_REQUEST_TIMEOUT_FAST = 15

QUERY_EXTRACTION_PROMPT = (
    "You are a search query optimizer. Convert the user's message into a clean, focused "
    "web search query (max 10 words). Rules:\n"
    "- Remove filler words (you know, like, something, can you, tell me, search)\n"
    "- Add specific dates (today, 2026), event names, full names if applicable\n"
    "- For sports: include league name, team names, 'live score today'\n"
    "- For people: include full name and what user wants to know\n"
    "- Resolve references (he, that, it) from conversation history\n"
    "Output ONLY the search query. Nothing else."
)

class RealtimeGroqService(GroqService):
    def __init__(self, vector_store_service: VectorStoreService):
        super().__init__(vector_store_service)
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
            logger.info("Tavily search client initialized successfully")
        else:
            self.tavily_client = None
            logger.warning("TAVILY_API_KEY not set. Realtime search will be unavailable.")

        if GROQ_API_KEYS:
            self.fast_llm = ChatGroq(
                groq_api_key=GROQ_API_KEYS[0],
                model_name=INTENT_CLASSIFY_MODEL,
                temperature=0.0,
                request_timeout=GROQ_REQUEST_TIMEOUT_FAST,
                max_tokens=50,
            )
        else:
            self.fast_llm = None

    def extract_search_query(
        self, question: str, chat_history: Optional[List[Tuple]] = None
    ) -> str:
        if not self.fast_llm:
            return question
        
        q = question.strip()
        q_lower = q.lower()
        has_filler = any(p in q_lower for p in [
            "something like", "going on", "can you", "tell me",
            "search", "right now", "please"
        ])

        if len(q) < 30 and not has_filler:
            return q

        try:
            t0 = time.perf_counter()
            history_context = ""
            if chat_history:
                recent_chat_history = chat_history[-3:]
                parts = []
                for h, a in recent_chat_history:
                    parts.append(f"User: {h[:200]}")
                    parts.append(f"Assistant: {a[:200]}")
                history_context = "\n".join(parts)

            if history_context:
                full_prompt = (
                    f"{QUERY_EXTRACTION_PROMPT}\n\n"
                    f"Recent conversation:\n{history_context}\n\n"
                    f"User's latest message: {question}\n\n"
                    "Search query:"
                )
            else:
                full_prompt = (
                    f"{QUERY_EXTRACTION_PROMPT}\n\n"
                    f"User's message: {question}\n\n"
                    "Search query:"
                )

            response = self.fast_llm.invoke(full_prompt)
            extracted = response.content.strip().strip('"').strip("'")

            if extracted and 3 < len(extracted) < 200:
                logger.info(f"[REALTIME] Query extraction: {extracted} ({time.perf_counter()-t0:.3f}s)")
                return extracted
            else:
                logger.warning("[REALTIME] Query extraction returned unstable result, using raw question")
                return question
        except Exception as e:
            logger.warning(f"[REALTIME] Query extraction failed ({e}), using raw question")
            return question

    def search_tavily(self, query: str, num_results: int = 5) -> Tuple[str, Optional[dict]]:
        if not self.tavily_client:
            logger.warning("Tavily client not initialized. TAVILY_API_KEY not set.")
            return ("", None)
        if not query or not str(query).strip():
            return ("", None)

        try:
            t0 = time.perf_counter()
            response = with_retry(
                lambda: self.tavily_client.search(
                    query=query, 
                    search_depth="fast",
                    max_results=num_results,
                    include_answer=True,
                    include_raw_content=False
                ),
                max_retries=2,
                initial_delay=1.0
            )

            ai_answer = response.get("answer", "")
            results = response.get("results", [])

            if not results and not ai_answer:
                logger.warning(f"No Tavily search results for query: {query}")
                return ("", None)

            payload = {
                "query": query,
                "answer": ai_answer,
                "results": [
                    {
                        "title": r.get("title", "no title"),
                        "content": r.get("content", "")[:500],
                        "url": r.get("url", ""),
                        "score": round(float(r.get("score", 0)), 3)
                    } for r in results[:num_results]
                ]
            }

            parts = [f"WEB SEARCH RESULTS FOR: {query}"]
            if ai_answer:
                parts.append(f"AI-SYNTHESIZED ANSWER (use this as your primary source):\n{ai_answer}\n")
            
            parts.append("INDIVIDUAL SOURCES:")
            for i, result in enumerate(results[:num_results], 1):
                title = result.get("title", "no title")
                content = result.get("content", "")
                url = result.get("url", "")
                score = result.get("score", 0)
                parts.append(f"Source ({i}) (relevance: {score:.3f})")
                parts.append(f"Title: {title}")
                if content:
                    parts.append(f"Content: {content}")
                parts.append(f"URL: {url}")

            parts.append("END SEARCH RESULTS")
            formatted = "\n".join(parts)
            
            logger.info(f"TAVILY Results: {len(results)}, AI answer: {'yes' if ai_answer else 'no'}, formatted: {len(formatted)} chars ({time.perf_counter()-t0:.3f}s)")
            return (formatted, payload)

        except Exception as e:
            logger.error(f"Error performing Tavily search: {e}")
            return ("", None)

    def get_response(self, question: str, chat_history: Optional[List[Tuple]] = None, key_start_index: int = 0) -> str:
        try:
            search_query = self.extract_search_query(question, chat_history)
            logger.info(f"[REALTIME] Searching Tavily for: {search_query}")
            
            formatted_results, _ = self.search_tavily(search_query, num_results=5)
            
            if formatted_results:
                logger.info(f"[REALTIME] Tavily returned results (length: {len(formatted_results)} chars)")
            else:
                logger.warning(f"[REALTIME] Tavily returned no results for: {search_query}")

            extra_parts = [escape_curly_braces(formatted_results)] if formatted_results else None
            prompt, messages = self.build_prompt_and_messages(
                question, chat_history,
                extra_system_parts=extra_parts,
                mode_addendum=REALTIME_CHAT_ADDENDUM
            )

            t0 = time.perf_counter()
            response_content = self.invoke_llm(prompt, messages, question, key_start_index=key_start_index)
            logger.info(f"[TIMING] groq api: {time.perf_counter() - t0:.3f}s")
            logger.info(f"[RESPONSE] Realtime chat length: {len(response_content)} chars | Preview: {response_content[:120]}...")
            return response_content

        except AllGroqApisFailedError:
            raise
        except Exception as e:
            logger.error(f"Error in realtime get_response: {e}", exc_info=True)
            raise

    def prefetch_web_search(self, question: str, chat_history: Optional[List[Tuple]] = None) -> Tuple[str, Optional[dict]]:
        t0 = time.perf_counter()
        search_query = self.extract_search_query(question, chat_history)
        logger.info(f"[REALTIME] Prefetch: extracted query '{search_query[:50]}' in {time.perf_counter()-t0:.3f}s")
        
        formatted_results, payload = self.search_tavily(search_query, num_results=5)
        if formatted_results:
            logger.info(f"[REALTIME] Pre-fetch: Tavily returned {len(formatted_results)} chars in {time.perf_counter()-t0:.3f}s total")
            return (formatted_results, payload)
        
        logger.warning("[REALTIME] Pre-fetch failed.")
        return ("", None)

    def stream_response(self, question: str, chat_history: Optional[List[Tuple]] = None, key_start_index: int = 0) -> Iterator[Any]:
        try:
            yield {"activity": {"event": "extracting_query", "message": "Extracting search query..."}}
            search_query = self.extract_search_query(question, chat_history)
            logger.info(f"[REALTIME] Searching Tavily for: {search_query}")
            
            yield {"activity": {"event": "searching_web", "query": search_query, "message": f"Searching web for '{search_query}'..."}}
            formatted_results, payload = self.search_tavily(search_query, num_results=5)
            
            num_results = len(payload.get("results", [])) if payload else 0
            if formatted_results:
                logger.info(f"[REALTIME] Tavily returned results (length: {len(formatted_results)} chars)")
                yield {"activity": {"event": "search_completed", "message": f"Search completed: {num_results} results, {len(formatted_results)} chars of context"}}
            else:
                logger.warning(f"[REALTIME] Tavily returned no results for: {search_query}")
                yield {"activity": {"event": "search_completed", "message": "No search results found"}}

            if payload:
                yield {"search_results": payload}

            extra_parts = [escape_curly_braces(formatted_results)] if formatted_results else None
            prompt, messages = self.build_prompt_and_messages(
                question, chat_history,
                extra_system_parts=extra_parts,
                mode_addendum=REALTIME_CHAT_ADDENDUM
            )

            yield from self.stream_llm(prompt, messages, question, key_start_index=key_start_index)
            logger.info(f"[REALTIME] Stream completed for: {search_query}")

        except AllGroqApisFailedError:
            raise
        except Exception as e:
            logger.error(f"Error in realtime stream_response: {e}", exc_info=True)
            raise

    def stream_response_with_prefetched(
        self,
        question: str,
        chat_history: Optional[List[Tuple]] = None,
        formatted_results: Optional[str] = None,
        payload: Optional[dict] = None,
        key_start_index: int = 0
    ) -> Iterator[Any]:
        try:
            if payload:
                yield {"search_results": payload}
            
            extra_parts = [escape_curly_braces(formatted_results)] if formatted_results else None
            prompt, messages = self.build_prompt_and_messages(
                question, chat_history,
                extra_system_parts=extra_parts,
                mode_addendum=REALTIME_CHAT_ADDENDUM
            )

            yield from self.stream_llm(prompt, messages, question, key_start_index=key_start_index)
            logger.info("[REALTIME] Stream completed (pre-fetched results)")

        except AllGroqApisFailedError:
            raise
        except Exception as e:
            logger.error(f"Error in realtime stream_response_with_prefetched: {e}", exc_info=True)
            raise