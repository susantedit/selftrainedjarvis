from typing import List, Optional, Iterator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import time
import logging
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information
from app.utils.retry import with_retry
from config import (
    GROQ_API_KEYS,
    GROQ_MODEL,
    JARVIS_SYSTEM_PROMPT,
    GENERAL_CHAT_ADDENDUM,
    GROQ_REQUEST_TIMEOUT
)

logger = logging.getLogger("J.A.R.V.I.S")

ALL_APIS_FAILED_MESSAGE = (
    "I'm unable to process your request at the moment. All API services are "
    "temporarily unavailable. Please try again in a few minutes."
)

class AllGroqApisFailedError(Exception):
    pass

def escape_curly_braces(text: str) -> str:
    if not text:
        return text
    return text.replace("{", "{{").replace("}", "}}")

REPEAT_WINDOW = 100
REPEAT_THRESHOLD = 2
REPEAT_CHECK_INTERVAL = 200

def _detect_repetition_loop(text: str) -> bool:
    if len(text) < REPEAT_WINDOW * REPEAT_THRESHOLD:
        return False
    phrase = text[-REPEAT_WINDOW:]
    return text.count(phrase) >= REPEAT_THRESHOLD

def truncate_at_repetition(text: str) -> str:
    if len(text) < REPEAT_WINDOW * REPEAT_THRESHOLD:
        return text
    phrase = text[-REPEAT_WINDOW:]
    if text.count(phrase) >= REPEAT_THRESHOLD:
        first = text.find(phrase)
        second = text.find(phrase, first + 1)
        if second > first:
            return text[:second].rstrip()
    return text

def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "tokens per day" in msg

def log_timing(label: str, elapsed: float, extra: str = ""):
    msg = f"[TIMING] {label}: {elapsed:.3f}s"
    if extra:
        msg += f" ({extra})"
    logger.info(msg)

def _mask_api_key(key: str) -> str:
    if not key or len(key) < 12:
        return "masked***"
    return f"{key[:4]}...{key[-4:]}"

class GroqService:
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service
        if not GROQ_API_KEYS:
            raise ValueError(
                "No Groq API keys configured. Set GROQ_API_KEY (and optionally GROQ_API_KEY_2, GROQ_API_KEY_3, ...) in .env"
            )
        
        self.llms = []
        for key in GROQ_API_KEYS:
            self.llms.append(ChatGroq(
                groq_api_key=key,
                model=GROQ_MODEL,
                temperature=0.5,
                max_tokens=1024,
                request_timeout=GROQ_REQUEST_TIMEOUT,
                model_kwargs={"frequency_penalty": 0.7}
            ))
        
        logger.info(f"Initialized GroqService with {len(GROQ_API_KEYS)} API key(s) (primary + fallbacks)")

    def invoke_llm(
        self,
        prompt: ChatPromptTemplate,
        messages: list,
        question: str,
        key_start_index: int = 0
    ) -> str:
        n = len(self.llms)
        last_exc = None
        keys_tried = []

        for i in range(n):
            idx = (key_start_index + i) % n
            keys_tried.append(idx)
            masked_key = _mask_api_key(GROQ_API_KEYS[idx])
            logger.info(f"Trying API key #{idx+1}/{n}: {masked_key}")

            def invoke_with_key():
                chain = prompt | self.llms[idx]
                return chain.invoke({"history": messages, "question": question})

            try:
                response = with_retry(
                    invoke_with_key,
                    max_retries=2,
                    initial_delay=1.0
                )
                
                if i > 0:
                    logger.info(f"Fallback successful: API key #{idx+1}/{n} succeeded: {masked_key}")
                
                text = response.content
                if _detect_repetition_loop(text):
                    logger.warning(f"[LLM] Repetition loop detected, truncating response ({len(text)} chars)")
                    text = truncate_at_repetition(text)
                return text

            except Exception as e:
                last_exc = e
                if _is_rate_limit_error(e):
                    logger.warning(f"API key #{idx+1}/{n} rate limited: {masked_key}")
                else:
                    logger.warning(f"API key #{idx+1}/{n} failed: {masked_key} ({str(e)[:100]})")
                
                if i < n - 1:
                    logger.info("Falling back to next API key...")
                    continue

        masked_all = ", ".join(_mask_api_key(GROQ_API_KEYS[j]) for j in keys_tried)
        logger.error(f"All {n} API key(s) failed. Tried: {masked_all}")
        raise AllGroqApisFailedError(ALL_APIS_FAILED_MESSAGE) from last_exc

    def stream_llm(
        self,
        prompt: ChatPromptTemplate,
        messages: list,
        question: str,
        key_start_index: int = 0
    ) -> Iterator[str]:
        n = len(self.llms)
        last_exc = None
        
        for i in range(n):
            idx = (key_start_index + i) % n
            masked_key = _mask_api_key(GROQ_API_KEYS[idx])
            logger.info(f"Streaming with API key #{idx+1}/{n}: {masked_key}")
            
            try:
                chain = prompt | self.llms[idx]
                chunk_count = 0
                first_chunk_time = None
                stream_start = time.perf_counter()
                
                accumulated = ""
                last_check_len = 0
                repetition_stopped = False
                
                for chunk in chain.stream({"history": messages, "question": question}):
                    content = ""
                    if hasattr(chunk, "content"):
                        content = chunk.content or ""
                    elif isinstance(chunk, dict) and "content" in chunk:
                        content = chunk.get("content", "")
                    
                    if isinstance(content, str) and content:
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter() - stream_start
                            log_timing("first chunk", first_chunk_time)
                        
                        chunk_count += 1
                        accumulated += content
                        
                        if len(accumulated) - last_check_len > REPEAT_CHECK_INTERVAL:
                            last_check_len = len(accumulated)
                            if _detect_repetition_loop(accumulated):
                                logger.warning(f"[STREAM] Repetition loop detected after {len(accumulated)} chars, stopping")
                                repetition_stopped = True
                                break
                        
                        yield content
                
                total_stream_time = time.perf_counter() - stream_start
                log_timing("groq stream total", total_stream_time, f"chunks: {chunk_count}{', TRUNCATED-REPETITION' if repetition_stopped else ''}")
                
                if i > 0 and chunk_count > 0:
                    logger.info(f"Fallback successful: API key #{idx+1}/{n} streamed: {masked_key}")
                return

            except Exception as e:
                last_exc = e
                if _is_rate_limit_error(e):
                    logger.warning(f"API key #{idx+1}/{n} rate limited: {masked_key}")
                else:
                    logger.warning(f"API key #{idx+1}/{n} failed: {masked_key} ({str(e)[:100]})")
                
                if i < n - 1:
                    logger.info("Falling back to next API key for stream...")
                    continue
                break

        logger.error(f"All {n} API key(s) failed during stream.")
        raise AllGroqApisFailedError(ALL_APIS_FAILED_MESSAGE) from last_exc

    def build_prompt_and_messages(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        extra_system_parts: Optional[List[str]] = None,
        mode_addendum: str = ""
    ) -> tuple:
        context = ""
        context_sources = []
        t0 = time.perf_counter()
        
        try:
            retriever = self.vector_store_service.get_retriever(k=10)
            context_docs = retriever.invoke(question)
            if context_docs:
                context = "\n".join([doc.page_content for doc in context_docs])
                context_sources = [doc.metadata.get("source", "unknown") for doc in context_docs]
                logger.info(f"[CONTEXT] Retrieved {len(context_docs)} chunks from sources: {context_sources}")
            else:
                logger.info("[CONTEXT] No relevant chunks found for query")
        except Exception as retrieval_err:
            logger.warning(f"Vector store retrieval failed, using empty context: {retrieval_err}")
        finally:
            log_timing("vector db", time.perf_counter() - t0)

        # Always append full learning data so personal facts are never missed
        from config import load_user_context
        full_user_context = load_user_context()

        time_info = get_time_information()
        system_message = JARVIS_SYSTEM_PROMPT
        system_message += f"\n\nCurrent time and date: {time_info}"

        if full_user_context:
            system_message += f"\n\n=== USER PROFILE & KNOWLEDGE BASE ===\n{escape_curly_braces(full_user_context)}"
        
        if context and context not in full_user_context:
            system_message += f"\n\n=== RELEVANT CONTEXT (vector search) ===\n{escape_curly_braces(context)}"
        
        if extra_system_parts:
            system_message += "\n\n" + "\n\n".join(extra_system_parts)
            
        if mode_addendum:
            system_message += f"\n\n{mode_addendum}"
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        
        messages = []
        if chat_history:
            for human_msg, ai_msg in chat_history:
                messages.append(HumanMessage(content=human_msg))
                messages.append(AIMessage(content=ai_msg))
                
        logger.info(f"[PROMPT] System message length: {len(system_message)} chars | History pairs: {len(chat_history) if chat_history else 0} | Question: {question[:100]}...")
        return prompt, messages

    def get_response(
        self, 
        question: str, 
        chat_history: Optional[List[tuple]] = None, 
        key_start_index: int = 0
    ) -> str:
        try:
            prompt, messages = self.build_prompt_and_messages(
                question, chat_history, mode_addendum=GENERAL_CHAT_ADDENDUM
            )
            
            t0 = time.perf_counter()
            result = self.invoke_llm(prompt, messages, question, key_start_index=key_start_index)
            log_timing("groq api", time.perf_counter() - t0)
            
            logger.info(f"[RESPONSE] General chat length: {len(result)} chars | Preview: {result[:120]}...")
            return result
            
        except AllGroqApisFailedError:
            raise
        except Exception as e:
            raise Exception(f"Error getting response from Groq: {str(e)}") from e

    def stream_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        key_start_index: int = 0
    ) -> Iterator[str]:
        try:
            prompt, messages = self.build_prompt_and_messages(
                question, chat_history, mode_addendum=GENERAL_CHAT_ADDENDUM
            )
            
            yield {"activity": {"event": "context_retrieved", "message": "Retrieved relevant context from knowledge base"}}
            
            yield from self.stream_llm(prompt, messages, question, key_start_index=key_start_index)
            
        except AllGroqApisFailedError:
            raise
        except Exception as e:
            raise Exception(f"Error streaming response from Groq: {str(e)}") from e