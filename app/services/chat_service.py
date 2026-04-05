import base64
import json
import logging
import time
import uuid
import threading
from pathlib import Path
from typing import List, Optional, Dict, Iterator, Any, Union

from config import CHATS_DATA_DIR, CAMERA_CAPTURES_DIR, MAX_CHAT_HISTORY_TURNS, GROQ_API_KEYS
from app.models import ChatMessage
from app.services.groq_service import GroqService
from app.services.realtime_service import RealtimeGroqService
from app.services.brain_service import BrainService
from app.services.decision_types import (
    CATEGORY_GENERAL, CATEGORY_REALTIME, CATEGORY_CAMERA, 
    CATEGORY_TASK, CATEGORY_MIXED, HEAVY_INTENTS, INSTANT_INTENTS
)
from app.services.task_executor import TaskExecutor, TaskResponse
from app.services.task_manager import TaskManager
from app.services.vision_service import VisionService
from app.utils.key_rotation import get_next_key_pair

CAMERA_BYPASS_TOKEN = "!!CAMTOKEN!!"
JARVIS_BRAIN_SEARCH_TIMEOUT = 15
SAVE_EVERY_N_CHUNKS = 10

logger = logging.getLogger("J.A.R.V.I.S")

def _save_camera_image(img_base64: str, session_id: str) -> Optional[Path]:
    if not img_base64 or not CAMERA_CAPTURES_DIR:
        return None
    
    raw = img_base64.split(",", 1)[-1] if "," in img_base64 else img_base64
    try:
        data = base64.b64decode(raw)
        if len(data) < 100:
            logger.warning("[VISION] Captured image very small (%d bytes), may be invalid", len(data))
            
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        safe_id = (session_id or "unknown").replace("/", "").replace("\\", "")[:16]
        filename = f"cam_{safe_id}_{ts}.jpg"
        path = CAMERA_CAPTURES_DIR / filename
        path.write_bytes(data)
        
        logger.info("[VISION] Saved camera capture: %s (%d bytes)", filename, len(data))
        return path
    except Exception as e:
        logger.warning("[VISION] Failed to save camera image: %s", e)
        return None

class ChatService:
    def __init__(
        self,
        groq_service: GroqService,
        brain_service: BrainService = None,
        realtime_service: RealtimeGroqService = None,
        task_executor: TaskExecutor = None,
        vision_service: VisionService = None,
        task_manager: TaskManager = None
    ):
        self.groq_service = groq_service
        self.realtime_service = realtime_service
        self.brain_service = brain_service
        self.task_executor = task_executor
        self.vision_service = vision_service
        self.task_manager = task_manager
        self.sessions: Dict[str, List[ChatMessage]] = {}
        self.save_lock = threading.Lock()

    def load_session_from_disk(self, session_id: str) -> bool:
        safe_session_id = session_id.replace("/", "").replace("\\", "")
        filename = f"chat_{safe_session_id}.json"
        filepath = CHATS_DATA_DIR / filename
        
        if not filepath.exists():
            return False
            
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                chat_dict = json.load(f)
            
            messages = []
            for msg in chat_dict.get("messages", []):
                if not isinstance(msg, dict): continue
                role = msg.get("role")
                role = role if role in ("user", "assistant") else "user"
                content = msg.get("content")
                content = content if isinstance(content, str) else str(content or "")
                messages.append(ChatMessage(role=role, content=content))
                
            self.sessions[session_id] = messages
            return True
        except Exception as e:
            logger.warning("Failed to load session %s from disk: %s", session_id, e)
            return False

    def validate_session_id(self, session_id: str) -> bool:
        if not session_id or not session_id.strip():
            return False
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            return False
        if len(session_id) > 255:
            return False
        return True

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        t0 = time.perf_counter()
        if not session_id:
            new_id = str(uuid.uuid4())
            self.sessions[new_id] = []
            logger.info("[TIMING] session_get_or_create: %.3fs (new)", time.perf_counter() - t0)
            return new_id
            
        if not self.validate_session_id(session_id):
            raise ValueError(f"Invalid session id format: {session_id}")
            
        if session_id in self.sessions:
            logger.info("[TIMING] session_get_or_create: %.3fs (memory)", time.perf_counter() - t0)
            return session_id
            
        if self.load_session_from_disk(session_id):
            logger.info("[TIMING] session_get_or_create: %.3fs (disk)", time.perf_counter() - t0)
            return session_id
            
        self.sessions[session_id] = []
        logger.info("[TIMING] session_get_or_create: %.3fs (new id)", time.perf_counter() - t0)
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(ChatMessage(role=role, content=content))

    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        return self.sessions.get(session_id, [])

    def format_history_for_llm(self, session_id: str, exclude_last: bool = False) -> List[tuple]:
        messages = self.get_chat_history(session_id)
        history = []
        messages_to_process = messages[:-1] if exclude_last else messages
        
        i = 0
        while i < len(messages_to_process) - 1:
            user_msg = messages_to_process[i]
            ai_msg = messages_to_process[i+1]
            if user_msg.role == "user" and ai_msg.role == "assistant":
                u_content = user_msg.content if isinstance(user_msg.content, str) else str(user_msg.content or "")
                a_content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content or "")
                history.append((u_content, a_content))
                i += 2
            else:
                i += 1
                
        if len(history) > MAX_CHAT_HISTORY_TURNS:
            history = history[-MAX_CHAT_HISTORY_TURNS:]
        return history

    def process_message(self, session_id: str, user_message: str) -> str:
        self.add_message(session_id, "user", user_message)
        logger.info("[GENERAL] Session: %s | User: %.20s", session_id[:12], user_message)
        
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        logger.info("[GENERAL] History pairs sent to LLM: %d", len(chat_history))
        
        chat_idx = get_next_key_pair(len(GROQ_API_KEYS), need_brain=False)
        response = self.groq_service.get_response(
            question=user_message, 
            chat_history=chat_history, 
            key_start_index=chat_idx
        )
        
        self.add_message(session_id, "assistant", response)
        logger.info("[GENERAL] Response length: %d chars", len(response))
        return response

    def process_realtime_message(self, session_id: str, user_message: str) -> str:
        if not self.realtime_service:
            raise ValueError("Realtime service is not initialized.")
            
        self.add_message(session_id, "user", user_message)
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        chat_idx = get_next_key_pair(len(GROQ_API_KEYS), need_brain=False)
        
        response = self.realtime_service.get_response(
            question=user_message, 
            chat_history=chat_history, 
            key_start_index=chat_idx
        )
        
        self.add_message(session_id, "assistant", response)
        return response

    def process_message_stream(self, session_id: str, user_message: str) -> Iterator[Union[str, Dict[str, Any]]]:
        logger.info("[GENERAL-STREAM] Session: %s | User: %.20s", session_id[:12], user_message)
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", "")
        
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        yield {"activity": {"event": "query_detected", "message": user_message}}
        yield {"activity": {"event": "routing", "route": "general"}}
        yield {"activity": {"event": "streaming_started", "route": "general"}}
        
        chat_idx = get_next_key_pair(len(GROQ_API_KEYS), need_brain=False)
        chunk_count = 0
        t0 = time.perf_counter()
        
        try:
            for chunk in self.groq_service.stream_response(
                question=user_message, chat_history=chat_history, key_start_index=chat_idx
            ):
                if isinstance(chunk, dict):
                    yield chunk
                    continue
                
                if chunk_count == 0:
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    yield {"activity": {"event": "first_chunk", "route": "general", "elapsed_ms": elapsed_ms}}
                
                self.sessions[session_id][-1].content += chunk
                chunk_count += 1
                if chunk_count % SAVE_EVERY_N_CHUNKS == 0:
                    self.save_chat_session(session_id, log_timing=False)
                yield chunk
        finally:
            final_response = self.sessions[session_id][-1].content
            logger.info("[GENERAL-STREAM] Completed | Chunks: %d", chunk_count)
            self.save_chat_session(session_id)

    def process_realtime_message_stream(self, session_id: str, user_message: str) -> Iterator[Union[str, Dict[str, Any]]]:
        if not self.realtime_service:
            raise ValueError("Realtime service is not initialized.")
            
        logger.info("[REALTIME-STREAM] Session: %s | User: %.20s", session_id[:12], user_message)
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", "")
        
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        yield {"activity": {"event": "query_detected", "message": user_message}}
        yield {"activity": {"event": "routing", "route": "realtime"}}
        yield {"activity": {"event": "streaming_started", "route": "realtime"}}
        
        chat_idx = get_next_key_pair(len(GROQ_API_KEYS), need_brain=False)
        chunk_count = 0
        t0 = time.perf_counter()
        
        try:
            for chunk in self.realtime_service.stream_response(
                question=user_message, chat_history=chat_history, key_start_index=chat_idx
            ):
                if isinstance(chunk, dict):
                    yield chunk
                    continue
                
                if chunk_count == 0:
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    yield {"activity": {"event": "first_chunk", "route": "realtime", "elapsed_ms": elapsed_ms}}
                
                self.sessions[session_id][-1].content += chunk
                chunk_count += 1
                yield chunk
        finally:
            self.save_chat_session(session_id)

    def process_jarvis_message_stream(
        self, session_id: str, user_message: str, img_base64: Optional[str] = None
    ) -> Iterator[Union[str, Dict[str, Any]]]:
        t_jarvis = time.perf_counter()
        logger.info("[JARVIS-STREAM] Session: %s | Image: %s", session_id[:12], "yes" if img_base64 else "no")
        
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", "")
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        
        yield {"activity": {"event": "query_detected", "message": user_message}}
        
        if img_base64 and CAMERA_BYPASS_TOKEN in user_message:
            yield {"activity": {"event": "decision", "query_type": "camera", "reasoning": "Image attached"}}
            yield {"activity": {"event": "routing", "route": "vision"}}
            _save_camera_image(img_base64, session_id)
            
            if self.vision_service:
                prompt = user_message.replace(CAMERA_BYPASS_TOKEN, "").strip() or "What do you see?"
                text = self.vision_service.describe_image(img_base64, prompt)
            else:
                text = "Vision is not available."
                
            self.sessions[session_id][-1].content = text
            yield text
            self.save_chat_session(session_id)
            return

        brain_idx, chat_idx = get_next_key_pair(len(GROQ_API_KEYS), need_brain=bool(self.brain_service))
        category = CATEGORY_GENERAL
        
        if self.brain_service:
            category, method, elapsed = self.brain_service.classify_primary(user_message, chat_history, key_index=brain_idx)
            yield {"activity": {"event": "decision", "query_type": category, "reasoning": method.capitalize()}}

        if category == CATEGORY_CAMERA:
            yield {"activity": {"event": "routing", "route": "camera"}}
            if img_base64:
                _save_camera_image(img_base64, session_id)
                text = self.vision_service.describe_image(img_base64, user_message) if self.vision_service else "Vision unavailable"
            else:
                text = "Let me take a look..."
                yield {"actions": {"cam": {"action": "open_and_capture", "resend_message": user_message}}}
            
            self.sessions[session_id][-1].content = text
            yield text
            self.save_chat_session(session_id)
            return

        if category in (CATEGORY_TASK, CATEGORY_MIXED):
            yield {"activity": {"event": "routing", "route": "task" if category == CATEGORY_TASK else "mixed"}}
            task_types, task_method, task_elapsed = self.brain_service.classify_task(user_message, chat_history, key_index=brain_idx)
            
            intents = self.brain_service.extract_task_payloads(user_message, task_types, chat_history)
            yield {"activity": {"event": "tasks_executing", "message": "Running tasks..."}}

            # Split intents: instant (open/play/search) vs heavy (image/content)
            instant_intents = [(t, p) for t, p in intents if t not in HEAVY_INTENTS]
            heavy_intents = [(t, p) for t, p in intents if t in HEAVY_INTENTS]

            instant_response = self.task_executor.execute(instant_intents, chat_history)

            # Build and yield actions dict from instant response
            actions_dict = {}
            if instant_response.wopens: actions_dict["wopens"] = instant_response.wopens
            if instant_response.plays: actions_dict["plays"] = instant_response.plays
            if instant_response.googlesearches: actions_dict["googlesearches"] = instant_response.googlesearches
            if instant_response.youtubesearches: actions_dict["youtubesearches"] = instant_response.youtubesearches
            if instant_response.images:
                # images are (url, bytes) tuples — send only the URL to frontend
                actions_dict["images"] = [img[0] if isinstance(img, tuple) else img for img in instant_response.images]
            if instant_response.contents: actions_dict["contents"] = instant_response.contents
            if instant_response.cam: actions_dict["cam"] = instant_response.cam
            if actions_dict:
                yield {"actions": actions_dict}

            # Submit heavy tasks (image gen, content) to background task manager
            bg_task_ids = []
            if self.task_manager and heavy_intents:
                for intent_type, payload in heavy_intents:
                    task_id = self.task_manager.submit(intent_type, payload, chat_history)
                    bg_task_ids.append({
                        "task_id": task_id,
                        "type": intent_type,
                        "label": payload.get("prompt", payload.get("message", ""))[:80],
                    })
                if bg_task_ids:
                    yield {"background_tasks": bg_task_ids}
                    yield {"activity": {"event": "background_dispatched", "message": f"Dispatched {len(bg_task_ids)} background task(s)"}}

            if category == CATEGORY_MIXED:
                stream_svc = self.realtime_service if self.realtime_service else self.groq_service
                yield {"activity": {"event": "streaming_started", "route": "mixed"}}
                for chunk in stream_svc.stream_response(user_message, chat_history, chat_idx):
                    if isinstance(chunk, dict): yield chunk
                    else:
                        self.sessions[session_id][-1].content += chunk
                        yield chunk
            else:
                self.sessions[session_id][-1].content = instant_response.text
                yield instant_response.text
            
            self.save_chat_session(session_id)
            return

        # Default Flow (General/Realtime)
        use_realtime = category == CATEGORY_REALTIME and self.realtime_service
        route_name = "realtime" if use_realtime else "general"
        yield {"activity": {"event": "routing", "route": route_name}}
        yield {"activity": {"event": "streaming_started", "route": route_name}}
        
        stream_svc = self.realtime_service if use_realtime else self.groq_service
        chunk_count = 0
        t0 = time.perf_counter()
        for chunk in stream_svc.stream_response(user_message, chat_history, chat_idx):
            if isinstance(chunk, dict): yield chunk
            else:
                if chunk_count == 0:
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    yield {"activity": {"event": "first_chunk", "route": route_name, "elapsed_ms": elapsed_ms}}
                self.sessions[session_id][-1].content += chunk
                chunk_count += 1
                yield chunk
        self.save_chat_session(session_id)

    def save_chat_session(self, session_id: str, log_timing: bool = True):
        if session_id not in self.sessions:
            return
            
        safe_id = session_id.replace("/", "").replace("\\", "")
        filepath = CHATS_DATA_DIR / f"chat_{safe_id}.json"
        
        chat_dict = {
            "session_id": session_id,
            "messages": [{"role": m.role, "content": m.content} for m in self.sessions[session_id]]
        }
        
        with self.save_lock:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(chat_dict, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error("Failed to save chat session %s: %s", session_id, e)