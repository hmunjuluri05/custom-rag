from typing import Any, Dict, List, Optional, Union
import logging
import time
from datetime import datetime
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseMessage, LLMResult

logger = logging.getLogger(__name__)


class RAGSystemCallbackHandler(BaseCallbackHandler):
    """Comprehensive callback handler for monitoring LLM and embedding operations"""

    def __init__(self, log_level: str = "INFO"):
        super().__init__()
        self.log_level = getattr(logging, log_level.upper())
        self.logger = logging.getLogger(f"{__name__}.RAGSystem")
        self.logger.setLevel(self.log_level)

        # Metrics tracking
        self.metrics = {
            "llm_calls": 0,
            "embedding_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "errors": 0,
            "start_time": datetime.now(),
            "call_times": []
        }

        self._current_call_start = None

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Called when LLM starts running."""
        self._current_call_start = time.time()
        self.metrics["llm_calls"] += 1

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"🚀 LLM Call #{self.metrics['llm_calls']} started")
            self.logger.debug(f"Model: {serialized.get('model_name', 'unknown')}")
            self.logger.debug(f"Prompts count: {len(prompts)}")

            # Log first prompt (truncated for readability)
            if prompts:
                prompt_preview = prompts[0][:200] + "..." if len(prompts[0]) > 200 else prompts[0]
                self.logger.debug(f"Prompt preview: {prompt_preview}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM ends running."""
        if self._current_call_start:
            call_time = time.time() - self._current_call_start
            self.metrics["call_times"].append(call_time)
            self._current_call_start = None

        # Track token usage
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]
            tokens_used = token_usage.get("total_tokens", 0)
            self.metrics["total_tokens"] += tokens_used

            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"✅ LLM Call completed in {call_time:.2f}s")
                self.logger.info(f"Tokens used: {tokens_used} (Total: {self.metrics['total_tokens']})")

        # Log response preview
        if self.logger.isEnabledFor(logging.DEBUG) and response.generations:
            for i, generation_list in enumerate(response.generations):
                for j, generation in enumerate(generation_list):
                    response_preview = generation.text[:200] + "..." if len(generation.text) > 200 else generation.text
                    self.logger.debug(f"Response {i}-{j} preview: {response_preview}")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Called when LLM errors."""
        self.metrics["errors"] += 1
        self.logger.error(f"❌ LLM Error #{self.metrics['errors']}: {str(error)}")

        if self._current_call_start:
            call_time = time.time() - self._current_call_start
            self.logger.error(f"Error occurred after {call_time:.2f}s")
            self._current_call_start = None

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Called when chain starts running."""
        if self.logger.isEnabledFor(logging.DEBUG):
            chain_name = serialized.get("name", "unknown_chain")
            self.logger.debug(f"🔗 Chain '{chain_name}' started")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when chain ends running."""
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("🔗 Chain completed successfully")

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Called when chain errors."""
        self.metrics["errors"] += 1
        self.logger.error(f"❌ Chain Error: {str(error)}")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Called when tool starts running."""
        if self.logger.isEnabledFor(logging.DEBUG):
            tool_name = serialized.get("name", "unknown_tool")
            self.logger.debug(f"🔧 Tool '{tool_name}' started with input: {input_str[:100]}...")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when tool ends running."""
        if self.logger.isEnabledFor(logging.DEBUG):
            output_preview = output[:100] + "..." if len(output) > 100 else output
            self.logger.debug(f"🔧 Tool completed with output: {output_preview}")

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Called when tool errors."""
        self.metrics["errors"] += 1
        self.logger.error(f"❌ Tool Error: {str(error)}")

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Called when arbitrary text is generated."""
        if self.logger.isEnabledFor(logging.DEBUG):
            text_preview = text[:100] + "..." if len(text) > 100 else text
            self.logger.debug(f"📝 Text generated: {text_preview}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics about LLM usage"""
        current_time = datetime.now()
        duration = (current_time - self.metrics["start_time"]).total_seconds()

        call_times = self.metrics["call_times"]
        avg_call_time = sum(call_times) / len(call_times) if call_times else 0

        return {
            "session_duration_seconds": duration,
            "llm_calls": self.metrics["llm_calls"],
            "embedding_calls": self.metrics["embedding_calls"],
            "total_tokens": self.metrics["total_tokens"],
            "total_cost": self.metrics["total_cost"],
            "errors": self.metrics["errors"],
            "average_call_time_seconds": avg_call_time,
            "total_call_time_seconds": sum(call_times),
            "calls_per_minute": (self.metrics["llm_calls"] / duration * 60) if duration > 0 else 0,
            "start_time": self.metrics["start_time"].isoformat(),
            "current_time": current_time.isoformat()
        }

    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            "llm_calls": 0,
            "embedding_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "errors": 0,
            "start_time": datetime.now(),
            "call_times": []
        }
        self.logger.info("📊 Metrics reset")


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler specifically for streaming responses"""

    def __init__(self, on_token_callback=None):
        super().__init__()
        self.on_token_callback = on_token_callback
        self.logger = logging.getLogger(f"{__name__}.Streaming")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when LLM generates a new token during streaming"""
        if self.on_token_callback:
            self.on_token_callback(token)

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"🔄 New token: '{token}'")


class EmbeddingCallbackHandler(BaseCallbackHandler):
    """Specialized callback handler for embedding operations"""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.Embeddings")
        self.embedding_metrics = {
            "calls": 0,
            "total_texts": 0,
            "total_time": 0.0,
            "errors": 0
        }
        self._start_time = None

    def on_embedding_start(self, texts: List[str], **kwargs: Any) -> None:
        """Called when embedding starts"""
        self._start_time = time.time()
        self.embedding_metrics["calls"] += 1
        self.embedding_metrics["total_texts"] += len(texts)

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"🔢 Embedding Call #{self.embedding_metrics['calls']} started")
            self.logger.debug(f"Texts to embed: {len(texts)}")

    def on_embedding_end(self, embeddings: List[List[float]], **kwargs: Any) -> None:
        """Called when embedding ends"""
        if self._start_time:
            call_time = time.time() - self._start_time
            self.embedding_metrics["total_time"] += call_time

            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"✅ Embedding completed in {call_time:.2f}s")
                self.logger.info(f"Generated {len(embeddings)} embeddings")

    def on_embedding_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Called when embedding errors"""
        self.embedding_metrics["errors"] += 1
        self.logger.error(f"❌ Embedding Error: {str(error)}")

    def get_embedding_metrics(self) -> Dict[str, Any]:
        """Get embedding-specific metrics"""
        avg_time = (self.embedding_metrics["total_time"] / self.embedding_metrics["calls"]
                   if self.embedding_metrics["calls"] > 0 else 0)

        return {
            "embedding_calls": self.embedding_metrics["calls"],
            "total_texts_embedded": self.embedding_metrics["total_texts"],
            "total_embedding_time": self.embedding_metrics["total_time"],
            "average_embedding_time": avg_time,
            "embedding_errors": self.embedding_metrics["errors"]
        }


class CallbackManager:
    """Centralized manager for all RAG system callbacks"""

    def __init__(self, log_level: str = "INFO", enable_streaming: bool = True):
        self.rag_callback = RAGSystemCallbackHandler(log_level)
        self.embedding_callback = EmbeddingCallbackHandler()
        self.streaming_callback = StreamingCallbackHandler() if enable_streaming else None

        self.logger = logging.getLogger(f"{__name__}.Manager")

    def get_callbacks(self, include_streaming: bool = False) -> List[BaseCallbackHandler]:
        """Get list of callbacks for LangChain operations"""
        callbacks = [self.rag_callback]

        if include_streaming and self.streaming_callback:
            callbacks.append(self.streaming_callback)

        return callbacks

    def get_embedding_callbacks(self) -> List[BaseCallbackHandler]:
        """Get callbacks specifically for embedding operations"""
        return [self.embedding_callback]

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all callbacks"""
        rag_metrics = self.rag_callback.get_metrics()
        embedding_metrics = self.embedding_callback.get_embedding_metrics()

        return {
            "rag_system": rag_metrics,
            "embeddings": embedding_metrics,
            "combined": {
                "total_operations": rag_metrics["llm_calls"] + embedding_metrics["embedding_calls"],
                "total_errors": rag_metrics["errors"] + embedding_metrics["embedding_errors"],
                "success_rate": self._calculate_success_rate(rag_metrics, embedding_metrics)
            }
        }

    def _calculate_success_rate(self, rag_metrics: Dict, embedding_metrics: Dict) -> float:
        """Calculate overall success rate"""
        total_operations = rag_metrics["llm_calls"] + embedding_metrics["embedding_calls"]
        total_errors = rag_metrics["errors"] + embedding_metrics["embedding_errors"]

        if total_operations == 0:
            return 100.0

        return ((total_operations - total_errors) / total_operations) * 100

    def reset_all_metrics(self):
        """Reset metrics for all callbacks"""
        self.rag_callback.reset_metrics()
        self.embedding_callback.embedding_metrics = {
            "calls": 0,
            "total_texts": 0,
            "total_time": 0.0,
            "errors": 0
        }
        self.logger.info("📊 All callback metrics reset")

    def set_streaming_token_callback(self, callback_fn):
        """Set callback function for streaming tokens"""
        if self.streaming_callback:
            self.streaming_callback.on_token_callback = callback_fn