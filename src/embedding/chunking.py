from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import re
import logging
from enum import Enum

# LangChain imports
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    # LangChain-based strategies
    RECURSIVE_CHARACTER = "recursive_character"  # Most popular and effective
    CHARACTER = "character"
    TOKEN_BASED = "token_based"
    SENTENCE_TRANSFORMERS_TOKEN = "sentence_transformers_token"

    # Custom strategies (legacy)
    WORD_BASED = "word_based"
    SENTENCE_BASED = "sentence_based"
    PARAGRAPH_BASED = "paragraph_based"
    SEMANTIC_BASED = "semantic_based"
    FIXED_SIZE = "fixed_size"

    # LLM-based intelligent chunking strategies
    LLM_SEMANTIC = "llm_semantic"  # Full LLM-based semantic chunking (with optional metadata)
    LLM_ENHANCED = "llm_enhanced"  # Hybrid: rule-based + LLM refinement (with optional metadata)

class ChunkingConfig:
    """Configuration for chunking strategies"""

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        max_chunk_size: int = 4000,
        preserve_sentences: bool = True,
        preserve_paragraphs: bool = False,
        # LangChain-specific parameters
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        strip_whitespace: bool = True,
        # Token-based parameters
        model_name: str = "gpt-3.5-turbo",
        encoding_name: Optional[str] = None,
        # Token-based chunking parameters
        tokens_per_chunk: Optional[int] = None,
        # LLM-based chunking parameters
        llm_service: Any = None,
        document_type: Optional[str] = None,  # e.g., "legal", "technical", "financial"
        metadata_detail: str = "basic"  # "basic", "detailed", "comprehensive"
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.preserve_sentences = preserve_sentences
        self.preserve_paragraphs = preserve_paragraphs

        # LangChain-specific
        self.separators = separators
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace

        # Token-based
        self.model_name = model_name
        self.encoding_name = encoding_name

        # Token-based chunking
        self.tokens_per_chunk = tokens_per_chunk or chunk_size

        # LLM-based chunking
        self.llm_service = llm_service
        self.document_type = document_type
        self.metadata_detail = metadata_detail

class BaseChunker(ABC):
    """Abstract base class for text chunkers"""

    def __init__(self, config: ChunkingConfig):
        self.config = config

    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks"""
        pass

    def _create_chunk(self, text: str, metadata: Dict[str, Any], chunk_index: int, total_chunks: int, start_word: int = None, end_word: int = None) -> Dict[str, Any]:
        """Create a chunk with metadata"""
        word_count = len(text.split())

        # Use None to indicate positions should be calculated later
        # If explicitly passed (even as 0), use those values
        return {
            "text": text.strip(),
            "metadata": {
                **metadata,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "chunking_strategy": self.config.strategy.value,
                "chunk_size_config": self.config.chunk_size,
                "actual_chunk_size": word_count,
                "start_word": start_word,
                "end_word": end_word
            }
        }

    def _calculate_word_positions(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate cumulative word positions for chunks that don't have them"""
        cumulative_position = 0

        for chunk in chunks:
            metadata = chunk["metadata"]
            word_count = metadata["actual_chunk_size"]

            # Only calculate if positions weren't explicitly set
            if metadata.get("start_word") is None:
                metadata["start_word"] = cumulative_position
                metadata["end_word"] = cumulative_position + word_count
                cumulative_position += word_count

        return chunks

class WordBasedChunker(BaseChunker):
    """Word-based chunking with overlap"""

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        words = text.split()

        if len(words) <= self.config.chunk_size:
            return [self._create_chunk(text, metadata, 0, 1, start_word=0, end_word=len(words))]

        chunks = []
        step_size = self.config.chunk_size - self.config.chunk_overlap
        chunk_index = 0
        start = 0

        while start < len(words):
            end = min(start + self.config.chunk_size, len(words))
            chunk_words = words[start:end]
            actual_end = end  # Track actual end position

            # Preserve sentence boundaries if enabled
            if self.config.preserve_sentences and end < len(words):
                chunk_text = " ".join(chunk_words)
                last_sentence_end = max(
                    chunk_text.rfind('.'),
                    chunk_text.rfind('!'),
                    chunk_text.rfind('?')
                )

                if last_sentence_end > len(chunk_text) * 0.5:  # Only if we find a sentence end in the latter half
                    chunk_text = chunk_text[:last_sentence_end + 1]
                    chunk_words = chunk_text.split()
                    actual_end = start + len(chunk_words)

            chunk_text = " ".join(chunk_words)

            # Add chunk if it has content (don't enforce min_chunk_size for small documents)
            if chunk_words:
                chunks.append(self._create_chunk(chunk_text, metadata, chunk_index, 0, start_word=start, end_word=actual_end))
                chunk_index += 1

            start += step_size

            # Prevent infinite loop - ensure we make progress
            if step_size <= 0:
                break

            if end >= len(words):
                break

        # Update total chunks count
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)

        # Calculate cumulative word positions
        self._calculate_word_positions(chunks)

        return chunks

class SentenceBasedChunker(BaseChunker):
    """Sentence-based chunking"""

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be improved with spaCy or NLTK
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If adding this sentence would exceed the limit, finalize current chunk
            if current_word_count + sentence_words > self.config.chunk_size and current_chunk:
                chunk_text = ". ".join(current_chunk) + "."
                chunks.append(self._create_chunk(chunk_text, metadata, chunk_index, 0))
                chunk_index += 1

                # Start new chunk with overlap
                overlap_sentences = max(1, len(current_chunk) // 4)  # 25% overlap
                current_chunk = current_chunk[-overlap_sentences:] if self.config.chunk_overlap > 0 else []
                current_word_count = sum(len(s.split()) for s in current_chunk)

            current_chunk.append(sentence)
            current_word_count += sentence_words

        # Add remaining sentences as final chunk
        if current_chunk:
            chunk_text = ". ".join(current_chunk) + "."
            chunks.append(self._create_chunk(chunk_text, metadata, chunk_index, 0))

        # Update total chunks count
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)

        # Calculate cumulative word positions
        self._calculate_word_positions(chunks)

        return chunks

class ParagraphBasedChunker(BaseChunker):
    """Paragraph-based chunking"""

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph_words = len(paragraph.split())

            # If this single paragraph is too large, split it with word-based chunking
            if paragraph_words > self.config.max_chunk_size:
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, metadata, chunk_index, 0))
                    chunk_index += 1
                    current_chunk = []
                    current_word_count = 0

                # Use word-based chunking for large paragraph
                word_chunker = WordBasedChunker(self.config)
                para_chunks = word_chunker.chunk_text(paragraph, metadata)
                for chunk in para_chunks:
                    chunk["metadata"]["chunk_index"] = chunk_index
                    chunks.append(chunk)
                    chunk_index += 1
                continue

            # If adding this paragraph would exceed the limit, finalize current chunk
            if current_word_count + paragraph_words > self.config.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, metadata, chunk_index, 0))
                chunk_index += 1

                # Start new chunk with overlap
                if self.config.chunk_overlap > 0 and current_chunk:
                    current_chunk = current_chunk[-1:]  # Keep last paragraph for overlap
                    current_word_count = len(current_chunk[0].split())
                else:
                    current_chunk = []
                    current_word_count = 0

            current_chunk.append(paragraph)
            current_word_count += paragraph_words

        # Add remaining paragraphs as final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, metadata, chunk_index, 0))

        # Update total chunks count
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)

        # Calculate cumulative word positions
        self._calculate_word_positions(chunks)

        return chunks

class FixedSizeChunker(BaseChunker):
    """Fixed character-size chunking"""

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        # Use character-based chunking with word boundaries
        chunk_size_chars = self.config.chunk_size * 5  # Roughly 5 chars per word
        overlap_chars = self.config.chunk_overlap * 5

        chunks = []
        chunk_index = 0
        start = 0

        while start < len(text):
            end = min(start + chunk_size_chars, len(text))

            # Try to end at word boundary
            if end < len(text) and not text[end].isspace():
                # Find last space before end
                last_space = text.rfind(' ', start, end)
                if last_space > start + chunk_size_chars * 0.5:  # Only if reasonable position
                    end = last_space

            chunk_text = text[start:end].strip()

            # Add chunk if it has content
            if chunk_text:
                chunks.append(self._create_chunk(chunk_text, metadata, chunk_index, 0))
                chunk_index += 1

            # Move start forward, ensuring progress
            new_start = end - overlap_chars
            if new_start <= start:  # Prevent infinite loop
                new_start = start + 1
            start = new_start

            if start >= len(text):
                break

        # Update total chunks count
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)

        # Calculate cumulative word positions
        self._calculate_word_positions(chunks)

        return chunks

# LangChain-based chunkers (defined before factory to avoid NameError)
class RecursiveCharacterChunker(BaseChunker):
    """LangChain RecursiveCharacterTextSplitter wrapper"""

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        try:
            # Default separators for recursive splitting
            separators = self.config.separators or ["\n\n", "\n", " ", ""]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                separators=separators,
                keep_separator=self.config.keep_separator,
                strip_whitespace=self.config.strip_whitespace
            )

            chunks = splitter.split_text(text)

            # Convert to our format
            formatted_chunks = []
            for i, chunk_text in enumerate(chunks):
                # Skip empty chunks
                if chunk_text.strip():
                    formatted_chunks.append(
                        self._create_chunk(chunk_text, metadata, i, len(chunks))
                    )

            # Calculate cumulative word positions
            self._calculate_word_positions(formatted_chunks)

            return formatted_chunks

        except Exception as e:
            logger.error(f"Error in RecursiveCharacterChunker: {str(e)}")
            # Fallback to simple word-based chunking
            return WordBasedChunker(self.config).chunk_text(text, metadata)

class CharacterChunker(BaseChunker):
    """LangChain CharacterTextSplitter wrapper"""

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        try:
            separator = self.config.separators[0] if self.config.separators else "\n\n"

            splitter = CharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separator=separator,
                length_function=len,
                keep_separator=self.config.keep_separator,
                strip_whitespace=self.config.strip_whitespace
            )

            chunks = splitter.split_text(text)

            formatted_chunks = []
            for i, chunk_text in enumerate(chunks):
                # Skip empty chunks
                if chunk_text.strip():
                    formatted_chunks.append(
                        self._create_chunk(chunk_text, metadata, i, len(chunks))
                    )

            # Calculate cumulative word positions
            self._calculate_word_positions(formatted_chunks)

            return formatted_chunks

        except Exception as e:
            logger.error(f"Error in CharacterChunker: {str(e)}")
            return WordBasedChunker(self.config).chunk_text(text, metadata)

class TokenBasedChunker(BaseChunker):
    """LangChain TokenTextSplitter wrapper"""

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        try:
            splitter = TokenTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                model_name=self.config.model_name,
                encoding_name=self.config.encoding_name
            )

            chunks = splitter.split_text(text)

            formatted_chunks = []
            for i, chunk_text in enumerate(chunks):
                # Skip empty chunks
                if chunk_text.strip():
                    chunk = self._create_chunk(chunk_text, metadata, i, len(chunks))
                    # Add token-specific metadata
                    chunk["metadata"]["token_count"] = len(splitter.encode(chunk_text)) if hasattr(splitter, 'encode') else None
                    chunk["metadata"]["model_name"] = self.config.model_name
                    formatted_chunks.append(chunk)

            # Calculate cumulative word positions
            self._calculate_word_positions(formatted_chunks)

            return formatted_chunks

        except Exception as e:
            logger.error(f"Error in TokenBasedChunker: {str(e)}")
            return WordBasedChunker(self.config).chunk_text(text, metadata)

class SentenceTransformersTokenChunker(BaseChunker):
    """LangChain SentenceTransformersTokenTextSplitter wrapper"""

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        try:
            # SentenceTransformers not available - fallback to tiktoken
            from langchain_text_splitters import TokenTextSplitter
            splitter = TokenTextSplitter(
                chunk_size=self.config.tokens_per_chunk,
                chunk_overlap=self.config.chunk_overlap
            )

            chunks = splitter.split_text(text)

            formatted_chunks = []
            for i, chunk_text in enumerate(chunks):
                # Skip empty chunks
                if chunk_text.strip():
                    chunk = self._create_chunk(chunk_text, metadata, i, len(chunks))
                    # Add token-specific metadata
                    chunk["metadata"]["tokens_per_chunk"] = self.config.tokens_per_chunk
                    chunk["metadata"]["model_name"] = "tiktoken"
                    formatted_chunks.append(chunk)

            # Calculate cumulative word positions
            self._calculate_word_positions(formatted_chunks)

            return formatted_chunks

        except Exception as e:
            logger.error(f"Error in SentenceTransformersTokenChunker: {str(e)}")
            return WordBasedChunker(self.config).chunk_text(text, metadata)


# LLM-Based Chunking Implementations

class LLMSemanticChunker(BaseChunker):
    """Full LLM-based semantic chunking - analyzes document and creates optimal chunks with optional metadata enrichment"""

    async def chunk_text_async(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Async version of chunk_text for LLM calls"""
        if not text.strip():
            return []

        if not self.config.llm_service:
            logger.warning("No LLM service provided for LLM Semantic Chunking, falling back to recursive character")
            fallback_config = ChunkingConfig(strategy=ChunkingStrategy.RECURSIVE_CHARACTER, chunk_size=self.config.chunk_size)
            return RecursiveCharacterChunker(fallback_config).chunk_text(text, metadata)

        try:
            # Prepare document type hint
            doc_type = self.config.document_type or "general"
            detail_level = self.config.metadata_detail or "basic"

            # Create LLM prompt for semantic chunking with metadata based on detail level
            if detail_level == "none":
                metadata_instruction = "- A brief title (1 line)"
            elif detail_level == "basic":
                metadata_instruction = """- A brief title (1 line)
   - Key topics/keywords (3-5 words)"""
            elif detail_level == "detailed":
                metadata_instruction = """- A descriptive title (1 line)
   - Key topics/keywords (5-7 words)
   - Main topic/category
   - Key entities (people, places, organizations)"""
            else:  # comprehensive
                metadata_instruction = """- A descriptive title (1 line)
   - Key topics/keywords (7-10 words)
   - Main topic/category
   - Key entities (people, places, organizations)
   - Sentiment (positive/negative/neutral)
   - Important facts or figures"""

            # Create LLM prompt for semantic chunking
            prompt = f"""Analyze this document and split it into semantically meaningful chunks.

Document Type: {doc_type}
Target Chunk Size: {self.config.chunk_size} words (flexible based on semantic boundaries)
Min Chunk Size: {self.config.min_chunk_size} words
Max Chunk Size: {self.config.max_chunk_size} words

Instructions:
1. Identify natural semantic boundaries (topics, sections, concepts)
2. Keep related information together
3. Preserve context within each chunk
4. For each chunk, provide:
   - The chunk text
{metadata_instruction}

Document:
{text[:8000]}

Respond in JSON format:
{{
  "chunks": [
    {{
      "text": "chunk content here",
      "title": "Brief chunk title"{', "keywords": "keyword1, keyword2"' if detail_level != "none" else ''}{', "topic": "Main topic", "entities": "entity1, entity2"' if detail_level in ["detailed", "comprehensive"] else ''}{', "sentiment": "neutral", "facts": "key facts"' if detail_level == "comprehensive" else ''}
    }}
  ]
}}"""

            # Call LLM
            context = ""  # No context needed for this task
            response = await self.config.llm_service.generate_response(context, prompt)

            # Parse LLM response
            import json
            try:
                # Try to extract JSON from response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    llm_chunks = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except:
                logger.error(f"Failed to parse LLM response as JSON, falling back")
                fallback_config = ChunkingConfig(strategy=ChunkingStrategy.RECURSIVE_CHARACTER, chunk_size=self.config.chunk_size)
                return RecursiveCharacterChunker(fallback_config).chunk_text(text, metadata)

            # Format chunks
            formatted_chunks = []
            for i, llm_chunk in enumerate(llm_chunks.get("chunks", [])):
                chunk_text = llm_chunk.get("text", "")
                if len(chunk_text.split()) >= self.config.min_chunk_size:
                    chunk = self._create_chunk(chunk_text, metadata, i, len(llm_chunks.get("chunks", [])))
                    # Add LLM-generated metadata based on detail level
                    chunk["metadata"]["llm_title"] = llm_chunk.get("title", "")
                    if detail_level != "none":
                        chunk["metadata"]["llm_keywords"] = llm_chunk.get("keywords", "")
                        chunk["metadata"]["llm_topic"] = llm_chunk.get("topic", "")
                    if detail_level in ["detailed", "comprehensive"]:
                        chunk["metadata"]["llm_entities"] = llm_chunk.get("entities", "")
                    if detail_level == "comprehensive":
                        chunk["metadata"]["llm_sentiment"] = llm_chunk.get("sentiment", "")
                        chunk["metadata"]["llm_facts"] = llm_chunk.get("facts", "")
                    chunk["metadata"]["chunking_method"] = "llm_semantic"
                    chunk["metadata"]["metadata_detail"] = detail_level
                    formatted_chunks.append(chunk)

            # Calculate word positions
            self._calculate_word_positions(formatted_chunks)

            return formatted_chunks

        except Exception as e:
            logger.error(f"Error in LLM Semantic Chunking: {str(e)}")
            # Fallback to recursive character
            fallback_config = ChunkingConfig(strategy=ChunkingStrategy.RECURSIVE_CHARACTER, chunk_size=self.config.chunk_size)
            return RecursiveCharacterChunker(fallback_config).chunk_text(text, metadata)

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Synchronous wrapper - runs async version"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.chunk_text_async(text, metadata))


class LLMEnhancedChunker(BaseChunker):
    """Hybrid chunking: fast rule-based splitting + LLM boundary refinement with optional metadata enrichment"""

    async def chunk_text_async(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Async version for LLM calls"""
        if not text.strip():
            return []

        # Step 1: Fast rule-based chunking
        base_config = ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        base_chunks = RecursiveCharacterChunker(base_config).chunk_text(text, metadata)

        # If no LLM service, return base chunks
        if not self.config.llm_service:
            logger.warning("No LLM service for LLM Enhanced Chunking, returning base chunks")
            return base_chunks

        try:
            # Step 2: LLM refines boundaries
            doc_type = self.config.document_type or "general"
            detail_level = self.config.metadata_detail or "none"

            # Prepare chunk preview for LLM
            chunk_previews = []
            for i, chunk in enumerate(base_chunks[:10]):  # Limit to first 10 for analysis
                preview = chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                chunk_previews.append(f"Chunk {i}: {preview}")

            prompt = f"""Review these document chunks and suggest improvements to chunk boundaries.

Document Type: {doc_type}

Current Chunks:
{chr(10).join(chunk_previews)}

Instructions:
1. Identify if any chunks should be merged (same topic)
2. Identify if any chunks should be split (multiple topics)
3. Suggest better semantic boundaries

Respond with JSON:
{{
  "suggestions": [
    {{"action": "merge", "chunks": [0, 1], "reason": "Same topic"}},
    {{"action": "split", "chunk": 3, "position": 150, "reason": "Topic change"}}
  ]
}}"""

            context = ""
            response = await self.config.llm_service.generate_response(context, prompt)

            # Parse suggestions
            import json
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    suggestions = json.loads(response[json_start:json_end])

                    # Apply suggestions (simplified - merge only for now)
                    for suggestion in suggestions.get("suggestions", [])[:3]:  # Limit to 3 suggestions
                        if suggestion.get("action") == "merge":
                            chunk_ids = suggestion.get("chunks", [])
                            if len(chunk_ids) == 2 and all(0 <= i < len(base_chunks) for i in chunk_ids):
                                # Mark chunks for metadata update
                                base_chunks[chunk_ids[0]]["metadata"]["llm_refined"] = True
                                base_chunks[chunk_ids[0]]["metadata"]["llm_suggestion"] = suggestion.get("reason", "")
            except:
                logger.warning("Could not parse LLM refinement suggestions")

            # Step 3: Optional metadata enrichment
            if detail_level and detail_level != "none":
                await self._enrich_metadata(base_chunks, detail_level)

            # Add refinement metadata
            for chunk in base_chunks:
                chunk["metadata"]["chunking_method"] = "llm_enhanced"
                chunk["metadata"]["metadata_detail"] = detail_level
                if "llm_refined" not in chunk["metadata"]:
                    chunk["metadata"]["llm_refined"] = False

            return base_chunks

        except Exception as e:
            logger.error(f"Error in LLM Enhanced Chunking: {str(e)}")
            return base_chunks

    async def _enrich_metadata(self, chunks: List[Dict[str, Any]], detail_level: str):
        """Enrich chunks with LLM-generated metadata"""
        for i, chunk in enumerate(chunks[:5]):  # Limit to first 5 chunks
            chunk_text = chunk["text"][:1000]

            if detail_level == "basic":
                prompt = f"""Analyze this text chunk and provide:
1. A brief summary (1 sentence)
2. Main keywords (3-5 words)

Chunk: {chunk_text}

Respond in format:
Summary: <summary>
Keywords: <keyword1>, <keyword2>, <keyword3>"""

            elif detail_level == "detailed":
                prompt = f"""Analyze this text chunk and provide:
1. A summary (2-3 sentences)
2. Main keywords (5-7 words)
3. Primary topic/category
4. Key entities (people, places, organizations)

Chunk: {chunk_text}

Respond in format:
Summary: <summary>
Keywords: <keywords>
Topic: <topic>
Entities: <entities>"""

            else:  # comprehensive
                prompt = f"""Analyze this text chunk comprehensively:
1. Summary (3-4 sentences)
2. Keywords (7-10 words)
3. Topic/Category
4. Key entities
5. Sentiment (positive/negative/neutral)
6. Important facts or figures

Chunk: {chunk_text}

Respond in format:
Summary: <summary>
Keywords: <keywords>
Topic: <topic>
Entities: <entities>
Sentiment: <sentiment>
Facts: <facts>"""

            try:
                context = ""
                response = await self.config.llm_service.generate_response(context, prompt)

                # Parse and add metadata
                chunk["metadata"]["llm_summary"] = self._extract_field(response, "Summary")
                chunk["metadata"]["llm_keywords"] = self._extract_field(response, "Keywords")

                if detail_level in ["detailed", "comprehensive"]:
                    chunk["metadata"]["llm_topic"] = self._extract_field(response, "Topic")
                    chunk["metadata"]["llm_entities"] = self._extract_field(response, "Entities")

                if detail_level == "comprehensive":
                    chunk["metadata"]["llm_sentiment"] = self._extract_field(response, "Sentiment")
                    chunk["metadata"]["llm_facts"] = self._extract_field(response, "Facts")
            except Exception as e:
                logger.warning(f"Failed to enrich metadata for chunk {i}: {e}")

    def _extract_field(self, response: str, field_name: str) -> str:
        """Extract a field from LLM response"""
        try:
            lines = response.split('\n')
            for line in lines:
                if line.strip().startswith(f"{field_name}:"):
                    return line.split(':', 1)[1].strip()
            return ""
        except:
            return ""

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Synchronous wrapper"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.chunk_text_async(text, metadata))


# ChunkerFactory - must be defined after all chunker classes

class ChunkerFactory:
    """Factory for creating chunkers"""

    _chunkers = {
        # Custom strategies
        ChunkingStrategy.WORD_BASED: WordBasedChunker,
        ChunkingStrategy.SENTENCE_BASED: SentenceBasedChunker,
        ChunkingStrategy.PARAGRAPH_BASED: ParagraphBasedChunker,
        ChunkingStrategy.FIXED_SIZE: FixedSizeChunker,
        ChunkingStrategy.SEMANTIC_BASED: WordBasedChunker,  # Fallback to word-based for now
        # LangChain strategies
        ChunkingStrategy.RECURSIVE_CHARACTER: RecursiveCharacterChunker,
        ChunkingStrategy.CHARACTER: CharacterChunker,
        ChunkingStrategy.TOKEN_BASED: TokenBasedChunker,
        ChunkingStrategy.SENTENCE_TRANSFORMERS_TOKEN: SentenceTransformersTokenChunker,
        # LLM-based strategies (with optional metadata enrichment)
        ChunkingStrategy.LLM_SEMANTIC: LLMSemanticChunker,
        ChunkingStrategy.LLM_ENHANCED: LLMEnhancedChunker
    }

    @classmethod
    def create_chunker(cls, config: ChunkingConfig) -> BaseChunker:
        """Create a chunker based on configuration"""
        chunker_class = cls._chunkers.get(config.strategy, WordBasedChunker)
        return chunker_class(config)

    @classmethod
    def get_available_strategies(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about available chunking strategies"""
        return {
            ChunkingStrategy.WORD_BASED.value: {
                "name": "Word-based",
                "description": "Split text based on word count with configurable overlap",
                "parameters": ["chunk_size", "chunk_overlap", "preserve_sentences"],
                "recommended_for": "General purpose, balanced performance"
            },
            ChunkingStrategy.SENTENCE_BASED.value: {
                "name": "Sentence-based",
                "description": "Split text at sentence boundaries, preserving semantic meaning",
                "parameters": ["chunk_size", "chunk_overlap"],
                "recommended_for": "Preserving sentence context, Q&A systems"
            },
            ChunkingStrategy.PARAGRAPH_BASED.value: {
                "name": "Paragraph-based",
                "description": "Split text at paragraph boundaries, maintaining topical coherence",
                "parameters": ["chunk_size", "chunk_overlap"],
                "recommended_for": "Structured documents, maintaining topic coherence"
            },
            ChunkingStrategy.FIXED_SIZE.value: {
                "name": "Fixed Character Size",
                "description": "Split text into fixed character-length chunks",
                "parameters": ["chunk_size", "chunk_overlap"],
                "recommended_for": "Consistent chunk sizes, character limits"
            },
            ChunkingStrategy.SEMANTIC_BASED.value: {
                "name": "Semantic-based",
                "description": "Split text based on semantic similarity (advanced)",
                "parameters": ["chunk_size", "chunk_overlap"],
                "recommended_for": "Advanced use cases, semantic coherence"
            },
            # LangChain-based strategies
            ChunkingStrategy.RECURSIVE_CHARACTER.value: {
                "name": "Recursive Character (Recommended)",
                "description": "LangChain's most effective splitter - recursively splits by multiple separators",
                "parameters": ["chunk_size", "chunk_overlap", "separators"],
                "recommended_for": "Most use cases, intelligent splitting, production ready"
            },
            ChunkingStrategy.CHARACTER.value: {
                "name": "Character-based",
                "description": "LangChain's simple character splitter with single separator",
                "parameters": ["chunk_size", "chunk_overlap", "separator"],
                "recommended_for": "Simple splitting, specific separator needs"
            },
            ChunkingStrategy.TOKEN_BASED.value: {
                "name": "Token-based (GPT Models)",
                "description": "Split by token count using tiktoken (GPT-3.5, GPT-4 compatible)",
                "parameters": ["chunk_size", "chunk_overlap", "model_name"],
                "recommended_for": "LLM integration, precise token control, API limits"
            },
            ChunkingStrategy.SENTENCE_TRANSFORMERS_TOKEN.value: {
                "name": "SentenceTransformers Token",
                "description": "Split by SentenceTransers model token limits",
                "parameters": ["tokens_per_chunk", "chunk_overlap", "model_name"],
                "recommended_for": "SentenceTransformers models, embedding optimization"
            },
            # LLM-based strategies (with optional metadata enrichment)
            ChunkingStrategy.LLM_SEMANTIC.value: {
                "name": "LLM Semantic Chunking (AI-Powered)",
                "description": "AI analyzes document structure and creates semantically meaningful chunks with optional metadata enrichment",
                "parameters": ["chunk_size", "document_type", "metadata_detail"],
                "recommended_for": "High-value documents, complex analysis, best quality chunks"
            },
            ChunkingStrategy.LLM_ENHANCED.value: {
                "name": "LLM Enhanced Chunking (Hybrid)",
                "description": "Fast rule-based chunking refined by AI for better boundaries with optional metadata enrichment",
                "parameters": ["chunk_size", "chunk_overlap", "document_type", "metadata_detail"],
                "recommended_for": "Balance between speed and quality, production use"
            }
        }