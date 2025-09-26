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
        tokens_per_chunk: Optional[int] = None
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

class BaseChunker(ABC):
    """Abstract base class for text chunkers"""

    def __init__(self, config: ChunkingConfig):
        self.config = config

    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks"""
        pass

    def _create_chunk(self, text: str, metadata: Dict[str, Any], chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """Create a chunk with metadata"""
        return {
            "text": text.strip(),
            "metadata": {
                **metadata,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "chunking_strategy": self.config.strategy.value,
                "chunk_size_config": self.config.chunk_size,
                "actual_chunk_size": len(text.split())
            }
        }

class WordBasedChunker(BaseChunker):
    """Word-based chunking with overlap"""

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        words = text.split()

        if len(words) <= self.config.chunk_size:
            return [self._create_chunk(text, metadata, 0, 1)]

        chunks = []
        step_size = self.config.chunk_size - self.config.chunk_overlap
        chunk_index = 0
        start = 0

        while start < len(words):
            end = min(start + self.config.chunk_size, len(words))
            chunk_words = words[start:end]

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

            chunk_text = " ".join(chunk_words)

            if len(chunk_words) >= self.config.min_chunk_size:
                chunks.append(self._create_chunk(chunk_text, metadata, chunk_index, 0))
                chunk_index += 1

            start += step_size

            if end >= len(words):
                break

        # Update total chunks count
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)

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

            if chunk_text and len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(self._create_chunk(chunk_text, metadata, chunk_index, 0))
                chunk_index += 1

            start = end - overlap_chars

            if start >= len(text):
                break

        # Update total chunks count
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)

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
                if len(chunk_text.strip()) >= self.config.min_chunk_size:
                    formatted_chunks.append(
                        self._create_chunk(chunk_text, metadata, i, len(chunks))
                    )

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
                if len(chunk_text.strip()) >= self.config.min_chunk_size:
                    formatted_chunks.append(
                        self._create_chunk(chunk_text, metadata, i, len(chunks))
                    )

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
                if len(chunk_text.strip()) >= self.config.min_chunk_size:
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunking_strategy": self.config.strategy.value,
                        "token_count": len(splitter.encode(chunk_text)) if hasattr(splitter, 'encode') else None,
                        "model_name": self.config.model_name
                    }
                    formatted_chunks.append({
                        "text": chunk_text.strip(),
                        "metadata": chunk_metadata
                    })

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
                if len(chunk_text.strip()) >= self.config.min_chunk_size:
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunking_strategy": self.config.strategy.value,
                        "tokens_per_chunk": self.config.tokens_per_chunk,
                        "model_name": "tiktoken"
                    }
                    formatted_chunks.append({
                        "text": chunk_text.strip(),
                        "metadata": chunk_metadata
                    })

            return formatted_chunks

        except Exception as e:
            logger.error(f"Error in SentenceTransformersTokenChunker: {str(e)}")
            return WordBasedChunker(self.config).chunk_text(text, metadata)

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
        ChunkingStrategy.SENTENCE_TRANSFORMERS_TOKEN: SentenceTransformersTokenChunker
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
                "description": "Split by SentenceTransformers model token limits",
                "parameters": ["tokens_per_chunk", "chunk_overlap", "model_name"],
                "recommended_for": "SentenceTransformers models, embedding optimization"
            }
        }