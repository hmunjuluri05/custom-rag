"""
Comprehensive test script for Embedding models.

This script tests embedding models independently to ensure they are:
1. Properly abstracted and loosely coupled
2. Independently testable
3. Following the interface contracts
4. Working correctly with different providers

Usage:
    python test_embedding_models.py
    python test_embedding_models.py --provider openai
    python test_embedding_models.py --provider google
    python test_embedding_models.py --mock-only  # Test with mocks only
"""

import asyncio
import sys
import os
import argparse
import time
import numpy as np
from unittest.mock import Mock
from typing import Dict, Any, List

# Add project root to path
sys.path.append('.')

from src.embedding.interfaces import IEmbeddingModel, IEmbeddingModelFactory
from src.embedding.models import EmbeddingModelFactory, EmbeddingService, OpenAIEmbeddingModel, GoogleEmbeddingModel


class MockEmbeddingModel(IEmbeddingModel):
    """Mock embedding model for testing without external dependencies"""

    def __init__(self, model_name: str = "mock-embedding", dimension: int = 768):
        self.model_name = model_name
        self.dimension = dimension
        self.call_count = 0
        self.last_texts = None

    async def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        self.call_count += 1
        self.last_texts = texts
        # Generate deterministic mock embeddings
        embeddings = []
        for i, text in enumerate(texts):
            # Create a simple deterministic embedding based on text hash
            hash_val = hash(text) % 1000
            embedding = np.random.RandomState(hash_val).random(self.dimension)
            # Normalize to unit vector
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        return np.array(embeddings)

    async def encode_single(self, text: str, **kwargs) -> np.ndarray:
        result = await self.encode([text], **kwargs)
        return result[0]

    def get_dimension(self) -> int:
        return self.dimension

    def get_model_name(self) -> str:
        return self.model_name

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "mock",
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_input_tokens": 8192,
            "description": "Mock embedding model for testing",
            "use_cases": ["testing", "development"],
            "call_count": self.call_count
        }

    def validate_connection(self) -> bool:
        return True

    def get_max_batch_size(self) -> int:
        return 100

    def get_max_input_length(self) -> int:
        return 8192

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        # Calculate cosine similarity
        return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))


class EmbeddingModelTester:
    """Comprehensive tester for embedding models"""

    def __init__(self):
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }

    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {test_name}")
        if details:
            print(f"    {details}")

        if passed:
            self.test_results["passed"] += 1
        else:
            self.test_results["failed"] += 1
            self.test_results["errors"].append(f"{test_name}: {details}")

    async def test_embedding_model_interface_compliance(self, model: IEmbeddingModel, model_name: str):
        """Test that a model properly implements the IEmbeddingModel interface"""
        print(f"\nTesting Interface Compliance: {model_name}")
        print("-" * 50)

        # Test 1: Basic encoding
        try:
            test_texts = ["Hello world", "This is a test", "Embedding models are useful"]
            embeddings = await model.encode(test_texts)

            is_valid = (
                isinstance(embeddings, np.ndarray) and
                embeddings.shape[0] == len(test_texts) and
                embeddings.shape[1] == model.get_dimension()
            )

            self.log_test(
                "Batch Encoding",
                is_valid,
                f"Shape: {embeddings.shape}, Expected: ({len(test_texts)}, {model.get_dimension()})"
            )
        except Exception as e:
            self.log_test("Batch Encoding", False, f"Error: {str(e)}")

        # Test 2: Single text encoding
        try:
            single_embedding = await model.encode_single("Single test text")

            is_valid = (
                isinstance(single_embedding, np.ndarray) and
                single_embedding.shape == (model.get_dimension(),)
            )

            self.log_test(
                "Single Encoding",
                is_valid,
                f"Shape: {single_embedding.shape}, Expected: ({model.get_dimension()},)"
            )
        except Exception as e:
            self.log_test("Single Encoding", False, f"Error: {str(e)}")

        # Test 3: Dimension consistency
        try:
            dimension = model.get_dimension()
            self.log_test(
                "Dimension Info",
                isinstance(dimension, int) and dimension > 0,
                f"Dimension: {dimension}"
            )
        except Exception as e:
            self.log_test("Dimension Info", False, f"Error: {str(e)}")

        # Test 4: Model info
        try:
            info = model.get_model_info()
            required_fields = ["provider", "model_name", "dimension"]
            has_required = all(field in info for field in required_fields)

            self.log_test(
                "Model Info",
                isinstance(info, dict) and has_required,
                f"Info keys: {list(info.keys())}"
            )
        except Exception as e:
            self.log_test("Model Info", False, f"Error: {str(e)}")

        # Test 5: Connection validation
        try:
            is_valid = model.validate_connection()
            self.log_test(
                "Connection Validation",
                isinstance(is_valid, bool),
                f"Connection valid: {is_valid}"
            )
        except Exception as e:
            self.log_test("Connection Validation", False, f"Error: {str(e)}")

        # Test 6: Similarity calculation
        try:
            emb1 = await model.encode_single("Test text one")
            emb2 = await model.encode_single("Test text two")
            emb3 = await model.encode_single("Test text one")  # Same as emb1

            sim_different = model.calculate_similarity(emb1, emb2)
            sim_same = model.calculate_similarity(emb1, emb3)

            is_valid = (
                isinstance(sim_different, float) and
                isinstance(sim_same, float) and
                0 <= sim_different <= 1 and
                0 <= sim_same <= 1 and
                sim_same > sim_different  # Same text should be more similar
            )

            self.log_test(
                "Similarity Calculation",
                is_valid,
                f"Same text similarity: {sim_same:.3f}, Different: {sim_different:.3f}"
            )
        except Exception as e:
            self.log_test("Similarity Calculation", False, f"Error: {str(e)}")

    async def test_embedding_consistency(self, model: IEmbeddingModel, model_name: str):
        """Test embedding consistency and properties"""
        print(f"\nTesting Embedding Consistency: {model_name}")
        print("-" * 50)

        # Test 1: Deterministic embeddings
        try:
            text = "Consistent test text"
            emb1 = await model.encode_single(text)
            emb2 = await model.encode_single(text)

            # Check if embeddings are the same (or very close for some models)
            similarity = model.calculate_similarity(emb1, emb2)
            is_consistent = similarity > 0.99

            self.log_test(
                "Deterministic Embeddings",
                is_consistent,
                f"Self-similarity: {similarity:.6f}"
            )
        except Exception as e:
            self.log_test("Deterministic Embeddings", False, f"Error: {str(e)}")

        # Test 2: Embedding normalization
        try:
            embeddings = await model.encode(["Test text for normalization"])
            embedding = embeddings[0]
            norm = np.linalg.norm(embedding)

            # Many embedding models produce unit vectors
            is_normalized = 0.9 <= norm <= 1.1

            self.log_test(
                "Embedding Normalization",
                is_normalized,
                f"L2 norm: {norm:.6f}"
            )
        except Exception as e:
            self.log_test("Embedding Normalization", False, f"Error: {str(e)}")

        # Test 3: Semantic similarity
        try:
            similar_texts = ["The cat sits on the mat", "A cat is sitting on a mat"]
            different_texts = ["The cat sits on the mat", "Financial markets are volatile"]

            similar_embs = await model.encode(similar_texts)
            different_embs = await model.encode(different_texts)

            sim_similar = model.calculate_similarity(similar_embs[0], similar_embs[1])
            sim_different = model.calculate_similarity(different_embs[0], different_embs[1])

            # Similar texts should have higher similarity than different texts
            is_semantic = sim_similar > sim_different

            self.log_test(
                "Semantic Similarity",
                is_semantic,
                f"Similar texts: {sim_similar:.3f}, Different texts: {sim_different:.3f}"
            )
        except Exception as e:
            self.log_test("Semantic Similarity", False, f"Error: {str(e)}")

    async def test_batch_processing(self, model: IEmbeddingModel, model_name: str):
        """Test batch processing capabilities"""
        print(f"\nTesting Batch Processing: {model_name}")
        print("-" * 50)

        # Test 1: Different batch sizes
        try:
            texts = [f"Test text number {i}" for i in range(10)]

            # Test single item
            single_emb = await model.encode([texts[0]])

            # Test small batch
            small_batch = await model.encode(texts[:3])

            # Test larger batch
            large_batch = await model.encode(texts)

            all_valid = (
                single_emb.shape == (1, model.get_dimension()) and
                small_batch.shape == (3, model.get_dimension()) and
                large_batch.shape == (10, model.get_dimension())
            )

            self.log_test(
                "Batch Size Handling",
                all_valid,
                f"Shapes: {single_emb.shape}, {small_batch.shape}, {large_batch.shape}"
            )
        except Exception as e:
            self.log_test("Batch Size Handling", False, f"Error: {str(e)}")

        # Test 2: Empty input handling
        try:
            empty_result = await model.encode([])
            is_valid = isinstance(empty_result, np.ndarray) and empty_result.shape[0] == 0

            self.log_test(
                "Empty Input Handling",
                is_valid,
                f"Empty result shape: {empty_result.shape}"
            )
        except Exception as e:
            self.log_test("Empty Input Handling", False, f"Error: {str(e)}")

        # Test 3: Maximum batch size respect
        try:
            max_batch = model.get_max_batch_size()
            is_valid = isinstance(max_batch, int) and max_batch > 0

            self.log_test(
                "Max Batch Size Info",
                is_valid,
                f"Max batch size: {max_batch}"
            )
        except Exception as e:
            self.log_test("Max Batch Size Info", False, f"Error: {str(e)}")

    async def test_performance_characteristics(self, model: IEmbeddingModel, model_name: str):
        """Test performance characteristics"""
        print(f"\nTesting Performance: {model_name}")
        print("-" * 50)

        # Test 1: Encoding speed
        try:
            texts = ["Performance test text"] * 5
            start_time = time.time()
            embeddings = await model.encode(texts)
            encoding_time = time.time() - start_time

            # Should encode within reasonable time (10 seconds for 5 texts)
            is_fast_enough = encoding_time < 10.0

            self.log_test(
                "Encoding Speed",
                is_fast_enough,
                f"Time for 5 texts: {encoding_time:.2f}s"
            )
        except Exception as e:
            self.log_test("Encoding Speed", False, f"Error: {str(e)}")

        # Test 2: Memory efficiency (check if embeddings are reasonable size)
        try:
            texts = ["Memory test text"] * 10
            embeddings = await model.encode(texts)

            # Calculate memory usage (rough estimate)
            memory_bytes = embeddings.nbytes
            memory_mb = memory_bytes / (1024 * 1024)

            # Should be reasonable (less than 100MB for 10 embeddings)
            is_memory_efficient = memory_mb < 100

            self.log_test(
                "Memory Efficiency",
                is_memory_efficient,
                f"Memory for 10 embeddings: {memory_mb:.2f}MB"
            )
        except Exception as e:
            self.log_test("Memory Efficiency", False, f"Error: {str(e)}")

    async def test_error_handling(self, model: IEmbeddingModel, model_name: str):
        """Test error handling capabilities"""
        print(f"\nTesting Error Handling: {model_name}")
        print("-" * 50)

        # Test 1: Very long text handling
        try:
            max_length = model.get_max_input_length()
            long_text = "Very long text. " * (max_length // 10)  # Create long text

            if len(long_text) > max_length and not isinstance(model, MockEmbeddingModel):
                # Should either handle gracefully or give clear error
                try:
                    result = await model.encode([long_text])
                    self.log_test(
                        "Long Text Handling",
                        True,
                        "Handled long text gracefully"
                    )
                except Exception as e:
                    # Expected to fail, but should give clear error
                    error_is_clear = any(word in str(e).lower() for word in ['length', 'token', 'limit', 'exceed'])
                    self.log_test(
                        "Long Text Handling",
                        error_is_clear,
                        f"Clear error for long text: {str(e)[:100]}"
                    )
            else:
                # Mock or short text - should work
                result = await model.encode([long_text[:100]])
                self.log_test(
                    "Long Text Handling",
                    isinstance(result, np.ndarray),
                    "Handled text appropriately"
                )

        except Exception as e:
            self.log_test("Long Text Handling", False, f"Unexpected error: {str(e)}")

        # Test 2: Special characters
        try:
            special_texts = ["Text with émojis 😀🌟", "Special chars: @#$%^&*()", "Unicode: こんにちは"]
            result = await model.encode(special_texts)

            self.log_test(
                "Special Characters",
                isinstance(result, np.ndarray) and result.shape[0] == len(special_texts),
                "Handled special characters correctly"
            )
        except Exception as e:
            self.log_test("Special Characters", False, f"Error: {str(e)}")

    def test_factory_functionality(self):
        """Test embedding factory functionality"""
        print(f"\nTesting Factory Functionality")
        print("-" * 50)

        # Test 1: Available models
        try:
            models = EmbeddingModelFactory.get_available_models()
            self.log_test(
                "Available Models",
                isinstance(models, dict) and len(models) > 0,
                f"Found {len(models)} models"
            )
        except Exception as e:
            self.log_test("Available Models", False, f"Error: {str(e)}")

        # Test 2: Recommended model
        try:
            recommended = EmbeddingModelFactory.get_recommended_model()
            self.log_test(
                "Recommended Model",
                isinstance(recommended, str) and len(recommended) > 0,
                f"Recommended: {recommended}"
            )
        except Exception as e:
            self.log_test("Recommended Model", False, f"Error: {str(e)}")

        # Test 3: Factory interface compliance
        try:
            factory_methods = [
                hasattr(EmbeddingModelFactory, 'create_model'),
                hasattr(EmbeddingModelFactory, 'get_available_models'),
                hasattr(EmbeddingModelFactory, 'get_recommended_model')
            ]

            self.log_test(
                "Factory Interface",
                all(factory_methods),
                "Factory has required methods"
            )
        except Exception as e:
            self.log_test("Factory Interface", False, f"Error: {str(e)}")

    async def test_embedding_service_integration(self, service: EmbeddingService, service_name: str):
        """Test EmbeddingService integration"""
        print(f"\nTesting Service Integration: {service_name}")
        print("-" * 50)

        # Test 1: Text encoding through service
        try:
            texts = ["Service test text 1", "Service test text 2"]
            embeddings = await service.encode_texts(texts)

            is_valid = (
                isinstance(embeddings, list) and
                len(embeddings) == len(texts) and
                all(isinstance(emb, np.ndarray) for emb in embeddings)
            )

            self.log_test(
                "Service Text Encoding",
                is_valid,
                f"Encoded {len(embeddings)} texts through service"
            )
        except Exception as e:
            self.log_test("Service Text Encoding", False, f"Error: {str(e)}")

        # Test 2: Similarity search
        try:
            query = "Search query text"
            candidates = ["Relevant text about searching", "Unrelated text about cooking", "Another search result"]

            results = await service.find_similar_texts(query, candidates, top_k=2)

            is_valid = (
                isinstance(results, list) and
                len(results) <= 2 and
                all("similarity_score" in result for result in results) and
                all("text" in result for result in results)
            )

            self.log_test(
                "Similarity Search",
                is_valid,
                f"Found {len(results)} similar texts"
            )
        except Exception as e:
            self.log_test("Similarity Search", False, f"Error: {str(e)}")

        # Test 3: Model info through service
        try:
            info = service.get_model_info()
            self.log_test(
                "Service Model Info",
                isinstance(info, dict) and "model_name" in info,
                f"Model: {info.get('model_name', 'Unknown')}"
            )
        except Exception as e:
            self.log_test("Service Model Info", False, f"Error: {str(e)}")

    async def run_mock_tests(self):
        """Run tests with mock models"""
        print("=" * 60)
        print("TESTING WITH MOCK MODELS")
        print("=" * 60)

        # Test different mock configurations
        mock_models = [
            MockEmbeddingModel("mock-small", 384),
            MockEmbeddingModel("mock-large", 1536),
            MockEmbeddingModel("mock-huge", 3072)
        ]

        for mock_model in mock_models:
            await self.test_embedding_model_interface_compliance(mock_model, mock_model.get_model_name())
            await self.test_embedding_consistency(mock_model, mock_model.get_model_name())
            await self.test_batch_processing(mock_model, mock_model.get_model_name())
            await self.test_performance_characteristics(mock_model, mock_model.get_model_name())
            await self.test_error_handling(mock_model, mock_model.get_model_name())

        # Test mock service
        mock_service = EmbeddingService.__new__(EmbeddingService)  # Create without __init__
        mock_service.model = mock_models[0]
        mock_service.model_name = mock_models[0].get_model_name()

        await self.test_embedding_service_integration(mock_service, "Mock Service")

    async def run_real_tests(self, provider: str = None):
        """Run tests with real models (requires API keys)"""
        print("=" * 60)
        print(f"TESTING WITH REAL MODELS{f' ({provider.upper()})' if provider else ''}")
        print("=" * 60)

        # Check for environment variables
        api_key = os.getenv('API_KEY')
        base_url = os.getenv('BASE_URL')

        if not api_key:
            print("WARNING: No API_KEY found in environment. Skipping real model tests.")
            print("Set API_KEY environment variable to test with real models.")
            return

        # Test factory functionality
        self.test_factory_functionality()

        # Get available models to test
        try:
            available_models = EmbeddingModelFactory.get_available_models()
            models_to_test = []

            if provider:
                # Filter by provider
                models_to_test = [
                    name for name, info in available_models.items()
                    if info.get('provider', '').lower() == provider.lower()
                ]
            else:
                # Test a few representative models
                models_to_test = list(available_models.keys())[:3]  # Test first 3 models

            for model_name in models_to_test:
                try:
                    print(f"\nTesting model: {model_name}")

                    # Create model
                    model = EmbeddingModelFactory.create_model(
                        model_name=model_name,
                        api_key=api_key,
                        base_url=base_url
                    )

                    await self.test_embedding_model_interface_compliance(model, model_name)
                    await self.test_embedding_consistency(model, model_name)
                    await self.test_batch_processing(model, model_name)
                    await self.test_performance_characteristics(model, model_name)
                    await self.test_error_handling(model, model_name)

                    # Test service
                    service = EmbeddingService(
                        model_name=model_name,
                        api_key=api_key,
                        base_url=base_url
                    )
                    await self.test_embedding_service_integration(service, f"{model_name} Service")

                except Exception as e:
                    self.log_test(f"{model_name} Model Creation", False, f"Error: {str(e)}")

        except Exception as e:
            self.log_test("Real Model Testing", False, f"Error getting available models: {str(e)}")

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        total_tests = self.test_results["passed"] + self.test_results["failed"]
        pass_rate = (self.test_results["passed"] / total_tests * 100) if total_tests > 0 else 0

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.test_results['passed']}")
        print(f"Failed: {self.test_results['failed']}")
        print(f"Pass Rate: {pass_rate:.1f}%")

        if self.test_results["errors"]:
            print(f"\nFailures:")
            for error in self.test_results["errors"]:
                print(f"  - {error}")

        print(f"\nEmbedding Models are {'PROPERLY' if pass_rate >= 80 else 'NOT PROPERLY'} abstracted and testable!")


async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Test Embedding models")
    parser.add_argument('--provider', choices=['openai', 'google'], help='Test specific provider')
    parser.add_argument('--mock-only', action='store_true', help='Test with mocks only')
    parser.add_argument('--real-only', action='store_true', help='Test with real models only')

    args = parser.parse_args()

    tester = EmbeddingModelTester()

    print("Embedding Models Independence and Testability Test")
    print("=" * 60)

    if not args.real_only:
        await tester.run_mock_tests()

    if not args.mock_only:
        await tester.run_real_tests(args.provider)

    tester.print_summary()

    # Exit with error code if tests failed
    if tester.test_results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())