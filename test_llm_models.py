"""
Comprehensive test script for LLM models.

This script tests LLM models independently to ensure they are:
1. Properly abstracted and loosely coupled
2. Independently testable
3. Following the interface contracts
4. Working correctly with different providers

Usage:
    python test_llm_models.py
    python test_llm_models.py --provider openai
    python test_llm_models.py --provider google
    python test_llm_models.py --mock-only  # Test with mocks only
"""

import asyncio
import sys
import os
import argparse
import time
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.append('.')

from src.llm.interfaces import ILLMModel, ILLMModelFactory, ILLMService
from src.llm.models import LLMFactory, LLMService, OpenAILLMModel, GoogleLLMModel


class MockLLMModel(ILLMModel):
    """Mock LLM model for testing without external dependencies"""

    def __init__(self, model_name: str = "mock-llm", provider: str = "mock"):
        self.model_name = model_name
        self.provider = provider
        self.call_count = 0
        self.last_context = None
        self.last_query = None

    async def generate_response(self, context: str, query: str, **kwargs) -> str:
        self.call_count += 1
        self.last_context = context
        self.last_query = query
        return f"Mock response for query: '{query}' (call #{self.call_count})"

    async def generate_response_stream(self, context: str, query: str, **kwargs):
        self.call_count += 1
        self.last_context = context
        self.last_query = query
        chunks = ["Mock ", "streaming ", f"response for: '{query}'"]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.1)  # Simulate streaming delay

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "description": "Mock LLM model for testing",
            "cost": "Free",
            "framework": "mock",
            "call_count": self.call_count
        }

    def validate_connection(self) -> bool:
        return True

    def get_token_limit(self) -> int:
        return 4096

    def estimate_tokens(self, text: str) -> int:
        # Simple estimation: ~4 characters per token
        return len(text) // 4


class LLMModelTester:
    """Comprehensive tester for LLM models"""

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

    async def test_llm_model_interface_compliance(self, model: ILLMModel, model_name: str):
        """Test that a model properly implements the ILLMModel interface"""
        print(f"\nTesting Interface Compliance: {model_name}")
        print("-" * 50)

        # Test 1: Basic response generation
        try:
            response = await model.generate_response("Test context", "Test query")
            self.log_test(
                "Response Generation",
                isinstance(response, str) and len(response) > 0,
                f"Response: {response[:100]}..."
            )
        except Exception as e:
            self.log_test("Response Generation", False, f"Error: {str(e)}")

        # Test 2: Streaming response
        try:
            chunks = []
            async for chunk in model.generate_response_stream("Test context", "Test query"):
                chunks.append(chunk)

            self.log_test(
                "Streaming Response",
                len(chunks) > 0 and all(isinstance(c, str) for c in chunks),
                f"Received {len(chunks)} chunks"
            )
        except Exception as e:
            self.log_test("Streaming Response", False, f"Error: {str(e)}")

        # Test 3: Model info
        try:
            info = model.get_model_info()
            required_fields = ["provider", "model_name"]
            has_required = all(field in info for field in required_fields)

            self.log_test(
                "Model Info",
                isinstance(info, dict) and has_required,
                f"Info keys: {list(info.keys())}"
            )
        except Exception as e:
            self.log_test("Model Info", False, f"Error: {str(e)}")

        # Test 4: Connection validation
        try:
            is_valid = model.validate_connection()
            self.log_test(
                "Connection Validation",
                isinstance(is_valid, bool),
                f"Connection valid: {is_valid}"
            )
        except Exception as e:
            self.log_test("Connection Validation", False, f"Error: {str(e)}")

        # Test 5: Token estimation
        try:
            token_count = model.estimate_tokens("This is a test sentence.")
            self.log_test(
                "Token Estimation",
                isinstance(token_count, int) and token_count > 0,
                f"Estimated tokens: {token_count}"
            )
        except Exception as e:
            self.log_test("Token Estimation", False, f"Error: {str(e)}")

    async def test_llm_service_integration(self, service: ILLMService, service_name: str):
        """Test LLMService integration"""
        print(f"\nTesting Service Integration: {service_name}")
        print("-" * 50)

        # Test 1: Response with sources
        try:
            sources = [{"filename": "test.txt", "relevance": 0.9}]
            result = await service.generate_response_with_sources(
                "Test context", "Test query", sources
            )

            is_valid = (
                isinstance(result, dict) and
                "response" in result and
                "sources" in result and
                "query" in result
            )

            self.log_test(
                "Response with Sources",
                is_valid,
                f"Result keys: {list(result.keys())}"
            )
        except Exception as e:
            self.log_test("Response with Sources", False, f"Error: {str(e)}")

        # Test 2: Streaming response
        try:
            chunks = []
            async for chunk in service.generate_streaming_response("Test context", "Test query"):
                chunks.append(chunk)
                if len(chunks) >= 5:  # Limit for testing
                    break

            self.log_test(
                "Service Streaming",
                len(chunks) > 0,
                f"Received {len(chunks)} chunks"
            )
        except Exception as e:
            self.log_test("Service Streaming", False, f"Error: {str(e)}")

        # Test 3: Model info retrieval
        try:
            info = service.get_model_info()
            self.log_test(
                "Service Model Info",
                isinstance(info, dict) and "provider" in info,
                f"Provider: {info.get('provider', 'Unknown')}"
            )
        except Exception as e:
            self.log_test("Service Model Info", False, f"Error: {str(e)}")

    async def test_performance_characteristics(self, model: ILLMModel, model_name: str):
        """Test performance characteristics"""
        print(f"\nTesting Performance: {model_name}")
        print("-" * 50)

        # Test response time
        try:
            start_time = time.time()
            response = await model.generate_response("Short context", "Simple query")
            response_time = time.time() - start_time

            self.log_test(
                "Response Time",
                response_time < 30.0,  # Should respond within 30 seconds
                f"Response time: {response_time:.2f}s"
            )
        except Exception as e:
            self.log_test("Response Time", False, f"Error: {str(e)}")

        # Test with longer context
        try:
            long_context = "This is a longer context. " * 50
            response = await model.generate_response(long_context, "What is this about?")

            self.log_test(
                "Long Context Handling",
                isinstance(response, str) and len(response) > 0,
                f"Response length: {len(response)} chars"
            )
        except Exception as e:
            self.log_test("Long Context Handling", False, f"Error: {str(e)}")

    async def test_error_handling(self, model: ILLMModel, model_name: str):
        """Test error handling capabilities"""
        print(f"\nTesting Error Handling: {model_name}")
        print("-" * 50)

        # Test empty input handling
        try:
            response = await model.generate_response("", "")
            self.log_test(
                "Empty Input Handling",
                isinstance(response, str),
                "Handled empty input gracefully"
            )
        except Exception as e:
            self.log_test("Empty Input Handling", False, f"Error: {str(e)}")

        # Test very long input (if not mock)
        if not isinstance(model, MockLLMModel):
            try:
                very_long_input = "Very long text. " * 1000
                response = await model.generate_response(very_long_input, "Summarize this")

                self.log_test(
                    "Long Input Handling",
                    isinstance(response, str),
                    "Handled very long input"
                )
            except Exception as e:
                # This is expected to fail for most models due to token limits
                self.log_test(
                    "Long Input Handling",
                    "token" in str(e).lower() or "length" in str(e).lower(),
                    f"Expected token limit error: {str(e)[:100]}"
                )

    def test_factory_functionality(self):
        """Test LLM factory functionality"""
        print(f"\nTesting Factory Functionality")
        print("-" * 50)

        # Test 1: Available models
        try:
            models = LLMFactory.get_available_models()
            self.log_test(
                "Available Models",
                isinstance(models, dict) and len(models) > 0,
                f"Found {len(models)} models"
            )
        except Exception as e:
            self.log_test("Available Models", False, f"Error: {str(e)}")

        # Test 2: Factory interface compliance
        try:
            factory_methods = [
                hasattr(LLMFactory, 'create_model'),
                hasattr(LLMFactory, 'get_available_models')
            ]

            self.log_test(
                "Factory Interface",
                all(factory_methods),
                "Factory has required methods"
            )
        except Exception as e:
            self.log_test("Factory Interface", False, f"Error: {str(e)}")

    async def run_mock_tests(self):
        """Run tests with mock models"""
        print("=" * 60)
        print("TESTING WITH MOCK MODELS")
        print("=" * 60)

        # Test mock LLM model
        mock_model = MockLLMModel()
        await self.test_llm_model_interface_compliance(mock_model, "Mock LLM")
        await self.test_performance_characteristics(mock_model, "Mock LLM")
        await self.test_error_handling(mock_model, "Mock LLM")

        # Test mock service
        mock_service = LLMService.__new__(LLMService)  # Create without __init__
        mock_service.llm_model = mock_model
        mock_service.callback_manager = None

        # Mock the service methods
        async def mock_generate_response(context, query):
            return await mock_model.generate_response(context, query)

        async def mock_generate_response_with_sources(context, query, sources=None):
            response = await mock_model.generate_response(context, query)
            return {
                'response': response,
                'sources': sources or [],
                'query': query,
                'model_info': mock_model.get_model_info()
            }

        async def mock_generate_streaming_response(context, query):
            async for chunk in mock_model.generate_response_stream(context, query):
                yield {'type': 'chunk', 'content': chunk}

        mock_service.generate_response = mock_generate_response
        mock_service.generate_response_with_sources = mock_generate_response_with_sources
        mock_service.generate_streaming_response = mock_generate_streaming_response
        mock_service.get_model_info = mock_model.get_model_info

        await self.test_llm_service_integration(mock_service, "Mock Service")

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

        # Test available providers
        providers_to_test = []
        if provider:
            providers_to_test = [provider]
        else:
            # Test all available providers
            try:
                from src.config.model_config import get_model_config
                config = get_model_config()
                available_models = config.get_llm_models()
                providers_to_test = list(set(
                    info.get('provider', 'unknown')
                    for info in available_models.values()
                ))
            except Exception as e:
                print(f"Could not get available providers: {e}")
                providers_to_test = ['openai', 'google']

        for provider_name in providers_to_test:
            try:
                print(f"\nTesting {provider_name.upper()} models...")

                # Create LLM service for this provider
                from src.config.model_config import LLMProvider
                provider_enum = None
                if provider_name == 'openai':
                    provider_enum = LLMProvider.OPENAI
                elif provider_name == 'google':
                    provider_enum = LLMProvider.GOOGLE

                if provider_enum:
                    service = LLMService(
                        provider=provider_enum,
                        api_key=api_key,
                        base_url=base_url
                    )

                    await self.test_llm_service_integration(service, f"{provider_name} Service")
                    await self.test_performance_characteristics(service.llm_model, f"{provider_name} Model")
                    await self.test_error_handling(service.llm_model, f"{provider_name} Model")

            except Exception as e:
                self.log_test(f"{provider_name} Model Creation", False, f"Error: {str(e)}")

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

        print(f"\nLLM Models are {'PROPERLY' if pass_rate >= 80 else 'NOT PROPERLY'} abstracted and testable!")


async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Test LLM models")
    parser.add_argument('--provider', choices=['openai', 'google'], help='Test specific provider')
    parser.add_argument('--mock-only', action='store_true', help='Test with mocks only')
    parser.add_argument('--real-only', action='store_true', help='Test with real models only')

    args = parser.parse_args()

    tester = LLMModelTester()

    print("LLM Models Independence and Testability Test")
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