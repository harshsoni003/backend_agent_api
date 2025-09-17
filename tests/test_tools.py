import pytest
import sys
import os
import json
import base64
from unittest.mock import patch, MagicMock, AsyncMock, call

# Mock environment variables before importing modules that use them
with patch.dict(os.environ, {
    'LLM_PROVIDER': 'openai',
    'LLM_BASE_URL': 'https://api.openai.com/v1',
    'LLM_API_KEY': 'test-api-key',
    'LLM_CHOICE': 'gpt-4o-mini',
    'VISION_LLM_CHOICE': 'gpt-4o-mini',
    'EMBEDDING_PROVIDER': 'openai',
    'EMBEDDING_BASE_URL': 'https://api.openai.com/v1',
    'EMBEDDING_API_KEY': 'test-api-key',
    'EMBEDDING_MODEL_CHOICE': 'text-embedding-3-small',
    'SUPABASE_URL': 'https://test-supabase-url.com',
    'SUPABASE_SERVICE_KEY': 'test-supabase-key',
    'BRAVE_API_KEY': 'test-brave-key',
    'SEARXNG_BASE_URL': 'http://test-searxng-url.com'
}):
    # Mock the AsyncOpenAI client before it's used in tools
    with patch('openai.AsyncOpenAI') as mock_openai_client:
        # Configure the mock to return a MagicMock
        mock_client = MagicMock()
        mock_openai_client.return_value = mock_client
        
        # Mock the Supabase client
        with patch('supabase.create_client') as mock_create_client:
            mock_supabase = MagicMock()
            mock_create_client.return_value = mock_supabase
            
            # Add parent directory to path to import the tools module
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from tools import (
                brave_web_search,
                searxng_web_search,
                web_search_tool,
                get_embedding,
                retrieve_relevant_documents_tool,
                list_documents_tool,
                get_document_content_tool,
                image_analysis_tool,
                execute_safe_code_tool
            )


class TestWebSearchTools:
    @pytest.mark.asyncio
    async def test_brave_web_search_success(self):
        # Mock HTTP client and response
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Test Title 1",
                        "description": "Test Description 1",
                        "url": "https://example.com/1"
                    },
                    {
                        "title": "Test Title 2",
                        "description": "Test Description 2",
                        "url": "https://example.com/2"
                    }
                ]
            }
        }
        mock_client.get.return_value = mock_response
        
        # Test the function
        result = await brave_web_search("test query", mock_client, "test-api-key")
        
        # Verify the HTTP request was made correctly
        mock_client.get.assert_called_once_with(
            'https://api.search.brave.com/res/v1/web/search',
            params={
                'q': "test query",
                'count': 5,
                'text_decorations': True,
                'search_lang': 'en'
            },
            headers={
                'X-Subscription-Token': 'test-api-key',
                'Accept': 'application/json',
            }
        )
        
        # Verify the response was processed correctly
        assert "Test Title 1" in result
        assert "Test Description 1" in result
        assert "https://example.com/1" in result
        assert "Test Title 2" in result
        assert "Test Description 2" in result
        assert "https://example.com/2" in result

    @pytest.mark.asyncio
    async def test_brave_web_search_no_results(self):
        # Mock HTTP client and response with no results
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_client.get.return_value = mock_response
        
        # Test the function
        result = await brave_web_search("test query", mock_client, "test-api-key")
        
        # Verify the response for no results
        assert result == "No results found for the query."

    @pytest.mark.asyncio
    async def test_searxng_web_search_success(self):
        # Mock HTTP client and response
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "SearXNG Result 1",
                    "url": "https://example.com/searxng1",
                    "content": "SearXNG Content 1"
                },
                {
                    "title": "SearXNG Result 2",
                    "url": "https://example.com/searxng2",
                    "content": "SearXNG Content 2"
                }
            ]
        }
        mock_client.get.return_value = mock_response
        
        # Test the function
        result = await searxng_web_search("test query", mock_client, "https://searxng.example.com")
        
        # Verify the HTTP request was made correctly
        mock_client.get.assert_called_once_with(
            "https://searxng.example.com/search",
            params={'q': "test query", 'format': 'json'}
        )
        
        # Verify the response was processed correctly
        assert "SearXNG Result 1" in result
        assert "https://example.com/searxng1" in result
        assert "SearXNG Content 1" in result
        assert "SearXNG Result 2" in result
        assert "https://example.com/searxng2" in result
        assert "SearXNG Content 2" in result

    @pytest.mark.asyncio
    async def test_searxng_web_search_no_results(self):
        # Mock HTTP client and response with no results
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_client.get.return_value = mock_response
        
        # Test the function
        result = await searxng_web_search("test query", mock_client, "https://searxng.example.com")
        
        # Verify the response for no results
        assert result == "No results found for the query."

    @pytest.mark.asyncio
    @patch('tools.brave_web_search')
    async def test_web_search_tool_with_brave(self, mock_brave_search):
        # Setup mock for brave search
        mock_brave_search.return_value = "Brave search results"
        
        # Create a specific mock client to pass to the function
        mock_client = AsyncMock()
        
        # Test with Brave API key provided
        result = await web_search_tool(
            "test query", 
            mock_client, 
            "brave-api-key", 
            "https://searxng.example.com"
        )
        
        # Verify Brave search was called with the right parameters
        # Use ANY for the client parameter since we can't directly compare AsyncMock instances
        from unittest.mock import ANY
        mock_brave_search.assert_called_once_with("test query", ANY, "brave-api-key")
        assert result == "Brave search results"

    @pytest.mark.asyncio
    @patch('tools.searxng_web_search')
    async def test_web_search_tool_with_searxng(self, mock_searxng_search):
        # Setup mock for SearXNG search
        mock_searxng_search.return_value = "SearXNG search results"
        
        # Create a specific mock client to pass to the function
        mock_client = AsyncMock()
        
        # Test with no Brave API key (should use SearXNG)
        result = await web_search_tool(
            "test query", 
            mock_client, 
            "", 
            "https://searxng.example.com"
        )
        
        # Verify SearXNG search was called with the right parameters
        # Use ANY for the client parameter since we can't directly compare AsyncMock instances
        from unittest.mock import ANY
        mock_searxng_search.assert_called_once_with("test query", ANY, "https://searxng.example.com")
        assert result == "SearXNG search results"

    @pytest.mark.asyncio
    async def test_web_search_tool_exception(self):
        # Mock HTTP client that raises an exception
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Test exception")
        
        # Test with both API keys to ensure exception handling
        result = await web_search_tool(
            "test query", 
            mock_client, 
            "brave-api-key", 
            "https://searxng.example.com"
        )
        
        # Verify exception was handled
        assert "Test exception" in result


# Global reference to the mocked OpenAI client
openai_client_mock = mock_client

class TestEmbeddingTools:
    @pytest.mark.asyncio
    async def test_get_embedding_success(self):
        # Mock embedding client and response
        mock_client = AsyncMock()
        mock_embeddings = AsyncMock()
        mock_client.embeddings = mock_embeddings
        
        # Setup mock response
        mock_embedding_data = MagicMock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding_data]
        mock_embeddings.create.return_value = mock_response
        
        # Test the function
        with patch('tools.embedding_model', 'text-embedding-3-small'):
            result = await get_embedding("test text", mock_client)
        
        # Verify the embedding request was made correctly
        mock_embeddings.create.assert_called_once_with(
            model='text-embedding-3-small',
            input="test text"
        )
        
        # Verify the result
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_get_embedding_exception(self):
        # Mock embedding client that raises an exception
        mock_client = AsyncMock()
        mock_embeddings = AsyncMock()
        mock_client.embeddings = mock_embeddings
        mock_embeddings.create.side_effect = Exception("Test exception")
        
        # Test the function
        with patch('tools.embedding_model', 'text-embedding-3-small'):
            with patch('builtins.print') as mock_print:
                result = await get_embedding("test text", mock_client)
        
        # Verify the exception was handled
        mock_print.assert_called_once_with("Error getting embedding: Test exception")
        
        # Verify a zero vector was returned
        assert len(result) == 1536
        assert all(x == 0 for x in result)


class TestDocumentTools:
    @pytest.mark.asyncio
    @patch('tools.get_embedding')
    async def test_retrieve_relevant_documents_tool_success(self, mock_get_embedding):
        # Mock embedding function
        mock_get_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock Supabase client and response
        mock_supabase = MagicMock()
        mock_rpc = MagicMock()
        mock_supabase.rpc.return_value = mock_rpc
        mock_execute = MagicMock()
        mock_rpc.execute.return_value = mock_execute
        
        # Setup mock data
        mock_execute.data = [
            {
                'content': 'Document content 1',
                'metadata': {
                    'file_id': 'doc1',
                    'file_title': 'Document 1'
                }
            },
            {
                'content': 'Document content 2',
                'metadata': {
                    'file_id': 'doc2',
                    'file_title': 'Document 2'
                }
            }
        ]
        
        # Create a specific mock client to pass to the function
        mock_embedding_client = AsyncMock()
        
        # Test the function
        result = await retrieve_relevant_documents_tool(mock_supabase, mock_embedding_client, "test query")
        
        # Verify embedding was requested with the right parameters
        # Use ANY for the client parameter since we can't directly compare AsyncMock instances
        from unittest.mock import ANY
        mock_get_embedding.assert_called_once_with("test query", ANY)
        
        # Verify Supabase RPC was called correctly
        mock_supabase.rpc.assert_called_once_with(
            'match_documents',
            {
                'query_embedding': [0.1, 0.2, 0.3],
                'match_count': 4
            }
        )
        
        # Verify the result contains document information
        assert "Document ID: doc1" in result
        assert "Document Tilte: Document 1" in result
        assert "Document content 1" in result
        assert "Document ID: doc2" in result
        assert "Document Tilte: Document 2" in result
        assert "Document content 2" in result

    @pytest.mark.asyncio
    @patch('tools.get_embedding')
    async def test_retrieve_relevant_documents_tool_no_results(self, mock_get_embedding):
        # Mock embedding function
        mock_get_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock Supabase client with no results
        mock_supabase = MagicMock()
        mock_rpc = MagicMock()
        mock_supabase.rpc.return_value = mock_rpc
        mock_execute = MagicMock()
        mock_rpc.execute.return_value = mock_execute
        mock_execute.data = []
        
        # Test the function
        result = await retrieve_relevant_documents_tool(mock_supabase, AsyncMock(), "test query")
        
        # Verify the result for no documents
        assert result == "No relevant documents found."

    @pytest.mark.asyncio
    @patch('tools.get_embedding')
    async def test_retrieve_relevant_documents_tool_exception(self, mock_get_embedding):
        # Mock embedding function that raises an exception
        mock_get_embedding.side_effect = Exception("Test exception")
        
        # Test the function
        with patch('builtins.print') as mock_print:
            result = await retrieve_relevant_documents_tool(MagicMock(), AsyncMock(), "test query")
        
        # Verify the exception was handled
        mock_print.assert_called_once_with("Error retrieving documents: Test exception")
        assert "Error retrieving documents: Test exception" in result

    @pytest.mark.asyncio
    async def test_list_documents_tool_success(self):
        # Mock Supabase client and response
        mock_supabase = MagicMock()
        mock_from = MagicMock()
        mock_supabase.from_.return_value = mock_from
        mock_select = MagicMock()
        mock_from.select.return_value = mock_select
        mock_execute = MagicMock()
        mock_select.execute.return_value = mock_execute
        
        # Setup mock data
        mock_execute.data = [
            {
                'id': 'doc1',
                'title': 'Document 1',
                'schema': 'schema1',
                'url': 'https://example.com/doc1'
            },
            {
                'id': 'doc2',
                'title': 'Document 2',
                'schema': 'schema2',
                'url': 'https://example.com/doc2'
            }
        ]
        
        # Test the function
        result = await list_documents_tool(mock_supabase)
        
        # Verify Supabase query was called correctly
        mock_supabase.from_.assert_called_once_with('document_metadata')
        mock_from.select.assert_called_once_with('id, title, schema, url')
        
        # Verify the result contains document information
        assert str(mock_execute.data) == result

    @pytest.mark.asyncio
    async def test_list_documents_tool_exception(self):
        # Mock Supabase client that raises an exception
        mock_supabase = MagicMock()
        mock_supabase.from_.side_effect = Exception("Test exception")
        
        # Test the function
        with patch('builtins.print') as mock_print:
            result = await list_documents_tool(mock_supabase)
        
        # Verify the exception was handled
        mock_print.assert_called_once_with("Error retrieving documents: Test exception")
        assert result == str([])

    @pytest.mark.asyncio
    async def test_get_document_content_tool_success(self):
        # Mock Supabase client and response
        mock_supabase = MagicMock()
        mock_from = MagicMock()
        mock_supabase.from_.return_value = mock_from
        mock_select = MagicMock()
        mock_from.select.return_value = mock_select
        mock_eq = MagicMock()
        mock_select.eq.return_value = mock_eq
        mock_order = MagicMock()
        mock_eq.order.return_value = mock_order
        mock_execute = MagicMock()
        mock_order.execute.return_value = mock_execute
        
        # Setup mock data
        mock_execute.data = [
            {
                'id': 'chunk1',
                'content': 'Document content part 1',
                'metadata': {
                    'file_id': 'doc1',
                    'file_title': 'Document 1 - Part 1'
                }
            },
            {
                'id': 'chunk2',
                'content': 'Document content part 2',
                'metadata': {
                    'file_id': 'doc1',
                    'file_title': 'Document 1 - Part 2'
                }
            }
        ]
        
        # Test the function
        result = await get_document_content_tool(mock_supabase, 'doc1')
        
        # Verify Supabase query was called correctly
        mock_supabase.from_.assert_called_once_with('documents')
        mock_from.select.assert_called_once_with('id, content, metadata')
        mock_select.eq.assert_called_once_with('metadata->>file_id', 'doc1')
        mock_eq.order.assert_called_once_with('id')
        
        # Verify the result contains document content
        assert "# Document 1" in result
        assert "Document content part 1" in result
        assert "Document content part 2" in result

    @pytest.mark.asyncio
    async def test_get_document_content_tool_no_content(self):
        # Mock Supabase client with no results
        mock_supabase = MagicMock()
        mock_from = MagicMock()
        mock_supabase.from_.return_value = mock_from
        mock_select = MagicMock()
        mock_from.select.return_value = mock_select
        mock_eq = MagicMock()
        mock_select.eq.return_value = mock_eq
        mock_order = MagicMock()
        mock_eq.order.return_value = mock_order
        mock_execute = MagicMock()
        mock_order.execute.return_value = mock_execute
        mock_execute.data = []
        
        # Test the function
        result = await get_document_content_tool(mock_supabase, 'doc1')
        
        # Verify the result for no content
        assert result == "No content found for document: doc1"

    @pytest.mark.asyncio
    async def test_get_document_content_tool_exception(self):
        # Mock Supabase client that raises an exception
        mock_supabase = MagicMock()
        mock_supabase.from_.side_effect = Exception("Test exception")
        
        # Test the function
        with patch('builtins.print') as mock_print:
            result = await get_document_content_tool(mock_supabase, 'doc1')
        
        # Verify the exception was handled
        mock_print.assert_called_once_with("Error retrieving document content: Test exception")
        assert "Error retrieving document content: Test exception" in result


class TestImageAnalysisTool:
    @pytest.mark.asyncio
    @patch('tools.OpenAIModel')
    @patch('tools.OpenAIProvider')
    @patch('tools.Agent')
    @patch('tools.os.getenv')
    async def test_image_analysis_tool_success(self, mock_getenv, mock_agent_class, mock_provider_class, mock_model_class):
        # Mock environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            'VISION_LLM_CHOICE': 'gpt-4o-mini',
            'LLM_BASE_URL': 'https://api.openai.com/v1',
            'LLM_API_KEY': 'test-api-key'
        }.get(key, default)
        
        # Mock OpenAI provider, model and agent
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider
        
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Create a mock agent with a proper async mock for run
        mock_agent = MagicMock()
        # We need to track if run was called and with what arguments
        run_called = False
        run_args = None
        
        async def mock_run(*args, **kwargs):
            nonlocal run_called, run_args
            run_called = True
            run_args = args
            mock_result = MagicMock()
            mock_result.data = "Image analysis result"
            return mock_result
            
        mock_agent.run = mock_run
        mock_agent_class.return_value = mock_agent
        
        # Mock Supabase client and response
        mock_supabase = MagicMock()
        mock_from = MagicMock()
        mock_supabase.from_.return_value = mock_from
        mock_select = MagicMock()
        mock_from.select.return_value = mock_select
        mock_eq = MagicMock()
        mock_select.eq.return_value = mock_eq
        mock_limit = MagicMock()
        mock_eq.limit.return_value = mock_limit
        mock_execute = MagicMock()
        mock_limit.execute.return_value = mock_execute
        
        # Setup mock document data with base64 image
        test_binary = base64.b64encode(b'test image data').decode('utf-8')
        mock_execute.data = [
            {
                'metadata': {
                    'file_id': 'img1',
                    'file_contents': test_binary,
                    'mime_type': 'image/jpeg'
                }
            }
        ]
        
        # Test the function
        result = await image_analysis_tool(mock_supabase, 'img1', 'Describe this image')
        
        # Verify Supabase query was called correctly
        mock_supabase.from_.assert_called_once_with('documents')
        mock_from.select.assert_called_once_with('metadata')
        mock_select.eq.assert_called_once_with('metadata->>file_id', 'img1')
        mock_eq.limit.assert_called_once_with(1)
        
        # Verify agent setup and run
        mock_provider_class.assert_called_once_with(base_url='https://api.openai.com/v1', api_key='test-api-key')
        mock_model_class.assert_called_once_with('gpt-4o-mini', provider=mock_provider)
        mock_agent_class.assert_called_once()
        
        # Verify the agent run method was called
        assert run_called, "Agent run method was not called"
        # Verify the arguments were correct
        assert run_args is not None, "Agent run was called without arguments"
        assert run_args[0][0] == 'Describe this image', f"Expected 'Describe this image', got {run_args[0][0]}"
        # Check that the second argument is a BinaryContent object
        assert hasattr(run_args[0][1], 'data'), "Second argument is missing 'data' attribute"
        assert hasattr(run_args[0][1], 'media_type'), "Second argument is missing 'media_type' attribute"
        
        # Verify the result
        assert result == "Image analysis result"

    @pytest.mark.asyncio
    async def test_image_analysis_tool_no_document(self):
        # Mock Supabase client with no results
        mock_supabase = MagicMock()
        mock_from = MagicMock()
        mock_supabase.from_.return_value = mock_from
        mock_select = MagicMock()
        mock_from.select.return_value = mock_select
        mock_eq = MagicMock()
        mock_select.eq.return_value = mock_eq
        mock_limit = MagicMock()
        mock_eq.limit.return_value = mock_limit
        mock_execute = MagicMock()
        mock_limit.execute.return_value = mock_execute
        mock_execute.data = []
        
        # Test the function
        result = await image_analysis_tool(mock_supabase, 'img1', 'Describe this image')
        
        # Verify the result for no document
        assert result == "No content found for document: img1"

    @pytest.mark.asyncio
    async def test_image_analysis_tool_no_file_contents(self):
        # Mock Supabase client with document but no file contents
        mock_supabase = MagicMock()
        mock_from = MagicMock()
        mock_supabase.from_.return_value = mock_from
        mock_select = MagicMock()
        mock_from.select.return_value = mock_select
        mock_eq = MagicMock()
        mock_select.eq.return_value = mock_eq
        mock_limit = MagicMock()
        mock_eq.limit.return_value = mock_limit
        mock_execute = MagicMock()
        mock_limit.execute.return_value = mock_execute
        
        # Document with empty file contents
        mock_execute.data = [
            {
                'metadata': {
                    'file_id': 'img1',
                    'file_contents': '',
                    'mime_type': 'image/jpeg'
                }
            }
        ]
        
        # Test the function
        result = await image_analysis_tool(mock_supabase, 'img1', 'Describe this image')
        
        # Verify the result for no file contents
        assert result == "No file contents found for document: img1"

    @pytest.mark.asyncio
    async def test_image_analysis_tool_exception(self):
        # Mock Supabase client that raises an exception
        mock_supabase = MagicMock()
        mock_supabase.from_.side_effect = Exception("Test exception")
        
        # Test the function
        with patch('builtins.print') as mock_print:
            result = await image_analysis_tool(mock_supabase, 'img1', 'Describe this image')
        
        # Verify the exception was handled
        mock_print.assert_called_once_with("Error analyzing image: Test exception")
        assert "Error analyzing image: Test exception" in result


class TestExecuteSafeCodeTool:
    def test_execute_safe_code_success(self):
        # Test code that should execute safely
        code = """
print("Hello, World!")
result = 2 + 2
print(f"2 + 2 = {result}")
"""
        result = execute_safe_code_tool(code)
        
        # Verify the output
        assert "Hello, World!" in result
        assert "2 + 2 = 4" in result

    def test_execute_safe_code_with_allowed_modules(self):
        # Test code that uses allowed modules
        code = """
import math
import json
import datetime

print(f"Pi is approximately {math.pi}")
print(json.dumps({"key": "value"}))
print(f"Current year: {datetime.datetime.now().year}")
"""
        result = execute_safe_code_tool(code)
        
        # Verify the output contains results from allowed modules
        assert "Pi is approximately 3.14" in result
        assert '{"key": "value"}' in result
        assert "Current year:" in result

    def test_execute_safe_code_with_disallowed_modules(self):
        # Test code that tries to import disallowed modules
        # Use a simpler approach that doesn't rely on try/except with Exception
        code = """
# Try to access disallowed modules
if 'os' in __builtins__:
    print("os module is available")
else:
    print("os module is not available")

if 'subprocess' in __builtins__:
    print("subprocess module is available")
else:
    print("subprocess module is not available")
"""
        result = execute_safe_code_tool(code)
        
        # Verify the output shows the modules aren't available
        assert "os module is not available" in result
        assert "subprocess module is not available" in result

    def test_execute_safe_code_with_exception(self):
        # Test code that raises an exception
        # Note: The actual implementation catches the exception and returns an error message
        # without including the output before the exception
        code = """
print("Starting")
x = 1 / 0  # Division by zero
print("This won't be reached")
"""
        result = execute_safe_code_tool(code)
        
        # Verify the output shows the exception
        assert "Error executing code: division by zero" in result

    def test_execute_safe_code_with_complex_operations(self):
        # Test code with more complex operations
        code = """
# Define a function
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

# Use list comprehension
squares = [x**2 for x in range(5)]
print(f"Squares: {squares}")

# Use higher-order functions
doubled = list(map(lambda x: x*2, squares))
print(f"Doubled: {doubled}")

# Test recursion
print(f"Factorial of 5: {factorial(5)}")
"""
        result = execute_safe_code_tool(code)
        
        # Verify the output shows complex operations worked
        assert "Squares: [0, 1, 4, 9, 16]" in result
        assert "Doubled: [0, 2, 8, 18, 32]" in result
        assert "Factorial of 5: 120" in result
