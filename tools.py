from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_globals, safe_builtins, guarded_unpack_sequence
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, BinaryContent
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from httpx import AsyncClient
from supabase import Client
import base64
import json
import sys
import os
import re

embedding_model = os.getenv('EMBEDDING_MODEL_CHOICE') or 'text-embedding-3-small'

async def brave_web_search(query: str, http_client: AsyncClient, brave_api_key: str) -> str:
    """
    Helper function for web_search_tool - searches the web with the Brave API
    and returns a summary of all the top search results.

    Args are the same as the parent function except without the SearXNG base url.
    """
    headers = {
        'X-Subscription-Token': brave_api_key,
        'Accept': 'application/json',
    }
    
    response = await http_client.get(
        'https://api.search.brave.com/res/v1/web/search',
        params={
            'q': query,
            'count': 5,
            'text_decorations': True,
            'search_lang': 'en'
        },
        headers=headers
    )
    response.raise_for_status()
    data = response.json()

    results = []
    
    # Add web results in a nice formatted way
    web_results = data.get('web', {}).get('results', [])
    for item in web_results[:3]:
        title = item.get('title', '')
        description = item.get('description', '')
        url = item.get('url', '')
        if title and description:
            results.append(f"Title: {title}\nSummary: {description}\nSource: {url}\n")

    return "\n".join(results) if results else "No results found for the query."

async def searxng_web_search(query: str, http_client: AsyncClient, searxng_base_url: str) -> str:
    """
    Helper function for web_search_tool - searches the web with SearXNG
    and returns a list of the top search results with the most relevant snippet from each page.

    Args are the same as the parent function except without the Brave API key.
    """
    # Prepare the parameters for the request
    params = {'q': query, 'format': 'json'}
    
    # Make the request to SearXNG
    response = await http_client.get(f"{searxng_base_url}/search", params=params)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Parse the results
    data = response.json()
    
    results = ""
    for i, page in enumerate(data.get('results', []), 1):
        if i > 10:  # Limiting to the top 10 results, could make this a parameter for the function too
            break

        results += f"{i}. {page.get('title', 'No title')}"
        results += f"   URL: {page.get('url', 'No URL')}"
        results += f"   Content: {page.get('content', 'No content')[:300]}...\n\n"

    return results if results else "No results found for the query."

async def web_search_tool(query: str, http_client: AsyncClient, brave_api_key: str, searxng_base_url: str) -> str:
    """
    Search the web with a specific query and get a summary of the top search results.
    
    Args:
        query: The query for the web search
        http_client: The client for making HTTP requests to Brave or SearXNG
        brave_api_key: The optional key for Brave (will use SearXNG if this isn't defined)
        searxng_base_url: The optional base URL for SearXNG (will use Brave if this isn't defined)
        
    Returns:
        A summary of the web search.
        For Brave, this is a single paragraph.
        For SearXNG, this is a list of the top search results including the most relevant snippet from the page.
    """
    try:
        if brave_api_key:
            return await brave_web_search(query, http_client, brave_api_key)
        else:
            return await searxng_web_search(query, http_client, searxng_base_url)
    except Exception as e:
        print(f"Exception during websearch: {e}")
        return str(e)

async def get_embedding(text: str, embedding_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await embedding_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def retrieve_relevant_documents_tool(supabase: Client, embedding_client: AsyncOpenAI, user_query: str) -> str:
    """
    Function to retrieve relevant document chunks with RAG.
    This is called by the retrieve_relevant_documents tool for the agent.
    
    Returns:
        List[str]: List of relevant document chunks with metadata
    """    
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, embedding_client)
        
        # Query Supabase for relevant documents
        result = supabase.rpc(
            'match_documents',
            {
                'query_embedding': query_embedding,
                'match_count': 10
            }
        ).execute()
        
        if not result.data:
            return "No relevant documents found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# Document ID: {doc['metadata'].get('file_id', 'unknown')}      
# Document Tilte: {doc['metadata'].get('file_title', 'unknown')}
# Document URL: {doc['metadata'].get('file_url', 'unknown')}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return f"Error retrieving documents: {str(e)}" 

async def list_documents_tool(supabase: Client) -> List[str]:
    """
    Function to retrieve a list of all available documents.
    This is called by the list_documents tool for the agent.
    
    Returns:
        List[str]: List of documents including their metadata (URL/path, schema if applicable, etc.)
    """
    try:
        # Query Supabase for unique documents
        result = supabase.from_('document_metadata') \
            .select('id, title, schema, url') \
            .execute()
            
        return str(result.data)
        
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return str([])

async def get_document_content_tool(supabase: Client, document_id: str) -> str:
    """
    Retrieve the full content of a specific document by combining all its chunks.
    This is called by the get_document_content tool for the agent.
        
    Returns:
        str: The complete content of the document with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks for this document
        result = supabase.from_('documents') \
            .select('id, content, metadata') \
            .eq('metadata->>file_id', document_id) \
            .order('id') \
            .execute()
        
        if not result.data:
            return f"No content found for document: {document_id}"
            
        # Format the document with its title and all chunks
        document_title = result.data[0]['metadata']['file_title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {document_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together but limit the characters in case the document is massive
        return "\n\n".join(formatted_content)[:20000]
        
    except Exception as e:
        print(f"Error retrieving document content: {e}")
        return f"Error retrieving document content: {str(e)}"     

async def execute_sql_query_tool(supabase: Client, sql_query: str) -> str:
    """
    Run a SQL query - use this to query from the document_rows table once you know the file ID you are querying. 
    dataset_id is the file_id and you are always using the row_data for filtering, which is a jsonb field that has 
    all the keys from the file schema given in the document_metadata table.

    Example query:

    SELECT AVG((row_data->>'revenue')::numeric)
    FROM document_rows
    WHERE dataset_id = '123';

    Example query 2:

    SELECT 
        row_data->>'category' as category,
        SUM((row_data->>'sales')::numeric) as total_sales
    FROM dataset_rows
    WHERE dataset_id = '123'
    GROUP BY row_data->>'category';
    
    Args:
        supabase: The Supabase client
        sql_query: The SQL query to execute (must be read-only)
        
    Returns:
        str: The results of the SQL query in JSON format
    """
    try:
        # Validate that the query is read-only by checking for write operations
        sql_query = sql_query.strip()
        write_operations = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'GRANT', 'REVOKE']
        
        # Convert query to uppercase for case-insensitive comparison
        upper_query = sql_query.upper()
        
        # Check if any write operations are in the query
        for op in write_operations:
            pattern = r'\b' + op + r'\b'
            if re.search(pattern, upper_query):
                return f"Error: Write operation '{op}' detected. Only read-only queries are allowed."
        
        # Execute the query using the RPC function
        result = supabase.rpc(
            'execute_custom_sql',
            {"sql_query": sql_query}
        ).execute()
        
        # Check for errors in the response
        if result.data and 'error' in result.data:
            return f"SQL Error: {result.data['error']}"
        
        # Format the results nicely
        return json.dumps(result.data, indent=2)
        
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"

async def image_analysis_tool(supabase: Client, document_id: str, query: str) -> str:
    try:
        # Environment variables for the vision model
        llm = os.getenv('VISION_LLM_CHOICE', 'gpt-4o-mini')
        base_url = os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1')
        api_key = os.getenv('LLM_API_KEY', 'no-api-key-provided')

        # Define the vision agent based on the environment variables
        model = OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))
        vision_agent = Agent(
            model, 
            system_prompt="You are an image analyzer who looks at images provided and answers the accompanying query in detail."
        )

        # Get the binary of the file from the database
        result = supabase.from_('documents') \
            .select('metadata') \
            .eq('metadata->>file_id', document_id) \
            .limit(1) \
            .execute()

        if not result.data:
            return f"No content found for document: {document_id}"            

        # Get the binary and mime_type from the metadata
        metadata = result.data[0]['metadata']
        binary_str = metadata['file_contents']
        mime_type = metadata['mime_type']

        if not binary_str:
            return f"No file contents found for document: {document_id}"

        # Turn the binary string into binary and send it into the vision LLM
        binary = base64.b64decode(binary_str.encode('utf-8'))
        result = await vision_agent.run([query, BinaryContent(data=binary, media_type=mime_type)])

        return result.data

    except Exception as e:
        print(f"Error analyzing image: {e}")
        return f"Error analyzing image: {str(e)}"           

def execute_safe_code_tool(code: str) -> str:
    # Set up allowed modules
    allowed_modules = {
        # Core utilities
        'datetime': __import__('datetime'),
        'math': __import__('math'),
        'random': __import__('random'),
        'time': __import__('time'),
        'collections': __import__('collections'),
        'itertools': __import__('itertools'),
        'functools': __import__('functools'),
        'copy': __import__('copy'),
        're': __import__('re'),  # Regular expressions
        'json': __import__('json'),
        'csv': __import__('csv'),
        'uuid': __import__('uuid'),
        'string': __import__('string'),
        'statistics': __import__('statistics'),
        
        # Data structures and algorithms
        'heapq': __import__('heapq'),
        'bisect': __import__('bisect'),
        'array': __import__('array'),
        'enum': __import__('enum'),
        'dataclasses': __import__('dataclasses'),
        
        # Numeric/scientific (if installed)
        # 'numpy': __import__('numpy', fromlist=['*']),
        # 'pandas': __import__('pandas', fromlist=['*']),
        # 'scipy': __import__('scipy', fromlist=['*']),
        
        # File/IO (with careful restrictions)
        'io': __import__('io'),
        'base64': __import__('base64'),
        'hashlib': __import__('hashlib'),
        'tempfile': __import__('tempfile')
    }
    
    # Try to import optional modules that might not be installed
    try:
        allowed_modules['numpy'] = __import__('numpy')
    except ImportError:
        pass
        
    try:
        allowed_modules['pandas'] = __import__('pandas')
    except ImportError:
        pass
        
    try:
        allowed_modules['scipy'] = __import__('scipy')
    except ImportError:
        pass
    
    # Custom import function that only allows whitelisted modules
    def safe_import(name, *args, **kwargs):
        if name in allowed_modules:
            return allowed_modules[name]
        raise ImportError(f"Module {name} is not allowed")
    
    # Create a safe environment with minimal built-ins
    safe_builtins = {
        # Basic operations
        'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool, 
        'chr': chr, 'complex': complex, 'divmod': divmod, 'float': float, 
        'format': format, 'hex': hex, 'int': int, 'len': len, 'max': max, 
        'min': min, 'oct': oct, 'ord': ord, 'pow': pow, 'round': round,
        'sorted': sorted, 'sum': sum,
        
        # Types and conversions
        'bytes': bytes, 'dict': dict, 'frozenset': frozenset, 'list': list, 
        'repr': repr, 'set': set, 'slice': slice, 'str': str, 'tuple': tuple, 
        'type': type, 'zip': zip,
        
        # Iteration and generation
        'enumerate': enumerate, 'filter': filter, 'iter': iter, 'map': map,
        'next': next, 'range': range, 'reversed': reversed,
        
        # Other safe operations
        'getattr': getattr, 'hasattr': hasattr, 'hash': hash,
        'isinstance': isinstance, 'issubclass': issubclass,
        
        # Import handler
        '__import__': safe_import
    }
    
    # Set up output capture
    output = []
    def safe_print(*args, **kwargs):
        end = kwargs.get('end', '\n')
        sep = kwargs.get('sep', ' ')
        output.append(sep.join(str(arg) for arg in args) + end)
    
    # Create restricted globals
    restricted_globals = {
        '__builtins__': safe_builtins,
        'print': safe_print
    }
    
    try:
        # Execute the code with timeout
        exec(code, restricted_globals)
        return ''.join(output)
    except Exception as e:
        return f"Error executing code: {str(e)}"