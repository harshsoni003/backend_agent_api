from mem0 import Memory, AsyncMemory
from openai import AsyncOpenAI
from supabase import Client
import os

def get_agent_clients():
    # Embedding client setup
    base_url = os.getenv('EMBEDDING_BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('EMBEDDING_API_KEY', 'no-api-key-provided')
    
    embedding_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # Supabase client setup
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    supabase = Client(supabase_url, supabase_key)

    return embedding_client, supabase

def get_mem0_config():
    # Get LLM provider and configuration
    llm_provider = os.getenv('LLM_PROVIDER')
    llm_api_key = os.getenv('LLM_API_KEY')
    llm_model = os.getenv('LLM_CHOICE')
    
    # Get embedder provider and configuration
    embedding_provider = os.getenv('EMBEDDING_PROVIDER')
    embedding_api_key = os.getenv('EMBEDDING_API_KEY')
    embedding_model = os.getenv('EMBEDDING_MODEL_CHOICE')
    
    # Initialize config dictionary
    config = {}
    
    # Configure LLM based on provider
    if llm_provider == 'openai' or llm_provider == 'openrouter':
        config["llm"] = {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "temperature": 0.2,
                "max_tokens": 2000,
            }
        }
        
        # Set API key in environment if not already set
        if llm_provider == 'openai' and llm_api_key:
            os.environ["OPENAI_API_KEY"] = llm_api_key
            
        # For OpenRouter, set the specific API key
        if llm_provider == 'openrouter' and llm_api_key:
            os.environ["OPENROUTER_API_KEY"] = llm_api_key
    
    elif llm_provider == 'ollama':
        config["llm"] = {
            "provider": "ollama",
            "config": {
                "model": llm_model,
                "temperature": 0.2,
                "max_tokens": 2000,
            }
        }
        
        # Set base URL for Ollama if provided
        llm_base_url = os.getenv('LLM_BASE_URL')
        if llm_base_url:
            config["llm"]["config"]["ollama_base_url"] = llm_base_url.replace("/v1", "")
    
    # Configure embedder based on provider
    if embedding_provider == 'openai':
        config["embedder"] = {
            "provider": "openai",
            "config": {
                "model": embedding_model or "text-embedding-3-small",
                "embedding_dims": 1536  # Default for text-embedding-3-small
            }
        }
        
        # Set API key in environment if not already set
        if embedding_api_key:
            os.environ["OPENAI_API_KEY"] = embedding_api_key
    
    elif embedding_provider == 'ollama':
        config["embedder"] = {
            "provider": "ollama",
            "config": {
                "model": embedding_model or "nomic-embed-text",
                "embedding_dims": 768  # Default for nomic-embed-text
            }
        }
        
        # Set base URL for Ollama if provided
        embedding_base_url = os.getenv('EMBEDDING_BASE_URL')
        if embedding_base_url:
            config["embedder"]["config"]["ollama_base_url"] = embedding_base_url.replace("/v1", "")
    
    # Configure Supabase vector store
    config["vector_store"] = {
        "provider": "supabase",
        "config": {
            "connection_string": os.environ.get('DATABASE_URL', ''),
            "collection_name": "mem0_memories",
            "embedding_model_dims": 1536 if embedding_provider == "openai" else 768
        }
    }

    return config
    
def get_mem0_client():
    # Create and return the Memory client
    config = get_mem0_config()
    return Memory.from_config(config)

async def get_mem0_client_async():
    # Create and return the Memory client
    config = get_mem0_config()    
    return await AsyncMemory.from_config(config)
