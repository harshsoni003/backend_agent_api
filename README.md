# Jeannie-ai-GobelFlakes Agent API

A production-ready FastAPI backend that serves the Pydantic AI agent with capabilities including Agentic RAG, long-term memory, web search, image analysis, and code execution. This API provides real-time streaming responses and conversation management for AI agent applications.

## Agent Capabilities

- **Agentic RAG**: Query your documents with context-aware intelligence (local or Google Drive files)
- **Long-term Memory**: The agent remembers important information from previous conversations
- **Web Search**: Search the internet using Brave API or SearXNG
- **Image Analysis**: Analyze images with vision-capable LLMs
- **Code Execution**: Generate and run Python code safely
- **Multi-LLM Support**: Works with OpenAI, OpenRouter, or local Ollama models
- **Real-time Streaming**: Stream AI responses in real-time
- **Conversation Management**: Store and retrieve conversation history

## Project Structure

```
backend_agent_api/
├── .env.example               # Example environment variables
├── requirements.txt           # Project dependencies
├── agent_api.py               # FastAPI backend with agent endpoints
├── agent.py                   # Main Pydantic AI agent implementation
├── clients.py                 # Client config for LLMs, databases, and long term memory
├── db_utils.py                # Database utility functions
├── prompt.py                  # System prompt template
├── tools.py                   # Agent tool implementations
├── PLANNING.md                # Development planning document
├── TASKS.md                   # Task tracking document
└── tests/                     # Test suite
    ├── __init__.py
    ├── conftest.py            # Test configuration
    ├── test_clients.py        # Client tests
    └── test_tools.py          # Tool tests
```

## Setup Instructions

### Prerequisites

- Docker (recommended) or Python 3.11+

#### Additional requirements based on your setup:

If running the agent locally:
- [Local AI package installed](https://github.com/coleam00/local-ai-packaged) (recommended)
- OR locally hosted: Supabase, Ollama, SearXNG

If not running the agent locally:
- Supabase project
- API keys for LLM provider (OpenAI or OpenRouter)
- Brave API key
- [Optional] Google Drive API credentials (if using Google Drive RAG pipeline)

### Docker Setup (Recommended)

1. **Create a `.env` file** with your configuration:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Build and run with Docker:**
   ```bash
   # Build the image
   docker build -t agent-api .
   
   # Run with environment variables from .env file
   docker run -d \
     --name agent-api \
     -p 8001:8001 \
     --env-file .env \
     agent-api
   ```

3. **Access the API:**
   - API endpoint: http://localhost:8001
   - Health check: http://localhost:8001/health
   - API docs: http://localhost:8001/docs

### Manual Environment Setup (Alternative)

1. Clone the repository (if not already done)

2. Create and activate a virtual environment:

```bash
# Navigate to the project directory
cd 6_Agent_Deployment/backend_agent_api

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. Create a `.env` file by copying the `.env.example` file:

```bash
# From the main project directory
copy .env.example .env
# On macOS/Linux:
# cp .env.example .env
```

4. Configure your `.env` file with the appropriate values:

### Environment Configuration

#### LLM Configuration

```
# The provider for your LLM (openai, openrouter, or ollama)
LLM_PROVIDER=openai

# Base URL for the OpenAI compatible instance
# OpenAI: https://api.openai.com/v1
# Ollama: http://localhost:11434/v1
# OpenRouter: https://openrouter.ai/api/v1
LLM_BASE_URL=https://api.openai.com/v1

# API key for your LLM provider
LLM_API_KEY=your_api_key_here

# The LLM model to use (must support tools/function calling)
# OpenAI example: gpt-4o-mini
# OpenRouter example: anthropic/claude-3.7-sonnet
# Ollama example: qwen2.5:14b-instruct-8k
LLM_CHOICE=gpt-4o-mini

# Vision LLM for image analysis
VISION_LLM_CHOICE=gpt-4o-mini
```

#### Embedding Configuration

```
# The provider for your embedding model (openai or ollama)
EMBEDDING_PROVIDER=openai

# Base URL for the embedding model
EMBEDDING_BASE_URL=https://api.openai.com/v1

# API key for your embedding model provider
EMBEDDING_API_KEY=your_api_key_here

# The embedding model to use
# OpenAI example: text-embedding-3-small
# Ollama example: nomic-embed-text
EMBEDDING_MODEL_CHOICE=text-embedding-3-small
```

#### Database Configuration

```
# Postgres DB URL for mem0 (long-term memory)
# Format: postgresql://[user]:[password]@[host]:[port]/[database_name]
DATABASE_URL=postgresql://postgres:password@localhost:5432/mydb

# Supabase configuration for RAG
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
```

#### Web Search Configuration

```
# Brave API key (leave empty if using SearXNG)
BRAVE_API_KEY=your_brave_api_key

# SearXNG endpoint (leave empty if using Brave)
# Local AI setup: http://localhost:8080 or http://searxng:8080 if in Docker
SEARXNG_BASE_URL=
```

#### Agent Observability (Optional)

LangFuse provides detailed observability into your agent's execution, including conversation tracking, performance metrics, and debugging insights. This is completely optional and primarily useful for production deployments.

```
# LangFuse configuration (optional - leave empty to disable)
# Get your keys from https://cloud.langfuse.com/ after creating a project
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

**To enable LangFuse observability:**
1. Create a free account at [https://cloud.langfuse.com/](https://cloud.langfuse.com/)
2. Create a new project and obtain your public and secret keys
3. Add the keys to your `.env` file

**To disable LangFuse (default):**
- Simply leave the `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` empty in your `.env` file
- The agent will run normally without any observability overhead

## Database Setup

### Cloud Implementation

1. Create a Supabase project at [https://supabase.com](https://supabase.com) if you haven't already done so for the n8n prototype.

2. Create the necessary tables by running the SQL scripts in the `..\sql` directory:
   - Navigate to the SQL Editor in Supabase
   - Run each of the following SQL scripts:
     - `sql/1-user_profiles_requests.sql`: Creates tables for user profiles and requests
     - `sql/2-user_profiles_requests_rls.sql`: Sets up row-level security for user data
     - `sql/3-conversations_messages.sql`: Creates tables for conversations and messages
     - `sql/4-conversations_messages_rls.sql`: Sets up row-level security for conversations   
     - `sql/5-documents.sql`: Creates the documents table with vector embeddings
     - `sql/6-document_metadata.sql`: Creates the document metadata table
     - `sql/7-document_rows.sql`: Creates the table for tabular data
     - `sql/8-execute_sql_rpc.sql`: Creates the RPC function for executing SQL queries

   **Note:** You must execute the `execute_sql_rpc.sql` script even if you followed along with the prototype. This creates a secure RPC function that allows the agent to execute read-only SQL queries against your document data.

### Local Implementation

If you're using the Local AI package or a self-hosted Supabase instance:

1. Navigate to the SQL Editor tab in Supabase (http://localhost:8000 for your Supabase dashboard by default)

2. Run the same SQL scripts mentioned above

   > **Important:** For local Ollama implementations using models like nomic-embed-text, you'll need to modify the vector dimensions in the SQL scripts from 1536 to 768 (or whatever the dimensions are for your embedding model) before running them.

## Running the API

### With Docker (Recommended)

```bash
# Check if container is running
docker ps | grep agent-api

# View logs
docker logs -f agent-api

# Stop the container
docker stop agent-api

# Start the container again
docker start agent-api

# Remove the container
docker rm agent-api

# Rebuild and run
docker build -t agent-api .
docker run -d --name agent-api -p 8001:8001 --env-file .env agent-api
```

### Without Docker (Manual)

1. Activate the virtual environment:
   ```bash
   # Navigate to the project directory
   cd 6_Agent_Deployment/backend_agent_api
   
   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   # source venv/bin/activate
   ```

2. Start the FastAPI server:
   ```bash
   uvicorn agent_api:app --reload --port 8001
   ```

The API will be available at `http://localhost:8001` by default. Port 8001 is specified because port 8000 is the default and that is taken by the Supabase dashboard if you are using the local AI package or just self-hosting Supabase.

### API Endpoints

- **POST `/api/pydantic-agent`**: Main endpoint for agent interactions
  - Supports real-time streaming of AI responses
  - Manages conversation history automatically
  - Generates conversation titles based on context

### Code Execution MCP Server Setup (Optional)

To enable code execution, you need to install Deno and run the MCP server:

1. Install Deno by following the instructions at [https://github.com/denoland/deno/](https://github.com/denoland/deno/)

2. Run the MCP server in a separate terminal:
   ```bash
   deno run -N -R=node_modules -W=node_modules --node-modules-dir=auto jsr:@pydantic/mcp-run-python sse
   ```

3. Uncomment the `mcp_servers=[code_execution_server]` line in `agent.py` to enable code execution 

4. Comment out the other code execution tool

## Key Features

- **Real-time Streaming**: The API streams responses in real-time for a better user experience
- **Automatic Title Generation**: Conversation titles are generated automatically based on the initial messages
- **Session Management**: Supports multiple conversations per user
- **Database Persistence**: All conversations and messages are stored in the database
- **Error Handling**: Comprehensive error handling and logging

## Troubleshooting

- **Vector Dimensions Mismatch**: Ensure the embedding dimensions in your database match the model you're using. OpenAI models typically use 1536 dimensions, while Ollama's nomic-embed-text uses 768.

- **Function Calling Support**: Not all models support function calling. If using Ollama, make sure to use a model that supports this feature (like Qwen).

- **Database Connection**: For any errors connecting to the database, you can check the logs directly in Supabase. For cloud Supabase, go to "Logs" -> Postgres. For local Supabase, check the logs for the supabase-kong container as well as the supabase-db container.

- **Port Conflicts**: If port 8001 is already in use, you can specify a different port when starting the server.