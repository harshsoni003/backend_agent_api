from opentelemetry import trace
from dotenv import load_dotenv
import nest_asyncio
import logfire
import base64
import os

load_dotenv()

def scrubbing_callback(match: logfire.ScrubMatch):
    """Preserve the Langfuse session ID."""
    if (
        match.path == ("attributes", "langfuse.session.id")
        and match.pattern_match.group(0) == "session"
    ):
        # Return the original value to prevent redaction.
        return match.value

# Configure Langfuse for agent observability
def configure_langfuse():
    """
    Configure Langfuse for agent observability and tracing.
    
    Returns:
        trace.Tracer or None: A tracer instance if Langfuse is configured, None otherwise
    """
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    # If Langfuse credentials are not provided, return None
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        print("Langfuse credentials not found. Tracing disabled.")
        return None
    
    LANGFUSE_AUTH = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{LANGFUSE_HOST}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

    # Configure Logfire to work with Langfuse
    nest_asyncio.apply()
    logfire.configure(
        service_name='pydantic_ai_agent',
        send_to_logfire=False,
        scrubbing=logfire.ScrubbingOptions(callback=scrubbing_callback)
    )

    return trace.get_tracer("pydantic_ai_agent")