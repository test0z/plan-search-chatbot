# Backend

The backend provides a RESTful API for the chatbot, handling natural language processing, search, and responses through Azure OpenAI services.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Azure subscription with OpenAI service enabled
- uv
```bash
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/plan-search-chatbot.git
   cd plan-search-chatbot/app/backend
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file and add your Azure OpenAI credentials:
   ```
   AZURE_OPENAI_API_KEY=your-api-key-here
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2023-12-01-preview
   AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
   AZURE_OPENAI_QUERY_DEPLOYMENT_NAME=your-query-deployment-name

   # Google Search API Configuration
   GOOGLE_API_KEY=
   GOOGLE_CSE_ID=
   GOOGLE_MAX_RESULTS=10 # at most 10


   ```

3. Install backend dependencies using uv:
   ```bash
   uv pip install -e .
   ```
   
   For development dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```

### Running the Backend

Start the FastAPI server:
```bash
uv run run.py
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### Testing

The project includes a comprehensive test suite. To run the tests:

```bash
# Install test dependencies if you haven't already
uv pip install -e ".[dev]" 

# Run tests with verbose output
uv run pytest -v

```

### API Endpoints

- `GET /health` - Health check endpoint
- `POST /chat` - Chat endpoint for interacting with the AI assistant

### Testing with curl

You can enable streaming responses by setting the `stream` parameter to `true`:

```bash
# Test chat endpoint with streaming enabled
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "hello"
      }
    ],
    "max_tokens": 1500,
    "temperature": 0.7,
    "stream": true
  }' --no-buffer
```

When streaming is enabled, the response will be delivered as Server-Sent Events (SSE) with partial message chunks as they're generated. Each chunk is delivered in the format:

```
data: {"chunk": "Hello"}
data: {"chunk": "! How can I help you with Microsoft"}
data: {"chunk": " products today?"}
...
data: [DONE]
```

Your client code should parse these events to progressively display the response to the user.

## Azure OpenAI Configuration

This project uses Azure OpenAI services for the chatbot functionality. The configuration includes:

- **API Key**: Authentication for Azure OpenAI service
- **Endpoint**: Your Azure OpenAI service endpoint
- **Deployment Name**: The specific model deployment to use
- **API Version**: Azure OpenAI API version

Ensure your Azure OpenAI deployment has sufficient capacity and appropriate content safety settings for your use case.


