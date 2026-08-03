# Free Claude Code Gemini Bridge

A high performance bridge for running Claude Code using Gemini powered backends. This project provides a lean and lightweight pipeline optimized specifically for Google Gemini models.

## Project Status
The bridge is fully functional. It acts as a bridge for running Claude Code using Gemini powered backends. All redundant providers and legacy modules have been removed to ensure maximum performance and minimal latency.

## Architecture

The following diagram illustrates the request and response sequence within the proxy engine.

```mermaid
sequenceDiagram
    autonumber
    
    actor CLI as Claude Code
    
    box transparent Proxy Engine
        participant API as FastAPI
        participant Conv as Message Converter
        participant Provider as Target Provider
        participant Stream as SSE Builder
    end
    
    participant LLM as External LLM

    CLI->>API: POST /v1/messages (Anthropic Payload)
    API->>Conv: Intercept & flatten roles/tools
    Conv-->>API: Return standard OpenAI/Gemini schema
    
    API->>Provider: Initialize adapter execution
    Provider->>LLM: Async stream request (httpx)

    loop Real-Time Chunk Execution
        LLM-->>Provider: Yield raw delta chunks
        Provider->>Stream: Parse tokens & tool calls
        Stream-->>API: Emulate Anthropic SSE lifecycle
        API-->>CLI: Stream text/event-stream back
    end
```

### Data Flow Explanation
1. Request Interception: The Claude Code CLI sends an Anthropic-formatted POST request to the local server.
2. Schema Translation: The bridge maps Anthropic roles, instructions, and tool schemas into the standard Gemini format.
3. Provider Execution: The translated payload is dispatched to the configured Gemini API.
4. Streaming Translation: As the provider streams tokens, the bridge wraps them into the exact SSE format that the Anthropic CLI expects.
5. Tool Parsing: A heuristic parser extracts tool calls from the stream in real time to maintain interactivity.
6. Return Stream: The server returns a real-time SSE stream back to the Claude CLI.

## Setup
1. Configure your environment variables in the .env file.
2. Install dependencies using uv.
3. Start the server using the command python server.py.

## Usage
Once the server is running on localhost port 8082, point your Claude Code CLI to the local bridge:
export ANTHROPIC_BASE_URL=http://localhost:8082/v1
export ANTHROPIC_API_KEY=your_token
claude

## License
MIT
