# Free-Claude-Code (Gemini Optimized)

A high-performance bridge for running Claude Code using Gemini-powered backends. 

### Key Optimizations
* **Handshake Fix**: Resolved 401 unauthorized errors during the initial connection.
* **Tool-Index Offset**: Fixed collision errors by offsetting tool indices (+2), ensuring stable streaming for Thinking, Text, and Tool blocks.
* **Minimalist Build**: Stripped unnecessary bloat for a leaner local inference environment.

### Setup
1. Configure your environment in `.env`.
2. Run the server: `python server.py`
